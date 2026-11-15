"""BookClean PDF 程序入口。

用法:
    python app.py             启动桌面 GUI（M5.1：导入/配置/预检/输出目录/详情）
    python app.py --diagnose  运行环境诊断（依赖、功能链、磁盘、写权限）
    python app.py --smoke     打包冒烟验证（全链路小任务，headless，写 smoke_result.json）
    python app.py --version   显示版本
"""

import argparse
import importlib
import importlib.metadata
import json
import platform
import shutil
import struct
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bookclean import __version__  # noqa: E402
from bookclean.logging_setup import setup_logging  # noqa: E402
from bookclean.user_paths import UserPaths  # noqa: E402

RUNTIME_PACKAGES = ("numpy", "opencv-python-headless", "Pillow", "pypdfium2", "img2pdf")


def _check_environment() -> list[tuple[str, bool, str]]:
    checks = []
    bits = struct.calcsize("P") * 8
    ok = sys.version_info[:2] == (3, 12) and bits == 64
    checks.append((
        "Python 3.12 x64",
        ok,
        f"{platform.python_version()} {bits}bit @ {sys.executable}",
    ))
    checks.append(("操作系统", True, f"{platform.system()} {platform.release()} {platform.machine()}"))
    return checks


def _check_packages() -> list[tuple[str, bool, str]]:
    checks = []
    for name in RUNTIME_PACKAGES:
        mod_name = {"opencv-python-headless": "cv2", "Pillow": "PIL"}.get(name, name)
        try:
            # 先验证真实可导入（功能性判据）；打包后 dist-info 元数据可能不在 bundle 内，
            # 版本号尽力而为（metadata→模块 __version__→bundled），不因缺元数据误报失败。
            module = importlib.import_module(mod_name)
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                version = getattr(module, "__version__", None) or "bundled"
            checks.append((f"依赖 {name}", True, version))
        except Exception as exc:
            checks.append((f"依赖 {name}", False, f"{type(exc).__name__}: {exc}"))
    return checks


def _check_functional_chain() -> list[tuple[str, bool, str]]:
    """内存中验证核心链路: Pillow 图像 -> img2pdf 封装 -> pypdfium2 渲染 -> NumPy/OpenCV。"""
    try:
        import io

        import cv2
        import img2pdf
        import numpy as np
        import pypdfium2 as pdfium
        from PIL import Image

        img = Image.new("RGB", (32, 24), (205, 185, 140))
        png = io.BytesIO()
        img.save(png, format="PNG", dpi=(300, 300))
        pdf_bytes = img2pdf.convert(png.getvalue())
        pdf = pdfium.PdfDocument(pdf_bytes)
        try:
            if len(pdf) != 1:
                return [("功能链 PDF 封装/渲染", False, f"页数异常: {len(pdf)}")]
            bitmap = pdf[0].render(scale=300 / 72)
            arr = np.asarray(bitmap.to_pil())
        finally:
            pdf.close()
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        ok = gray.shape == (24, 32) and 0 < int(gray.mean()) < 255
        return [(
            "功能链 PDF 封装/渲染",
            ok,
            f"img2pdf {len(pdf_bytes)}B -> pypdfium2 渲染 {arr.shape} -> OpenCV 灰度均值 {gray.mean():.1f}",
        )]
    except Exception as exc:
        return [("功能链 PDF 封装/渲染", False, f"{type(exc).__name__}: {exc}")]


def _check_gui_toolkit() -> list[tuple[str, bool, str]]:
    try:
        import tkinter

        return [("Tkinter 可用", True, f"Tcl/Tk {tkinter.TkVersion}")]
    except Exception as exc:
        return [("Tkinter 可用", False, f"{type(exc).__name__}: {exc}")]


def _check_storage() -> list[tuple[str, bool, str]]:
    paths = UserPaths()
    checks = []
    free_gb = shutil.disk_usage(paths.data_root).free / 1024**3
    checks.append(("磁盘剩余空间", free_gb > 2.0, f"{free_gb:.1f} GB 可用 ({paths.data_root.drive})"))
    for target in (paths.tasks_root, paths.logs_dir):
        try:
            target.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target, prefix="m0_", suffix=".tmp", delete=True):
                pass
            checks.append((f"写权限 {target.name}/", True, str(target)))
        except Exception as exc:
            checks.append((f"写权限 {target.name}/", False, f"{type(exc).__name__}: {exc}"))
    return checks


def run_diagnostics() -> dict:
    checks = (
        _check_environment()
        + _check_packages()
        + _check_functional_chain()
        + _check_gui_toolkit()
        + _check_storage()
    )
    return {"ok": all(ok for _, ok, _ in checks), "checks": checks}


def _launch_gui(logger) -> int:
    """默认入口：启动桌面 GUI。失败时给出可操作的提示而非静默崩溃。"""
    try:
        from bookclean.ui import run_gui
    except Exception as exc:  # 导入期失败（如缺少 Tk）
        logger.error("GUI 模块导入失败: %s", exc)
        print(f"无法启动图形界面：{exc}\n请运行 `python app.py --diagnose` 检查环境。")
        return 1
    try:
        return run_gui()
    except Exception as exc:
        logger.exception("GUI 运行失败")
        print(f"图形界面运行出错：{exc}\n请运行 `python app.py --diagnose` 检查环境。")
        return 1


def _ensure_std_streams() -> None:
    """PyInstaller --windowed 下 sys.stdout/stderr 可能为 None。

    此时把标准流重定向到用户数据目录的 console.log，使 --diagnose/--smoke 输出与
    traceback 可被捕获而非崩溃（架构 §9 风险 5）。开发环境/有控制台时不改动。
    """
    if not getattr(sys, "frozen", False):
        return
    if sys.stdout is not None and sys.stderr is not None:
        return
    log_dir = UserPaths().logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    stream = open(log_dir / "console.log", "a", encoding="utf-8", errors="replace")
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


def _run_smoke() -> int:
    """打包冒烟：经真实应用服务路径跑全链路小任务，结果写 smoke_result.json。"""
    from bookclean.frozen_smoke import run_smoke

    result = run_smoke()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("SMOKE_OK" if result["ok"] else "SMOKE_FAIL")
    return 0 if result["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    _ensure_std_streams()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="bookclean", description="BookClean PDF — 本地扫描书清理与图像型 PDF 导出")
    parser.add_argument("--diagnose", action="store_true", help="运行环境诊断")
    parser.add_argument("--smoke", action="store_true", help="打包冒烟验证（全链路小任务，headless）")
    parser.add_argument("--version", action="version", version=f"BookClean PDF {__version__}")
    args = parser.parse_args(argv)

    logger = setup_logging(UserPaths().logs_dir)
    logger.info("BookClean PDF %s 启动, argv=%s", __version__, sys.argv[1:])

    if args.smoke:
        return _run_smoke()

    if args.diagnose:
        result = run_diagnostics()
        print(f"BookClean PDF {__version__} 环境诊断")
        for name, ok, detail in result["checks"]:
            print(f"  [{'通过' if ok else '失败'}] {name}: {detail}")
        print("诊断结果:", "全部通过" if result["ok"] else "存在失败项")
        logger.info("环境诊断: %s", "全部通过" if result["ok"] else "存在失败项")
        return 0 if result["ok"] else 1

    return _launch_gui(logger)


if __name__ == "__main__":
    sys.exit(main())