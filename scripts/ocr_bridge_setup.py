# -*- coding: utf-8 -*-
"""M6.5 PP-OCRv5 桥接配置：探测用户既有 DeepSeekOCR_v2 安装 → 生成桥接插件清单。

0 下载：引擎运行时与模型全部复用用户既有安装（只读）。本脚本只读取对方项目
（校验文件存在 + 计算模型 sha256 + 只读 import 验证），绝不修改其任何文件
（import 验证设 PYTHONDONTWRITEBYTECODE=1，不生成 __pycache__）。

用法:
  python scripts/ocr_bridge_setup.py [--source D:/LLM-OCR/DeepSeekOCR_v2]
         [--python <解释器>] [--models-dir <模型目录>] [--plugins-root temp/plugins]
默认自动探测：<source> 的上级目录下的 .venv-gpu/Scripts/python.exe 与 models/。
退出码: 0 配置成功 / 1 校验失败 / 2 未找到既有安装或参数无效。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bookclean.ocr.plugin_manager import PLUGIN_SCHEMA, PROTOCOL_VERSION, default_plugins_root

DEFAULT_SOURCE = Path("D:/LLM-OCR/DeepSeekOCR_v2")
MODEL_NAMES = ("PP-OCRv5_server_det", "PP-OCRv5_server_rec")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_python(source: Path) -> Path | None:
    for cand in (source.parent / ".venv-gpu" / "Scripts" / "python.exe",
                 source.parent / ".venv" / "Scripts" / "python.exe",
                 source / ".venv-gpu" / "Scripts" / "python.exe"):
        if cand.is_file():
            return cand
    return None


def _detect_models(source: Path) -> Path | None:
    for cand in (source.parent / "models", source / "models"):
        if all((cand / n / "inference.pdiparams").is_file() for n in MODEL_NAMES):
            return cand
    return None


def _paddleocr_version(python_exe: Path) -> str:
    try:
        out = subprocess.run(
            [str(python_exe), "-c",
             "from importlib.metadata import version; print(version('paddleocr'))"],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        v = out.stdout.strip()
        if out.returncode == 0 and v:
            return v
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _import_check(python_exe: Path, source: Path) -> tuple[bool, str]:
    """只读 import 验证（不加载模型、不初始化引擎、不写任何文件）。"""
    code = (
        "import sys; sys.dont_write_bytecode = True; "
        f"sys.path.insert(0, {str(source)!r}); "
        "from src import ocr_engine; "
        "assert hasattr(ocr_engine, 'init_paddle') and hasattr(ocr_engine, 'run_ocr'); "
        "print('IMPORT_OK')"
    )
    try:
        out = subprocess.run(
            [str(python_exe), "-c", code],
            capture_output=True, text=True, timeout=180,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if out.returncode == 0 and "IMPORT_OK" in out.stdout:
        return True, ""
    tail = (out.stderr or out.stdout or "").strip().splitlines()
    return False, " | ".join(tail[-3:]) or f"exit={out.returncode}"


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="配置 PP-OCRv5 桥接插件（复用既有安装，0 下载）")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE),
                        help="DeepSeekOCR_v2 项目目录（含 src/ocr_engine.py）")
    parser.add_argument("--python", default=None, help="对方项目的 Python 解释器（默认自动探测 .venv-gpu）")
    parser.add_argument("--models-dir", default=None, help="PP-OCRv5 server 模型目录（默认自动探测）")
    parser.add_argument("--plugins-root", default=None, help="BookClean 插件根目录（默认用户数据目录/plugins）")
    parser.add_argument("--skip-import-check", action="store_true",
                        help="跳过只读 import 验证（不推荐）")
    args = parser.parse_args(argv)

    source = Path(args.source).resolve()
    if not (source / "src" / "ocr_engine.py").is_file():
        print(f"错误: 未找到既有安装（{source}/src/ocr_engine.py 不存在）。"
              "桥接模式要求本机已安装 DeepSeekOCR_v2；可用 --source 指定路径。", file=sys.stderr)
        return 2
    python_exe = Path(args.python).resolve() if args.python else _detect_python(source)
    if python_exe is None or not python_exe.is_file():
        print(f"错误: 未找到对方项目的 Python 解释器（探测 {source.parent}/.venv-gpu 等）。"
              "可用 --python 指定。", file=sys.stderr)
        return 2
    models_dir = Path(args.models_dir).resolve() if args.models_dir else _detect_models(source)
    if models_dir is None:
        print("错误: 未找到 PP-OCRv5 server 模型（det/rec 的 inference.pdiparams）。"
              "可用 --models-dir 指定。桥接模式不下载模型。", file=sys.stderr)
        return 2

    print(f"既有安装: {source}")
    print(f"解释器:   {python_exe}")
    print(f"模型目录: {models_dir}")

    if not args.skip_import_check:
        ok, detail = _import_check(python_exe, source)
        if not ok:
            print(f"错误: 只读 import 验证失败: {detail}", file=sys.stderr)
            return 1
        print("只读 import 验证: 通过（init_paddle/run_ocr 可用，未写任何文件）")

    version = _paddleocr_version(python_exe)
    models = []
    for name in MODEL_NAMES:
        f = models_dir / name / "inference.pdiparams"
        models.append({
            "id": name.lower().replace("_", "-"),
            "path": str(f),
            "sha256": _sha256(f),
            "size_bytes": f.stat().st_size,
        })
        print(f"模型 {name}: {f.stat().st_size / 1e6:.1f}MB sha256 已记录")

    plugins_root = Path(args.plugins_root) if args.plugins_root else default_plugins_root()
    plugin_dir = plugins_root / "ocr-paddle-bridge" / version
    plugin_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": PLUGIN_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "engine_id": "ocr-paddle-bridge",
        "engine_version": version,
        "params_schema_version": 1,
        "python": str(python_exe),
        "worker": "paddle-bridge",
        "models": models,
        "extra": {"source_dir": str(source), "models_dir": str(models_dir)},
        "notes": "Read-only bridge to existing DeepSeekOCR_v2 installation (0 download); "
                 "CPU only; DeepSeek cloud proofreading path never imported/called",
    }
    mf = plugin_dir / "plugin.json"
    tmp = mf.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, mf)
    print(f"桥接插件已配置: {mf}")
    print("在 BookClean 设置中选择引擎“PP-OCRv5 桥接”或保持“自动”，点击“检测引擎”确认。")
    print("注意: PP-OCRv5 server 峰值内存约 5.7GB（实测），初始化约 11s；仅建议密集中文页选用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())