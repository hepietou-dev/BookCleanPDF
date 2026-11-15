"""打包（frozen）冒烟验证：经真实 AppController→TaskWorker→内核 全链路跑小任务。

不导入 Tkinter，可在 --windowed 无控制台环境下 headless 运行。M5.5 用于验证打包后
二进制（numpy/cv2/PIL/pypdfium2/pikepdf/lxml/img2pdf）与 frozen 路径解析真实可用。
复用与 GUI 完全相同的应用服务路径，不复制任何处理逻辑；原始输入只读（SHA256 核验）。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .app_controller import AppController, TaskSettings
from .task_events import TERMINAL_EVENTS, TaskCancelled, TaskCompleted, TaskFailed
from .user_paths import UserPaths, is_frozen

_MATRIX = [("preserve", "balanced"), ("grayscale", "balanced"),
           ("preserve", "compact"), ("preserve", "lossless")]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _make_pages(dir_path: Path, n: int = 3) -> list[Path]:
    """生成合成书页（泛黄背景+文本行+彩色插图+顶部黑边），触发分析/处理/保色/黑边。"""
    from PIL import Image, ImageDraw

    dir_path.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(n):
        img = Image.new("RGB", (1240, 1754), (245, 242, 232))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 1239, 55], fill=(18, 18, 18))
        y = 200
        for line in range(18):
            w = 700 + (line * 37) % 320
            d.rectangle([150, y, 150 + w, y + 22], fill=(40, 38, 36))
            y += 62
        d.rectangle([150, 1320, 620, 1520], fill=(70, 110, 160))
        p = dir_path / f"smoke_page_{i + 1:03d}.png"
        img.save(p, dpi=(300, 300))
        paths.append(p)
    return paths


def _drain_terminal(ctl: AppController, timeout: float = 300.0):
    """排空事件队列直到终止事件或 worker 结束（headless，不触碰 Tkinter）。"""
    deadline = time.monotonic() + timeout
    terminal = None
    while time.monotonic() < deadline:
        for ev in ctl.poll_events():
            if isinstance(ev, TERMINAL_EVENTS):
                terminal = ev
        if terminal is not None or not ctl.is_running():
            break
        time.sleep(0.05)
    for ev in ctl.poll_events():
        if isinstance(ev, TERMINAL_EVENTS):
            terminal = ev
    ctl.wait(60)
    return terminal


def _run_one(ctl: AppController, inputs: list[Path], color: str, compression: str,
             out_dir: Path) -> dict:
    ctl.clear_inputs()
    ctl.inspect_inputs(inputs)
    ctl.set_settings(TaskSettings(preset="auto", color_mode=color, compression=compression,
                                  border_clean=True, output_dir=str(out_dir), overwrite=True))
    vr = ctl.validate_start()
    if not vr.ok:
        return {"ok": False, "color": color, "compression": compression,
                "stage": "validate", "errors": vr.errors}
    t0 = time.monotonic()
    output = ctl.start_task()
    if output is None:
        return {"ok": False, "color": color, "compression": compression, "stage": "start"}
    terminal = _drain_terminal(ctl)
    ctl.clear_active()
    elapsed = round(time.monotonic() - t0, 2)
    base = {"color": color, "compression": compression, "elapsed_seconds": elapsed}
    if isinstance(terminal, TaskCompleted):
        view = ctl.load_result(terminal.task_dir, terminal.output_path)
        base.update({
            "terminal": "completed", "task_dir": terminal.task_dir,
            "output_path": terminal.output_path,
            "output_size_bytes": view.get("output_size_bytes"),
            "validation_status": view.get("validation_status"),
            "total_pages": view.get("total_pages"), "processed": view.get("processed"),
            "fallbacks": view.get("fallbacks"), "optimized": view.get("optimized"),
            "state": view.get("state"),
            "ok": bool(view.get("output_exists")) and view.get("validation_status") == "passed",
        })
        return base
    if isinstance(terminal, TaskFailed):
        base.update({"ok": False, "terminal": "failed", "error": terminal.error,
                     "exc_type": terminal.exc_type})
        return base
    if isinstance(terminal, TaskCancelled):
        base.update({"ok": False, "terminal": "cancelled"})
        return base
    base.update({"ok": False, "terminal": "timeout"})
    return base


def _reopen_render(pdf_path: str, expected: int = 3) -> dict:
    """用 pypdfium2 重开正式 PDF 并渲染首页，核验页数/可渲染（打包后 pdfium DLL 可用）。"""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    try:
        n = len(pdf)
        bmp = pdf[0].render(scale=1.0)
        pil = bmp.to_pil()
        return {"ok": n == expected and pil.size[0] > 0, "page_count": n, "render_size": list(pil.size)}
    finally:
        pdf.close()


def _make_text_page(dir_path: Path) -> Path:
    """生成含真实字形的中英混排页（供 OCR 冒烟；用系统字体，frozen 下可读）。"""
    from PIL import Image, ImageDraw, ImageFont

    dir_path.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1240, 1754), (255, 255, 255))
    d = ImageDraw.Draw(img)
    font = None
    for fp in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\arial.ttf"):
        try:
            font = ImageFont.truetype(fp, 42)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    lines = [
        "BookClean PDF searchable text layer test.",
        "The quick brown fox jumps over the lazy dog.",
        "中文正文测试：扫描优化后叠加可搜索文字层。",
        "第二章 1848 年至 1850 年的法兰西阶级斗争。",
        "Mixed 中英混排 line 1234567890.",
    ]
    y = 220
    for ln in lines:
        d.text((120, y), ln, fill=(0, 0, 0), font=font)
        y += 100
    p = dir_path / "ocr_smoke_page.png"
    img.save(p, dpi=(300, 300))
    return p


def _run_ocr_smoke(ctl: AppController, data_root: Path) -> dict:
    """frozen OCR 全链路冒烟：真实字形页 → OCR 插件 worker → 可搜索 PDF → 提取文字。

    证明打包后 app 能用插件独立 Python 启动 worker（M7.5：rapid_worker.py 已落盘）。
    需先把 ocr-rapid 插件装到 default_plugins_root()（frozen=%LOCALAPPDATA%/BookCleanPDF/plugins）。
    """
    import pypdfium2 as pdfium

    page = _make_text_page(data_root / "ocr_smoke_input")
    out_dir = data_root / "ocr_smoke_output"
    ctl.clear_inputs()
    ctl.inspect_inputs([page])
    ctl.set_settings(TaskSettings(preset="auto", color_mode="preserve", compression="balanced",
                                  border_clean=False, ocr_enabled=True, ocr_engine="ocr-rapid",
                                  output_dir=str(out_dir), overwrite=True))
    vr = ctl.validate_start()
    if not vr.ok:
        return {"ok": False, "stage": "validate", "errors": vr.errors}
    t0 = time.monotonic()
    if ctl.start_task() is None:
        return {"ok": False, "stage": "start"}
    term = _drain_terminal(ctl, timeout=300.0)
    ctl.clear_active()
    elapsed = round(time.monotonic() - t0, 2)
    if not isinstance(term, TaskCompleted):
        return {"ok": False, "terminal": type(term).__name__,
                "error": getattr(term, "error", None)}
    view = ctl.load_result(term.task_dir, term.output_path)
    extracted = 0
    try:
        pdf = pdfium.PdfDocument(term.output_path)
        try:
            for i in range(len(pdf)):
                extracted += len(pdf[i].get_textpage().get_text_bounded().strip())
        finally:
            pdf.close()
    except Exception as exc:
        return {"ok": False, "stage": "extract", "error": f"{type(exc).__name__}: {exc}"}
    ocr_report = {}
    rp = Path(term.task_dir) / "ocr_report.json"
    if rp.exists():
        ocr_report = json.loads(rp.read_text(encoding="utf-8"))
    return {
        "ok": bool(view.get("output_exists")) and view.get("validation_status") == "passed"
        and extracted > 0 and ocr_report.get("failed", 1) == 0,
        "terminal": "completed", "elapsed_seconds": elapsed,
        "output_path": term.output_path,
        "validation_status": view.get("validation_status"),
        "ocr_completed": ocr_report.get("completed"),
        "ocr_failed": ocr_report.get("failed"),
        "ocr_text_chars": ocr_report.get("total_text_chars"),
        "extracted_chars": extracted,
        "searchable": extracted > 0,
    }


def run_smoke(root: str | Path | None = None) -> dict:
    """运行打包冒烟矩阵 + 重开渲染 + 续跑复用 + 异常文件 + frozen 路径核验。

    root 仅供测试覆盖数据目录；默认用 UserPaths()（frozen 时=%LOCALAPPDATA%/BookCleanPDF）。
    结果写 <data_root>/smoke_result.json，并返回该 dict。
    """
    paths = UserPaths(root=root) if root is not None else UserPaths()
    paths.ensure_dirs()
    data_root = paths.data_root
    input_dir = data_root / "smoke_input"
    out_dir = data_root / "smoke_output"
    # 默认 3 页（冒烟）；BOOKCLEAN_SMOKE_PAGES 可放大为长任务以实测 frozen 峰值内存，默认行为不变。
    import os
    n_pages = max(1, int(os.environ.get("BOOKCLEAN_SMOKE_PAGES", "3")))
    pages = _make_pages(input_dir, n_pages)
    before = {p.name: _sha256(p) for p in pages}

    ctl = AppController(paths=paths)
    runs = [_run_one(ctl, pages, color, comp, out_dir) for color, comp in _MATRIX]

    reopen = {}
    resume = {}
    first = next((r for r in runs if r.get("ok") and r.get("terminal") == "completed"), None)
    if first:
        reopen = _reopen_render(first["output_path"], len(pages))
        # 续跑复用：删除正式输出模拟崩溃 → resume 应复用全部已验证页，绝不重跑导入
        out_pdf = Path(first["output_path"])
        if out_pdf.exists():
            out_pdf.unlink()
        t0 = time.monotonic()
        resumed = ctl.resume_task(first["task_dir"])
        if resumed is None:
            resume = {"ok": False, "reason": "resume_task 返回 None"}
        else:
            term = _drain_terminal(ctl)
            ctl.clear_active()
            if isinstance(term, TaskCompleted):
                view = ctl.load_result(term.task_dir, term.output_path)
                proc_report = {}
                pr = Path(term.task_dir) / "processing_report.json"
                if pr.exists():
                    proc_report = json.loads(pr.read_text(encoding="utf-8"))
                summary = proc_report.get("summary", proc_report)
                resume = {
                    "ok": bool(view.get("output_exists")) and view.get("validation_status") == "passed",
                    "state": view.get("state"), "reused": summary.get("reused"),
                    "newly_processed": summary.get("processed"),
                    "output_size_bytes": view.get("output_size_bytes"),
                    "elapsed_seconds": round(time.monotonic() - t0, 2),
                }
            else:
                resume = {"ok": False, "terminal": type(term).__name__}

    # 异常文件：损坏 PDF 应被逐文件捕获为 error 行，不中断
    corrupt = data_root / "smoke_corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.4 this is not a valid pdf body\n%%EOF")
    ctl.clear_inputs()
    infos = ctl.inspect_inputs([corrupt])
    error_file = {"ok": any(i.error for i in infos),
                  "errors": [i.error for i in infos if i.error]}

    after = {p.name: _sha256(p) for p in pages}
    originals_unchanged = before == after

    import os
    localappdata = os.environ.get("LOCALAPPDATA", "")
    frozen_paths_ok = (not is_frozen()) or (
        bool(localappdata) and str(data_root).startswith(localappdata))

    ocr_smoke = {}
    if os.environ.get("BOOKCLEAN_SMOKE_OCR") == "1":
        ocr_smoke = _run_ocr_smoke(ctl, data_root)

    ok = (all(r.get("ok") for r in runs) and reopen.get("ok", False)
          and resume.get("ok", False) and error_file["ok"] and originals_unchanged
          and frozen_paths_ok)
    if ocr_smoke:
        ok = ok and ocr_smoke.get("ok", False)

    result = {
        "ok": ok,
        "frozen": is_frozen(),
        "data_root": str(data_root),
        "tasks_root": str(paths.tasks_root),
        "logs_dir": str(paths.logs_dir),
        "frozen_paths_ok": frozen_paths_ok,
        "originals_unchanged": originals_unchanged,
        "runs": runs,
        "reopen_render": reopen,
        "resume_reuse": resume,
        "error_file": error_file,
    }
    if ocr_smoke:
        result["ocr_smoke"] = ocr_smoke
    (data_root / "smoke_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result