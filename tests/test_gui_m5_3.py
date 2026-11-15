"""M5.3 GUI 自动化测试：结果面板/友好错误/回退不计优化/打开按钮/页定位/查看结果。

真实 Tk（conftest 单一 session 解释器）。含真实内核端到端完成用例；结果面板渲染用例
直接驱动 show_result（确定性）。真实文件处理验证由控制器集成测试与真实演示覆盖，非模拟替代。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

from bookclean.app_controller import AppController, TaskSettings  # noqa: E402
from bookclean.task_events import TaskFailed, TaskState  # noqa: E402
from bookclean.user_paths import UserPaths  # noqa: E402


def _make_page(path: Path, idx: int) -> Path:
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(idx)
    arr = rng.integers(205, 245, size=(340, 460, 3), dtype=np.uint8)
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    for y in range(50, 290, 26):
        d.line([(40, y), (420, y)], fill=(25, 25, 25), width=2)
    im.save(path)
    return path


class _StubController:
    def __init__(self):
        self.opened: list[str] = []
        self.log_opened = 0

    def open_path(self, p):
        self.opened.append(str(p))
        return True

    def open_log(self):
        self.log_opened += 1
        return True

    def resolve_output_dir(self):
        return Path("/out")


def _result_dict(**over):
    base = {
        "task_id": "task_x", "manifest_exists": True, "total_pages": 3,
        "inputs": [{"name": "a.png", "kind": "image", "page_count": 1, "error": None}],
        "optimized": 2, "processed": 3, "fallbacks": 1, "failed": 0, "reused": 0,
        "fallback_breakdown": {"background": 0, "geometry": 1, "original": 0},
        "failed_pages": [],
        "fallback_pages": [{"display_index": 2, "level": "geometry", "reason": "text_core 0.75",
                            "source": "geometry", "image_path": "/x/processed_pages/page_000002.png",
                            "source_path": "/src2.png"}],
        "border": {"cleaned": 2, "skipped": 1, "failed": 0},
        "output_path": "/out/book.pdf", "output_exists": True, "output_size_bytes": 4191704,
        "export_status": "completed", "validation_status": "passed", "validation_issues": [],
        "color_mode": "preserve", "compression_mode": "balanced", "elapsed_seconds": 45.08,
        "report_path": "/out/export_report.json", "report_txt_path": "/out/export_report.txt",
        "log_path": "/logs/bookclean.log", "warnings": [],
    }
    base.update(over)
    return base


@pytest.fixture
def win(tk_root, tmp_path):
    from bookclean.ui.main_window import MainWindow
    ctl = AppController(paths=UserPaths(root=tmp_path / "data"))
    top = tk.Toplevel(tk_root)
    window = MainWindow(top, controller=ctl)
    top.update_idletasks()
    yield window
    try:
        if window._running:
            window.controller.cancel_task()
            window.controller.wait(timeout=30)
        window._cancel_poll()
        if window._drop is not None:
            window._drop.disable()
        if top.winfo_exists():
            top.destroy()
    except tk.TclError:
        pass


@pytest.fixture
def panels(tk_root):
    from bookclean.ui.result_panel import show_result
    created = []

    def _make(result, controller=None, terminal=None):
        w = show_result(tk_root, controller or _StubController(), result, terminal)
        created.append(w)
        tk_root.update_idletasks()
        return w

    yield _make
    for w in created:
        try:
            if w.winfo_exists():
                w.destroy()
        except tk.TclError:
            pass


def _silence(monkeypatch):
    for name in ("showinfo", "showerror", "askyesno"):
        monkeypatch.setattr(f"bookclean.ui.main_window.messagebox.{name}", lambda *a, **k: None)
    for name in ("showinfo", "showerror"):
        monkeypatch.setattr(f"bookclean.ui.result_panel.messagebox.{name}", lambda *a, **k: None)


def _import(win, tmp_path, n):
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    paths = [str(_make_page(src / f"p{i}.png", i)) for i in range(n)]
    win.import_panel.add_paths(paths)
    win.controller.set_settings(TaskSettings(output_dir=str(tmp_path / "out"), overwrite=True))


def _drive(win, timeout=180):
    deadline = time.monotonic() + timeout
    while win._running and time.monotonic() < deadline:
        win.root.update()
        time.sleep(0.01)


# ---- 真实端到端：完成 → 结果面板 ----------------------------------------

def test_completion_opens_result_panel(win, tmp_path, monkeypatch):
    _silence(monkeypatch)
    _import(win, tmp_path, 2)
    win._on_start()
    _drive(win)
    assert win.state == TaskState.COMPLETED
    assert win._result_win is not None and win._result_win.winfo_exists()
    assert "disabled" not in win.result_btn.state()
    summary = win._result_win.summary_text.get("1.0", "end")
    assert "完全优化" in summary
    assert win._last_result["output_exists"] is True
    assert win._last_result["validation_status"] == "passed"


def test_view_result_button_reopens_without_stacking(win, tmp_path, monkeypatch):
    _silence(monkeypatch)
    _import(win, tmp_path, 2)
    win._on_start()
    _drive(win)
    first = win._result_win
    win._on_show_result()
    assert win._result_win is not None and win._result_win.winfo_exists()
    assert not first.winfo_exists()  # 旧窗口被替换，不堆叠


# ---- 失败：友好提示，不甩 traceback --------------------------------------

def test_failed_event_friendly_error_no_traceback(win, monkeypatch):
    shown = {}
    monkeypatch.setattr("bookclean.ui.main_window.messagebox.showerror",
                        lambda title, msg, **k: shown.update(title=title, msg=msg))
    win._running = True
    win._set_running_ui(True)
    win._handle_event(TaskFailed(TaskState.EXPORTING, "ExportError: PDF 验证失败: size mismatch", "ExportError"))
    assert win.state == TaskState.FAILED
    assert shown["title"] == "处理失败"
    assert "建议" in shown["msg"]
    assert "size mismatch" in shown["msg"]
    assert "Traceback" not in shown["msg"]
    assert win._result_win is None  # task_dir 空 → 不打开结果面板


def test_failed_event_with_task_dir_opens_result(win, tmp_path, monkeypatch):
    _silence(monkeypatch)
    task_dir = tmp_path / "task_fail"
    task_dir.mkdir()
    manifest = {
        "task_id": "task_fail", "total_pages": 1, "inputs": [],
        "pages": [{"display_index": 1, "page_index": 0, "status": "completed",
                   "processing_status": "failed", "processing_error": "ValueError: bad",
                   "fallback_level": "none"}],
        "processing_summary": {"processed": 0, "fallbacks": 0, "failed": 1},
    }
    (task_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    win._running = True
    win._set_running_ui(True)
    win._handle_event(TaskFailed(TaskState.PROCESSING, "ValueError: bad", "ValueError", task_dir=str(task_dir)))
    assert win._result_win is not None and win._result_win.winfo_exists()
    assert win._last_result["failed"] == 1
    assert win._last_result["output_exists"] is False
    assert len(win._result_win.failed_tree.get_children()) == 1


# ---- 结果面板渲染（确定性） ----------------------------------------------

def test_result_panel_fallback_not_counted_optimized(panels):
    w = panels(_result_dict(), terminal=TaskState.COMPLETED)
    s = w.summary_text.get("1.0", "end")
    assert "完全优化：2" in s
    assert "回退：1" in s
    assert "几何 1" in s
    assert len(w.fallback_tree.get_children()) == 1
    assert len(w.failed_tree.get_children()) == 0


def test_result_panel_no_output_disables_pdf(panels):
    w = panels(_result_dict(output_exists=False, output_path="", validation_status=""),
               terminal=TaskState.CANCELLED)
    assert "disabled" in w.open_pdf_btn.state()
    assert "未生成正式 PDF" in w.summary_text.get("1.0", "end")


def test_result_panel_open_buttons_use_controller(panels, tmp_path):
    ctl = _StubController()
    report = tmp_path / "export_report.json"
    report.write_text("{}", encoding="utf-8")
    w = panels(_result_dict(report_path=str(report)), controller=ctl, terminal=TaskState.COMPLETED)
    w.open_pdf_btn.invoke()
    w.open_folder_btn.invoke()
    w.open_report_btn.invoke()
    w.open_log_btn.invoke()
    assert "/out/book.pdf" in ctl.opened
    assert str(Path("/out/book.pdf").parent) in ctl.opened
    assert str(report) in ctl.opened
    assert ctl.log_opened == 1


def test_result_panel_page_selection_shows_detail(panels):
    w = panels(_result_dict(), terminal=TaskState.COMPLETED)
    tree = w.fallback_tree
    children = tree.get_children()
    tree.selection_set(children[0])
    tree.event_generate("<<TreeviewSelect>>")
    w.update_idletasks()
    detail = w.detail_text.get("1.0", "end")
    assert "第 2 页" in detail
    assert "text_core 0.75" in detail
    assert "几何回退" in detail