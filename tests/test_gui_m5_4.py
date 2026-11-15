"""M5.4 GUI 自动化测试：历史面板渲染/恢复按钮启用/重开/续跑接线 + 结果页恢复按钮。

真实 Tk（conftest 单一 session 解释器 + 每用例 Toplevel）。历史面板渲染用确定性 stub；
主窗口续跑/重开用真实内核端到端（不模拟处理）。所有断言基于真实任务目录与派生视图。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

from bookclean.app_controller import AppController, TaskSettings  # noqa: E402
from bookclean.task_events import TaskState  # noqa: E402
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


def _silence(monkeypatch):
    for name in ("showinfo", "showerror", "askyesno"):
        monkeypatch.setattr(f"bookclean.ui.main_window.messagebox.{name}", lambda *a, **k: None)
    for name in ("showinfo", "showerror"):
        monkeypatch.setattr(f"bookclean.ui.history_panel.messagebox.{name}", lambda *a, **k: None)
        monkeypatch.setattr(f"bookclean.ui.result_panel.messagebox.{name}", lambda *a, **k: None)


class _HistStub:
    def __init__(self, entries):
        self._entries = entries

    def get_history(self):
        return self._entries


def _entries():
    return [
        {"task_dir": "/t/a", "label": "A", "state": "incomplete", "resumable": True,
         "total_pages": 3, "created_at": "2026-11-13T01:02:03+00:00", "output_path": "/out/a.pdf"},
        {"task_dir": "/t/b", "label": "B", "state": "completed", "resumable": False,
         "total_pages": 2, "created_at": "2026-11-13T01:00:00+00:00", "output_path": "/out/b.pdf"},
        {"task_dir": "/t/c", "label": "C", "state": "missing", "resumable": False,
         "total_pages": 0, "created_at": "2026-11-13T00:59:00+00:00", "output_path": ""},
    ]


@pytest.fixture
def hist(tk_root):
    from bookclean.ui.history_panel import show_history
    created = []

    def _make(entries=None, on_resume=None, on_reopen=None):
        w = show_history(tk_root, _HistStub(entries if entries is not None else _entries()),
                         on_resume=on_resume, on_reopen=on_reopen)
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


def _disabled(btn):
    return "disabled" in btn.state()


# ---- 历史面板渲染与按钮启用（确定性） ------------------------------------

def test_history_renders_entries(hist):
    w = hist()
    assert len(w.history_tree.get_children()) == 3
    first = w.history_tree.item(w.history_tree.get_children()[0], "values")
    assert first[0] == "A" and first[1] == "未完成" and str(first[2]) == "3"


def test_history_resume_button_enablement(hist):
    w = hist()
    tree = w.history_tree
    rows = tree.get_children()
    tree.selection_set(rows[0]); tree.event_generate("<<TreeviewSelect>>")  # incomplete
    assert not _disabled(w.resume_btn) and not _disabled(w.reopen_btn)
    tree.selection_set(rows[1]); tree.event_generate("<<TreeviewSelect>>")  # completed
    assert _disabled(w.resume_btn) and not _disabled(w.reopen_btn)
    tree.selection_set(rows[2]); tree.event_generate("<<TreeviewSelect>>")  # missing
    assert _disabled(w.resume_btn) and _disabled(w.reopen_btn)


def test_history_resume_and_reopen_callbacks(hist):
    resumed, reopened = [], []
    w = hist(on_resume=lambda td: resumed.append(td),
             on_reopen=lambda td, out: reopened.append((td, out)))
    tree = w.history_tree
    rows = tree.get_children()
    tree.selection_set(rows[0]); tree.event_generate("<<TreeviewSelect>>")
    w.resume_btn.invoke()
    assert resumed == ["/t/a"]
    tree.selection_set(rows[1]); tree.event_generate("<<TreeviewSelect>>")
    w.reopen_btn.invoke()
    assert reopened == [("/t/b", "/out/b.pdf")]


def test_history_empty(hist):
    w = hist(entries=[])
    assert len(w.history_tree.get_children()) == 0
    assert _disabled(w.resume_btn) and _disabled(w.reopen_btn)


# ---- 结果页恢复按钮（M5.4） ----------------------------------------------

def test_result_panel_resume_button(tk_root):
    from bookclean.ui.result_panel import show_result
    called = []
    result = {"task_dir": "/t/a", "task_id": "a", "resumable": True, "output_exists": False,
              "output_path": "", "total_pages": 3, "optimized": 1, "processed": 1, "fallbacks": 0,
              "failed": 0, "fallback_breakdown": {}, "failed_pages": [], "fallback_pages": [],
              "border": {}, "validation_status": "", "validation_issues": [], "inputs": [],
              "elapsed_seconds": 1.0, "report_path": "", "warnings": []}
    w = show_result(tk_root, _HistStub([]), result, TaskState.CANCELLED,
                    on_resume=lambda td: called.append(td))
    tk_root.update_idletasks()
    assert w.resume_btn is not None
    w.resume_btn.invoke()
    assert called == ["/t/a"]
    assert not w.winfo_exists()  # 续跑后关闭结果窗


def test_result_panel_no_resume_when_completed(tk_root):
    from bookclean.ui.result_panel import show_result
    result = {"task_dir": "/t/b", "resumable": False, "output_exists": True, "output_path": "/out/b.pdf",
              "total_pages": 2, "optimized": 2, "processed": 2, "fallbacks": 0, "failed": 0,
              "fallback_breakdown": {}, "failed_pages": [], "fallback_pages": [], "border": {},
              "validation_status": "passed", "validation_issues": [], "inputs": [],
              "elapsed_seconds": 1.0, "report_path": "", "warnings": []}
    w = show_result(tk_root, _HistStub([]), result, TaskState.COMPLETED, on_resume=lambda td: None)
    tk_root.update_idletasks()
    assert w.resume_btn is None
    w.destroy()


# ---- 主窗口：历史/重开/续跑接线（真实内核） ------------------------------

def test_main_window_history_opens(win, tmp_path, monkeypatch):
    _silence(monkeypatch)
    _import(win, tmp_path, 2)
    win._on_start()
    _drive(win)
    assert win.state == TaskState.COMPLETED
    win._on_history()
    assert win._history_win is not None and win._history_win.winfo_exists()
    assert len(win._history_win.history_tree.get_children()) >= 1


def test_main_window_reopen_completed(win, tmp_path, monkeypatch):
    _silence(monkeypatch)
    _import(win, tmp_path, 2)
    win._on_start()
    _drive(win)
    task_dir = win._last_result["task_dir"]
    out = win._last_result["output_path"]
    win._on_reopen(task_dir, out)
    assert win._result_win is not None and win._result_win.winfo_exists()
    assert win._last_result["state"] == "completed"
    assert win._last_terminal == TaskState.COMPLETED


def test_main_window_resume_after_export_loss(win, tmp_path, monkeypatch):
    _silence(monkeypatch)
    _import(win, tmp_path, 3)
    win._on_start()
    _drive(win)
    assert win.state == TaskState.COMPLETED
    task_dir = Path(win._last_result["task_dir"])
    out = Path(win._last_result["output_path"])
    # 模拟处理完成后崩溃：删除正式 PDF 与报告
    out.unlink(missing_ok=True)
    (out.parent / "export_report.json").unlink(missing_ok=True)
    (out.parent / "export_report.txt").unlink(missing_ok=True)

    win._on_resume(str(task_dir))
    assert win._running is True
    _drive(win)
    assert win.state == TaskState.COMPLETED
    r = win._last_result
    assert r["output_exists"] is True and r["processed"] == 3 and r["reused"] == 3
    assert r["validation_status"] == "passed"


def test_main_window_resume_guard_while_running(win, tmp_path, monkeypatch):
    _silence(monkeypatch)
    _import(win, tmp_path, 8)
    win._on_start()
    assert win._running is True
    # 运行中恢复被守卫拒绝（不抛异常，状态不变）
    win._on_resume(str(tmp_path / "whatever"))
    assert win._running is True
    win.controller.cancel_task()
    _drive(win)