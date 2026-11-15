"""M5.2 GUI 自动化测试：真实 Tk 窗口 + 真实后台 worker + root.after 轮询。

覆盖：开始锁定 UI/真实进度/完成复位、取消按钮、运行中关闭确认（同意/拒绝）、
刷新不重跑、终止事件处理、回退页不计为完全优化。无显示环境时整文件跳过。
进度通过 root.update() 驱动真实 after 回调链（_poll→_handle_event），非模拟。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

from bookclean.app_controller import AppController, TaskSettings  # noqa: E402
from bookclean.task_events import (  # noqa: E402
    PageProgress, StageSummary, TaskCompleted, TaskFailed, TaskState,
)
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


def _silence(monkeypatch):
    for name in ("showinfo", "showerror", "askyesno"):
        monkeypatch.setattr(f"bookclean.ui.main_window.messagebox.{name}", lambda *a, **k: None)


# ---- 开始 → 真实进度 → 完成复位 ----------------------------------------

def test_start_locks_ui_and_completes(win, tmp_path, monkeypatch):
    _silence(monkeypatch)
    _import(win, tmp_path, 3)
    win._on_start()
    assert win._running is True
    assert "disabled" in win.start_btn.state()
    assert "disabled" not in win.cancel_btn.state()
    assert "disabled" in win.import_panel._buttons[0].state()
    assert str(win.settings_panel._combos[0].cget("state")) == "disabled"

    seen_progress = False
    deadline = time.monotonic() + 180
    while win._running and time.monotonic() < deadline:
        win.root.update()
        if float(win.progress_bar.cget("value")) > 0:
            seen_progress = True
        time.sleep(0.01)
    assert win.state == TaskState.COMPLETED
    assert seen_progress, "进度条从未反映真实进度"
    assert "disabled" not in win.start_btn.state()
    assert "disabled" in win.cancel_btn.state()
    assert "disabled" not in win.import_panel._buttons[0].state()
    assert str(win.settings_panel._combos[0].cget("state")) == "readonly"
    assert list((tmp_path / "out").glob("*.pdf"))


def test_cancel_button_stops_task(win, tmp_path, monkeypatch):
    _silence(monkeypatch)
    _import(win, tmp_path, 8)
    win._on_start()
    assert win._running
    win._on_cancel()
    assert "正在取消" in win.status_var.get()
    _drive(win)
    assert win.state == TaskState.CANCELLED
    assert not win._running
    assert "disabled" not in win.start_btn.state()
    assert not list((tmp_path / "out").glob("*.pdf"))


# ---- 运行中关闭 ---------------------------------------------------------

def test_close_while_running_confirms_and_cancels(win, tmp_path, monkeypatch):
    _silence(monkeypatch)
    monkeypatch.setattr("bookclean.ui.main_window.messagebox.askyesno", lambda *a, **k: True)
    _import(win, tmp_path, 8)
    win._on_start()
    assert win._running
    top = win.root
    win._on_close()
    assert not win.controller.is_running()
    assert not top.winfo_exists()


def test_close_while_running_declined_keeps_window(win, tmp_path, monkeypatch):
    _silence(monkeypatch)
    monkeypatch.setattr("bookclean.ui.main_window.messagebox.askyesno", lambda *a, **k: False)
    _import(win, tmp_path, 8)
    win._on_start()
    win._on_close()  # 拒绝 → 不销毁
    assert win.root.winfo_exists()
    assert win._running
    win.controller.cancel_task()
    _drive(win)
    assert win.state == TaskState.CANCELLED


# ---- 刷新不重跑 ---------------------------------------------------------

def test_refresh_does_not_rerun(win, tmp_path, monkeypatch):
    _silence(monkeypatch)
    _import(win, tmp_path, 2)
    win._on_start()
    _drive(win)
    assert win.state == TaskState.COMPLETED
    tasks_root = win.controller.paths.tasks_root
    before = len(list(tasks_root.glob("task_*")))
    pdfs_before = list((tmp_path / "out").glob("*.pdf"))
    for _ in range(20):
        win._update_readiness()
        win._poll()
        win.root.update()
    assert len(list(tasks_root.glob("task_*"))) == before
    assert list((tmp_path / "out").glob("*.pdf")) == pdfs_before
    assert not win.controller.is_running()


# ---- 终止事件处理（确定性注入） ----------------------------------------

def test_failed_event_handler(win, monkeypatch):
    shown = {}
    monkeypatch.setattr("bookclean.ui.main_window.messagebox.showerror",
                        lambda title, msg, **k: shown.update(title=title, msg=msg))
    win._running = True
    win._set_running_ui(True)
    win._handle_event(TaskFailed(TaskState.EXPORTING, "ExportError: boom", "ExportError"))
    assert win.state == TaskState.FAILED
    assert not win._running
    assert "disabled" not in win.start_btn.state()
    assert shown.get("title") == "处理失败"
    assert "boom" in shown.get("msg", "")


def test_completed_event_handler(win, monkeypatch):
    shown = {}
    monkeypatch.setattr("bookclean.ui.main_window.messagebox.showinfo",
                        lambda title, msg, **k: shown.update(title=title, msg=msg))
    win._running = True
    win._set_running_ui(True)
    win._handle_event(TaskCompleted("X:/out.pdf", {"output_size_bytes": 1234}))
    assert win.state == TaskState.COMPLETED
    assert not win._running
    assert shown.get("title") == "处理完成"


# ---- 回退页不计为完全优化 ----------------------------------------------

def test_fallback_summary_not_counted_optimized(win):
    win._handle_event(StageSummary(TaskState.PROCESSING,
                                   {"processed": 4, "fallbacks": 4, "failed": 0, "reused": 0}))
    text = win.counts_var.get()
    assert "优化 0" in text
    assert "回退 4" in text


def test_live_progress_counters_and_bar(win):
    win._handle_event(PageProgress(TaskState.PROCESSING, done=3, total=4,
                                   counters={"processed": 1, "fallbacks": 2, "failed": 0}))
    assert float(win.progress_bar.cget("value")) == pytest.approx(75.0)
    text = win.counts_var.get()
    assert "优化 1" in text and "回退 2" in text and "3/4" in text