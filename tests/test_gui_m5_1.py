"""M5.1 GUI 自动化测试：真实 Tk 窗口/控件，导入/设置/状态/详情/拖拽/关闭。

用单一 Tk 解释器（session 级）+ 每用例 Toplevel，避免反复创建 Tk 解释器在 Windows 上
偶发的 init.tcl/tcl_findLibrary 初始化抖动。无可用显示环境时整文件跳过（不伪造通过）。
拖拽通过“调用真实 WM_DROPFILES 回调入口”（MainWindow._on_drop / ImportPanel.add_paths）
+ 子类化生命周期验证；物理 OS 拖拽另行人工核验。
"""

import sys

import pytest

tk = pytest.importorskip("tkinter")

from bookclean.app_controller import AppController, TaskSettings  # noqa: E402
from bookclean.task_events import TaskState  # noqa: E402
from bookclean.user_paths import UserPaths  # noqa: E402


def _make_image(path):
    from PIL import Image
    Image.new("RGB", (30, 20), (200, 120, 80)).save(path)
    return path


def _make_pdf(path, pages=2):
    import io
    import img2pdf
    from PIL import Image
    bufs = []
    for i in range(pages):
        b = io.BytesIO()
        Image.new("RGB", (30, 20), (i * 40, 100, 150)).save(b, format="PNG")
        bufs.append(b.getvalue())
    path.write_bytes(img2pdf.convert(*bufs))
    return path


@pytest.fixture
def win(tk_root, tmp_path):
    from bookclean.ui.main_window import MainWindow
    ctl = AppController(paths=UserPaths(root=tmp_path / "data"))
    toplevel = tk.Toplevel(tk_root)
    window = MainWindow(toplevel, controller=ctl)
    toplevel.update_idletasks()
    yield window
    # 清理：若仍有后台任务则协作取消并等待，避免关闭确认模态框阻塞测试。
    try:
        if window._running:
            window.controller.cancel_task()
            window.controller.wait(timeout=30)
        window._cancel_poll()
        if window._drop is not None:
            window._drop.disable()
        toplevel.destroy()
    except tk.TclError:
        pass


@pytest.fixture
def samples(tmp_path):
    return {
        "img": _make_image(tmp_path / "a.png"),
        "pdf": _make_pdf(tmp_path / "book.pdf", pages=2),
        "bad": (tmp_path / "bad.txt"),
    }


# ---- 主窗口状态 ---------------------------------------------------------

def test_initial_pending(win):
    assert win.state == TaskState.PENDING
    assert "disabled" in win.start_btn.state()
    assert "disabled" in win.cancel_btn.state()  # M5.1 不伪造运行中取消


def test_import_updates_readiness(win, samples):
    samples["bad"].write_text("x")
    win.import_panel.add_paths([samples["img"], samples["pdf"], samples["bad"]])
    win.root.update_idletasks()
    assert len(win.import_panel.tree.get_children()) == 3
    assert win.state == TaskState.READY
    assert "disabled" not in win.start_btn.state()
    assert "有效 2" in win.import_panel.summary.cget("text")


def test_error_row_tagged(win, samples):
    samples["bad"].write_text("x")
    win.import_panel.add_paths([samples["bad"]])
    win.root.update_idletasks()
    iid = win.import_panel.tree.get_children()[0]
    assert "error" in win.import_panel.tree.item(iid, "tags")


def test_clear_resets_pending(win, samples):
    win.import_panel.add_paths([samples["img"]])
    win.import_panel._clear()
    win.root.update_idletasks()
    assert len(win.import_panel.tree.get_children()) == 0
    assert win.state == TaskState.PENDING


def test_remove_selected(win, samples):
    win.import_panel.add_paths([samples["img"], samples["pdf"]])
    win.root.update_idletasks()
    win.import_panel.tree.selection_set("0")
    win.import_panel._remove_selected()
    win.root.update_idletasks()
    assert len(win.import_panel.tree.get_children()) == 1


# ---- 设置面板 -----------------------------------------------------------

def test_settings_sync(win):
    win.settings_panel.quality_var.set("紧凑（体积最小）")
    win.settings_panel._sync()
    assert win.controller.settings.compression == "compact"
    win.settings_panel.color_var.set("灰度")
    win.settings_panel._sync()
    assert win.controller.settings.color_mode == "grayscale"
    win.settings_panel.preset_var.set("旧书翻新")
    win.settings_panel._sync()
    assert win.controller.settings.preset == "old_book"


def test_border_toggle(win):
    win.settings_panel.border_var.set(False)
    win.settings_panel._sync()
    assert win.controller.settings.border_clean is False
    _, _, border = win.controller.build_configs()
    assert border is None


def test_output_dir_change(win, tmp_path, monkeypatch):
    target = tmp_path / "chosen_out"
    monkeypatch.setattr("bookclean.ui.import_panel.filedialog.askdirectory", lambda **k: str(target))
    win.import_panel._change_output_dir()
    assert win.controller.settings.output_dir == str(target)
    assert win.import_panel.out_var.get() == str(target)


# ---- 开始处理（预检，monkeypatch 掉模态框） ----------------------------

def test_start_launches_background_task(win, samples, tmp_path, monkeypatch):
    """M5.2：开始处理真正启动后台 worker，运行中锁定 UI，完成后复位并产出 PDF。"""
    import time
    monkeypatch.setattr("bookclean.ui.main_window.messagebox.showinfo", lambda *a, **k: None)
    monkeypatch.setattr("bookclean.ui.main_window.messagebox.showerror", lambda *a, **k: None)
    win.import_panel.add_paths([samples["img"]])
    win.controller.set_settings(TaskSettings(output_dir=str(tmp_path / "out")))
    win._on_start()
    assert win._running is True
    assert "disabled" in win.start_btn.state()
    assert "disabled" not in win.cancel_btn.state()
    deadline = time.monotonic() + 120
    while win._running and time.monotonic() < deadline:
        for ev in win.controller.poll_events():
            win._handle_event(ev)
        if win._running:
            time.sleep(0.02)
    assert win.state == TaskState.COMPLETED
    assert "disabled" not in win.start_btn.state()
    assert "disabled" in win.cancel_btn.state()
    assert list((tmp_path / "out").glob("*.pdf"))


def test_start_preflight_error(win, monkeypatch):
    shown = {}
    monkeypatch.setattr("bookclean.ui.main_window.messagebox.showerror",
                        lambda title, msg, **k: shown.update(title=title, msg=msg))
    win._on_start()  # 无输入
    assert "预检未通过" in win.status_var.get()
    assert shown.get("title") == "无法开始"


# ---- 详情对话框 ---------------------------------------------------------

def test_details_dialog(win, samples):
    win.import_panel.add_paths([samples["img"], samples["pdf"]])
    win._on_details()
    win.root.update_idletasks()
    tops = [w for w in win.root.winfo_children() if w.winfo_class() == "Toplevel"]
    assert len(tops) == 1
    texts = [w for w in tops[0].winfo_children() if w.winfo_class() == "Text"]
    assert texts and "任务详情" in texts[0].get("1.0", "end")
    tops[0].destroy()


# ---- 拖拽：回调入口 + 子类化生命周期 -----------------------------------

def test_drop_handler_adds_files(win, samples):
    win._on_drop([samples["img"], samples["pdf"]])  # 真实 WM_DROPFILES 最终调用此入口
    win.root.update_idletasks()
    assert len(win.import_panel.tree.get_children()) == 2
    assert win.state == TaskState.READY


def test_dnd_lifecycle(win):
    from bookclean.ui.dnd import enable_drop
    if sys.platform != "win32":
        assert enable_drop(win.root, lambda paths: None) is None
        return
    received = []
    target = enable_drop(win.root, lambda paths: received.extend(paths))
    assert target is not None  # 子类化成功
    assert target._enabled is True
    target.disable()
    assert target._enabled is False
    target.disable()  # 幂等，不崩溃


def test_close_disables_drop(win):
    drop = win._drop
    win._on_close()
    if drop is not None:
        assert drop._enabled is False


# ---- 高 DPI -------------------------------------------------------------

def test_dpi_scaling(win):
    from bookclean.ui.dpi import apply_scaling
    apply_scaling(win.root)
    scaling = float(win.root.tk.call("tk", "scaling"))
    assert scaling > 0