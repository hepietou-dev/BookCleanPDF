"""M6.4 GUI 集成测试：OCR 设置组/引擎管理/进度阶段/结果统计/历史快照/隔离。

真实 Tk（conftest 单一 session 解释器）+ 每用例 Toplevel。结果视图用真实任务目录 manifest
派生（非模拟）。隔离断言：OCR 关时构造 GUI/导入控制器绝不导入 bookclean.ocr；引擎状态
查询为惰性导入（仅用户交互时）。无可用显示环境时 GUI 用例跳过（不伪造通过）。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

tk = pytest.importorskip("tkinter")

from bookclean.app_controller import AppController, TaskSettings  # noqa: E402
from bookclean.task_events import PageProgress, StageSummary, TaskState  # noqa: E402
from bookclean.task_result import build_result_view  # noqa: E402
from bookclean.user_paths import UserPaths  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def win(tk_root, tmp_path):
    from bookclean.ui.main_window import MainWindow
    ctl = AppController(paths=UserPaths(root=tmp_path / "data"))
    toplevel = tk.Toplevel(tk_root)
    window = MainWindow(toplevel, controller=ctl)
    toplevel.update_idletasks()
    yield window
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


# ---- 设置组 -------------------------------------------------------------

def test_settings_ocr_group_default_off(win):
    sp = win.settings_panel
    assert sp.ocr_enabled_var.get() is False
    assert win.controller.settings.ocr_enabled is False
    assert win.controller.settings.ocr_engine == "auto"
    assert str(sp.ocr_combo.cget("state")) == "disabled"  # 未勾选时引擎下拉禁用


def test_settings_ocr_toggle_syncs_controller(win):
    sp = win.settings_panel
    sp.ocr_enabled_var.set(True)
    sp._on_ocr_toggle()
    assert win.controller.settings.ocr_enabled is True
    assert str(sp.ocr_combo.cget("state")) == "readonly"


def test_settings_ocr_engine_mapping(win):
    sp = win.settings_panel
    sp.ocr_enabled_var.set(True)
    sp.ocr_engine_var.set("RapidOCR")
    sp._sync()
    assert win.controller.settings.ocr_engine == "ocr-rapid"
    sp.ocr_engine_var.set("自动（推荐）")
    sp._sync()
    assert win.controller.settings.ocr_engine == "auto"


def test_settings_ocr_status_refresh_detects_plugin(win):
    sp = win.settings_panel
    sp.ocr_enabled_var.set(True)
    sp._refresh_ocr_status()
    text = sp.ocr_status_var.get()
    # 开发环境默认插件目录含真实 ocr-rapid 插件
    assert "已就绪" in text and "ocr-rapid" in text


def test_settings_ocr_manage_opens_dir(win, monkeypatch):
    sp = win.settings_panel
    opened = []
    monkeypatch.setattr(win.controller, "open_path", lambda p: opened.append(str(p)) or True)
    monkeypatch.setattr("bookclean.ui.settings_panel.messagebox.showinfo", lambda *a, **k: None)
    sp.ocr_enabled_var.set(True)
    sp._on_manage_engines()
    assert opened and "plugins" in opened[0]


def test_settings_set_enabled_locks_ocr(win):
    sp = win.settings_panel
    sp.ocr_enabled_var.set(True)
    sp._update_ocr_widgets()
    sp.set_enabled(False)
    assert str(sp.ocr_combo.cget("state")) == "disabled"
    assert "disabled" in sp._ocr_check.state()
    sp.set_enabled(True)
    assert str(sp.ocr_combo.cget("state")) == "readonly"


def test_settings_collect_includes_ocr(win):
    sp = win.settings_panel
    sp.ocr_enabled_var.set(True)
    sp.ocr_engine_var.set("RapidOCR")
    s = sp.collect()
    assert s.ocr_enabled is True and s.ocr_engine == "ocr-rapid"


# ---- 进度阶段 -----------------------------------------------------------

def test_main_window_ocr_running_state():
    from bookclean.ui.main_window import _RUNNING, _STATE_LABEL
    assert TaskState.OCR_RECOGNIZING in _RUNNING
    assert _STATE_LABEL[TaskState.OCR_RECOGNIZING] == "OCR 识别"


def test_main_window_ocr_progress_counts(win):
    ev = PageProgress(TaskState.OCR_RECOGNIZING, 3, 10, counters={"completed": 3, "failed": 1, "skipped": 0})
    win._counters = ev.counters
    win._update_counts(ev)
    text = win.counts_var.get()
    assert "OCR 识别 3" in text and "失败 1" in text and "3/10" in text


def test_main_window_ocr_summary_reuse(win):
    ev = StageSummary(TaskState.OCR_RECOGNIZING,
                      {"completed": 8, "failed": 0, "skipped": 0, "reused": 5, "total_text_chars": 1234})
    win._update_summary_counts(ev)
    text = win.counts_var.get()
    assert "复用已识别 5 页" in text and "1234" in text


def test_main_window_details_includes_ocr(win):
    win.controller.set_settings(TaskSettings(ocr_enabled=True, ocr_engine="ocr-rapid"))
    win._on_details()
    # 详情对话框为最新 Toplevel，含 OCR 设置行
    tops = [w for w in win.root.winfo_children() if isinstance(w, tk.Toplevel)]
    detail = tops[-1]
    texts = [w for w in detail.winfo_children() if isinstance(w, tk.Text)]
    body = texts[0].get("1.0", "end")
    assert "可搜索 PDF（OCR）：开" in body and "ocr-rapid" in body
    detail.destroy()


# ---- 结果视图（真实 manifest 派生） -------------------------------------

def _write_manifest(task_dir: Path, ocr_summary=None, ocr_status="", ocr_failed_page=False):
    task_dir.mkdir(parents=True, exist_ok=True)
    pages = [{"display_index": 1, "page_index": 0, "status": "completed",
              "processing_status": "completed", "image_path": "original_pages/p1.png"}]
    if ocr_failed_page:
        pages[0]["ocr_status"] = "failed"
        pages[0]["ocr_error"] = "WORKER_CRASHED"
    manifest = {"task_id": "t", "total_pages": 1, "pages": pages,
                "processing_summary": {"elapsed_seconds": 2.0},
                "inputs": [{"path": "a.png", "kind": "image", "page_count": 1}]}
    if ocr_summary is not None:
        manifest["ocr_summary"] = ocr_summary
    if ocr_status:
        manifest["ocr_status"] = ocr_status
    (task_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


_OCR_SUMMARY = {"total_pages": 1, "completed": 1, "failed": 0, "skipped": 0, "reused": 0,
                "total_text_chars": 390, "coverage": 1.0, "engine_id": "ocr-rapid",
                "engine_version": "1.4.4", "restarts": 0, "elapsed_seconds": 3.5, "avg_page_ms": 3500.0}


def test_result_view_ocr_block(tmp_path):
    td = tmp_path / "task"
    _write_manifest(td, _OCR_SUMMARY, "completed")
    view = build_result_view(td)
    ocr = view["ocr"]
    assert ocr["enabled"] is True and ocr["status"] == "completed"
    assert ocr["engine_id"] == "ocr-rapid" and ocr["completed"] == 1
    assert ocr["total_text_chars"] == 390 and ocr["coverage"] == 1.0
    assert view["elapsed_seconds"] == pytest.approx(5.5)  # 2.0 处理 + 3.5 OCR


def test_result_view_ocr_failed_page(tmp_path):
    td = tmp_path / "task"
    _write_manifest(td, {**_OCR_SUMMARY, "completed": 0, "failed": 1}, "completed_with_errors",
                    ocr_failed_page=True)
    view = build_result_view(td)
    assert view["failed"] == 1
    assert any(p["stage"] == "ocr" and p["reason"] == "WORKER_CRASHED" for p in view["failed_pages"])


def test_result_view_ocr_off(tmp_path):
    td = tmp_path / "task"
    _write_manifest(td)
    view = build_result_view(td)
    assert view["ocr"]["enabled"] is False
    assert view["failed"] == 0


def test_result_view_ocr_degrade(tmp_path):
    td = tmp_path / "task"
    _write_manifest(td, {**_OCR_SUMMARY, "completed": 0, "skipped": 1, "engine_id": "",
                         "reason": "未安装 OCR 插件"}, "unavailable")
    view = build_result_view(td)
    assert view["ocr"]["status"] == "unavailable"
    assert "未安装" in view["ocr"]["reason"]


# ---- 结果面板 -----------------------------------------------------------

def _result_dict(**over):
    base = {
        "task_id": "t", "manifest_exists": True, "total_pages": 1, "inputs": [],
        "optimized": 1, "processed": 1, "fallbacks": 0, "failed": 0, "reused": 0,
        "fallback_breakdown": {"background": 0, "geometry": 0, "original": 0},
        "failed_pages": [], "fallback_pages": [],
        "border": {"cleaned": 0, "skipped": 0, "failed": 0},
        "ocr": {"enabled": False},
        "output_path": "/out/book.pdf", "output_exists": True, "output_size_bytes": 100,
        "export_status": "completed", "validation_status": "passed", "validation_issues": [],
        "color_mode": "preserve", "compression_mode": "balanced", "elapsed_seconds": 5.5,
        "report_path": "/out/export_report.json", "report_txt_path": "/out/export_report.txt",
        "warnings": [],
    }
    base.update(over)
    return base


class _StubController:
    def open_path(self, p):
        return True

    def open_log(self):
        return True

    def resolve_output_dir(self):
        return Path("/out")


def test_result_panel_ocr_stats(tk_root):
    from bookclean.ui.result_panel import show_result
    top = tk.Toplevel(tk_root)
    result = _result_dict(ocr={"enabled": True, "status": "completed", "engine_id": "ocr-rapid",
                               "engine_version": "1.4.4", "completed": 1, "failed": 0, "skipped": 0,
                               "reused": 0, "total_text_chars": 390, "coverage": 1.0,
                               "elapsed_seconds": 3.5, "reason": ""})
    win = show_result(top, _StubController(), result, TaskState.COMPLETED)
    body = win.summary_text.get("1.0", "end")
    assert "OCR 文字层：completed" in body and "ocr-rapid 1.4.4" in body and "390" in body
    win.destroy()
    top.destroy()


def test_result_panel_ocr_off_line(tk_root):
    from bookclean.ui.result_panel import show_result
    top = tk.Toplevel(tk_root)
    win = show_result(top, _StubController(), _result_dict(), TaskState.COMPLETED)
    body = win.summary_text.get("1.0", "end")
    assert "OCR 文字层：关" in body
    win.destroy()
    top.destroy()


def test_result_panel_ocr_failed_stage_label(tk_root):
    from bookclean.ui.result_panel import show_result
    top = tk.Toplevel(tk_root)
    result = _result_dict(failed=1, failed_pages=[
        {"display_index": 1, "stage": "ocr", "reason": "WORKER_CRASHED",
         "image_path": "/x/p1.png", "source_path": "/a.png"}])
    win = show_result(top, _StubController(), result, TaskState.COMPLETED)
    rows = [win.failed_tree.item(i)["values"] for i in win.failed_tree.get_children()]
    assert any("OCR" in str(r) for r in rows)
    win.destroy()
    top.destroy()


# ---- 历史面板 OCR 列 ----------------------------------------------------

class _HistoryController:
    def get_history(self):
        return [{"task_dir": "/t1", "label": "book", "state": "completed", "resumable": False,
                 "total_pages": 3, "created_at": "2026-11-13T10:00:00", "output_path": "/out.pdf",
                 "settings": {"ocr_enabled": True, "ocr_engine": "ocr-rapid"}},
                {"task_dir": "/t2", "label": "plain", "state": "completed", "resumable": False,
                 "total_pages": 2, "created_at": "2026-11-13T11:00:00", "output_path": "/out2.pdf",
                 "settings": {"ocr_enabled": False}}]


def test_history_ocr_column(tk_root):
    from bookclean.ui.history_panel import show_history
    top = tk.Toplevel(tk_root)
    win = show_history(top, _HistoryController())
    rows = [win.history_tree.item(i)["values"] for i in win.history_tree.get_children()]
    assert rows[0][3] == "开" and rows[1][3] == "关"
    win.destroy()
    top.destroy()


# ---- 控制器引擎管理助手 -------------------------------------------------

def test_controller_ocr_status_installed():
    ctl = AppController()
    status = ctl.ocr_plugin_status("auto")
    assert status["installed"] is True and status["engine_id"] == "ocr-rapid"


def test_controller_ocr_status_missing(tmp_path, monkeypatch):
    import bookclean.ocr.plugin_manager as pm
    monkeypatch.setattr(pm, "default_plugins_root", lambda: tmp_path / "empty")
    ctl = AppController()
    status = ctl.ocr_plugin_status("auto")
    assert status["installed"] is False and status["reason"]


def test_controller_install_help_text():
    text = AppController.ocr_install_help()
    assert "离线" in text and "ocr-rapid" in text and "plugin.json" in text


def test_controller_open_plugins_dir(monkeypatch):
    ctl = AppController()
    opened = []
    monkeypatch.setattr(ctl, "open_path", lambda p: opened.append(str(p)) or True)
    assert ctl.open_plugins_dir() is True
    assert opened and "plugins" in opened[0]


# ---- 隔离（OCR 关不导入任何 OCR 代码；引擎查询惰性） ---------------------

def test_import_controller_and_gui_no_cr_when_off():
    code = (
        "import sys; sys.path.insert(0, 'src');"
        "import bookclean.app_controller, bookclean.task_result;"
        "import bookclean.ui.settings_panel, bookclean.ui.main_window;"
        "import bookclean.ui.result_panel, bookclean.ui.history_panel;"
        "bad=[m for m in sys.modules if m.startswith('bookclean.ocr')];"
        "assert not bad, bad; print('OK')"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_ocr_status_query_is_lazy_import():
    code = (
        "import sys; sys.path.insert(0, 'src');"
        "from bookclean.app_controller import AppController;"
        "assert not [m for m in sys.modules if m.startswith('bookclean.ocr')];"
        "c=AppController(); c.ocr_plugin_status('auto');"
        "assert any(m.startswith('bookclean.ocr') for m in sys.modules);"
        "print('OK')"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout