"""M5.4 控制器恢复/历史集成测试：真实内核端到端，不模拟处理。

覆盖架构 §10 M5.4 关键项：未完成续跑、页复用（recovery 8 维）、崩溃恢复（导入即登记索引）、
已完成重开、已完成/缺失拒绝恢复、重复启动守卫、再导入不覆盖既有任务数据、设置持久化。
所有断言基于真实生成的任务目录/processed_pages/processing_report/PDF/验证报告。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from bookclean.app_controller import AppController, TaskSettings
from bookclean.task_events import StageChanged, TaskCancelled, TaskCompleted, TaskFailed, TaskState
from bookclean.user_paths import UserPaths


def _make_page(path: Path, idx: int) -> Path:
    rng = np.random.default_rng(idx)
    arr = rng.integers(205, 245, size=(360, 480, 3), dtype=np.uint8)
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    for y in range(50, 300, 26):
        d.line([(40, y), (440, y)], fill=(25, 25, 25), width=2)
    im.save(path)
    return path


def _controller(tmp_path, n=3):
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    inputs = [str(_make_page(src / f"p{i}.png", i)) for i in range(n)]
    ctl = AppController(paths=UserPaths(root=tmp_path / "data"))
    ctl.inspect_inputs(inputs)
    ctl.set_settings(TaskSettings(output_dir=str(tmp_path / "out"), overwrite=True))
    return ctl


def _run_to_terminal(ctl, timeout=180):
    deadline = time.monotonic() + timeout
    terminal = None
    while time.monotonic() < deadline:
        for ev in ctl.poll_events():
            if isinstance(ev, (TaskCompleted, TaskFailed, TaskCancelled)):
                terminal = ev
        if terminal is not None:
            break
        time.sleep(0.03)
    ctl.wait(timeout=20)
    return terminal


def _complete(ctl):
    out = ctl.start_task()
    term = _run_to_terminal(ctl)
    assert isinstance(term, TaskCompleted), term
    ctl.clear_active()
    return Path(term.task_dir), Path(out)


def _drop_output(out: Path):
    """模拟处理完成后、导出前崩溃：删除正式 PDF 与导出报告（任务目录页面数据保留）。"""
    out.unlink(missing_ok=True)
    (out.parent / "export_report.json").unlink(missing_ok=True)
    (out.parent / "export_report.txt").unlink(missing_ok=True)


def _proc_report(task_dir: Path) -> dict:
    return json.loads((task_dir / "processing_report.json").read_text(encoding="utf-8"))


# ---- 未完成续跑 + 页复用（崩溃后重导出，全部复用） ------------------------

def test_resume_reuses_all_pages_after_export_loss(tmp_path):
    ctl = _controller(tmp_path, n=3)
    task_dir, out = _complete(ctl)
    assert ctl.load_task(task_dir, out)["state"] == "completed"
    originals = sorted(p.name for p in (task_dir / "original_pages").glob("*.png"))

    _drop_output(out)  # 崩溃：无正式输出
    view = ctl.load_task(task_dir, out)
    assert view["state"] == "incomplete" and view["resumable"] is True

    resume_out = ctl.resume_task(task_dir)
    assert resume_out == str(out)  # 沿用历史目标输出
    term = _run_to_terminal(ctl)
    assert isinstance(term, TaskCompleted), term
    ctl.clear_active()

    rep = _proc_report(task_dir)
    assert rep["reused"] == 3 and rep["processed"] == 0  # 全部复用，未重处理
    r = ctl.load_task(task_dir, resume_out)
    assert r["processed"] == 3 and r["reused"] == 3  # 逐页总数正确（非 summary 少计）
    assert r["output_exists"] is True and r["validation_status"] == "passed"
    assert r["state"] == "completed"
    # 原件只读：original_pages 未被续跑改动
    assert sorted(p.name for p in (task_dir / "original_pages").glob("*.png")) == originals


# ---- 页复用 + 缺失页重处理（recovery 8 维真实校验） ----------------------

def test_resume_reprocesses_missing_processed_page(tmp_path):
    ctl = _controller(tmp_path, n=3)
    task_dir, out = _complete(ctl)
    # 删除第 3 页处理结果 → recovery 判不可复用 → 续跑重处理该页，其余复用
    (task_dir / "processed_pages" / "page_000003.png").unlink()
    _drop_output(out)
    assert ctl.load_task(task_dir, out)["resumable"] is True

    resume_out = ctl.resume_task(task_dir)
    assert resume_out is not None
    term = _run_to_terminal(ctl)
    assert isinstance(term, TaskCompleted), term
    ctl.clear_active()

    rep = _proc_report(task_dir)
    assert rep["reused"] == 2 and rep["processed"] == 1
    assert (task_dir / "processed_pages" / "page_000003.png").exists()  # 已重建
    r = ctl.load_task(task_dir, resume_out)
    assert r["processed"] == 3 and r["output_exists"] is True


# ---- 取消 → 恢复 → 完成（崩溃/取消恢复，导入即登记索引） ------------------

def test_cancel_then_resume_completes(tmp_path):
    ctl = _controller(tmp_path, n=6)
    out = ctl.start_task()
    deadline = time.monotonic() + 60
    cancelled = False
    terminal = None
    while time.monotonic() < deadline:
        for ev in ctl.poll_events():
            if isinstance(ev, StageChanged) and ev.state == TaskState.PROCESSING and not cancelled:
                ctl.cancel_task()
                cancelled = True
            if isinstance(ev, (TaskCompleted, TaskFailed, TaskCancelled)):
                terminal = ev
        if terminal is not None:
            break
        time.sleep(0.02)
    ctl.wait(timeout=20)
    assert cancelled and isinstance(terminal, TaskCancelled), terminal
    task_dir = Path(terminal.task_dir)
    ctl.clear_active()

    # 导入即登记索引：取消后历史已有该任务，且可恢复
    hist = ctl.get_history()
    entry = next(e for e in hist if e["task_dir"] == str(task_dir))
    assert entry["state"] == "cancelled" and entry["resumable"] is True

    resume_out = ctl.resume_task(task_dir)
    assert resume_out is not None
    term2 = _run_to_terminal(ctl)
    assert isinstance(term2, TaskCompleted), term2
    ctl.clear_active()
    r = ctl.load_task(task_dir, resume_out)
    assert r["processed"] == 6 and r["output_exists"] is True and r["state"] == "completed"


# ---- 已完成重开 / 拒绝恢复 ----------------------------------------------

def test_completed_reopen_and_resume_refused(tmp_path):
    ctl = _controller(tmp_path, n=2)
    task_dir, out = _complete(ctl)
    r = ctl.load_task(task_dir, out)  # 重开：只读派生
    assert r["state"] == "completed" and r["output_exists"] is True
    assert r["resumable"] is False
    assert ctl.resume_task(task_dir) is None  # 已完成不可续跑


def test_resume_refuses_missing_task(tmp_path):
    ctl = _controller(tmp_path, n=1)
    assert ctl.resume_task(tmp_path / "no_such_task") is None


def test_resume_guard_while_running(tmp_path):
    ctl = _controller(tmp_path, n=8)
    ctl.start_task()
    try:
        assert ctl.is_running()
        assert ctl.resume_task(tmp_path / "whatever") is None  # 重复启动守卫
    finally:
        ctl.cancel_task()
        ctl.wait(timeout=20)
        ctl.clear_active()


# ---- 再导入不覆盖既有任务数据 --------------------------------------------

def test_reimport_does_not_overwrite_previous_task(tmp_path):
    ctl = _controller(tmp_path, n=2)
    task_dir1, out1 = _complete(ctl)
    manifest1 = json.loads((task_dir1 / "manifest.json").read_text(encoding="utf-8"))
    processed1 = sorted(p.name for p in (task_dir1 / "processed_pages").glob("*.png"))

    out2 = ctl.start_task()  # 同一输入再次处理 → 新任务目录
    term2 = _run_to_terminal(ctl)
    assert isinstance(term2, TaskCompleted), term2
    task_dir2 = Path(term2.task_dir)
    ctl.clear_active()

    assert task_dir2 != task_dir1  # 独立任务目录
    # 既有任务数据未被覆盖
    manifest1_after = json.loads((task_dir1 / "manifest.json").read_text(encoding="utf-8"))
    assert manifest1_after == manifest1
    assert sorted(p.name for p in (task_dir1 / "processed_pages").glob("*.png")) == processed1
    assert all(p["processing_status"] == "completed" for p in manifest1_after["pages"])


# ---- 历史索引派生状态 ----------------------------------------------------

def test_get_history_derives_state(tmp_path):
    ctl = _controller(tmp_path, n=3)
    task_dir, out = _complete(ctl)
    hist = ctl.get_history()
    assert len(hist) >= 1
    e = next(x for x in hist if x["task_dir"] == str(task_dir))
    assert e["state"] == "completed" and e["resumable"] is False
    assert e["total_pages"] == 3 and e["label"]

    _drop_output(out)
    e2 = next(x for x in ctl.get_history() if x["task_dir"] == str(task_dir))
    assert e2["state"] == "incomplete" and e2["resumable"] is True  # 以 Manifest 派生，非索引 last_state


# ---- 设置持久化 ----------------------------------------------------------

def test_settings_persistence_roundtrip(tmp_path):
    ctl = AppController(paths=UserPaths(root=tmp_path / "data"))
    ctl.set_settings(TaskSettings(preset="old_book", compression="compact", color_mode="grayscale",
                                  border_clean=False, output_dir=str(tmp_path / "out")))
    ctl.save_settings()
    ctl2 = AppController(paths=UserPaths(root=tmp_path / "data"))
    assert ctl2.load_settings() is True
    s = ctl2.settings
    assert s.preset == "old_book" and s.compression == "compact"
    assert s.color_mode == "grayscale" and s.border_clean is False


def test_load_settings_absent_returns_false(tmp_path):
    ctl = AppController(paths=UserPaths(root=tmp_path / "fresh"))
    assert ctl.load_settings() is False
    assert ctl.settings.preset == "auto"  # 保留默认


def test_load_task_missing_graceful(tmp_path):
    ctl = _controller(tmp_path, n=1)
    r = ctl.load_task(tmp_path / "nope", tmp_path / "none.pdf")
    assert r["manifest_exists"] is False and r["state"] == "missing"
    assert "log_path" in r