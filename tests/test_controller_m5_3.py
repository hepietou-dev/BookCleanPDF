"""M5.3 控制器结果接口测试：真实内核任务→load_result/open_log，终止事件携带 task_dir。

不依赖 Tkinter；通过真实 start_task/poll_events 跑完整流水线，断言基于真实生成的 PDF/报告。
"""
from __future__ import annotations

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


def _controller(tmp_path, n=2):
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
    ctl.wait(timeout=15)
    return terminal


def test_load_result_after_real_completion(tmp_path):
    ctl = _controller(tmp_path, n=2)
    out = ctl.start_task()
    term = _run_to_terminal(ctl)
    assert isinstance(term, TaskCompleted), term
    assert term.task_dir and (Path(term.task_dir) / "manifest.json").exists()
    r = ctl.load_result(term.task_dir, out)
    assert r["output_exists"] is True
    assert r["output_size_bytes"] > 0
    assert r["validation_status"] == "passed"
    assert r["total_pages"] == 2
    assert r["optimized"] + r["fallbacks"] == r["processed"]  # 回退不计完全优化
    assert r["failed"] == 0
    assert Path(r["report_path"]).exists()
    assert r["log_path"].endswith("bookclean.log")
    ctl.clear_active()


def test_load_result_failed_task_no_final_output(tmp_path):
    ctl = _controller(tmp_path, n=2)
    out = ctl.default_output_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()  # 输出路径被目录占用 → 导出 os.replace 真实失败
    ctl.start_task()
    term = _run_to_terminal(ctl)
    assert isinstance(term, TaskFailed), term
    assert term.task_dir
    r = ctl.load_result(term.task_dir, str(out))
    assert r["output_exists"] is False  # 目录占用，无正式 PDF
    assert out.is_dir()  # 既有目录未被破坏
    ctl.clear_active()


def test_load_result_cancelled_partial(tmp_path):
    ctl = _controller(tmp_path, n=8)
    out = ctl.start_task()
    # 待导入完成进入 PROCESSING 后再取消，确保 task_dir/manifest 已落盘（非导入期即中止）
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
    ctl.wait(timeout=15)
    assert cancelled, "未观察到 PROCESSING 阶段"
    assert isinstance(terminal, TaskCancelled), terminal
    assert terminal.task_dir
    r = ctl.load_result(terminal.task_dir, out)
    assert r["output_exists"] is False
    assert r["manifest_exists"] is True
    ctl.clear_active()


def test_open_log_returns_bool_without_raise(tmp_path):
    ctl = _controller(tmp_path, n=1)
    result = ctl.open_log()
    assert isinstance(result, bool)  # 日志不存在时返回 False，不抛出到 UI


def test_load_result_degrades_on_missing_task(tmp_path):
    ctl = _controller(tmp_path, n=1)
    r = ctl.load_result(tmp_path / "no_such_task", tmp_path / "none.pdf")
    assert r["manifest_exists"] is False
    assert r["output_exists"] is False
    assert "log_path" in r