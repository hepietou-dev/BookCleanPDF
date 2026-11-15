"""M5.2 控制器后台任务编排测试：真实文件 start/cancel/poll/wait/clear，防重复启动。

不依赖 Tkinter；通过 poll_events 排空真实 worker 事件，断言基于真实生成的 PDF。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from bookclean.app_controller import AppController, TaskSettings
from bookclean.task_events import TaskCancelled, TaskCompleted, TaskFailed
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
    ctl.wait(timeout=15)
    return terminal


def test_start_task_produces_real_pdf(tmp_path):
    ctl = _controller(tmp_path, n=3)
    out = ctl.start_task()
    assert out is not None
    term = _run_to_terminal(ctl)
    assert isinstance(term, TaskCompleted), term
    assert Path(out).exists() and Path(out).stat().st_size > 0
    assert not Path(out).with_name(Path(out).stem + ".partial.pdf").exists()
    ctl.clear_active()
    assert not ctl.is_running()
    assert ctl.poll_events() == []


def test_start_refuses_duplicate_while_running(tmp_path):
    ctl = _controller(tmp_path, n=8)
    first = ctl.start_task()
    assert first is not None
    assert ctl.is_running()
    second = ctl.start_task()  # 活跃任务期间拒绝重复启动
    assert second is None
    _run_to_terminal(ctl)
    ctl.clear_active()


def test_start_invalid_returns_none(tmp_path):
    ctl = AppController(paths=UserPaths(root=tmp_path / "data"))
    assert ctl.start_task() is None  # 无输入
    assert not ctl.is_running()


def test_cancel_stops_task_no_output(tmp_path):
    ctl = _controller(tmp_path, n=8)
    out = ctl.start_task()
    assert ctl.cancel_task() is True
    term = _run_to_terminal(ctl)
    assert isinstance(term, TaskCancelled), term
    assert not Path(out).exists()
    ctl.clear_active()


def test_cancel_when_idle_false(tmp_path):
    ctl = _controller(tmp_path, n=1)
    assert ctl.cancel_task() is False


def test_poll_empty_when_idle(tmp_path):
    ctl = _controller(tmp_path, n=1)
    assert ctl.poll_events() == []
    assert ctl.wait(timeout=1) is True


def test_failure_surfaces_as_event(tmp_path):
    ctl = _controller(tmp_path, n=2)
    out = ctl.default_output_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()  # 输出路径被目录占用：预检(overwrite=True)通过，但导出 os.replace 真实失败
    started = ctl.start_task()
    assert started is not None
    term = _run_to_terminal(ctl)
    assert isinstance(term, TaskFailed), term
    assert out.is_dir()  # 既有目录未被破坏
    assert not out.with_name(out.stem + ".partial.pdf").exists()
    ctl.clear_active()