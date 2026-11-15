"""M5.2 worker 单元测试：真实文件处理、事件协议、取消矩阵、失败归类。

不依赖 Tkinter；worker 同步或后台线程运行，事件经 queue.Queue 收集。
取消覆盖：导入(回调钩子) / 处理(原生 cancel_event) / 导出 / 验证(回调钩子 + .partial 清理)。
所有断言基于真实生成的图片与真实内核链路，不用模拟处理代替。
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from bookclean.config import BorderCleanConfig, PDFExportConfig
from bookclean.pdf_exporter import PDFExporter
from bookclean.presets import get_preset
from bookclean.task_events import (
    PageProgress,
    StageChanged,
    StageSummary,
    TaskCancelled,
    TaskCompleted,
    TaskFailed,
    TaskState,
)
from bookclean.task_worker import TaskWorker


def _make_page(path: Path, idx: int) -> Path:
    rng = np.random.default_rng(idx)
    arr = rng.integers(205, 245, size=(380, 500, 3), dtype=np.uint8)
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    for y in range(60, 320, 28):
        d.line([(50, y), (450, y)], fill=(30, 30, 30), width=2)
    d.rectangle([0, 0, 499, 379], outline=(0, 0, 0), width=8)
    im.save(path)
    return path


def _build_worker(tmp_path, n=3, *, cls=TaskWorker, border=True, output_path=None):
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    inputs = [str(_make_page(src / f"p{i}.png", i)) for i in range(n)]
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir(exist_ok=True)
    out = Path(output_path) if output_path else tmp_path / "out" / "out.pdf"
    cfg = get_preset("auto")
    cfg.output_dir = str(tmp_path / "out")
    cfg.validate()
    q: queue.Queue = queue.Queue()
    cancel = threading.Event()
    w = cls(
        input_paths=inputs, total_pages=n,
        task_config=cfg,
        export_config=PDFExportConfig(overwrite=True),
        border_config=BorderCleanConfig() if border else None,
        output_path=out, tasks_root=tasks_root,
        event_queue=q, cancel_event=cancel,
    )
    return w, q, cancel, out


def _drain(q):
    events = []
    while True:
        try:
            events.append(q.get_nowait())
        except queue.Empty:
            break
    return events


def _terminal(events):
    for e in events:
        if isinstance(e, (TaskCompleted, TaskFailed, TaskCancelled)):
            return e
    return None


def _partial(out: Path) -> Path:
    return out.with_name(out.stem + ".partial.pdf")


# ---- 完整链路与事件协议 -------------------------------------------------

def test_full_pipeline_real_pdf(tmp_path):
    w, q, cancel, out = _build_worker(tmp_path, n=3)
    w.run()
    events = _drain(q)
    term = _terminal(events)
    assert isinstance(term, TaskCompleted), term
    assert out.exists() and out.stat().st_size > 0
    assert not _partial(out).exists()

    stages = [e.state for e in events if isinstance(e, StageChanged)]
    assert stages == [TaskState.IMPORTING, TaskState.PROCESSING, TaskState.BORDER_CLEANING,
                      TaskState.EXPORTING, TaskState.VALIDATING]

    for st in (TaskState.IMPORTING, TaskState.PROCESSING, TaskState.EXPORTING, TaskState.VALIDATING):
        pp = [e for e in events if isinstance(e, PageProgress) and e.state == st]
        assert pp, f"阶段 {st} 无逐页进度"
        assert pp[-1].done == pp[-1].total
        assert all(a.done <= b.done for a, b in zip(pp, pp[1:]))

    proc_pp = [e for e in events if isinstance(e, PageProgress) and e.state == TaskState.PROCESSING]
    c = proc_pp[-1].counters
    assert c["processed"] + c["fallbacks"] + c["failed"] == proc_pp[-1].done == 3
    summ = {e.state: e.summary for e in events if isinstance(e, StageSummary)}
    assert c["fallbacks"] == summ[TaskState.PROCESSING]["fallbacks"]
    assert c["processed"] == summ[TaskState.PROCESSING]["processed"] - summ[TaskState.PROCESSING]["fallbacks"]
    assert summ[TaskState.EXPORTING]["total_pages"] == 3


def test_border_disabled_skips_stage(tmp_path):
    w, q, cancel, out = _build_worker(tmp_path, n=2, border=False)
    w.run()
    events = _drain(q)
    stages = [e.state for e in events if isinstance(e, StageChanged)]
    assert TaskState.BORDER_CLEANING not in stages
    assert isinstance(_terminal(events), TaskCompleted)
    assert out.exists()


# ---- 取消矩阵 -----------------------------------------------------------

def test_cancel_before_start_import_hook(tmp_path):
    w, q, cancel, out = _build_worker(tmp_path, n=3)
    cancel.set()
    w.run()
    term = _terminal(_drain(q))
    assert isinstance(term, TaskCancelled)
    assert term.state == TaskState.IMPORTING
    assert not out.exists()
    assert not _partial(out).exists()


def test_cancel_during_processing_native(tmp_path):
    w, q, cancel, out = _build_worker(tmp_path, n=8)
    t = threading.Thread(target=w.run, daemon=True)
    t.start()
    deadline = time.monotonic() + 120
    cancelled = False
    while time.monotonic() < deadline:
        try:
            ev = q.get(timeout=0.2)
        except queue.Empty:
            continue
        if isinstance(ev, StageChanged) and ev.state == TaskState.PROCESSING:
            cancel.set()
        if isinstance(ev, TaskCancelled):
            cancelled = True
            break
        if isinstance(ev, (TaskCompleted, TaskFailed)):
            break
    t.join(timeout=30)
    assert cancelled, "处理阶段未能协作取消"
    assert not out.exists()
    assert not _partial(out).exists()


class _CancelAtExport(TaskWorker):
    def _run_export(self):
        self.cancel.set()
        super()._run_export()


def test_cancel_at_export_hook_no_partial(tmp_path):
    w, q, cancel, out = _build_worker(tmp_path, n=3, cls=_CancelAtExport)
    w.run()
    term = _terminal(_drain(q))
    assert isinstance(term, TaskCancelled)
    assert term.state == TaskState.EXPORTING
    assert not out.exists()
    assert not _partial(out).exists()  # build 在 try 之前中止，.partial 尚未写


class _CancelAtVerify(TaskWorker):
    def _verify_cb(self, i, n):
        super()._verify_cb(i, n)  # 首次调用发出 VALIDATING 阶段
        self.cancel.set()         # 下一次验证回调即中止


def test_cancel_at_verify_hook_cleans_partial(tmp_path):
    w, q, cancel, out = _build_worker(tmp_path, n=3, cls=_CancelAtVerify)
    w.run()
    term = _terminal(_drain(q))
    assert isinstance(term, TaskCancelled)
    assert term.state == TaskState.VALIDATING
    assert not out.exists()
    assert not _partial(out).exists()  # export 的 except BaseException 清理 .partial


# ---- 失败归类与原件保护 -------------------------------------------------

def test_failure_emits_taskfailed_and_preserves_input(tmp_path):
    w, q, cancel, out = _build_worker(tmp_path, n=2)
    w.output_path = Path(w.input_paths[0])  # 输出=输入 → 导出守卫拒绝（原件只读保护）
    w.run()
    term = _terminal(_drain(q))
    assert isinstance(term, TaskFailed)
    assert term.exc_type == "ExportError"
    assert term.state == TaskState.EXPORTING
    assert Path(w.input_paths[0]).exists()  # 原件未被改动


# ---- 内核回调钩子中止契约（真实任务目录，确定性验证 .partial 清理） ------

def test_export_progress_hook_abort_cleans_partial(tmp_path):
    w, q, cancel, out = _build_worker(tmp_path, n=2)
    w.run()
    task_dir = w._task_dir
    out2 = tmp_path / "out" / "hook.pdf"

    class Abort(Exception):
        pass

    def cb(rec):
        raise Abort()

    with pytest.raises(Abort):
        PDFExporter(PDFExportConfig(overwrite=True)).export(task_dir, out2, progress_cb=cb)
    assert not out2.exists()
    assert not _partial(out2).exists()


def test_export_verify_hook_abort_cleans_partial(tmp_path):
    w, q, cancel, out = _build_worker(tmp_path, n=2)
    w.run()
    task_dir = w._task_dir
    out2 = tmp_path / "out" / "vhook.pdf"

    class Abort(Exception):
        pass

    def vcb(i, n):
        raise Abort()

    with pytest.raises(Abort):
        PDFExporter(PDFExportConfig(overwrite=True)).export(task_dir, out2, verify_progress_cb=vcb)
    assert not out2.exists()
    assert not _partial(out2).exists()


def test_grayscale_verify_hook_abort_cleans_partial(tmp_path):
    """灰度二次验证（verify_grayscale）的回调中止也须传播并清理 .partial（句柄不泄漏）。"""
    w, q, cancel, out = _build_worker(tmp_path, n=2)
    w.run()
    task_dir = w._task_dir
    out2 = tmp_path / "out" / "ghook.pdf"

    class Abort(Exception):
        pass

    calls = {"n": 0}

    def vcb(i, n):
        calls["n"] += 1
        if calls["n"] > 2:  # 前 2 次为 verify() 逐页；第 3 次进入 verify_grayscale
            raise Abort()

    with pytest.raises(Abort):
        PDFExporter(PDFExportConfig(overwrite=True, color_mode="grayscale")).export(
            task_dir, out2, verify_progress_cb=vcb)
    assert calls["n"] == 3, "应已进入灰度二次验证"
    assert not out2.exists()
    assert not _partial(out2).exists()