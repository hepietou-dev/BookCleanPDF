"""GUI 任务状态机词汇 + worker→UI 事件协议（集中定义，避免硬编码字符串）。

状态由现有 Manifest 字段派生，不引入新的任务目录结构。
事件由后台 worker 线程产生、只经 queue.Queue 传递；UI 在主线程经 root.after 排空消费。
事件对象为不可变 dataclass，worker 绝不触碰 Tkinter 部件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaskState(str, Enum):
    PENDING = "pending"                    # 空闲：无输入或尚未校验
    IMPORTING = "importing"                # 正在读取输入元信息（页数/大小）
    READY = "ready"                        # 输入与配置预检通过，可开始处理
    PROCESSING = "processing"              # M2/M3 逐页处理中
    BORDER_CLEANING = "border_cleaning"    # M4.8.1 扫描黑边清除中
    OCR_RECOGNIZING = "ocr_recognizing"    # M6.3 可选本地 OCR 文字识别中
    EXPORTING = "exporting"                # M4 PDF 导出中
    VALIDATING = "validating"              # PDF 验证中
    COMPLETED = "completed"                # 已 os.replace 出最终 PDF（.partial 不算）
    FAILED = "failed"                      # 失败（带上下文错误）
    CANCELLED = "cancelled"                # 用户取消


@dataclass(frozen=True)
class StageChanged:
    """阶段切换（worker 进入某内核阶段时发出）。"""
    state: TaskState
    label: str = ""


@dataclass(frozen=True)
class PageProgress:
    """逐页真实进度：done/total 来自内核回调累计，非定时器伪造。"""
    state: TaskState
    done: int
    total: int
    page_index: int | None = None
    status: str = ""
    counters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class StageSummary:
    """阶段结束时的内核 summary（processed/failed/fallbacks/reused/cleaned/skipped…）。"""
    state: TaskState
    summary: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TaskFailed:
    state: TaskState
    error: str
    exc_type: str = ""
    task_dir: str = ""


@dataclass(frozen=True)
class TaskCancelled:
    state: TaskState
    task_dir: str = ""


@dataclass(frozen=True)
class TaskCompleted:
    output_path: str
    summary: dict = field(default_factory=dict)
    task_dir: str = ""


WorkerEvent = StageChanged | PageProgress | StageSummary | TaskFailed | TaskCancelled | TaskCompleted
TERMINAL_EVENTS = (TaskFailed, TaskCancelled, TaskCompleted)