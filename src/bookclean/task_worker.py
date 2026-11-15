"""后台任务 worker：单线程顺序调用内核，progress_cb 只入队，绝不触碰 Tkinter。

线程模型（M5.0 §5）：每个活跃任务一个 worker 线程；threading.Event 作取消信号；
queue.Queue 作单向事件通道；UI 主线程经 root.after 排空消费。耗时操作（PDF 渲染/
图像处理/PDF 写入）全部在本线程。

取消语义（如实，不伪造）：
- 处理/黑边：内核原生 cancel_event，逐页协作取消。
- 导入/导出/验证：无原生 cancel_event，用回调钩子——progress_cb/verify_progress_cb
  内检测取消即 raise _Cancelled。导出 build 在 try 之前抛出时 .partial 尚未写（干净）；
  verify 抛出走 export 的 except BaseException 清理 .partial（干净）。钩子中止=放弃整次
  导出，**不跳过/不弱化任何验证**。导入取消=放弃该部分任务目录（原件只读）。
- 无原生暂停：不提供真 Pause；处理阶段取消后已验证页可由 M5.4 resume 复用。

本模块不 import Tkinter/ui；只依赖内核 + 标准库。
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path

from . import task_history
from .border_detector import BorderCleaningRunner
from .config import BorderCleanConfig, OcrConfig, PDFExportConfig, TaskConfig
from .pdf_exporter import PDFExporter
from .processing_pipeline import ProcessingRunner
from .task_events import (
    PageProgress,
    StageChanged,
    StageSummary,
    TaskCancelled,
    TaskCompleted,
    TaskFailed,
    TaskState,
)
from .task_manager import ImportTaskManager

logger = logging.getLogger("bookclean")


class _Cancelled(Exception):
    """回调钩子协作中止信号（仅 worker 内部使用）。"""


class TaskWorker:
    def __init__(
        self,
        *,
        input_paths: list[str],
        total_pages: int,
        task_config: TaskConfig,
        export_config: PDFExportConfig,
        border_config: BorderCleanConfig | None,
        output_path: Path,
        tasks_root: Path,
        event_queue: queue.Queue,
        cancel_event: threading.Event,
        resume_task_dir: Path | None = None,
        history_path: Path | None = None,
        settings: dict | None = None,
        label: str = "",
        ocr_config: OcrConfig | None = None,
    ):
        self.input_paths = input_paths
        self.total_pages = total_pages
        self.task_config = task_config
        self.export_config = export_config
        self.border_config = border_config
        self.ocr_config = ocr_config
        self.output_path = Path(output_path)
        self.tasks_root = Path(tasks_root)
        self.q = event_queue
        self.cancel = cancel_event
        self.resume_task_dir = Path(resume_task_dir) if resume_task_dir else None
        self.history_path = Path(history_path) if history_path else None
        self.settings = settings
        self.label = label
        self._stage = TaskState.PENDING
        self._import_done = 0
        self._proc_done = 0
        self._proc = {"processed": 0, "failed": 0, "fallbacks": 0}
        self._border_done = 0
        self._border = {"cleaned": 0, "skipped": 0, "failed": 0}
        self._ocr_done = 0
        self._ocr = {"completed": 0, "failed": 0, "skipped": 0}
        self._ocr_text_source = None
        self._export_done = 0
        self._verify_started = False
        self._task_dir: Path | None = None

    # ---- 事件与取消 -----------------------------------------------------

    def _emit(self, event) -> None:
        self.q.put(event)

    def _td(self) -> str:
        return str(self._task_dir or "")

    def _check_cancel(self) -> None:
        if self.cancel.is_set():
            raise _Cancelled()

    def _set_stage(self, state: TaskState, label: str = "") -> None:
        self._stage = state
        self._emit(StageChanged(state, label))

    # ---- 历史索引（M5.4：导入即登记，崩溃中途仍可续跑；终止更新状态） ----

    def _record_running(self) -> None:
        if self.history_path is None or self._task_dir is None:
            return
        try:
            task_history.record_task(
                self.history_path, task_dir=self._task_dir,
                output_path=str(self.output_path), settings=self.settings,
                label=self.label, last_state="running")
        except OSError as exc:
            logger.warning("登记历史索引失败: %s", exc)

    def _terminate(self, event) -> None:
        """终止事件：先更新历史索引状态，再入队（索引失败不影响事件传递）。"""
        state = {TaskCompleted: "completed", TaskFailed: "failed",
                 TaskCancelled: "cancelled"}.get(type(event), "")
        if state and self.history_path is not None and self._task_dir is not None:
            try:
                out = str(self.output_path) if isinstance(event, TaskCompleted) else None
                task_history.update_task_state(self.history_path, self._task_dir, state, output_path=out)
            except OSError as exc:
                logger.warning("更新历史索引状态失败: %s", exc)
        self._emit(event)

    # ---- 各阶段回调（worker 线程，只入队） ------------------------------

    def _import_cb(self, entry: dict) -> None:
        self._check_cancel()
        self._import_done += 1
        self._emit(PageProgress(TaskState.IMPORTING, self._import_done, self.total_pages,
                                entry.get("display_index"), entry.get("status", "")))

    def _proc_cb(self, entry: dict) -> None:
        self._proc_done += 1
        status = entry.get("processing_status", "")
        if status == "completed":
            # 与内核 fallbacks 口径一致：任何非 none 回退（background/geometry/original）都不计为完全优化
            if entry.get("fallback_level", "none") != "none":
                self._proc["fallbacks"] += 1
            else:
                self._proc["processed"] += 1
        elif status == "failed":
            self._proc["failed"] += 1
        self._emit(PageProgress(TaskState.PROCESSING, self._proc_done, self.total_pages,
                                entry.get("display_index"), status, dict(self._proc)))

    def _border_cb(self, entry: dict) -> None:
        self._border_done += 1
        status = entry.get("border_status", "")
        if status == "completed":
            self._border["cleaned"] += 1
        elif status == "skipped":
            self._border["skipped"] += 1
        elif status == "failed":
            self._border["failed"] += 1
        self._emit(PageProgress(TaskState.BORDER_CLEANING, self._border_done, self.total_pages,
                                entry.get("display_index"), status, dict(self._border)))

    def _ocr_cb(self, entry: dict) -> None:
        self._ocr_done += 1
        status = entry.get("ocr_status", "")
        if status == "completed":
            self._ocr["completed"] += 1
        elif status == "failed":
            self._ocr["failed"] += 1
        elif status == "skipped":
            self._ocr["skipped"] += 1
        self._emit(PageProgress(TaskState.OCR_RECOGNIZING, self._ocr_done, self.total_pages,
                                entry.get("display_index"), status, dict(self._ocr)))

    def _export_cb(self, rec: dict) -> None:
        self._check_cancel()
        self._export_done += 1
        self._emit(PageProgress(TaskState.EXPORTING, self._export_done, self.total_pages,
                                rec.get("page", rec.get("display_index")), rec.get("page_status", "")))

    def _verify_cb(self, i: int, n: int) -> None:
        self._check_cancel()
        if not self._verify_started:
            self._verify_started = True
            self._set_stage(TaskState.VALIDATING)
        self._emit(PageProgress(TaskState.VALIDATING, i, n))

    # ---- 主流程 ---------------------------------------------------------

    def run(self) -> None:
        resume = self.resume_task_dir is not None
        try:
            if resume:
                self._begin_resume()
            else:
                self._run_import()
            if self.cancel.is_set():
                self._terminate(TaskCancelled(self._stage, task_dir=self._td()))
                return
            self._run_processing(resume=resume)
            if self.cancel.is_set():
                self._terminate(TaskCancelled(TaskState.PROCESSING, task_dir=self._td()))
                return
            if self.border_config is not None:
                self._run_border()
                if self.cancel.is_set():
                    self._terminate(TaskCancelled(TaskState.BORDER_CLEANING, task_dir=self._td()))
                    return
            if self.ocr_config is not None and self.ocr_config.enabled:
                self._run_ocr(resume=resume)
                if self.cancel.is_set():
                    self._terminate(TaskCancelled(TaskState.OCR_RECOGNIZING, task_dir=self._td()))
                    return
            self._run_export()
        except _Cancelled:
            self._terminate(TaskCancelled(self._stage, task_dir=self._td()))
        except Exception as exc:  # 阶段级异常→带上下文 ErrorEvent，不甩 traceback 到 UI
            logger.exception("任务失败（阶段 %s）", self._stage)
            self._terminate(TaskFailed(self._stage, f"{type(exc).__name__}: {exc}", type(exc).__name__, task_dir=self._td()))

    def _begin_resume(self) -> None:
        """续跑：复用既有任务目录，绝不重跑 M1 导入。读 manifest 取总页数并登记索引。"""
        self._task_dir = Path(self.resume_task_dir)
        manifest = json.loads((self._task_dir / "manifest.json").read_text(encoding="utf-8"))
        self.total_pages = int(manifest.get("total_pages", len(manifest.get("pages", []))))
        self._record_running()

    def _run_import(self) -> dict:
        self._set_stage(TaskState.IMPORTING, "导入与渲染输入")
        result = ImportTaskManager(self.tasks_root).run(
            self.input_paths, dpi=self.task_config.render_dpi, progress_cb=self._import_cb)
        self._task_dir = Path(result["task_dir"])
        manifest = result["manifest"]
        self.total_pages = manifest["total_pages"]
        self._record_running()  # 导入建立任务目录即登记，中途崩溃仍有索引可续跑
        imported = sum(1 for p in manifest["pages"] if p.get("status") == "completed")
        self._emit(StageSummary(TaskState.IMPORTING, {
            "total_pages": self.total_pages, "imported": imported,
            "file_errors": len(manifest.get("file_errors", [])),
        }))
        return manifest

    def _run_processing(self, resume: bool = False) -> dict:
        self._set_stage(TaskState.PROCESSING, "逐页图像修复")
        result = ProcessingRunner().run(
            self._task_dir, self.task_config,
            progress_cb=self._proc_cb, cancel_event=self.cancel, resume=resume)
        self._emit(StageSummary(TaskState.PROCESSING, result["summary"]))
        return result["manifest"]

    def _run_border(self) -> None:
        self._set_stage(TaskState.BORDER_CLEANING, "扫描黑边保守清除")
        result = BorderCleaningRunner().run(
            self._task_dir, self.border_config,
            progress_cb=self._border_cb, cancel_event=self.cancel)
        self._emit(StageSummary(TaskState.BORDER_CLEANING, result["summary"]))

    def _run_ocr(self, resume: bool = False) -> None:
        # 惰性导入：OCR 关时绝不导入任何 OCR 代码（导出路径与 M5 完全一致）
        from .ocr_stage import OcrStageRunner
        self._set_stage(TaskState.OCR_RECOGNIZING, "OCR 文字识别")
        result = OcrStageRunner().run(
            self._task_dir, self.ocr_config,
            progress_cb=self._ocr_cb, cancel_event=self.cancel, resume=resume)
        self._ocr_text_source = result.get("text_source")
        self._emit(StageSummary(TaskState.OCR_RECOGNIZING, result["summary"]))

    def _run_export(self) -> None:
        self._set_stage(TaskState.EXPORTING, "合成 PDF")
        info = PDFExporter(self.export_config).export(
            self._task_dir, self.output_path,
            progress_cb=self._export_cb, verify_progress_cb=self._verify_cb,
            text_source=self._ocr_text_source)
        summary = info.get("summary", {})
        self._emit(StageSummary(TaskState.EXPORTING, summary))
        if info.get("export_status") == "completed" and self.output_path.exists():
            self._terminate(TaskCompleted(str(self.output_path), {
                "validation_status": info.get("validation_status"),
                "output_size_bytes": info.get("output_size_bytes"),
                "summary": summary,
            }, task_dir=self._td()))
        else:
            self._terminate(TaskFailed(TaskState.EXPORTING, "导出未产生通过验证的正式 PDF", task_dir=self._td()))