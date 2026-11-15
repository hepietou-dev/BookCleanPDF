"""RapidOCR Provider：经插件 worker 子进程提供 OCR（M6.0 §4.1/§6）。

主进程职责：插件校验、图像 sha256、结果组装、批处理/取消编排、落盘（由调用方）。
worker 职责：模型加载与推理（只写 stdout）。崩溃/超时由 WorkerClient 重启一次。
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Iterable

from .plugin_manager import PluginInfo, PluginManager, PluginError
from .provider import (
    Availability,
    BatchSummary,
    EngineVersion,
    InitResult,
    OcrError,
    OcrLine,
    OcrParams,
    PageOcrResult,
    STATUS_COMPLETED,
    STATUS_FAILED,
)
from .worker_client import WorkerClient, WorkerError

logger = logging.getLogger("bookclean")

WORKER_SCRIPT = Path(__file__).resolve().parent / "rapid_worker.py"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class RapidOcrProvider:
    engine_id = "ocr-rapid"
    expected_worker_kind = "rapid"
    worker_script = WORKER_SCRIPT

    def __init__(self, plugin: PluginInfo, params: OcrParams | None = None):
        if plugin.worker_kind != self.expected_worker_kind:
            raise PluginError(f"插件 worker 种类不匹配: {plugin.worker_kind!r}")
        self._plugin = plugin
        self._params = params or OcrParams()
        self._cancel = threading.Event()
        model_id, model_sha = plugin.model_fingerprint()
        self._version = EngineVersion(
            engine_id=plugin.engine_id,
            engine_version=plugin.engine_version,
            model_id=model_id,
            model_sha256=model_sha,
            params_schema_version=plugin.params_schema_version,
        )
        self._client = WorkerClient(
            [str(plugin.python_exe), "-u", str(self.worker_script)],
            cwd=plugin.plugin_dir,
            page_timeout_s=self._params.page_timeout_s,
        )
        self._initialized = False

    # ---- 可用性与版本 ---------------------------------------------------

    def check_available(self) -> Availability:
        problems = PluginManager.verify_models(self._plugin)
        return Availability(
            installed=True,
            engine_id=self._plugin.engine_id,
            engine_version=self._plugin.engine_version,
            models_ok=not problems,
            reason="；".join(problems),
            plugin_dir=str(self._plugin.plugin_dir),
        )

    def get_version(self) -> EngineVersion:
        return self._version

    # ---- 初始化 ---------------------------------------------------------

    def _extra_init_params(self) -> dict:
        """子类注入引擎专属 init 参数（RapidOCR 无额外参数）。"""
        return {}

    def initialize(self, params: OcrParams | None = None) -> InitResult:
        if params is not None:
            self._params = params
            self._client.page_timeout_s = params.page_timeout_s
        problems = PluginManager.verify_models(self._plugin)
        if problems:
            return InitResult(ok=False, error=OcrError("MODEL_MISMATCH", "；".join(problems)))
        init_params = self._params.to_dict()
        init_params.update(self._extra_init_params())
        for m in self._plugin.models:
            key = {"det": "det_model_path", "rec": "rec_model_path", "cls": "cls_model_path"}
            for prefix, param_name in key.items():
                if m["id"].lower().find(prefix) >= 0 and param_name not in init_params:
                    init_params[param_name] = m["path"]
        t0 = time.perf_counter()
        try:
            ready = self._client.initialize(init_params)
        except WorkerError as exc:
            return InitResult(ok=False, error=OcrError(exc.code, str(exc)))
        self._initialized = True
        reported = str(ready.get("engine_version") or self._plugin.engine_version)
        if reported != self._version.engine_version:
            logger.info("插件清单版本 %s 与 runtime 实际版本 %s 不一致，以实际为准",
                        self._version.engine_version, reported)
            self._version = EngineVersion(
                engine_id=self._version.engine_id, engine_version=reported,
                model_id=self._version.model_id, model_sha256=self._version.model_sha256,
                params_schema_version=self._version.params_schema_version,
            )
        return InitResult(
            ok=True, engine_version=self._version, device=str(ready.get("device", "cpu")),
            init_seconds=float(ready.get("init_seconds", time.perf_counter() - t0)),
        )

    # ---- 识别 -----------------------------------------------------------

    def recognize_page(self, image_path: str | Path, page_no: int) -> PageOcrResult:
        # worker 的 cwd 是插件目录，相对路径必须先转绝对
        image_path = Path(image_path).resolve()
        if not self._initialized:
            return PageOcrResult(
                page_no=page_no, status=STATUS_FAILED,
                error=OcrError("NOT_INITIALIZED", "Provider 尚未 initialize"),
                engine_version=self._version,
            )
        try:
            image_sha = _sha256_file(image_path)
        except OSError as exc:
            return PageOcrResult(
                page_no=page_no, status=STATUS_FAILED,
                error=OcrError("IMAGE_UNREADABLE", f"{type(exc).__name__}: {exc}"),
                engine_version=self._version,
            )
        try:
            resp = self._client.recognize(page_no, image_path)
        except WorkerError as exc:
            return PageOcrResult(
                page_no=page_no, status=STATUS_FAILED, image_sha256=image_sha,
                error=OcrError(exc.code, str(exc)), engine_version=self._version,
            )
        return self._to_result(resp, page_no, image_sha)

    def _to_result(self, resp: dict, page_no: int, image_sha: str) -> PageOcrResult:
        status = resp.get("status")
        if status != STATUS_COMPLETED:
            err = resp.get("error") or {}
            return PageOcrResult(
                page_no=page_no, status=STATUS_FAILED, image_sha256=image_sha,
                error=OcrError(str(err.get("code", "ENGINE_ERROR")),
                               str(err.get("message", "引擎报告失败"))),
                engine_version=self._version,
            )
        lines = tuple(
            OcrLine(
                line_id=int(l["line_id"]), text=str(l["text"]),
                score=float(l["score"]) if l.get("score") is not None else None,
                bbox_px=tuple(float(v) for v in l["bbox_px"]),
                polygon_px=tuple(tuple(float(v) for v in p) for p in l["polygon_px"])
                if l.get("polygon_px") else None,
            )
            for l in resp.get("lines", [])
        )
        return PageOcrResult(
            page_no=page_no, status=STATUS_COMPLETED, lines=lines,
            reading_order=tuple(int(x) for x in resp.get("reading_order", [])),
            image_width_px=int(resp.get("image_width_px", 0)),
            image_height_px=int(resp.get("image_height_px", 0)),
            image_sha256=image_sha,
            elapsed_s=float(resp.get("elapsed_s", 0.0)),
            engine_version=self._version,
        )

    def recognize_batch(
        self,
        pages: Iterable[tuple[int, Path]],
        on_result: Callable[[PageOcrResult], None],
        cancel_event=None,
    ) -> BatchSummary:
        """流式逐页：每页独立结果回调；页间检查取消；失败不中断后续页。"""
        issued = completed = failed = 0
        t0 = time.perf_counter()
        for page_no, image_path in pages:
            if self._cancel.is_set() or (cancel_event is not None and cancel_event.is_set()):
                break
            result = self.recognize_page(image_path, page_no)
            issued += 1
            if result.status == STATUS_COMPLETED:
                completed += 1
            else:
                failed += 1
            on_result(result)
        return BatchSummary(
            issued=issued, completed=completed, failed=failed,
            elapsed_s=round(time.perf_counter() - t0, 3),
        )

    # ---- 取消与关闭 -----------------------------------------------------

    def cancel(self) -> None:
        self._cancel.set()
        try:
            self._client.cancel()
        except WorkerError:
            pass  # 进程已退出时取消无意义

    def shutdown(self) -> None:
        self._client.shutdown()
        self._initialized = False