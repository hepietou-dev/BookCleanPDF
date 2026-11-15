"""确定性 FakeProvider：测试与无插件演示用（不启动子进程、不依赖任何 OCR 库）。

输出完全由 page_no 派生，可配置失败页/空页，用于验证批处理/取消/失败隔离逻辑。
绝不用于真实识别结果展示（文本明确标注 fake）。
"""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path
from typing import Callable, Iterable

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

_VERSION = EngineVersion(
    engine_id="fake", engine_version="0.0.0", model_id="fake",
    model_sha256="0" * 64, params_schema_version=1,
)


class FakeProvider:
    engine_id = "fake"

    def __init__(self, fail_pages: Iterable[int] = (), empty_pages: Iterable[int] = (),
                 delay_s: float = 0.0):
        self._fail = set(fail_pages)
        self._empty = set(empty_pages)
        self._delay = float(delay_s)
        self._cancel = threading.Event()
        self._initialized = False

    def check_available(self) -> Availability:
        return Availability(installed=True, engine_id="fake", engine_version="0.0.0",
                            models_ok=True, reason="测试用假引擎")

    def get_version(self) -> EngineVersion:
        return _VERSION

    def initialize(self, params: OcrParams | None = None) -> InitResult:
        self._initialized = True
        return InitResult(ok=True, engine_version=_VERSION, device="cpu", init_seconds=0.0)

    def recognize_page(self, image_path: str | Path, page_no: int) -> PageOcrResult:
        if not self._initialized:
            return PageOcrResult(page_no=page_no, status=STATUS_FAILED,
                                 error=OcrError("NOT_INITIALIZED", "Provider 尚未 initialize"),
                                 engine_version=_VERSION)
        if self._delay:
            time.sleep(self._delay)
        image_path = Path(image_path)
        sha = ""
        width = height = 0
        if image_path.is_file():
            sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
            try:
                from PIL import Image
                with Image.open(image_path) as im:
                    width, height = im.size
            except Exception:  # noqa: BLE001 — fake 引擎对损坏图像也给出确定行为
                pass
        if page_no in self._fail:
            return PageOcrResult(page_no=page_no, status=STATUS_FAILED, image_sha256=sha,
                                 error=OcrError("ENGINE_ERROR", f"fake 失败页 {page_no}"),
                                 engine_version=_VERSION)
        lines: tuple[OcrLine, ...] = ()
        if page_no not in self._empty:
            lines = (
                OcrLine(line_id=0, text=f"fake line page {page_no}", score=0.99,
                        bbox_px=(10.0, 10.0, 200.0, 40.0)),
                OcrLine(line_id=1, text=f"第二行 {page_no}", score=0.98,
                        bbox_px=(10.0, 50.0, 180.0, 80.0)),
            )
        return PageOcrResult(
            page_no=page_no, status=STATUS_COMPLETED, lines=lines,
            reading_order=tuple(l.line_id for l in lines),
            image_width_px=width, image_height_px=height, image_sha256=sha,
            elapsed_s=self._delay, engine_version=_VERSION,
        )

    def recognize_batch(
        self,
        pages: Iterable[tuple[int, Path]],
        on_result: Callable[[PageOcrResult], None],
        cancel_event=None,
    ) -> BatchSummary:
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
        return BatchSummary(issued=issued, completed=completed, failed=failed,
                            elapsed_s=round(time.perf_counter() - t0, 3))

    def cancel(self) -> None:
        self._cancel.set()

    def shutdown(self) -> None:
        self._initialized = False