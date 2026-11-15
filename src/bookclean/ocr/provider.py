"""OCR Provider 接口与统一结果结构（M6.0 §6 设计落地）。

所有 dataclass 均可 JSON 序列化（to_dict/from_dict），供逐页缓存与 IPC 使用。
字段语义：引擎未提供的数据一律 None，绝不编造（score/polygon 等）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol, runtime_checkable

OCR_SCHEMA_VERSION = 1

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


@dataclass(frozen=True)
class OcrError:
    code: str
    message: str

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message}

    @classmethod
    def from_dict(cls, d: dict) -> "OcrError":
        return cls(code=str(d.get("code", "UNKNOWN")), message=str(d.get("message", "")))


@dataclass(frozen=True)
class EngineVersion:
    """引擎身份指纹：参与 OCR 缓存键（引擎/模型/参数任一变化即失效）。"""

    engine_id: str
    engine_version: str
    model_id: str
    model_sha256: str
    params_schema_version: int = 1

    def to_dict(self) -> dict:
        return {
            "engine_id": self.engine_id,
            "engine_version": self.engine_version,
            "model_id": self.model_id,
            "model_sha256": self.model_sha256,
            "params_schema_version": self.params_schema_version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EngineVersion":
        return cls(
            engine_id=str(d.get("engine_id", "")),
            engine_version=str(d.get("engine_version", "")),
            model_id=str(d.get("model_id", "")),
            model_sha256=str(d.get("model_sha256", "")),
            params_schema_version=int(d.get("params_schema_version", 1)),
        )

    def fingerprint(self) -> str:
        return "|".join([
            self.engine_id, self.engine_version, self.model_id,
            self.model_sha256, str(self.params_schema_version),
        ])


@dataclass(frozen=True)
class OcrParams:
    threads: int = 4
    lang: str = "auto"
    text_score: float = 0.5
    page_timeout_s: float = 120.0

    def to_dict(self) -> dict:
        return {
            "threads": self.threads,
            "lang": self.lang,
            "text_score": self.text_score,
            "page_timeout_s": self.page_timeout_s,
        }


@dataclass(frozen=True)
class Availability:
    installed: bool
    engine_id: str = ""
    engine_version: str = ""
    models_ok: bool = False
    reason: str = ""
    plugin_dir: str = ""

    def to_dict(self) -> dict:
        return {
            "installed": self.installed,
            "engine_id": self.engine_id,
            "engine_version": self.engine_version,
            "models_ok": self.models_ok,
            "reason": self.reason,
            "plugin_dir": self.plugin_dir,
        }


@dataclass(frozen=True)
class InitResult:
    ok: bool
    engine_version: EngineVersion | None = None
    device: str = "cpu"
    init_seconds: float = 0.0
    error: OcrError | None = None


@dataclass(frozen=True)
class OcrLine:
    line_id: int
    text: str
    score: float | None
    bbox_px: tuple[float, float, float, float]
    polygon_px: tuple[tuple[float, float], ...] | None = None

    def to_dict(self) -> dict:
        return {
            "line_id": self.line_id,
            "text": self.text,
            "score": self.score,
            "bbox_px": list(self.bbox_px),
            "polygon_px": [list(p) for p in self.polygon_px] if self.polygon_px else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OcrLine":
        poly = d.get("polygon_px")
        return cls(
            line_id=int(d["line_id"]),
            text=str(d["text"]),
            score=float(d["score"]) if d.get("score") is not None else None,
            bbox_px=tuple(float(v) for v in d["bbox_px"]),
            polygon_px=tuple(tuple(float(v) for v in p) for p in poly) if poly else None,
        )


@dataclass(frozen=True)
class PageOcrResult:
    page_no: int
    status: str
    lines: tuple[OcrLine, ...] = ()
    reading_order: tuple[int, ...] = ()
    image_width_px: int = 0
    image_height_px: int = 0
    image_sha256: str = ""
    elapsed_s: float = 0.0
    engine_version: EngineVersion | None = None
    error: OcrError | None = None
    schema_version: int = OCR_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "page_no": self.page_no,
            "status": self.status,
            "lines": [l.to_dict() for l in self.lines],
            "reading_order": list(self.reading_order),
            "image_width_px": self.image_width_px,
            "image_height_px": self.image_height_px,
            "image_sha256": self.image_sha256,
            "elapsed_s": self.elapsed_s,
            "engine_version": self.engine_version.to_dict() if self.engine_version else None,
            "error": self.error.to_dict() if self.error else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PageOcrResult":
        ev = d.get("engine_version")
        err = d.get("error")
        return cls(
            page_no=int(d["page_no"]),
            status=str(d["status"]),
            lines=tuple(OcrLine.from_dict(x) for x in d.get("lines", [])),
            reading_order=tuple(int(x) for x in d.get("reading_order", [])),
            image_width_px=int(d.get("image_width_px", 0)),
            image_height_px=int(d.get("image_height_px", 0)),
            image_sha256=str(d.get("image_sha256", "")),
            elapsed_s=float(d.get("elapsed_s", 0.0)),
            engine_version=EngineVersion.from_dict(ev) if ev else None,
            error=OcrError.from_dict(err) if err else None,
            schema_version=int(d.get("schema_version", OCR_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class BatchSummary:
    issued: int = 0
    completed: int = 0
    failed: int = 0
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "issued": self.issued,
            "completed": self.completed,
            "failed": self.failed,
            "elapsed_s": self.elapsed_s,
        }


@runtime_checkable
class OcrProvider(Protocol):
    """引擎无关接口。实现：RapidOcrProvider（插件 worker）/ FakeProvider（测试）。

    失败语义：recognize_page 永不向调用方抛引擎异常——失败以 status=failed 结果返回；
    仅 initialize 失败/插件缺失以 InitResult(ok=False) 或异常（PluginError）表达。
    """

    engine_id: str

    def check_available(self) -> Availability: ...

    def get_version(self) -> EngineVersion: ...

    def initialize(self, params: OcrParams | None = None) -> InitResult: ...

    def recognize_page(self, image_path: str | Path, page_no: int) -> PageOcrResult: ...

    def recognize_batch(
        self,
        pages: Iterable[tuple[int, Path]],
        on_result: Callable[[PageOcrResult], None],
        cancel_event=None,
    ) -> BatchSummary: ...

    def cancel(self) -> None: ...

    def shutdown(self) -> None: ...