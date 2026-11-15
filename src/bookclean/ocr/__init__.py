"""M6 OCR 可选能力包。

设计约束（docs/m6_0_ocr_architecture.md）：
- 本包不被 M1–M5 任何模块导入；OCR 关闭/缺插件时导出行为与 M5 完全一致。
- 基础包零 OCR 运行时依赖：引擎在独立插件 runtime 中以子进程运行。
- 所有结果数据来自引擎真实输出，绝不虚构；引擎未提供的字段为 None。
"""

from .provider import (
    OCR_SCHEMA_VERSION,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_PROCESSING,
    STATUS_SKIPPED,
    Availability,
    BatchSummary,
    EngineVersion,
    InitResult,
    OcrError,
    OcrLine,
    OcrParams,
    OcrProvider,
    PageOcrResult,
)

__all__ = [
    "OCR_SCHEMA_VERSION",
    "STATUS_PENDING",
    "STATUS_PROCESSING",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_SKIPPED",
    "Availability",
    "BatchSummary",
    "EngineVersion",
    "InitResult",
    "OcrError",
    "OcrLine",
    "OcrParams",
    "OcrProvider",
    "PageOcrResult",
]