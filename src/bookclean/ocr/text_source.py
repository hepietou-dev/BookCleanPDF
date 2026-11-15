"""M6.2 OCR 结果 → 文字层数据适配器（OCR 侧）。

导出内核（pdf_exporter）只依赖中性的 bookclean.text_layer，**不导入本模块**；
本模块把引擎结果 PageOcrResult 转成 PageText，并以导出页索引（0 基）为键提供
text_source 钩子所需的数据。failed/skipped/缺失页返回 None ⇒ 该页不注入文字层，
但正常导出不受影响（OCR 失败不阻塞导出）。
"""

from __future__ import annotations

from pathlib import Path

from ..text_layer import PageText, TextLine
from .provider import STATUS_COMPLETED, PageOcrResult


def page_text_from_result(result: PageOcrResult) -> PageText | None:
    """completed 且有行 ⇒ PageText；否则 None（不注入）。保留引擎原文与像素坐标。"""
    if result.status != STATUS_COMPLETED or not result.lines:
        return None
    lines = [TextLine(text=line.text, bbox_px=tuple(line.bbox_px)) for line in result.lines]
    return PageText(
        lines=lines,
        image_width_px=result.image_width_px,
        image_height_px=result.image_height_px,
    )


class OcrTextSource:
    """text_source 钩子实现：按导出页索引（0 基）返回 PageText 或 None。

    pages 键为 Manifest 的 page_index（0 基），与 PDFExporter.build 传入的 page_index 对齐。
    """

    def __init__(self, pages: dict[int, PageText]):
        self._pages = dict(pages)

    @classmethod
    def from_ocr_results(cls, results: dict[int, PageOcrResult]) -> "OcrTextSource":
        pages: dict[int, PageText] = {}
        for page_index, result in results.items():
            pt = page_text_from_result(result)
            if pt is not None:
                pages[page_index] = pt
        return cls(pages)

    def __call__(self, page_index: int, image_path: Path, layout) -> PageText | None:
        return self._pages.get(page_index)

    def __len__(self) -> int:
        return len(self._pages)