"""PDF 读取与逐页渲染（pypdfium2）。

已实测的 pypdfium2 4.30 行为：
- page.get_size() 返回应用 /Rotate 后的视觉尺寸（点）；get_rotation() 返回 /Rotate 角度；
- page.render(scale=dpi/72) 自动应用页面 /Rotate，输出即位图最终方向；
- 密码错误/缺失抛 PdfiumError("Incorrect password error")；损坏抛 "Data format error"；
- 零页 PDF 在打开时即抛 PdfiumError；
- PdfDocument / PdfPage / PdfBitmap 均有 close()，必须显式释放。
"""

import logging
from pathlib import Path
from typing import Callable, Iterator

import pypdfium2 as pdfium

from .errors import PDFCorruptError, PDFError, PDFPasswordError
from .models import PageData
from .utils.image_utils import to_rgb_array

logger = logging.getLogger("bookclean")

OnError = Callable[[int, Exception], None]


class PDFReader:
    def __init__(self, password: str | None = None):
        self._password = password

    def _open(self, pdf_path: str | Path) -> pdfium.PdfDocument:
        path = Path(pdf_path)
        if not path.is_file():
            raise PDFError(f"PDF 文件不存在: {path}")
        try:
            pdf = pdfium.PdfDocument(str(path), password=self._password)
        except pdfium.PdfiumError as exc:
            if "Incorrect password" in str(exc):
                raise PDFPasswordError(f"PDF 受密码保护或密码错误: {path}") from exc
            raise PDFCorruptError(f"PDF 损坏或无法读取（含零页文档）: {path} ({exc})") from exc
        if len(pdf) == 0:
            pdf.close()
            raise PDFError(f"PDF 页数为零: {path}")
        return pdf

    def get_page_count(self, pdf_path: str | Path) -> int:
        pdf = self._open(pdf_path)
        try:
            return len(pdf)
        finally:
            pdf.close()

    def get_page_info(self, pdf_path: str | Path, page_index: int) -> dict:
        pdf = self._open(pdf_path)
        try:
            self._check_index(pdf, page_index, pdf_path)
            page = pdf[page_index]
            try:
                width_pt, height_pt = page.get_size()
                return {
                    "page_index": page_index,
                    "width_pt": float(width_pt),
                    "height_pt": float(height_pt),
                    "rotation": int(page.get_rotation()),
                }
            finally:
                page.close()
        finally:
            pdf.close()

    def render_page(self, pdf_path: str | Path, page_index: int, dpi: int = 300) -> PageData:
        pdf = self._open(pdf_path)
        try:
            self._check_index(pdf, page_index, pdf_path)
            return self._render_one(pdf, Path(pdf_path), page_index, dpi)
        finally:
            pdf.close()

    def iter_pages(
        self,
        pdf_path: str | Path,
        dpi: int = 300,
        on_error: OnError | None = None,
    ) -> Iterator[PageData]:
        """逐页渲染。单页失败通过 on_error(页索引, 异常) 报告并继续下一页。"""
        path = Path(pdf_path)
        pdf = self._open(path)
        try:
            for index in range(len(pdf)):
                try:
                    yield self._render_one(pdf, path, index, dpi)
                except Exception as exc:
                    logger.error("PDF 渲染失败 %s 第 %d 页: %s", path.name, index + 1, exc)
                    if on_error is not None:
                        on_error(index, exc)
        finally:
            pdf.close()

    @staticmethod
    def _check_index(pdf: pdfium.PdfDocument, page_index: int, path: Path) -> None:
        if not 0 <= page_index < len(pdf):
            raise PDFError(f"页索引越界: {path} 第 {page_index} 页（共 {len(pdf)} 页）")

    @staticmethod
    def _render_one(pdf: pdfium.PdfDocument, path: Path, index: int, dpi: int) -> PageData:
        page = pdf[index]
        try:
            width_pt, height_pt = page.get_size()
            rotation = int(page.get_rotation())
            bitmap = page.render(scale=dpi / 72)
            try:
                arr = to_rgb_array(bitmap.to_pil())
            finally:
                bitmap.close()
        finally:
            page.close()
        height, width = arr.shape[:2]
        return PageData(
            page_index=index,
            image=arr,
            source_path=str(path),
            source_kind="pdf",
            original_width=width,
            original_height=height,
            source_dpi=float(dpi),
            color_mode="RGB",
            metadata={
                "source_page_index": index,
                "width_pt": float(width_pt),
                "height_pt": float(height_pt),
                "rotation": rotation,
                "render_dpi": dpi,
            },
        )