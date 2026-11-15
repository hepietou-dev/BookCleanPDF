"""M4.1 页面布局与物理尺寸管理。

尺寸来源优先级：
- PDF 来源：M1 Manifest metadata.width_pt/height_pt（pypdfium2 get_size 已应用 /Rotate，
  此处不得再次交换宽高）。
- 图片来源：原始文件的可信扫描 DPI（150..1200 且 x≈y）反推物理尺寸；
  72/96 等手机默认值与缺失 DPI 不可信，按 Auto 长边参考尺寸估算（默认 250mm）。
- 标准纸张（A4/A5/ISO B5/Letter）按图像方向自动横竖；Custom 按用户给定尺寸。

放置：Contain（默认，等比缩放居中白边）/ Cover（等比覆盖，可能裁切，警告）。
禁止 Stretch。有效 DPI 按图像像素 / PDF 中实际显示尺寸计算，不采信文件 DPI 标称。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .config import PDFExportConfig

logger = logging.getLogger("bookclean")

MM_TO_PT = 72.0 / 25.4

PAPER_SIZES_MM = {
    "A4": (210.0, 297.0),
    "A5": (148.0, 210.0),
    "B5": (176.0, 250.0),  # ISO B5
    "Letter": (215.9, 279.4),
}

TRUSTED_DPI_MIN = 150.0
TRUSTED_DPI_MAX = 1200.0
_DPI_ANISOTROPY_TOL = 0.01  # x/y 相对差超过 1% 视为不可信


@dataclass
class PageLayout:
    page_index: int
    page_width_pt: float
    page_height_pt: float
    image_width_px: int
    image_height_px: int
    image_display_width_pt: float
    image_display_height_pt: float
    effective_dpi_x: float
    effective_dpi_y: float
    fit_mode: str
    source_page_size: str
    estimated_size: bool
    warnings: list[str] = field(default_factory=list)


def _trusted_dpi(dpi: float | None) -> bool:
    return dpi is not None and TRUSTED_DPI_MIN <= dpi <= TRUSTED_DPI_MAX


def _read_source_dpi(source_path: str) -> tuple[float | None, bool]:
    """读取原始图片文件的真实 DPI。返回 (dpi, 是否成功读取文件)。"""
    try:
        with Image.open(source_path) as im:
            dpi = im.info.get("dpi")
    except Exception:
        return None, False
    if not dpi:
        return None, True
    x, y = float(dpi[0]), float(dpi[1])
    if x <= 0 or y <= 0 or abs(x - y) / max(x, y) > _DPI_ANISOTROPY_TOL:
        return None, True
    return x, True


class PageLayoutResolver:
    def __init__(self, config: PDFExportConfig):
        config.validate()
        self.config = config

    def resolve(self, page: dict, image_width_px: int, image_height_px: int) -> PageLayout:
        """按 Manifest 页记录与最终图像实际像素尺寸计算页面布局。

        像素尺寸必须取最终图像（M2 裁边可能改变尺寸），不能用 M1 原始宽高。
        """
        if image_width_px <= 0 or image_height_px <= 0:
            raise ValueError(f"非法图像尺寸: {image_width_px}x{image_height_px}")
        warnings: list[str] = []
        mode = self.config.page_size_mode
        meta = page.get("metadata") or {}

        if mode in PAPER_SIZES_MM:
            pw, ph = self._standard_size(mode, image_width_px, image_height_px)
            source, estimated = f"standard_{mode}", False
        elif mode == "custom":
            pw = self.config.custom_width_mm * MM_TO_PT
            ph = self.config.custom_height_mm * MM_TO_PT
            source, estimated = "custom", False
        elif mode == "preserve" and page.get("source_kind") == "pdf":
            pw, ph, source, estimated = self._pdf_size(page, meta, image_width_px, image_height_px, warnings)
        elif mode == "preserve":
            pw, ph, source, estimated = self._image_size(page, image_width_px, image_height_px, warnings)
        else:  # auto：全部页面按长边估算
            pw, ph = self._auto_size(image_width_px, image_height_px)
            source, estimated = "auto_estimated", True
            warnings.append(f"页面尺寸按长边 {self.config.auto_long_edge_mm:g}mm 自动估算，非原始纸张尺寸")

        display_w, display_h = self._fit(pw, ph, image_width_px, image_height_px, warnings)
        return PageLayout(
            page_index=page["page_index"],
            page_width_pt=pw,
            page_height_pt=ph,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            image_display_width_pt=display_w,
            image_display_height_pt=display_h,
            effective_dpi_x=image_width_px / display_w * 72.0,
            effective_dpi_y=image_height_px / display_h * 72.0,
            fit_mode=self.config.fit_mode,
            source_page_size=source,
            estimated_size=estimated,
            warnings=warnings,
        )

    def _standard_size(self, mode: str, wpx: int, hpx: int) -> tuple[float, float]:
        wmm, hmm = PAPER_SIZES_MM[mode]
        if wpx > hpx:  # 横图 -> 横版纸张
            wmm, hmm = hmm, wmm
        return wmm * MM_TO_PT, hmm * MM_TO_PT

    def _pdf_size(self, page: dict, meta: dict, wpx: int, hpx: int, warnings: list[str]) -> tuple[float, float, str, bool]:
        w, h = meta.get("width_pt"), meta.get("height_pt")
        if isinstance(w, (int, float)) and isinstance(h, (int, float)) and w > 0 and h > 0:
            # get_size() 已含 /Rotate 视觉方向，rotation 仅作记录，不再交换
            return float(w), float(h), "pdf_original", False
        dpi = page.get("dpi")
        if _trusted_dpi(dpi):
            warnings.append("PDF 页缺少物理尺寸元数据，按渲染 DPI 反推")
            return wpx / dpi * 72.0, hpx / dpi * 72.0, "pdf_render_dpi", True
        warnings.append("PDF 页缺少物理尺寸元数据，按长边自动估算")
        pw, ph = self._auto_size(wpx, hpx)
        return pw, ph, "auto_estimated", True

    def _image_size(self, page: dict, wpx: int, hpx: int, warnings: list[str]) -> tuple[float, float, str, bool]:
        dpi, readable = _read_source_dpi(page.get("source_path", ""))
        if readable:
            if _trusted_dpi(dpi):
                return wpx / dpi * 72.0, hpx / dpi * 72.0, "trusted_dpi", False
            pw, ph = self._auto_size(wpx, hpx)
            reason = "无 DPI 元数据" if dpi is None else f"DPI {dpi:g} 不在可信扫描范围"
            warnings.append(f"{reason}，页面尺寸按长边 {self.config.auto_long_edge_mm:g}mm 估算，非原始纸张尺寸")
            return pw, ph, "auto_estimated", True
        # 原始文件不可访问：只能用任务记录 DPI，无法核实，按估算对待
        dpi = page.get("dpi")
        if _trusted_dpi(dpi):
            warnings.append("原始图片不可访问，DPI 取自任务记录（未核实），物理尺寸为推算值")
            return wpx / dpi * 72.0, hpx / dpi * 72.0, "task_recorded_dpi", True
        pw, ph = self._auto_size(wpx, hpx)
        warnings.append(f"原始图片不可访问且任务 DPI 不可信，按长边 {self.config.auto_long_edge_mm:g}mm 估算")
        return pw, ph, "auto_estimated", True

    def _auto_size(self, wpx: int, hpx: int) -> tuple[float, float]:
        long_edge_pt = self.config.auto_long_edge_mm * MM_TO_PT
        if wpx >= hpx:
            return long_edge_pt, long_edge_pt * hpx / wpx
        return long_edge_pt * wpx / hpx, long_edge_pt

    def _fit(self, pw: float, ph: float, wpx: int, hpx: int, warnings: list[str]) -> tuple[float, float]:
        if self.config.fit_mode == "cover":
            scale = max(pw / wpx, ph / hpx)
            warnings.append("Cover 模式等比覆盖页面，边缘可能被裁切")
        else:
            scale = min(pw / wpx, ph / hpx)
        return wpx * scale, hpx * scale