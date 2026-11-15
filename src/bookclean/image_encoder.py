"""M4.2 页面图像编码与压缩。

独立于 M2 修复：不修改已通过 M3 质量检查的图像内容，只做编码与（可选）下采样。
三种模式：
- lossless：PNG/JPEG 原始字节直通（img2pdf 已验证原样嵌入），不做有损重编码，不下采样。
- balanced（默认）：按实际像素判定灰度/彩色；无损 PNG 与高质量 JPEG(q95, 4:4:4) 比体积取小，
  JPEG 设质量守卫（细小文字伪影超阈值则回退 PNG）。
- compact：JPEG q88–90（文字页保护取 90），可按目标 DPI 适度下采样；不二值化、不用 JBIG2。

分辨率：preserve（默认）/200/300/400，只允许下采样，不上采样。Lanczos 单次缩放。
灰度判定按像素 R==G==B 全等，而非 page_type（真实书纸张带色调，color_area_ratio=0 仍可能是彩色像素）。
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .config import PDFExportConfig
from .color_converter import to_grayscale
from .page_layout import PageLayout

logger = logging.getLogger("bookclean")

BALANCED_JPEG_QUALITY = 95
COMPACT_JPEG_QUALITY_COLOR = 88
COMPACT_JPEG_QUALITY_TEXT = 90
# JPEG 质量守卫（启发式像素口径，非 OCR 证明）：解码后与源平均绝对差超阈值则回退 PNG
JPEG_GUARD_MEAN_DIFF_MAX = 2.5
MIN_DOWNSAMPLE_EDGE_PX = 16


@dataclass
class EncodedImage:
    data: bytes
    format: str  # "png" / "jpeg"
    compression_method: str
    jpeg_quality: int | None
    width_px: int
    height_px: int
    original_width_px: int
    original_height_px: int
    effective_dpi_x: float
    effective_dpi_y: float
    color_mode: str  # "RGB" / "L" / "1"
    original_size_bytes: int
    encoded_size_bytes: int
    downsampled: bool
    warnings: list[str] = field(default_factory=list)
    source_color_mode: str | None = None  # 转换前 PIL 模式（grayscale 模式记录）
    already_grayscale: bool = False
    grayscale_converted: bool = False


def _is_exactly_gray(arr: np.ndarray) -> bool:
    return bool(np.array_equal(arr[:, :, 0], arr[:, :, 1]) and np.array_equal(arr[:, :, 1], arr[:, :, 2]))


def _jpeg_bytes(im: Image.Image, quality: int, dpi: float) -> bytes:
    buf = io.BytesIO()
    try:
        im.save(buf, format="JPEG", quality=quality, subsampling=0, optimize=True, dpi=(dpi, dpi))
    except OSError:
        # 极高熵图像（如纯噪声）下 libjpeg 的 optimize 两遍编码可能失败；退化为单遍（仍为合法 JPEG）
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, subsampling=0, optimize=False, dpi=(dpi, dpi))
    return buf.getvalue()


def _png_bytes(im: Image.Image, dpi: float) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG", dpi=(dpi, dpi))
    return buf.getvalue()


class ImageEncoder:
    def __init__(self, config: PDFExportConfig):
        config.validate()
        self.config = config

    def encode(self, image_path: str | Path, layout: PageLayout, page_type: str | None = None) -> EncodedImage:
        """逐页编码。任意时刻内存中只有一页的解码像素。"""
        image_path = Path(image_path)
        raw = image_path.read_bytes()
        im = Image.open(io.BytesIO(raw))
        im.load()
        warnings: list[str] = []
        orig_w, orig_h = im.size
        src_format = (im.format or "PNG").upper()

        if self.config.color_mode == "grayscale":
            return self._encode_grayscale(raw, im, layout, warnings)

        if self.config.compression == "lossless":
            return self._lossless(raw, im, src_format, layout, warnings)

        # 颜色内容判定（按实际像素）
        if im.mode == "1":
            # 1-bit：无损单色 PNG 直通（JPEG 不支持单色；不重二值化、不下采样）
            if self.config.target_dpi is not None:
                warnings.append("1-bit 页面保留原始像素，忽略 target_dpi")
            eff_x = orig_w / layout.image_display_width_pt * 72.0
            eff_y = orig_h / layout.image_display_height_pt * 72.0
            data = raw if src_format == "PNG" else _png_bytes(im, eff_x)
            return self._result(data, "png", "png_mono_lossless", None, im, layout,
                                orig_w, orig_h, eff_x, eff_y, "1", len(raw), False, warnings)

        rgb = im.convert("RGB")
        arr = np.asarray(rgb)
        gray = _is_exactly_gray(arr)
        color_mode = "L" if gray else "RGB"
        im = rgb

        # 分辨率处理：只降采样
        im, downsampled, eff_x, eff_y = self._resize(im, layout, warnings)
        if downsampled and color_mode == "L":
            im = im.convert("L")

        if self.config.compression == "balanced":
            return self._balanced(raw, im, src_format, gray, layout, page_type,
                                  orig_w, orig_h, eff_x, eff_y, downsampled, len(raw), warnings)
        return self._compact(im, gray, layout, page_type,
                             orig_w, orig_h, eff_x, eff_y, downsampled, len(raw), warnings)

    # ---------- 灰度模式（M4.7） ----------

    @staticmethod
    def _gray_source_array(im: Image.Image) -> np.ndarray:
        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
            return np.asarray(im.convert("RGBA"))
        if im.mode == "1":
            return np.asarray(im.convert("L"))
        if im.mode == "L":
            return np.asarray(im)
        return np.asarray(im.convert("RGB"))

    def _encode_grayscale(self, raw, im, layout, warnings) -> EncodedImage:
        """整页转 8-bit 单通道灰度后编码；三模式均不产生 RGB / 1-bit 输出。"""
        source_mode = im.mode
        res = to_grayscale(self._gray_source_array(im))
        gray_im = Image.fromarray(res.gray)
        meta = dict(source_color_mode=source_mode,
                    already_grayscale=res.already_grayscale,
                    grayscale_converted=res.converted)
        orig_w, orig_h = im.size

        if self.config.compression == "lossless":
            if self.config.target_dpi is not None:
                warnings.append("lossless 模式保留原始像素，忽略 target_dpi（不下采样）")
            eff_x = gray_im.width / layout.image_display_width_pt * 72.0
            eff_y = gray_im.height / layout.image_display_height_pt * 72.0
            data = _png_bytes(gray_im, eff_x)
            return self._result(data, "png", "png_gray_lossless", None, gray_im, layout,
                                orig_w, orig_h, eff_x, eff_y, "L", len(raw), False, warnings, **meta)

        gray_im, downsampled, eff_x, eff_y = self._resize(gray_im, layout, warnings)
        if gray_im.mode != "L":
            gray_im = gray_im.convert("L")
        quality = (BALANCED_JPEG_QUALITY if self.config.compression == "balanced"
                   else COMPACT_JPEG_QUALITY_TEXT)
        return self._choose(None, gray_im, "PNG", True, layout,
                            orig_w, orig_h, eff_x, eff_y, downsampled, len(raw), warnings,
                            quality=quality, **meta)

    # ---------- 模式实现 ----------

    def _lossless(self, raw, im, src_format, layout, warnings) -> EncodedImage:
        if self.config.target_dpi is not None:
            warnings.append("lossless 模式保留原始像素，忽略 target_dpi（不下采样）")
        eff_x = im.width / layout.image_display_width_pt * 72.0
        eff_y = im.height / layout.image_display_height_pt * 72.0
        if im.mode == "1":
            method = "png_mono_lossless"
            data, fmt = (raw, "png") if src_format == "PNG" else (_png_bytes(im, eff_x), "png")
        elif src_format in ("PNG", "JPEG"):
            data, fmt = raw, src_format.lower()
            method = f"{fmt}_lossless_passthrough"
        else:
            data, fmt = _png_bytes(im, eff_x), "png"
            method = "png_lossless_reencode"
        color_mode = im.mode if im.mode in ("RGB", "L", "1") else "RGB"
        return self._result(data, fmt, method, None, im, layout, im.width, im.height,
                            eff_x, eff_y, color_mode, len(raw), False, warnings)

    def _balanced(self, raw, im, src_format, gray, layout, page_type,
                  orig_w, orig_h, eff_x, eff_y, downsampled, original_size, warnings) -> EncodedImage:
        return self._choose(raw, im, src_format, gray, layout,
                            orig_w, orig_h, eff_x, eff_y, downsampled, original_size, warnings,
                            quality=BALANCED_JPEG_QUALITY)

    def _compact(self, im, gray, layout, page_type,
                 orig_w, orig_h, eff_x, eff_y, downsampled, original_size, warnings) -> EncodedImage:
        is_text = page_type in ("pure_text",)
        quality = COMPACT_JPEG_QUALITY_TEXT if (gray or is_text) else COMPACT_JPEG_QUALITY_COLOR
        # compact 同样取无损 PNG 与 JPEG 的较小者，保证不会比 balanced 更大（平坦图形 PNG 更优）
        return self._choose(None, im, "PNG", gray, layout,
                            orig_w, orig_h, eff_x, eff_y, downsampled, original_size, warnings,
                            quality=quality)

    def _choose(self, raw, im, src_format, gray, layout,
                orig_w, orig_h, eff_x, eff_y, downsampled, original_size, warnings, quality,
                source_color_mode=None, already_grayscale=False, grayscale_converted=False) -> EncodedImage:
        # 无损候选：灰度页用单通道 PNG（对 R==G==B 内容无损且更小），彩色页用 PNG
        if gray:
            lossless_data = _png_bytes(im.convert("L"), eff_x)
            lossless_method = "png_gray_lossless"
        elif raw is not None and src_format == "PNG" and not downsampled:
            lossless_data = raw  # 未缩放彩色 PNG 直通
            lossless_method = "png_lossless"
        else:
            lossless_data = _png_bytes(im, eff_x)
            lossless_method = "png_lossless"
        # 有损候选：JPEG 4:4:4
        jpeg_im = im.convert("L") if gray else im
        jpeg_data = _jpeg_bytes(jpeg_im, quality, eff_x)
        jpeg_method = "jpeg_gray" if gray else "jpeg_rgb"

        # 质量守卫：JPEG 解码与源差异过大（细小文字伪影）-> 强制无损
        guard_diff = self._jpeg_mean_diff(jpeg_data, jpeg_im)
        if guard_diff > JPEG_GUARD_MEAN_DIFF_MAX:
            warnings.append(f"JPEG 平均像素差 {guard_diff:.2f} 超阈值，回退无损 PNG 保护文字")
            chosen = (lossless_data, "png", lossless_method, None)
        elif len(jpeg_data) < len(lossless_data):
            chosen = (jpeg_data, "jpeg", jpeg_method, quality)
        else:
            chosen = (lossless_data, "png", lossless_method, None)

        data, fmt, method, q = chosen
        color_mode = "L" if gray else "RGB"
        return self._result(data, fmt, method, q, im, layout, orig_w, orig_h,
                            eff_x, eff_y, color_mode, original_size, downsampled, warnings,
                            source_color_mode=source_color_mode, already_grayscale=already_grayscale,
                            grayscale_converted=grayscale_converted)

    # ---------- 工具 ----------

    def _resize(self, im: Image.Image, layout: PageLayout, warnings: list[str]):
        eff_x = im.width / layout.image_display_width_pt * 72.0
        eff_y = im.height / layout.image_display_height_pt * 72.0
        target = self.config.target_dpi
        if target is None or eff_x <= target:
            return im, False, eff_x, eff_y
        factor = target / eff_x
        new_w = max(MIN_DOWNSAMPLE_EDGE_PX, round(im.width * factor))
        new_h = max(MIN_DOWNSAMPLE_EDGE_PX, round(im.height * factor))
        if new_w >= im.width and new_h >= im.height:
            return im, False, eff_x, eff_y
        resized = im.resize((new_w, new_h), Image.LANCZOS)  # 单次缩放
        nx = new_w / layout.image_display_width_pt * 72.0
        ny = new_h / layout.image_display_height_pt * 72.0
        warnings.append(f"下采样 {im.width}x{im.height} -> {new_w}x{new_h}（有效 DPI {eff_x:.0f} -> {nx:.0f}）")
        return resized, True, nx, ny

    @staticmethod
    def _jpeg_mean_diff(jpeg_data: bytes, reference: Image.Image) -> float:
        dec = Image.open(io.BytesIO(jpeg_data)).convert(reference.mode)
        a = np.asarray(reference, dtype=np.int16)
        b = np.asarray(dec, dtype=np.int16)
        return float(np.abs(a - b).mean())

    @staticmethod
    def _result(data, fmt, method, quality, im, layout, orig_w, orig_h,
                eff_x, eff_y, color_mode, original_size, downsampled, warnings,
                source_color_mode=None, already_grayscale=False, grayscale_converted=False) -> EncodedImage:
        return EncodedImage(
            data=data, format=fmt, compression_method=method, jpeg_quality=quality,
            width_px=im.width, height_px=im.height,
            original_width_px=orig_w, original_height_px=orig_h,
            effective_dpi_x=eff_x, effective_dpi_y=eff_y, color_mode=color_mode,
            original_size_bytes=original_size, encoded_size_bytes=len(data),
            downsampled=downsampled, warnings=warnings,
            source_color_mode=source_color_mode, already_grayscale=already_grayscale,
            grayscale_converted=grayscale_converted,
        )