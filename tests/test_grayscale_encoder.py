"""M4.7.2 灰度编码测试：三模式单通道 8-bit / lossless 像素无损 / 已灰度跳过 / RGBA / 1-bit / 体积实测。"""

import io

import numpy as np
import pytest
from PIL import Image

from bookclean.color_converter import to_grayscale
from bookclean.config import PDFExportConfig
from bookclean.image_encoder import ImageEncoder
from bookclean.page_layout import PageLayout


def make_layout(w, h, dpi=150.0):
    dw, dh = w / dpi * 72.0, h / dpi * 72.0
    return PageLayout(
        page_index=0, page_width_pt=dw, page_height_pt=dh,
        image_width_px=w, image_height_px=h,
        image_display_width_pt=dw, image_display_height_pt=dh,
        effective_dpi_x=dpi, effective_dpi_y=dpi,
        fit_mode="contain", source_page_size="trusted_dpi", estimated_size=False,
    )


def color_page(w=400, h=560):
    rng = np.random.default_rng(7)
    arr = np.full((h, w, 3), 250, dtype=np.uint8)
    arr[:, :, 0] = 235  # 带色调纸张
    arr[:, :, 1] = 245
    arr[:, :, 2] = 240
    arr[40:h - 40:10, 30:w - 30] = (20, 30, 30)      # 文字
    arr[100:200, 50:200] = (200, 30, 30)            # 红章
    arr[300:400, 100:300] = rng.integers(0, 256, (100, 200, 3), dtype=np.uint8)  # 照片
    return Image.fromarray(arr)


def gray_page(w=400, h=560):
    arr = np.full((h, w), 250, dtype=np.uint8)
    arr[40:h - 40:10, 30:w - 30] = 20
    return Image.fromarray(arr)


def _save(tmp_path, im, name="page.png"):
    p = tmp_path / name
    im.save(p)
    return p


def _decode(enc):
    return Image.open(io.BytesIO(enc.data))


@pytest.mark.parametrize("compression", ["lossless", "balanced", "compact"])
def test_grayscale_single_channel_8bit(tmp_path, compression):
    src = color_page()
    p = _save(tmp_path, src)
    cfg = PDFExportConfig(compression=compression, color_mode="grayscale")
    enc = ImageEncoder(cfg).encode(p, make_layout(*src.size))
    assert enc.color_mode == "L"
    dec = _decode(enc)
    arr = np.asarray(dec.convert("L"))
    assert arr.ndim == 2 and arr.dtype == np.uint8
    assert dec.mode in ("L", "1") and dec.mode != "1"  # 8-bit，非 1-bit
    assert enc.grayscale_converted and enc.source_color_mode == "RGB"


@pytest.mark.parametrize("compression", ["lossless", "balanced", "compact"])
def test_grayscale_never_rgb(tmp_path, compression):
    src = color_page()
    p = _save(tmp_path, src)
    cfg = PDFExportConfig(compression=compression, color_mode="grayscale")
    enc = ImageEncoder(cfg).encode(p, make_layout(*src.size))
    dec = _decode(enc)
    assert dec.mode != "RGB" and dec.mode != "RGBA"
    assert enc.format in ("png", "jpeg")


def test_lossless_grayscale_pixel_exact(tmp_path):
    src = color_page()
    p = _save(tmp_path, src)
    cfg = PDFExportConfig(compression="lossless", color_mode="grayscale")
    enc = ImageEncoder(cfg).encode(p, make_layout(*src.size))
    assert enc.format == "png" and enc.jpeg_quality is None
    expect = to_grayscale(np.asarray(src.convert("RGB"))).gray
    got = np.asarray(_decode(enc).convert("L"))
    assert np.array_equal(got, expect)  # 像素级无损


def test_lossless_grayscale_no_downsample(tmp_path):
    src = color_page()
    p = _save(tmp_path, src)
    cfg = PDFExportConfig(compression="lossless", color_mode="grayscale", target_dpi=200)
    enc = ImageEncoder(cfg).encode(p, make_layout(*src.size))
    assert not enc.downsampled
    assert enc.width_px == src.width and enc.height_px == src.height


@pytest.mark.parametrize("compression", ["balanced", "compact"])
def test_gray_jpeg_is_single_channel(tmp_path, compression):
    src = color_page()
    p = _save(tmp_path, src)
    cfg = PDFExportConfig(compression=compression, color_mode="grayscale")
    enc = ImageEncoder(cfg).encode(p, make_layout(*src.size))
    if enc.format == "jpeg":
        dec = _decode(enc)
        assert dec.mode == "L"  # 灰度 JPEG，无色度通道
        assert enc.jpeg_quality in (95, 90)


def test_already_grayscale_L_skips(tmp_path):
    src = gray_page()
    p = _save(tmp_path, src)
    cfg = PDFExportConfig(compression="balanced", color_mode="grayscale")
    enc = ImageEncoder(cfg).encode(p, make_layout(*src.size))
    assert enc.already_grayscale and not enc.grayscale_converted
    assert enc.source_color_mode == "L"
    assert enc.color_mode == "L"


def test_already_grayscale_rgb_equal(tmp_path):
    g = gray_page()
    src = Image.merge("RGB", (g, g, g))
    p = _save(tmp_path, src)
    cfg = PDFExportConfig(compression="balanced", color_mode="grayscale")
    enc = ImageEncoder(cfg).encode(p, make_layout(*src.size))
    assert enc.already_grayscale and not enc.grayscale_converted
    assert enc.source_color_mode == "RGB"


def test_rgba_transparent_white(tmp_path):
    rgba = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    p = _save(tmp_path, rgba, "rgba.png")
    cfg = PDFExportConfig(compression="lossless", color_mode="grayscale")
    enc = ImageEncoder(cfg).encode(p, make_layout(200, 200))
    got = np.asarray(_decode(enc).convert("L"))
    assert int(got[0, 0]) == 255
    assert enc.source_color_mode == "RGBA"


def test_1bit_becomes_8bit(tmp_path):
    mono = gray_page().convert("1")
    p = _save(tmp_path, mono, "mono.png")
    cfg = PDFExportConfig(compression="lossless", color_mode="grayscale")
    enc = ImageEncoder(cfg).encode(p, make_layout(*mono.size))
    dec = _decode(enc)
    assert dec.mode == "L"  # 8-bit，而非 1-bit
    assert enc.color_mode == "L"


def test_preserve_unchanged_color(tmp_path):
    src = color_page()
    p = _save(tmp_path, src)
    cfg = PDFExportConfig(compression="balanced", color_mode="preserve")
    enc = ImageEncoder(cfg).encode(p, make_layout(*src.size))
    assert enc.color_mode == "RGB"  # preserve 不转灰度
    assert not enc.grayscale_converted


def test_size_ordering_measured(tmp_path):
    src = color_page()
    p = _save(tmp_path, src)
    sizes = {}
    for comp in ("lossless", "balanced", "compact"):
        cfg = PDFExportConfig(compression=comp, color_mode="grayscale")
        enc = ImageEncoder(cfg).encode(p, make_layout(*src.size))
        sizes[comp] = enc.encoded_size_bytes
    # 实测：有损模式取 min(PNG,JPEG)，故不会超过无损 PNG；质量守卫回退 PNG 时允许相等。
    # 不预设 balanced/compact 相对大小（任务要求实际测量）。
    assert sizes["lossless"] >= sizes["balanced"]
    assert sizes["lossless"] >= sizes["compact"]


def test_compact_downsample_only(tmp_path):
    src = color_page(800, 1120)
    p = _save(tmp_path, src)
    cfg = PDFExportConfig(compression="compact", color_mode="grayscale", target_dpi=200)
    enc = ImageEncoder(cfg).encode(p, make_layout(*src.size, dpi=300))
    assert enc.downsampled
    assert enc.width_px < src.width
    assert enc.color_mode == "L"