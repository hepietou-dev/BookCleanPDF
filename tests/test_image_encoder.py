"""M4.2 图像编码测试：三模式 / 灰度判定 / DPI 只降采样 / 质量守卫 / 体积与保真。"""

import io

import numpy as np
import pytest
from PIL import Image

import bookclean.image_encoder as ie
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


def gray_text_image(w=600, h=800):
    arr = np.full((h, w), 255, dtype=np.uint8)
    arr[60:h - 60:12, 50:w - 50] = 20
    return Image.fromarray(arr).convert("RGB")  # R==G==B 精确灰度


def tinted_image(w=600, h=800):
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    arr[:, :, 0] = 210  # 青绿色调纸张
    arr[:, :, 1] = 235
    arr[:, :, 2] = 232
    arr[60:h - 60:12, 50:w - 50] = (20, 30, 30)
    return Image.fromarray(arr)


def flat_graphics(w=600, h=800):
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    arr[100:300, 100:500] = (220, 30, 30)
    arr[400:600, 100:500] = (30, 30, 220)
    return Image.fromarray(arr)


def color_photo_image(w=600, h=800, seed=42):
    """带青绿基调的照片式纹理：平滑渐变 + 中频细节 + 轻噪声，使 JPEG 体积优于 PNG 且守卫不触发。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 180 + 40 * np.sin(xx / 120.0) + 25 * np.cos(yy / 150.0)
    g = 210 + 30 * np.sin(xx / 90.0 + 1.0) + 20 * np.sin(yy / 110.0)
    b = 205 + 25 * np.cos(yy / 100.0) + 20 * np.cos(xx / 130.0)
    arr = np.stack([r, g, b], axis=-1)
    arr += (4 * np.sin(xx / 9.0) * np.cos(yy / 11.0))[..., None]
    arr += rng.normal(0, 2.5, (h, w, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def gray_photo_image(w=600, h=800, seed=7):
    """精确灰度（R==G==B）的照片式纹理，使灰度 JPEG 体积优于单通道 PNG。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    lum = 180 + 40 * np.sin(xx / 120.0) + 25 * np.cos(yy / 150.0)
    lum += 4 * np.sin(xx / 9.0) * np.cos(yy / 11.0)
    lum += rng.normal(0, 2.5, (h, w))
    lum = np.clip(lum, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([lum, lum, lum], axis=-1))


def pure_noise_image(w=300, h=400, seed=3):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def save_png(im, tmp_path, name="page.png", dpi=150):
    p = tmp_path / name
    im.save(p, format="PNG", dpi=(dpi, dpi))
    return p


def encode(path, layout, mode="balanced", page_type=None, **cfg):
    return ImageEncoder(PDFExportConfig(compression=mode, **cfg)).encode(path, layout, page_type)


# ---------- Lossless ----------

def test_lossless_png_bytes_identical(tmp_path):
    p = save_png(gray_text_image(), tmp_path)
    enc = encode(p, make_layout(600, 800), "lossless")
    assert enc.data == p.read_bytes()
    assert enc.format == "png"
    assert enc.compression_method == "png_lossless_passthrough"
    assert enc.jpeg_quality is None
    assert not enc.downsampled


def test_lossless_pixel_identical_after_embed(tmp_path):
    import img2pdf
    import pypdfium2 as pdfium
    p = save_png(tinted_image(), tmp_path)
    enc = encode(p, make_layout(600, 800), "lossless")
    pdf = tmp_path / "out.pdf"
    pdf.write_bytes(img2pdf.convert(enc.data))
    doc = pdfium.PdfDocument(str(pdf))
    arr = np.asarray(doc[0].render(scale=150 / 72).to_pil().convert("RGB"))
    src = np.asarray(Image.open(p).convert("RGB"))
    assert np.array_equal(arr, src)
    doc.close()


def test_lossless_preserves_color(tmp_path):
    p = save_png(tinted_image(), tmp_path)
    enc = encode(p, make_layout(600, 800), "lossless")
    assert enc.color_mode == "RGB"


def test_lossless_ignores_target_dpi(tmp_path):
    p = save_png(gray_text_image(), tmp_path)
    enc = encode(p, make_layout(600, 800), "lossless", target_dpi=200)
    assert not enc.downsampled
    assert (enc.width_px, enc.height_px) == (600, 800)
    assert any("lossless" in w for w in enc.warnings)


def test_lossless_jpeg_passthrough(tmp_path):
    p = tmp_path / "in.jpg"
    tinted_image().save(p, format="JPEG", quality=90)
    enc = encode(p, make_layout(600, 800), "lossless")
    assert enc.data == p.read_bytes()
    assert enc.format == "jpeg"


def test_lossless_1bit(tmp_path):
    p = tmp_path / "bw.png"
    gray_text_image().convert("1").save(p)
    enc = encode(p, make_layout(600, 800), "lossless")
    assert enc.compression_method == "png_mono_lossless"
    assert enc.color_mode == "1"
    assert enc.data == p.read_bytes()


# ---------- Balanced ----------

def test_balanced_gray_uses_gray_encoding(tmp_path):
    p = save_png(gray_text_image(), tmp_path)
    enc = encode(p, make_layout(600, 800), "balanced")
    assert enc.color_mode == "L"
    assert enc.compression_method in ("jpeg_gray", "png_gray_lossless")
    assert enc.encoded_size_bytes < p.stat().st_size  # 小于原始 RGB PNG


def test_balanced_color_rgb_jpeg_preserves_tint(tmp_path):
    p = save_png(color_photo_image(), tmp_path)
    enc = encode(p, make_layout(600, 800), "balanced")
    assert enc.compression_method == "jpeg_rgb"
    assert enc.color_mode == "RGB"
    assert enc.jpeg_quality == 95
    dec = np.asarray(Image.open(io.BytesIO(enc.data)).convert("RGB")).astype(np.int16)
    src = np.asarray(Image.open(p).convert("RGB")).astype(np.int16)
    # 色调保留：背景仍偏青绿（G,B > R）
    bg = dec[:40, :40].reshape(-1, 3).mean(axis=0)
    assert bg[1] > bg[0] + 10 and bg[2] > bg[0] + 10
    assert np.abs(dec - src).mean() < 3.0


def test_balanced_png_wins_for_flat_graphics(tmp_path):
    p = save_png(flat_graphics(), tmp_path)
    enc = encode(p, make_layout(600, 800), "balanced")
    # 纯色块 PNG 远小于 JPEG -> 应选无损 PNG
    assert enc.format == "png"
    assert enc.compression_method == "png_lossless"


def test_balanced_quality_guard_falls_back(tmp_path, monkeypatch):
    p = save_png(gray_text_image(), tmp_path)
    monkeypatch.setattr(ie, "JPEG_GUARD_MEAN_DIFF_MAX", 0.0)  # 强制守卫触发
    enc = encode(p, make_layout(600, 800), "balanced")
    assert enc.format == "png"
    assert any("守卫" in w or "回退" in w for w in enc.warnings)


def test_balanced_1bit_passthrough(tmp_path):
    p = tmp_path / "bw.png"
    gray_text_image().convert("1").save(p)
    enc = encode(p, make_layout(600, 800), "balanced")
    assert enc.compression_method == "png_mono_lossless"
    assert enc.format == "png"


def test_balanced_subsampling_444(tmp_path):
    # 验证 JPEG 无色度下采样：解析 SOF0 段采样因子均为 0x11
    p = save_png(color_photo_image(), tmp_path)
    enc = encode(p, make_layout(600, 800), "balanced")
    data = enc.data
    i = 2
    found = False
    while i < len(data) - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC2):  # SOF0 / SOF2
            ncomp = data[i + 9]
            for c in range(ncomp):
                factor = data[i + 11 + c * 3]
                assert factor == 0x11, f"分量 {c} 采样因子 {factor:#x} != 4:4:4"
            found = True
            break
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seglen = int.from_bytes(data[i + 2:i + 4], "big")
        i += 2 + seglen
    assert found, "未找到 SOF 段"


# ---------- Compact ----------

def test_compact_smaller_than_balanced(tmp_path):
    p = save_png(color_photo_image(900, 1200), tmp_path)
    lay = make_layout(900, 1200)
    loss = encode(p, lay, "lossless")
    bal = encode(p, lay, "balanced")
    comp = encode(p, lay, "compact")
    assert comp.encoded_size_bytes < bal.encoded_size_bytes < loss.encoded_size_bytes


def test_compact_text_quality_90(tmp_path):
    p = save_png(gray_photo_image(), tmp_path)
    enc = encode(p, make_layout(600, 800), "compact", page_type="pure_text")
    assert enc.jpeg_quality == 90


def test_compact_color_quality_88(tmp_path):
    p = save_png(color_photo_image(), tmp_path)
    enc = encode(p, make_layout(600, 800), "compact", page_type="photo_dominated")
    assert enc.jpeg_quality == 88


def test_compact_no_binarization(tmp_path):
    p = save_png(gray_photo_image(), tmp_path)
    enc = encode(p, make_layout(600, 800), "compact")
    assert enc.color_mode in ("L", "RGB")
    assert enc.color_mode != "1"
    assert enc.format == "jpeg"


def test_high_entropy_noise_does_not_crash(tmp_path):
    # 纯噪声下 libjpeg optimize 两遍编码可能失败，_jpeg_bytes 应自动退化而不崩溃
    im = pure_noise_image()
    jb = ie._jpeg_bytes(im, 95, 150.0)
    assert Image.open(io.BytesIO(jb)).size == (300, 400)
    p = save_png(im, tmp_path)
    for mode in ("balanced", "compact"):
        enc = encode(p, make_layout(300, 400), mode)
        assert Image.open(io.BytesIO(enc.data)).size == (300, 400)


# ---------- 分辨率 ----------

def test_downsample_300_to_200(tmp_path):
    p = save_png(gray_text_image(900, 1200), tmp_path, dpi=300)
    lay = make_layout(900, 1200, dpi=300)
    enc = encode(p, lay, "balanced", target_dpi=200)
    assert enc.downsampled
    assert (enc.width_px, enc.height_px) == (600, 800)
    assert enc.effective_dpi_x == pytest.approx(200, abs=1.0)
    assert (enc.original_width_px, enc.original_height_px) == (900, 1200)


def test_no_upsample(tmp_path):
    p = save_png(gray_text_image(600, 800), tmp_path, dpi=150)
    lay = make_layout(600, 800, dpi=150)
    enc = encode(p, lay, "balanced", target_dpi=300)
    assert not enc.downsampled
    assert (enc.width_px, enc.height_px) == (600, 800)
    assert enc.effective_dpi_x == pytest.approx(150, abs=0.5)


def test_preserve_no_resize(tmp_path):
    p = save_png(gray_text_image(600, 800), tmp_path)
    enc = encode(p, make_layout(600, 800), "balanced", target_dpi=None)
    assert not enc.downsampled
    assert (enc.width_px, enc.height_px) == (600, 800)


def test_downsample_color_png_candidate_consistent(tmp_path):
    # 下采样后若选 PNG，其尺寸必须是缩放后尺寸（不能混入全分辨率原图）
    p = save_png(flat_graphics(900, 1200), tmp_path, dpi=300)
    lay = make_layout(900, 1200, dpi=300)
    enc = encode(p, lay, "balanced", target_dpi=200)
    assert enc.downsampled
    dec = Image.open(io.BytesIO(enc.data))
    assert dec.size == (enc.width_px, enc.height_px) == (600, 800)


def test_original_size_bytes_accurate(tmp_path):
    p = save_png(tinted_image(), tmp_path)
    for mode in ("lossless", "balanced", "compact"):
        enc = encode(p, make_layout(600, 800), mode)
        assert enc.original_size_bytes == p.stat().st_size


# ---------- 真实书页面 ----------

REAL_TASK = "temp/m3_real_book/tasks/task_20261111_230829_24bdec"


def _real_page(tmp_path):
    import json
    from pathlib import Path
    task = Path(REAL_TASK)
    if not (task / "manifest.json").exists():
        pytest.skip("真实书任务目录不存在")
    m = json.loads((task / "manifest.json").read_text(encoding="utf-8"))
    p = m["pages"][0]
    return task / p["processed_image_path"], p


def test_real_book_three_modes_ordering(tmp_path):
    from bookclean.page_layout import PageLayoutResolver
    fp, p = _real_page(tmp_path)
    im = Image.open(fp)
    lay = PageLayoutResolver(PDFExportConfig()).resolve(p, im.width, im.height)
    im.close()
    sizes = {}
    for mode in ("lossless", "balanced", "compact"):
        enc = encode(fp, lay, mode, p.get("page_type"))
        sizes[mode] = enc.encoded_size_bytes
    assert sizes["compact"] < sizes["balanced"] < sizes["lossless"]


def test_real_book_balanced_preserves_tint(tmp_path):
    from bookclean.page_layout import PageLayoutResolver
    fp, p = _real_page(tmp_path)
    im = Image.open(fp)
    lay = PageLayoutResolver(PDFExportConfig()).resolve(p, im.width, im.height)
    im.close()
    enc = encode(fp, lay, "balanced", p.get("page_type"))
    assert enc.color_mode == "RGB"  # 真实页带色调，不能误判灰度
    dec = np.asarray(Image.open(io.BytesIO(enc.data)).convert("RGB")).astype(np.int16)
    src = np.asarray(Image.open(fp).convert("RGB")).astype(np.int16)
    assert np.abs(dec - src).mean() < 2.0