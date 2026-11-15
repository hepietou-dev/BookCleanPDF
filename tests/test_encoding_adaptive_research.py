"""M4.8.3 逐页自适应编码研究结论锁定测试。

研究结论（temp/m4_8_review/encoding_research.json，真实 120 页 + 合成 10 页实测）：
在用户选定档位内，自适应降低 JPEG 质量虽可大幅减体积（guard_min 24–47%），但会牺牲文字锐度
与彩色保真——而全局 mean-diff 守卫（≤2.5）捕捉不到这类局部退化（真实纯文字页 q82 均差仅 0.25
仍远低于阈值，文字边缘却已软化）。唯一内容安全的自适应（text_safe）整体仅省 0.85–1.61%，
且全部来自 6/120 稀有照片页，不稳定、不具普适性。故保留固定策略（balanced q95 / compact 文字灰度
q90、彩色 q88；灰度 compact 恒 q90）。

本模块锁定该决策的两个关键事实（非 OCR，启发式像素口径）：
1. mean-diff 守卫是全局上界，通过守卫不等于保真——低质量仍可大幅退化却过关。
2. 编码器刻意保持固定质量，放弃「守卫内最小体积」，以保护文字/彩色内容。
"""

import io

import numpy as np
from PIL import Image

import bookclean.image_encoder as ie
from bookclean.config import PDFExportConfig
from bookclean.image_encoder import ImageEncoder
from bookclean.page_layout import PageLayout
from scripts.make_m2_samples import text_page


def _layout(w, h, dpi=150.0):
    dw, dh = w / dpi * 72.0, h / dpi * 72.0
    return PageLayout(0, dw, dh, w, h, dw, dh, dpi, dpi,
                      "contain", "trusted_dpi", False)


def _save(im, tmp_path, name="page.png"):
    p = tmp_path / name
    im.save(p, format="PNG", dpi=(150, 150))
    return p


def _text_gray_image():
    """真实感渲染文字页 -> 精确灰度（R==G==B），抗锯齿 glyph 边缘。"""
    rgb, _ = text_page(1000, 1400, lang="en")
    lum = np.round(rgb.astype(np.float64).mean(axis=2)).astype(np.uint8)
    return Image.fromarray(np.stack([lum, lum, lum], axis=-1))


def _gray_texture_image(w=600, h=800, seed=7):
    """JPEG 友好的精确灰度纹理（平滑渐变 + 中频 + 轻噪声），使编码选 JPEG 而非 PNG。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    lum = (180 + 40 * np.sin(xx / 120.0) + 25 * np.cos(yy / 150.0)
           + 4 * np.sin(xx / 9.0) * np.cos(yy / 11.0) + rng.normal(0, 2.5, (h, w)))
    lum = np.clip(lum, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([lum, lum, lum], axis=-1))


def _mean_diff(jpeg_data, ref_im):
    return ImageEncoder._jpeg_mean_diff(jpeg_data, ref_im)


# ---------- 1. 守卫是全局上界，通过守卫 != 保真 ----------

def test_mean_diff_guard_is_global_not_fidelity_proof():
    im = _text_gray_image().convert("L")
    jb90 = ie._jpeg_bytes(im, 90, 150.0)
    jb82 = ie._jpeg_bytes(im, 82, 150.0)
    md90 = _mean_diff(jb90, im)
    md82 = _mean_diff(jb82, im)
    # q82 保真明显劣化（均差至少高 30%）……
    assert md82 > md90 * 1.3
    # ……却仍通过全局守卫：守卫无法阻止质量下探，故不能据此自适应降质量
    assert md82 <= ie.JPEG_GUARD_MEAN_DIFF_MAX
    assert md90 <= ie.JPEG_GUARD_MEAN_DIFF_MAX


def test_guard_threshold_far_above_text_page_diff():
    # 文字页均差远低于阈值（守卫几乎不触发），证明守卫对文字页降质量毫无约束力
    im = _text_gray_image().convert("L")
    md82 = _mean_diff(ie._jpeg_bytes(im, 82, 150.0), im)
    assert md82 < ie.JPEG_GUARD_MEAN_DIFF_MAX / 2


# ---------- 2. 编码器刻意保持固定质量，放弃守卫内最小体积 ----------

def test_compact_keeps_q90_forgoes_guard_minimum(tmp_path):
    tex = _gray_texture_image()
    p = _save(tex, tmp_path)
    enc = ImageEncoder(PDFExportConfig(compression="compact")).encode(
        p, _layout(600, 800), page_type="pure_text")
    assert enc.format == "jpeg" and enc.jpeg_quality == 90
    # 存在更小的 q82 候选且通过守卫——自适应 guard_min 会选它，但固定策略刻意不选
    jb82 = ie._jpeg_bytes(tex.convert("L"), 82, 150.0)
    assert len(jb82) < enc.encoded_size_bytes
    assert _mean_diff(jb82, tex.convert("L")) <= ie.JPEG_GUARD_MEAN_DIFF_MAX


def test_balanced_keeps_q95_forgoes_guard_minimum(tmp_path):
    tex = _gray_texture_image()
    p = _save(tex, tmp_path)
    enc = ImageEncoder(PDFExportConfig(compression="balanced")).encode(
        p, _layout(600, 800), page_type="pure_text")
    assert enc.format == "jpeg" and enc.jpeg_quality == 95
    jb90 = ie._jpeg_bytes(tex.convert("L"), 90, 150.0)
    assert len(jb90) < enc.encoded_size_bytes
    assert _mean_diff(jb90, tex.convert("L")) <= ie.JPEG_GUARD_MEAN_DIFF_MAX


def test_compact_color_photo_keeps_q88_not_lower(tmp_path):
    # 彩色照片页固定 q88（研究：4:2:0 / 更低质量整体仅省 ≤0.24%，不采纳）
    from scripts.make_m2_samples import _photo_array
    rgb = _photo_array(800, 600, color=True, seed=11)
    p = _save(Image.fromarray(rgb), tmp_path, "photo.png")
    enc = ImageEncoder(PDFExportConfig(compression="compact")).encode(
        p, _layout(800, 600), page_type="photo_dominated")
    assert enc.color_mode == "RGB"
    if enc.format == "jpeg":
        assert enc.jpeg_quality == 88