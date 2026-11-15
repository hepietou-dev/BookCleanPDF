"""M2.2 背景清理测试：背景改善 + 文字/彩色保真（用真实前景掩膜与深色核心像素衡量）。"""

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from bookclean.background_cleaner import BackgroundCleaner
from bookclean.config import TaskConfig
from bookclean.models import PageData
from bookclean.page_analyzer import PageAnalyzer
from scripts.make_m2_samples import (
    _font, color_content, shadow, specks, text_page, yellow,
)


def _pd(rgb):
    return PageData(0, rgb, "synthetic", "image", rgb.shape[1], rgb.shape[0], 150.0, "RGB")


def _gray(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)


def _core_keep(before, after, thresh=128, keep_thresh=160):
    core = _gray(before) < thresh
    return float((_gray(after) < keep_thresh)[core].sum() / max(1, core.sum()))


@pytest.fixture(scope="module")
def base():
    return text_page()


@pytest.fixture(scope="module")
def cleaner():
    return BackgroundCleaner()


@pytest.fixture(scope="module")
def analyzer():
    return PageAnalyzer()


@pytest.fixture(scope="module")
def cfg():
    return TaskConfig(background_strength=0.8, yellow_removal_strength=0.8, denoise_strength=0.3)


def test_yellow_background_whitened(base, cleaner, analyzer, cfg):
    rgb, m = yellow(base[0], base[1], paper=(232, 210, 150), noise=2.5, gradient=0.12)
    res = cleaner.process(rgb, analyzer.analyze(_pd(rgb)), cfg)
    assert res.params["bg_after"]["mean"] > 230
    assert res.params["bg_after"]["std"] < res.params["bg_before"]["std"] * 0.5
    a2 = analyzer.photometrics(res.image)
    assert a2.yellow_index < 0.3


def test_yellow_text_preserved(base, cleaner, analyzer, cfg):
    rgb, m = yellow(base[0], base[1], paper=(232, 210, 150), noise=2.5, gradient=0.12)
    res = cleaner.process(rgb, analyzer.analyze(_pd(rgb)), cfg)
    assert _core_keep(rgb, res.image) >= 0.98
    core = _gray(rgb) < 128
    assert abs(float(_gray(res.image)[core].mean()) - float(_gray(rgb)[core].mean())) < 15


def test_gradient_normalized(base, cleaner, analyzer, cfg):
    rgb, m = yellow(base[0], base[1], paper=(235, 220, 170), gradient=0.2)
    res = cleaner.process(rgb, analyzer.analyze(_pd(rgb)), cfg)
    assert res.params["bg_after"]["std"] < res.params["bg_before"]["std"] * 0.5


def test_shadow_normalized(base, cleaner, analyzer, cfg):
    rgb, m = shadow(base[0], base[1], kind="spine", strength=0.4)
    res = cleaner.process(rgb, analyzer.analyze(_pd(rgb)), cfg)
    assert res.params["bg_after"]["std"] < res.params["bg_before"]["std"] * 0.6
    assert _core_keep(rgb, res.image) >= 0.98


def test_specks_removed_punctuation_kept(base, cleaner, analyzer, cfg):
    rgb, m, sm = specks(base[0], base[1], n=200)
    res = cleaner.process(rgb, analyzer.analyze(_pd(rgb)), cfg)
    removed = next(s["isolated_specks_removed"] for s in res.steps if s["op"] == "denoise")
    assert removed > 50
    # 正文中的标点（句号等）属于近文字小连通域，必须保留；只在真实文字掩膜内衡量
    text_core = m & (_gray(rgb) < 128)
    keep = (_gray(res.image) < 160)[text_core].sum() / max(1, text_core.sum())
    assert keep >= 0.98


def test_color_preserved(base, cleaner, analyzer, cfg):
    rgb, m, cm = color_content(base[0], base[1])
    res = cleaner.process(rgb, analyzer.analyze(_pd(rgb)), cfg)
    chroma_b = (rgb.astype(int).max(2) - rgb.astype(int).min(2))[cm].mean()
    chroma_a = (res.image.astype(int).max(2) - res.image.astype(int).min(2))[cm].mean()
    assert chroma_a / chroma_b >= 0.9
    # 彩色区域未被灰度化
    assert (res.image.astype(int).max(2) - res.image.astype(int).min(2))[cm].mean() > 30


def test_clean_minimal_change(base, cleaner, analyzer, cfg):
    res = cleaner.process(base[0], analyzer.analyze(_pd(base[0])), cfg)
    diff = np.abs(res.image.astype(int) - base[0].astype(int)).mean()
    assert diff < 3
    assert not next(s["executed"] for s in res.steps if s["op"] == "background_normalize")
    assert not next(s["executed"] for s in res.steps if s["op"] == "deyellow")


def test_light_text_preserved(cleaner, analyzer, cfg):
    im = Image.new("RGB", (1000, 1400), (255, 255, 255))
    d = ImageDraw.Draw(im)
    f = _font("times.ttf", 30)
    for i in range(20):
        d.text((90, 90 + i * 40), "Light gray footnote text 12345, with punctuation.", font=f, fill=(165, 165, 165))
    rgb = np.ascontiguousarray(np.asarray(im, np.uint8))
    res = cleaner.process(rgb, analyzer.analyze(_pd(rgb)), cfg)
    core = _gray(rgb) < 200
    keep = (_gray(res.image) < 210)[core].sum() / max(1, core.sum())
    assert keep >= 0.95


def test_input_not_modified(base, cleaner, analyzer, cfg):
    rgb, m = yellow(base[0], base[1], paper=(232, 210, 150))
    snapshot = rgb.copy()
    cleaner.process(rgb, analyzer.analyze(_pd(rgb)), cfg)
    assert np.array_equal(rgb, snapshot)


def test_zero_strength_noop(base, cleaner, analyzer):
    cfg0 = TaskConfig(background_strength=0.0, yellow_removal_strength=0.0, denoise_strength=0.0)
    rgb, m = yellow(base[0], base[1], paper=(232, 210, 150))
    res = cleaner.process(rgb, analyzer.analyze(_pd(rgb)), cfg0)
    assert not res.executed
    assert np.array_equal(res.image, rgb)


def test_output_rgb_uint8_contiguous(base, cleaner, analyzer, cfg):
    rgb, m = yellow(base[0], base[1], paper=(232, 210, 150))
    res = cleaner.process(rgb, analyzer.analyze(_pd(rgb)), cfg)
    assert res.image.dtype == np.uint8
    assert res.image.ndim == 3 and res.image.shape[2] == 3
    assert res.image.flags["C_CONTIGUOUS"]