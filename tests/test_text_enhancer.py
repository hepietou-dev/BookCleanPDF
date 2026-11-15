"""M2.3 文字增强测试：Clean Gray 保抗锯齿、Sauvola 二值化、自动保色。"""

import cv2
import numpy as np
import pytest

from bookclean.config import TaskConfig
from bookclean.models import PageData
from bookclean.page_analyzer import PageAnalyzer
from bookclean.text_enhancer import TextEnhancer, sauvola_threshold
from scripts.make_m2_samples import color_content, text_page, yellow


def _pd(rgb):
    return PageData(0, rgb, "synthetic", "image", rgb.shape[1], rgb.shape[0], 150.0, "RGB")


def _gray(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)


@pytest.fixture(scope="module")
def base():
    return text_page()


@pytest.fixture(scope="module")
def enhancer():
    return TextEnhancer()


@pytest.fixture(scope="module")
def analyzer():
    return PageAnalyzer()


def test_clean_gray_keeps_antialiasing(base, enhancer, analyzer):
    res = enhancer.process(base[0], analyzer.analyze(_pd(base[0])), TaskConfig())
    g = _gray(res.image)
    mid = ((g > 30) & (g < 220)).sum()
    assert mid > 1000  # 保留抗锯齿灰阶，而非强制二值
    assert res.params["mode"] == "clean_gray"


def test_clean_gray_text_preserved(base, enhancer, analyzer):
    res = enhancer.process(base[0], analyzer.analyze(_pd(base[0])), TaskConfig())
    core = _gray(base[0]) < 128
    keep = (_gray(res.image) < 160)[core].sum() / core.sum()
    assert keep >= 0.98


def test_clean_gray_sharpen_effect(base, enhancer, analyzer):
    a = analyzer.analyze(_pd(base[0]))
    res = enhancer.process(base[0], a, TaskConfig(sharpen_strength=0.3))
    assert res.executed
    assert a.sharpness > 0


def test_color_page_preserves_rgb(base, enhancer, analyzer):
    rgb, m, cm = color_content(base[0], base[1])
    res = enhancer.process(rgb, analyzer.analyze(_pd(rgb)), TaskConfig())
    assert res.params["mode"] == "preserve_color"
    assert res.params["color_preserved"]
    chroma_b = (rgb.astype(int).max(2) - rgb.astype(int).min(2))[cm].mean()
    chroma_a = (res.image.astype(int).max(2) - res.image.astype(int).min(2))[cm].mean()
    assert chroma_a / chroma_b >= 0.9


def test_adaptive_bw_clean_text(base, enhancer, analyzer):
    cfg = TaskConfig(output_mode="adaptive_bw", enable_binarization=True)
    res = enhancer.process(base[0], analyzer.analyze(_pd(base[0])), cfg)
    assert res.params["mode"] == "adaptive_bw"
    g = _gray(res.image)
    assert set(np.unique(g)) <= {0, 255}
    core = _gray(base[0]) < 128
    bg = _gray(base[0]) > 200
    assert (g[core] == 0).mean() >= 0.9
    assert (g[bg] == 255).mean() >= 0.9


def test_adaptive_bw_unsuitable_color_falls_back(base, enhancer, analyzer):
    rgb, m, cm = color_content(base[0], base[1])
    cfg = TaskConfig(output_mode="adaptive_bw", enable_binarization=True)
    res = enhancer.process(rgb, analyzer.analyze(_pd(rgb)), cfg)
    assert res.params["mode"] == "clean_gray"
    assert "binarization_skipped_unsuitable" in res.warnings
    assert len(np.unique(_gray(res.image))) > 2  # 未二值化


def test_binarization_disabled_by_default(base, enhancer, analyzer):
    res = enhancer.process(base[0], analyzer.analyze(_pd(base[0])), TaskConfig(output_mode="adaptive_bw"))
    assert res.params["mode"] == "clean_gray"


def test_zero_sharpen_noop(base, enhancer, analyzer):
    res = enhancer.process(base[0], analyzer.analyze(_pd(base[0])), TaskConfig(sharpen_strength=0.0))
    assert not res.executed
    assert np.array_equal(res.image, base[0])


def test_sauvola_handles_gradient():
    rng = np.random.default_rng(0)
    g = np.full((200, 200), 200, np.uint8)
    g[:, 100:] = 120  # 背景渐变
    g[50:60, 50:150] = 30  # 深色文字条
    bw = sauvola_threshold(g, 25, 0.2)
    assert bw[55, 75] == 0  # 文字判黑
    assert bw[10, 10] == 255  # 背景判白
    assert bw[10, 150] == 255


def test_output_dtype_contiguous(base, enhancer, analyzer):
    res = enhancer.process(base[0], analyzer.analyze(_pd(base[0])), TaskConfig())
    assert res.image.dtype == np.uint8
    assert res.image.shape[2] == 3
    assert res.image.flags["C_CONTIGUOUS"]