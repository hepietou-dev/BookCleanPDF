"""M2.1 页面分析模块测试。全部使用离线合成样本与真实前景掩膜。"""

import numpy as np
import pytest

from bookclean.models import PageData
from bookclean.page_analyzer import PageAnalyzer
from scripts.make_m2_samples import (
    black_border, color_content, perspective, rotate, shadow, specks, text_page, yellow,
)


def _pd(rgb):
    return PageData(0, rgb, "synthetic", "image", rgb.shape[1], rgb.shape[0], 150.0, "RGB")


@pytest.fixture(scope="module")
def base():
    return text_page()


@pytest.fixture(scope="module")
def analyzer():
    return PageAnalyzer()


def test_clean_page_level_and_confident(base, analyzer):
    a = analyzer.analyze(_pd(base[0]))
    assert abs(a.skew_angle) < 0.3
    assert a.skew_confidence >= 0.5
    assert a.background_brightness > 250
    assert a.background_variance < 3
    assert a.yellow_index < 0.1
    assert a.color_ratio < 0.003
    assert a.foreground_ratio > 0.02


@pytest.mark.parametrize("angle", [2, 3, 5, 7, -2, -3, -5, -7])
def test_skew_estimation_accuracy(base, analyzer, angle):
    rgb, m = rotate(base[0], base[1], float(angle))
    a = analyzer.analyze(_pd(rgb))
    assert abs(a.skew_angle - angle) <= 0.4, f"估计 {a.skew_angle} 真值 {angle}"
    assert a.skew_confidence >= 0.5
    assert "deskew" in a.recommended_steps


def test_blank_page_low_confidence(analyzer):
    a = analyzer.analyze(_pd(np.full((1400, 1000, 3), 255, np.uint8)))
    assert a.skew_confidence < 0.4
    assert "deskew" not in a.recommended_steps


def test_yellow_index(base, analyzer):
    heavy, _ = yellow(base[0], base[1], paper=(232, 210, 150), noise=2.5, gradient=0.12)
    light, _ = yellow(base[0], base[1], paper=(248, 240, 214), noise=1.5)
    ah = analyzer.analyze(_pd(heavy))
    al = analyzer.analyze(_pd(light))
    assert ah.yellow_index > 0.5
    assert 0.15 < al.yellow_index < 0.6
    assert ah.yellow_index > al.yellow_index
    assert "deyellow" in ah.recommended_steps
    # 泛黄纸张本身不应被判为彩色内容
    assert ah.color_ratio < 0.005


def test_color_content_detected(base, analyzer):
    rgb, _, _ = color_content(base[0], base[1])
    a = analyzer.analyze(_pd(rgb))
    assert a.color_ratio > 0.02
    assert "preserve_color" in a.recommended_steps


def test_shadow_variance(base, analyzer):
    rgb, _ = shadow(base[0], base[1], kind="spine", strength=0.4)
    a = analyzer.analyze(_pd(rgb))
    assert a.background_variance > 15
    assert "background_normalize" in a.recommended_steps


def test_skew_with_shadow(base, analyzer):
    rgb, m = shadow(base[0], base[1], kind="spine", strength=0.4)
    rgb, m = rotate(rgb, m, 3.0)
    a = analyzer.analyze(_pd(rgb))
    assert abs(a.skew_angle - 3.0) <= 0.5
    assert a.skew_confidence >= 0.5


def test_dark_border_flag(base, analyzer):
    rgb, _ = black_border(base[0], base[1])
    a = analyzer.analyze(_pd(rgb))
    assert a.dark_border
    assert "crop" in a.recommended_steps


def test_perspective_quad(base, analyzer):
    rgb, _, _ = perspective(base[0], base[1])
    a = analyzer.analyze(_pd(rgb))
    assert a.page_quad is not None
    assert len(a.page_quad) == 4
    assert a.perspective_confidence >= 0.6
    assert a.perspective_score >= 0.12
    assert "perspective" in a.recommended_steps


def test_normal_scan_no_perspective(base, analyzer):
    a = analyzer.analyze(_pd(base[0]))
    assert a.perspective_score < 0.12
    assert "perspective" not in a.recommended_steps


def test_clean_page_low_noise(base, analyzer):
    a = analyzer.analyze(_pd(base[0]))
    assert a.noise_level < 0.15


def test_photometrics_reanalysis(base, analyzer):
    rgb, _ = yellow(base[0], base[1], paper=(232, 210, 150))
    a = analyzer.photometrics(rgb)
    assert a.yellow_index > 0.5
    assert a.skew_angle == 0.0  # 光度重分析不做倾斜检测