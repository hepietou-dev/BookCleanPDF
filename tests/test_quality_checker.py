"""M2.3 质量检查与回退决策测试。"""

import cv2
import numpy as np
import pytest

from bookclean.background_cleaner import BackgroundCleaner
from bookclean.config import TaskConfig
from bookclean.models import PageData
from bookclean.page_analyzer import PageAnalyzer
from bookclean.quality_checker import QualityChecker
from scripts.make_m2_samples import color_content, rotate, text_page, yellow


def _pd(rgb):
    return PageData(0, rgb, "synthetic", "image", rgb.shape[1], rgb.shape[0], 150.0, "RGB")


def _gray(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)


@pytest.fixture(scope="module")
def base():
    return text_page()


@pytest.fixture(scope="module")
def checker():
    return QualityChecker()


@pytest.fixture(scope="module")
def analyzer():
    return PageAnalyzer()


def test_accept_good_cleaning(base, checker, analyzer):
    rgb, m = yellow(base[0], base[1], paper=(232, 210, 150), noise=2.0)
    cfg = TaskConfig(background_strength=0.8, yellow_removal_strength=0.8, denoise_strength=0.2)
    cand = BackgroundCleaner().process(rgb, analyzer.analyze(_pd(rgb)), cfg).image
    report = checker.evaluate(rgb, cand, analyzer.analyze(_pd(rgb)), cfg)
    assert report["decision"] == "accept", report["reasons"]
    assert "metrics" in report and "checks" in report


def test_blank_fallback(base, checker, analyzer):
    blank = np.full_like(base[0], 255)
    report = checker.evaluate(base[0], blank, analyzer.analyze(_pd(base[0])), TaskConfig())
    assert report["decision"] == "fallback_original"
    assert any(c["name"] == "blank" and not c["passed"] for c in report["checks"])


def test_foreground_loss_fallback(base, checker, analyzer):
    rgb = base[0]
    fg = _gray(rgb) < 128
    cand = rgb.copy()
    cand[fg] = 255  # 抹掉文字
    report = checker.evaluate(rgb, cand, analyzer.analyze(_pd(rgb)), TaskConfig())
    assert report["decision"] == "fallback_original"
    assert any(c["name"] == "foreground" and not c["passed"] for c in report["checks"])


def test_overblur_retry(base, checker, analyzer):
    cand = cv2.GaussianBlur(base[0], (0, 0), sigmaX=4.0)
    report = checker.evaluate(base[0], cand, analyzer.analyze(_pd(base[0])), TaskConfig())
    assert report["decision"] in ("retry_lower_strength", "fallback_original")
    assert any(c["name"] == "sharpness" and not c["passed"] for c in report["checks"])


def test_color_loss_flagged(base, checker, analyzer):
    rgb, m, cm = color_content(base[0], base[1])
    cand = cv2.cvtColor(_gray(rgb), cv2.COLOR_GRAY2RGB)  # 强制灰度
    report = checker.evaluate(rgb, cand, analyzer.analyze(_pd(rgb)), TaskConfig())
    assert any(c["name"] == "color" and not c["passed"] for c in report["checks"])
    assert report["decision"] != "accept"


def test_size_collapse_fallback(base, checker, analyzer):
    cand = base[0][:100, :100]
    report = checker.evaluate(base[0], cand, analyzer.analyze(_pd(base[0])), TaskConfig())
    assert report["decision"] == "fallback_original"
    assert any(c["name"] == "size" and not c["passed"] for c in report["checks"])


def test_geometry_crop_accept(base, checker, analyzer):
    rgb = base[0]
    cand = rgb[50:1350, 50:950]  # 合理裁边
    report = checker.evaluate(rgb, cand, analyzer.analyze(_pd(rgb)), TaskConfig(), geometry_applied=True)
    assert report["decision"] == "accept", report["reasons"]


def test_skew_residual_flagged(base, checker, analyzer):
    rgb, m = rotate(base[0], base[1], 5.0)
    report = checker.evaluate(base[0], rgb, analyzer.analyze(_pd(base[0])), TaskConfig(),
                              geometry_applied=True, deskew_applied=True)
    assert any(c["name"] == "skew_residual" and not c["passed"] for c in report["checks"])


def test_report_structure(base, checker, analyzer):
    report = checker.evaluate(base[0], base[0], analyzer.analyze(_pd(base[0])), TaskConfig())
    assert report["decision"] == "accept"
    names = {c["name"] for c in report["checks"]}
    assert {"size", "blank", "foreground", "background"} <= names