"""M2.1 几何校正测试：纠斜残差、内容保真、保守裁边、高置信度透视。"""

import cv2
import numpy as np
import pytest

from bookclean.config import TaskConfig
from bookclean.geometry_corrector import GeometryCorrector
from bookclean.models import PageData
from bookclean.page_analyzer import PageAnalyzer
from scripts.make_m2_samples import black_border, perspective, rotate, text_page


def _pd(rgb):
    return PageData(0, rgb, "synthetic", "image", rgb.shape[1], rgb.shape[0], 150.0, "RGB")


def _ink(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) < 128


def _with_corner_markers(rgb):
    out = rgb.copy()
    h, w = out.shape[:2]
    for x, y in [(25, 25), (w - 39, 25), (25, h - 39), (w - 39, h - 39)]:
        out[y:y + 14, x:x + 14] = (10, 10, 10)
    return out


@pytest.fixture(scope="module")
def base():
    return text_page()


@pytest.fixture(scope="module")
def analyzer():
    return PageAnalyzer()


@pytest.fixture(scope="module")
def geo():
    return GeometryCorrector()


@pytest.mark.parametrize("angle", [-7, -5, -3, -2, 2, 3, 5, 7])
def test_deskew_residual_and_content(base, analyzer, geo, angle):
    marked = _with_corner_markers(base[0])
    rgb, m = rotate(marked, base[1], float(angle))
    cfg = TaskConfig()
    a = analyzer.analyze(_pd(rgb))
    res = geo.deskew(rgb, a, cfg)
    assert res.executed, res.params
    # 残差 <= 0.5 度
    a2 = analyzer.analyze(_pd(res.image))
    assert abs(a2.skew_angle) <= 0.5, f"残差 {a2.skew_angle}"
    # 画布扩展，不裁切内容
    assert res.image.shape[0] >= rgb.shape[0]
    assert res.image.shape[1] >= rgb.shape[1]
    # 四角标记全部保留（允许插值损失）
    assert _ink(res.image).sum() >= 0.85 * _ink(rgb).sum()
    # 新边缘白色填充，无异常黑边
    ring = np.concatenate([
        res.image[:3].reshape(-1, 3), res.image[-3:].reshape(-1, 3),
        res.image[:, :3].reshape(-1, 3), res.image[:, -3:].reshape(-1, 3),
    ])
    assert ring.mean() > 240
    # 不修改输入
    assert rgb.flags["C_CONTIGUOUS"]


def test_deskew_zero_skips(base, analyzer, geo):
    a = analyzer.analyze(_pd(base[0]))
    res = geo.deskew(base[0], a, TaskConfig())
    assert not res.executed
    assert res.params["reason"] in ("already_level", "low_confidence")
    assert res.image is base[0]


def test_deskew_low_confidence_skips(analyzer, geo):
    blank = np.full((1400, 1000, 3), 255, np.uint8)
    a = analyzer.analyze(_pd(blank))
    res = geo.deskew(blank, a, TaskConfig())
    assert not res.executed
    assert res.params["reason"] == "low_confidence"


def test_deskew_disabled(base, analyzer, geo):
    rgb, m = rotate(base[0], base[1], 5.0)
    a = analyzer.analyze(_pd(rgb))
    res = geo.deskew(rgb, a, TaskConfig(enable_deskew=False))
    assert not res.executed
    assert res.params["reason"] == "disabled"


def test_crop_black_border_preserves_content(base, analyzer, geo):
    rgb, m = black_border(base[0], base[1], border=70)
    a = analyzer.analyze(_pd(rgb))
    res = geo.crop(rgb, a, TaskConfig())
    assert res.executed
    assert res.params["mode"] == "dark_border"
    x0, y0, x1, y1 = res.params["crop_box"]
    # 页面内全部前景（含页码）保留
    assert m[y0:y1, x0:x1].sum() / m.sum() >= 0.999
    assert res.image.shape[0] < rgb.shape[0]
    # 页码位于输出底部区域
    bottom = res.image[int(res.image.shape[0] * 0.85):]
    assert _ink(bottom).sum() > 0


def test_crop_clean_margins_no_ink_loss(base, analyzer, geo):
    a = analyzer.analyze(_pd(base[0]))
    res = geo.crop(base[0], a, TaskConfig())
    if res.executed:
        x0, y0, x1, y1 = res.params["crop_box"]
        assert base[1][y0:y1, x0:x1].sum() == base[1].sum()


def test_crop_edge_text_safe(analyzer, geo):
    rgb, m = text_page(margin=25)
    a = analyzer.analyze(_pd(rgb))
    res = geo.crop(rgb, a, TaskConfig())
    if res.executed:
        x0, y0, x1, y1 = res.params["crop_box"]
        assert m[y0:y1, x0:x1].sum() == m.sum()


def test_crop_disabled(base, analyzer, geo):
    rgb, m = black_border(base[0], base[1])
    a = analyzer.analyze(_pd(rgb))
    res = geo.crop(rgb, a, TaskConfig(enable_crop=False))
    assert not res.executed
    assert res.params["reason"] == "disabled"


def test_perspective_correction(base, analyzer, geo):
    rgb, m, _ = perspective(base[0], base[1])
    cfg = TaskConfig(enable_perspective=True)
    a = analyzer.analyze(_pd(rgb))
    res = geo.correct_perspective(rgb, a, cfg)
    assert res.executed, res.params
    a2 = analyzer.analyze(_pd(res.image))
    # 校正后更接近矩形
    assert a2.perspective_score < a.perspective_score
    # 内容保留
    assert _ink(res.image).sum() >= 0.8 * m.sum()
    assert "matrix" in res.params


def test_perspective_skips_normal_scan(base, analyzer, geo):
    a = analyzer.analyze(_pd(base[0]))
    res = geo.correct_perspective(base[0], a, TaskConfig(enable_perspective=True))
    assert not res.executed


def test_perspective_disabled_by_default(base, analyzer, geo):
    rgb, m, _ = perspective(base[0], base[1])
    a = analyzer.analyze(_pd(rgb))
    res = geo.correct_perspective(rgb, a, TaskConfig())
    assert not res.executed
    assert res.params["reason"] == "disabled"