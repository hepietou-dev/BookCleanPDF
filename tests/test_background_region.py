"""M3.2 区域化背景清理专项测试：可信纸张背景改善 + 彩色/灰度/特殊内容保真 + 失败保守跳过。

用带真值掩膜的合成样本量化（色度保留率/文字核心保留率/背景均匀度），不只凭肉眼。
直接测试 BackgroundCleaner（流水线决策属 M3.3/M3.4）。
"""

import cv2
import numpy as np
import pytest

from bookclean.background_cleaner import BackgroundCleaner
from bookclean.config import TaskConfig
from bookclean.content_analyzer import ContentAnalyzer
from bookclean.models import PageData
from bookclean.page_analyzer import PageAnalyzer
from scripts.make_m2_samples import (
    build_region_samples, text_page, color_content, shadow, _photo_array,
)


def _pd(rgb):
    return PageData(0, rgb, "synthetic", "image", rgb.shape[1], rgb.shape[0], 150.0, "RGB")


def _gray(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)


def _chroma_ret(before, after, mask):
    if not mask.any():
        return 1.0
    cb = before.astype(int).max(2) - before.astype(int).min(2)
    ca = after.astype(int).max(2) - after.astype(int).min(2)
    return float(ca[mask].mean() / max(1e-6, cb[mask].mean()))


def _text_core_keep(before, after, text_mask):
    core = text_mask & (_gray(before) < 128)
    if not core.any():
        return 1.0
    return float((_gray(after) < 160)[core].sum() / core.sum())


@pytest.fixture(scope="module")
def samples():
    return build_region_samples()


@pytest.fixture(scope="module")
def cleaner():
    return BackgroundCleaner()


@pytest.fixture(scope="module")
def analyzer():
    return PageAnalyzer()


@pytest.fixture(scope="module")
def cfg():
    return TaskConfig(background_strength=0.8, yellow_removal_strength=0.8, denoise_strength=0.3)


def _run(cleaner, analyzer, cfg, rgb):
    return cleaner.process(rgb, analyzer.analyze(_pd(rgb)), cfg)


def test_yellow_color_map_paper_improved_color_kept(samples, cleaner, analyzer, cfg):
    s = samples["yellow_color_map"]
    res = _run(cleaner, analyzer, cfg, s["image"])
    assert res.params["bg_after"]["mean"] > res.params["bg_before"]["mean"] + 5
    assert _chroma_ret(s["image"], res.image, s["regions"]["color"]) >= 0.95
    assert _text_core_keep(s["image"], res.image, s["regions"]["text"]) >= 0.99


def test_yellow_color_photo_paper_improved_color_kept(samples, cleaner, analyzer, cfg):
    s = samples["yellow_color_photo"]
    res = _run(cleaner, analyzer, cfg, s["image"])
    assert res.params["bg_after"]["mean"] > res.params["bg_before"]["mean"] + 5
    assert _chroma_ret(s["image"], res.image, s["regions"]["color"]) >= 0.95
    assert _text_core_keep(s["image"], res.image, s["regions"]["text"]) >= 0.99


def test_yellow_red_seal_preserved_paper_improved(samples, cleaner, analyzer, cfg):
    s = samples["yellow_red_seal"]
    res = _run(cleaner, analyzer, cfg, s["image"])
    assert _chroma_ret(s["image"], res.image, s["regions"]["special"]) >= 0.95
    assert res.params["bg_after"]["mean"] > res.params["bg_before"]["mean"] + 5
    assert _text_core_keep(s["image"], res.image, s["regions"]["text"]) >= 0.99


def test_color_annotation_preserved(samples, cleaner, analyzer, cfg):
    s = samples["color_annotation"]
    res = _run(cleaner, analyzer, cfg, s["image"])
    assert _chroma_ret(s["image"], res.image, s["regions"]["special"]) >= 0.95
    assert _text_core_keep(s["image"], res.image, s["regions"]["text"]) >= 0.99


def test_grayscale_photo_levels_kept(samples, cleaner, analyzer, cfg):
    s = samples["grayscale_photo"]
    res = _run(cleaner, analyzer, cfg, s["image"])
    ill = s["regions"]["illustration"]
    std_b = float(_gray(s["image"])[ill].std())
    std_a = float(_gray(res.image)[ill].std())
    assert std_a >= 0.9 * std_b  # 灰度层次保留
    assert _text_core_keep(s["image"], res.image, s["regions"]["text"]) >= 0.99


def test_color_chart_black_text_kept(samples, cleaner, analyzer, cfg):
    s = samples["color_chart"]
    res = _run(cleaner, analyzer, cfg, s["image"])
    assert _chroma_ret(s["image"], res.image, s["regions"]["color"]) >= 0.95
    assert _text_core_keep(s["image"], res.image, s["regions"]["text"]) >= 0.99


def test_large_color_illustration_kept(samples, cleaner, analyzer, cfg):
    s = samples["large_color_illustration"]
    res = _run(cleaner, analyzer, cfg, s["image"])
    assert _chroma_ret(s["image"], res.image, s["regions"]["color"]) >= 0.95
    # 该样本为白纸（非泛黄）：清洁纸张不应被无意义改变，更不应变暗
    assert res.params["bg_after"]["mean"] >= res.params["bg_before"]["mean"] - 2
    assert _text_core_keep(s["image"], res.image, s["regions"]["text"]) >= 0.99


def test_color_illustration_with_shadow(cleaner, analyzer, cfg):
    rgb, m = text_page()
    rgb, m, cm = color_content(rgb, m)
    rgb, m = shadow(rgb, m, kind="spine", strength=0.4)
    res = _run(cleaner, analyzer, cfg, rgb)
    assert _chroma_ret(rgb, res.image, cm) >= 0.95
    assert res.params["bg_after"]["std"] < res.params["bg_before"]["std"]


def test_photo_dominated_skips_normalization(cleaner, analyzer, cfg):
    rgb = _photo_array(900, 1200, color=True, seed=3)
    res = _run(cleaner, analyzer, cfg, rgb)
    norm = next(s for s in res.steps if s["op"] == "background_normalize")
    assert not norm["executed"]
    assert norm["reason"] == "photo_dominated"
    assert not res.params["region_info"]["paper_ok"]


def test_low_confidence_skips_normalization(samples, cleaner, analyzer, cfg):
    rgb = samples["yellow_color_map"]["image"]
    regions = ContentAnalyzer().analyze(rgb)
    regions.confidence = 0.3  # 人为降低置信度，验证门槛
    res = cleaner.process(rgb, analyzer.analyze(_pd(rgb)), cfg, regions=regions)
    norm = next(s for s in res.steps if s["op"] == "background_normalize")
    assert not norm["executed"]
    assert norm["reason"] == "low_region_confidence"


def test_insufficient_paper_area_skips(samples, cleaner, analyzer, cfg):
    rgb = samples["pure_bw_text"]["image"]
    regions = ContentAnalyzer().analyze(rgb)
    regions.paper_mask = np.zeros_like(regions.paper_mask)  # 纸张面积不足
    res = cleaner.process(rgb, analyzer.analyze(_pd(rgb)), cfg, regions=regions)
    norm = next(s for s in res.steps if s["op"] == "background_normalize")
    assert not norm["executed"]
    assert norm["reason"] == "insufficient_paper_area"


def test_clean_page_minimal_change(samples, cleaner, analyzer, cfg):
    rgb = samples["pure_bw_text"]["image"]
    res = _run(cleaner, analyzer, cfg, rgb)
    diff = np.abs(res.image.astype(int) - rgb.astype(int)).mean()
    assert diff < 3
    assert not next(s["executed"] for s in res.steps if s["op"] == "background_normalize")


def test_text_core_protected_all_samples(samples, cleaner, analyzer, cfg):
    for name, s in samples.items():
        res = _run(cleaner, analyzer, cfg, s["image"])
        assert _text_core_keep(s["image"], res.image, s["regions"]["text"]) >= 0.99, name


def test_region_info_and_regions_attached(samples, cleaner, analyzer, cfg):
    s = samples["yellow_color_map"]
    res = _run(cleaner, analyzer, cfg, s["image"])
    ri = res.params["region_info"]
    assert 0.0 <= ri["confidence"] <= 1.0
    assert 0.0 <= ri["paper_area_ratio"] <= 1.0
    assert isinstance(ri["flags"], list)
    assert res.regions is not None
    assert res.regions.paper_mask.shape == s["image"].shape[:2]


def test_color_protection_check_recorded(samples, cleaner, analyzer, cfg):
    s = samples["yellow_color_map"]
    res = _run(cleaner, analyzer, cfg, s["image"])
    chk = next(s for s in res.steps if s["op"] == "color_protection_check")
    assert chk["chroma_retention"] >= 0.95
    assert chk["passed"]
    assert "color_area_retention" in chk and "edge_retention" in chk and "median_color_shift" in chk
    assert not res.suggest_fallback


def test_input_not_modified(samples, cleaner, analyzer, cfg):
    rgb = samples["yellow_color_photo"]["image"]
    snapshot = rgb.copy()
    _run(cleaner, analyzer, cfg, rgb)
    assert np.array_equal(rgb, snapshot)


def test_output_rgb_uint8_contiguous(samples, cleaner, analyzer, cfg):
    res = _run(cleaner, analyzer, cfg, samples["yellow_color_map"]["image"])
    assert res.image.dtype == np.uint8
    assert res.image.ndim == 3 and res.image.shape[2] == 3
    assert res.image.flags["C_CONTIGUOUS"]