"""M3.1 区域掩膜测试：用真值掩膜量化六区域识别（IoU/召回/精度），不只凭肉眼。"""

import cv2
import numpy as np
import pytest

from bookclean.content_analyzer import ContentAnalyzer, RegionParams, PageRegionMasks
from scripts.make_m2_samples import build_region_samples, text_page, shadow, yellow, _photo_array

REGIONS = ["paper", "text", "color", "illustration", "special"]


@pytest.fixture(scope="module")
def samples():
    return build_region_samples()


@pytest.fixture(scope="module")
def masks(samples):
    ca = ContentAnalyzer()
    return {name: ca.analyze(s["image"]) for name, s in samples.items()}


def _pred(m, r):
    return getattr(m, r + "_mask") > 0


def _recall(m, s, r):
    t = s["regions"][r]
    return float((_pred(m, r) & t).sum() / t.sum()) if t.sum() else 1.0


def _precision(m, s, r):
    p = _pred(m, r)
    return float((p & s["regions"][r]).sum() / p.sum()) if p.sum() else 1.0


def _iou(m, s, r):
    p, t = _pred(m, r), s["regions"][r]
    u = (p | t).sum()
    return float((p & t).sum() / u) if u else 1.0


def _protected(m):
    return ((m.text_mask > 0) | (m.color_mask > 0) | (m.illustration_mask > 0)
            | (m.special_mask > 0) | (m.uncertain_mask > 0))


def test_mask_dtype_shape_values(samples, masks):
    for name, s in samples.items():
        m = masks[name]
        h, w = s["image"].shape[:2]
        for r in REGIONS + ["uncertain"]:
            arr = getattr(m, r + "_mask")
            assert arr.dtype == np.uint8, (name, r, arr.dtype)
            assert arr.shape == (h, w), (name, r, arr.shape)
            assert arr.flags["C_CONTIGUOUS"]
            assert set(np.unique(arr)).issubset({0, 255}), (name, r)


def test_metadata_complete(masks):
    for name, m in masks.items():
        assert 0.0 <= m.confidence <= 1.0
        md = m.metadata
        assert set(md["area_ratios"]) == set(REGIONS + ["uncertain"])
        assert "params" in md and "scaled_thresholds" in md and "flags" in md
        assert isinstance(md["flags"], list)
        assert abs(sum(md["area_ratios"][r] for r in REGIONS + ["uncertain"]) - 1.0) < 0.02


def test_paper_background_recall_precision(samples, masks):
    for name, s in samples.items():
        assert _recall(masks[name], s, "paper") >= 0.95, name
        assert _precision(masks[name], s, "paper") >= 0.90, name


def test_text_core_always_protected(samples, masks):
    """文字核心像素绝不能被当作纸张（必须落在任一保护区）。"""
    for name, s in samples.items():
        gray = cv2.cvtColor(s["image"], cv2.COLOR_RGB2GRAY)
        core = s["regions"]["text"] & (gray < 128)
        if not core.sum():
            continue
        prot = (_protected(masks[name]) & core).sum() / core.sum()
        assert prot >= 0.99, (name, prot)


def test_text_identified_as_text(samples, masks):
    for name in ["yellow_black_text", "pure_bw_text", "color_chart", "yellow_color_map"]:
        assert _recall(masks[name], samples[name], "text") >= 0.85, name


def test_color_illustration_recall_precision(samples, masks):
    for name in ["yellow_color_map", "yellow_color_photo", "color_chart", "large_color_illustration"]:
        assert _recall(masks[name], samples[name], "color") >= 0.85, name
        assert _precision(masks[name], samples[name], "color") >= 0.95, name


def test_grayscale_not_misjudged_color(samples, masks):
    for name in ["grayscale_photo", "pure_bw_text", "yellow_black_text"]:
        assert masks[name].metadata["area_ratios"]["color"] < 0.005, name


def test_yellow_paper_not_misjudged_color(samples, masks):
    # 纯黄纸黑字无彩色内容；黄纸本身不应被判为彩色
    assert masks["yellow_black_text"].metadata["area_ratios"]["color"] < 0.001
    # 黄纸彩色地图：彩色精度高的同时纸张召回仍高（黄纸未被吞为彩色）
    assert _precision(masks["yellow_color_map"], samples["yellow_color_map"], "color") >= 0.95
    assert _recall(masks["yellow_color_map"], samples["yellow_color_map"], "paper") >= 0.95


def test_red_seal_protected_as_special(samples, masks):
    m, s = masks["yellow_red_seal"], samples["yellow_red_seal"]
    assert _recall(m, s, "special") >= 0.85
    # 印章不被当作纸张，也不被当作大幅彩色插图
    assert (_pred(m, "paper") & s["regions"]["special"]).sum() / s["regions"]["special"].sum() < 0.05
    assert _recall(m, s, "color") < 0.5 or m.metadata["area_ratios"]["color"] < 0.01


def test_color_annotation_protected(samples, masks):
    assert _recall(masks["color_annotation"], samples["color_annotation"], "special") >= 0.85


def test_grayscale_illustration_detected(samples, masks):
    m, s = masks["grayscale_photo"], samples["grayscale_photo"]
    assert _recall(m, s, "illustration") >= 0.80
    assert _precision(m, s, "illustration") >= 0.75
    assert m.metadata["area_ratios"]["color"] < 0.005


def test_few_color_elements_protected(samples, masks):
    m, s = masks["few_color_elements"], samples["few_color_elements"]
    assert _recall(m, s, "special") >= 0.85
    # 小彩色元素不应被当成大幅插图
    assert m.metadata["area_ratios"]["color"] < 0.01


def test_large_color_illustration(samples, masks):
    assert _recall(masks["large_color_illustration"], samples["large_color_illustration"], "color") >= 0.85


def test_uncertain_region_present_and_disjoint(samples, masks):
    for name, m in masks.items():
        assert (m.uncertain_mask > 0).any(), name
        # 纸张与不确定区域互斥
        assert not ((_pred(m, "paper")) & (m.uncertain_mask > 0)).any(), name


def test_shadow_not_illustration():
    rgb, _ = shadow(*text_page(), kind="spine", strength=0.4)
    m = ContentAnalyzer().analyze(rgb)
    assert m.metadata["area_ratios"]["illustration"] < 0.01
    assert m.metadata["area_ratios"]["color"] < 0.005


def test_heavy_yellow_not_illustration():
    rgb, _ = yellow(*text_page(), paper=(232, 210, 150), gradient=0.12)
    m = ContentAnalyzer().analyze(rgb)
    assert m.metadata["area_ratios"]["illustration"] < 0.01
    assert m.metadata["area_ratios"]["color"] < 0.005


def test_photo_dominated_failure_handling():
    """整页照片（几乎无纸张）：低置信度 + 失败标记，不强行给出大面积纸张。"""
    rgb = _photo_array(900, 1200, color=True, seed=3)
    m = ContentAnalyzer().analyze(rgb)
    assert m.confidence <= 0.6
    assert m.metadata["area_ratios"]["paper"] < 0.3
    assert any(f in m.metadata["flags"] for f in ("no_reliable_paper", "low_paper_area", "photo_dominated"))


def test_clean_page_high_confidence(masks):
    for name in ["pure_bw_text", "yellow_black_text", "color_chart"]:
        assert masks[name].confidence >= 0.9, name


def test_protection_weight_soft(samples, masks):
    ca = ContentAnalyzer()
    s = samples["yellow_color_photo"]
    m = masks["yellow_color_photo"]
    w = ca.protection_weight(m)
    assert w.dtype == np.float32
    assert w.shape == s["image"].shape[:2]
    assert 0.0 <= w.min() and w.max() <= 1.0
    # 彩色核心保护权重接近 1，远处纸张接近 0
    core = s["regions"]["color"]
    assert float(w[core].mean()) >= 0.95
    far_paper = s["regions"]["paper"] & (cv2.erode(s["regions"]["paper"].astype(np.uint8), np.ones((41, 41), np.uint8)) > 0)
    assert float(w[far_paper].mean()) <= 0.1
    # 软化：存在中间权重（非硬切割）
    assert ((w > 0.05) & (w < 0.95)).any()


def test_input_unmodified(samples):
    ca = ContentAnalyzer()
    s = samples["yellow_color_map"]
    before = s["image"].copy()
    ca.analyze(s["image"])
    assert np.array_equal(s["image"], before)


def test_masks_paper_content_partition(masks):
    for name, m in masks.items():
        paper = _pred(m, "paper")
        content = _protected(m)
        assert not (paper & content).any(), name
        assert (paper | content).mean() >= 0.98, name


def test_custom_params_recorded():
    ca = ContentAnalyzer(RegionParams(chroma_sat=80, color_area=3000))
    rgb, _ = text_page()
    m = ca.analyze(rgb)
    assert m.metadata["params"]["chroma_sat"] == 80
    assert m.metadata["params"]["color_area"] == 3000