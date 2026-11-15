"""M3.3 分区质量检查专项测试：五维分区评估 + 页面类型自适应 + 可解释风险评分 + 严重文字损伤不可抵消。

用带真值掩膜的合成样本量化（文字核心保留/彩色色度保留/插图灰度动态/纸张背景真实像素 std），
不只凭肉眼。背景指标依据可信 paper_mask，彩色/插图不再污染背景统计；照片页空白判定改用动态范围。
指标是图像风险提示，不等于 OCR 准确率/文字完整性证明。直接测试 QualityChecker（流水线决策属 M3.4）。
"""

import cv2
import numpy as np
import pytest

from bookclean.background_cleaner import BackgroundCleaner
from bookclean.config import TaskConfig
from bookclean.content_analyzer import ContentAnalyzer
from bookclean.models import PageData
from bookclean.page_analyzer import PageAnalyzer
from bookclean.quality_checker import QualityChecker, _DIM_WEIGHTS
from scripts.make_m2_samples import build_region_samples, _photo_array

_DIM_KEYS = {"paper_background", "text_preservation", "color_preservation",
             "illustration_preservation", "geometry_integrity"}


def _pd(rgb):
    return PageData(0, rgb, "synthetic", "image", rgb.shape[1], rgb.shape[0], 150.0, "RGB")


def _gray(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)


def _erode(mask, k=3):
    return cv2.erode(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


def _check(report, name):
    return next((c for c in report["checks"] if c["name"] == name), None)


@pytest.fixture(scope="module")
def samples():
    return build_region_samples()


@pytest.fixture(scope="module")
def checker():
    return QualityChecker()


@pytest.fixture(scope="module")
def analyzer():
    return PageAnalyzer()


@pytest.fixture(scope="module")
def content():
    return ContentAnalyzer()


@pytest.fixture(scope="module")
def cleaner():
    return BackgroundCleaner()


@pytest.fixture(scope="module")
def cfg():
    return TaskConfig(background_strength=0.8, yellow_removal_strength=0.8, denoise_strength=0.3)


def _clean(cleaner, analyzer, cfg, rgb, regions):
    return cleaner.process(rgb, analyzer.analyze(_pd(rgb)), cfg, regions=regions).image


# ---- 结构与可解释性 ----

def test_five_dimensions_present(samples, checker, analyzer, content, cleaner, cfg):
    rgb = samples["yellow_color_map"]["image"]
    reg = content.analyze(rgb)
    cand = _clean(cleaner, analyzer, cfg, rgb, reg)
    r = checker.evaluate(rgb, cand, analyzer.analyze(_pd(rgb)), cfg, regions=reg)
    assert set(r["dimensions"]) == _DIM_KEYS
    assert 0.0 <= r["risk_score"] <= 100.0
    assert "page_type" in r and "region_mask_confidence" in r
    assert isinstance(r["risk_contributions"], list) and len(r["risk_contributions"]) == 5


def test_risk_score_explainable(samples, checker, analyzer, content):
    rgb = samples["pure_bw_text"]["image"]
    reg = content.analyze(rgb)
    cand = rgb.copy()
    cand[reg.text_mask > 0] = 255  # 抹掉文字，制造可追踪风险
    r = checker.evaluate(rgb, cand, analyzer.analyze(_pd(rgb)), TaskConfig(), regions=reg)
    assert abs(sum(c["weight"] for c in r["risk_contributions"]) - 1.0) < 1e-6
    for c in r["risk_contributions"]:
        assert c["weight"] == _DIM_WEIGHTS[c["dimension"]]
        assert abs(c["contribution"] - 100.0 * c["penalty"] * c["weight"]) < 0.05
    assert r["risk_score"] == round(min(100.0, sum(c["contribution"] for c in r["risk_contributions"])), 2)
    assert r["risk_score"] > 0


def test_m2_report_keys_preserved(samples, checker, analyzer, content):
    rgb = samples["yellow_black_text"]["image"]
    reg = content.analyze(rgb)
    r = checker.evaluate(rgb, rgb, analyzer.analyze(_pd(rgb)), TaskConfig(), regions=reg)
    assert {"area_ratio", "fg_orig", "fg_cand", "bg_std_orig", "bg_std_cand",
            "color_orig", "color_cand"} <= set(r["metrics"])
    assert {"size", "blank", "foreground", "background"} <= {c["name"] for c in r["checks"]}


# ---- 页面类型自适应 ----

def test_page_type_classification(samples, checker, analyzer, content):
    expect = {"pure_bw_text": "pure_text", "yellow_black_text": "pure_text",
              "grayscale_photo": "grayscale_illustration",
              "yellow_color_map": "color_illustration",
              "large_color_illustration": "color_illustration"}
    for name, ptype in expect.items():
        rgb = samples[name]["image"]
        reg = content.analyze(rgb)
        r = checker.evaluate(rgb, rgb, analyzer.analyze(_pd(rgb)), TaskConfig(), regions=reg)
        assert r["page_type"] == ptype, name


def test_page_type_photo_dominated(checker, analyzer, content):
    ph = _photo_array(900, 1200, color=True, seed=3)
    reg = content.analyze(ph)
    r = checker.evaluate(ph, ph, analyzer.analyze(_pd(ph)), TaskConfig(), regions=reg)
    assert r["page_type"] == "photo_dominated"


def test_page_type_uncertain_on_low_confidence(samples, checker, analyzer, content):
    rgb = samples["pure_bw_text"]["image"]
    reg = content.analyze(rgb)
    reg.confidence = 0.3
    r = checker.evaluate(rgb, rgb, analyzer.analyze(_pd(rgb)), TaskConfig(), regions=reg)
    assert r["page_type"] == "uncertain"


# ---- A 纸张背景：依据真实纸张像素，不被彩色污染 ----

def test_background_std_on_paper_not_color(samples, checker, analyzer, content):
    rgb = samples["large_color_illustration"]["image"]  # 白纸 + 大面积彩色照片
    reg = content.analyze(rgb)
    r = checker.evaluate(rgb, rgb, analyzer.analyze(_pd(rgb)), TaskConfig(), regions=reg)
    pb = r["dimensions"]["paper_background"]
    assert pb["after"] is not None
    # 纸张真实像素 std 远小于含彩色的全局背景 std（彩色不再污染背景判定）
    assert pb["after"]["std"] < r["metrics"]["bg_std_cand"]
    assert _check(r, "background")["passed"]
    assert r["decision"] == "accept"


def test_yellow_paper_background_improved(samples, checker, analyzer, content, cleaner, cfg):
    rgb = samples["yellow_color_map"]["image"]
    reg = content.analyze(rgb)
    cand = _clean(cleaner, analyzer, cfg, rgb, reg)
    r = checker.evaluate(rgb, cand, analyzer.analyze(_pd(rgb)), cfg, regions=reg)
    pb = r["dimensions"]["paper_background"]
    assert pb["after"]["brightness"] > pb["before"]["brightness"] + 5
    assert _check(r, "background")["passed"]


def test_clean_paper_tiny_noise_no_spurious_retry(samples, checker, analyzer, content):
    # 干净白纸 std≈0：去噪/增强引入 <1 灰阶 std 属数值噪声，不应触发背景失败/无意义重试
    rgb = samples["pure_bw_text"]["image"]
    reg = content.analyze(rgb)
    cand = rgb.copy()
    paper = reg.paper_mask > 0
    rng = np.random.default_rng(0)
    cand[paper] = np.clip(rgb[paper].astype(int) - rng.integers(0, 2, int(paper.sum()))[:, None], 0, 255).astype(np.uint8)
    r = checker.evaluate(rgb, cand, analyzer.analyze(_pd(rgb)), TaskConfig(), regions=reg)
    pb = r["dimensions"]["paper_background"]
    assert pb["before"]["std"] < 0.5         # 原始纸张近乎纯净
    assert 0 < pb["after"]["std"] <= 1.0    # 候选仅极小绝对增量
    assert _check(r, "background")["passed"]
    assert r["decision"] == "accept"


# ---- B 文字保留：严重损伤不可被总分/好背景抵消 ----

def test_severe_text_not_offset_by_good_background(samples, checker, analyzer, content):
    rgb = samples["pure_bw_text"]["image"]  # 干净白纸，背景指标良好
    reg = content.analyze(rgb)
    cand = rgb.copy()
    cand[reg.text_mask > 0] = 255  # 仅抹掉文字，背景保持洁白
    r = checker.evaluate(rgb, cand, analyzer.analyze(_pd(rgb)), TaskConfig(), regions=reg)
    assert _check(r, "background")["passed"]            # 背景良好
    tc = _check(r, "text_core")
    assert tc is not None and not tc["passed"]          # 文字核心严重损伤
    assert tc["value"] < 0.5
    assert r["decision"] == "fallback_original"         # 即使背景好也强制回退


def test_text_contrast_reduction_flagged(samples, checker, analyzer, content):
    # 文字对比度降低：淡化到接近纸张（>=160）应被文字核心检查识别；轻度淡化不应误报
    rgb = samples["pure_bw_text"]["image"]
    reg = content.analyze(rgb)
    go = _gray(rgb)
    core_text = (reg.text_mask > 0) & (go < 128)
    faded = rgb.copy()
    faded[core_text] = 200  # 低对比度浅灰字
    r = checker.evaluate(rgb, faded, analyzer.analyze(_pd(rgb)), TaskConfig(), regions=reg)
    tc = _check(r, "text_core")
    assert tc is not None and not tc["passed"]
    assert r["decision"] == "fallback_original"
    mild = rgb.copy()
    mild[core_text] = 100  # 仍为深色字，仅轻度降对比
    r2 = checker.evaluate(rgb, mild, analyzer.analyze(_pd(rgb)), TaskConfig(), regions=reg)
    assert _check(r2, "text_core")["passed"]
    assert r2["decision"] != "fallback_original"  # 轻度降对比不算文字损伤（清晰度可能触发重试）


def test_text_core_preserved_on_cleaning(samples, checker, analyzer, content, cleaner, cfg):
    for name, s in samples.items():
        rgb = s["image"]
        reg = content.analyze(rgb)
        cand = _clean(cleaner, analyzer, cfg, rgb, reg)
        r = checker.evaluate(rgb, cand, analyzer.analyze(_pd(rgb)), cfg, regions=reg)
        tcr = r["dimensions"]["text_preservation"]["text_core_retention"]
        if tcr is not None:
            assert tcr >= 0.99, name


def test_text_core_cross_geometry_preserved(checker, analyzer, content):
    # 透视 warp 后几何校正：直接度量两侧腐蚀笔画核心的暗像素保留 >=0.99（真实保留），
    # 且检查器跨尺寸保守代理不应把透视页误判为严重文字损伤。
    import cv2 as _cv2
    from bookclean.geometry_corrector import GeometryCorrector
    from scripts.make_m2_samples import text_page, perspective
    base, m = text_page()
    warped, _, _ = perspective(base, m)
    cfg = TaskConfig()
    geo = GeometryCorrector()
    a = analyzer.analyze(_pd(warped))
    img = warped
    for fn in (geo.deskew, geo.crop, geo.correct_perspective):
        img = fn(img, a, cfg).image
    ro, rc = content.analyze(warped), content.analyze(img)
    go, gc = _gray(warped), _gray(img)
    core_o = _erode(ro.text_mask > 0, 3) & (go < 128)
    core_c = _erode(rc.text_mask > 0, 3)
    true_ret = float((gc < 160)[core_c].mean()) / max(1e-6, float((go < 128)[core_o].mean()))
    assert true_ret >= 0.99, true_ret
    r = checker.evaluate(warped, img, a, cfg, True, False, regions=ro)
    assert r["decision"] != "fallback_original"


# ---- C 彩色保留：按 color_mask 色度保留率 ----

def test_color_chroma_preserved_on_cleaning(samples, checker, analyzer, content, cleaner, cfg):
    rgb = samples["yellow_color_map"]["image"]
    reg = content.analyze(rgb)
    cand = _clean(cleaner, analyzer, cfg, rgb, reg)
    r = checker.evaluate(rgb, cand, analyzer.analyze(_pd(rgb)), cfg, regions=reg)
    assert r["dimensions"]["color_preservation"]["chroma_retention"] >= 0.95
    assert _check(r, "color")["passed"]
    assert r["decision"] == "accept"  # M3.0 彩色页误回退回归 guard


def test_color_fade_flagged(samples, checker, analyzer, content):
    rgb = samples["yellow_color_map"]["image"]
    reg = content.analyze(rgb)
    cm = reg.color_mask > 0
    cand = rgb.copy()
    cand[cm] = _gray(rgb)[cm][:, None]  # 彩色区域强制灰度（褪色）
    r = checker.evaluate(rgb, cand, analyzer.analyze(_pd(rgb)), TaskConfig(), regions=reg)
    assert r["dimensions"]["color_preservation"]["chroma_retention"] < 0.5
    assert not _check(r, "color")["passed"]
    assert r["decision"] != "accept"


def test_special_color_fade_flagged(samples, checker, analyzer, content):
    # 印章/批注属 special 彩色内容，褪色同样必须被检出（不能因面积小而跳过）
    for name in ("yellow_red_seal", "color_annotation", "few_color_elements"):
        rgb = samples[name]["image"]
        reg = content.analyze(rgb)
        sp = reg.special_mask > 0
        assert sp.any(), name
        cand = rgb.copy()
        cand[sp] = _gray(rgb)[sp][:, None]
        r = checker.evaluate(rgb, cand, analyzer.analyze(_pd(rgb)), TaskConfig(), regions=reg)
        assert not _check(r, "color")["passed"], name
        assert r["decision"] != "accept", name


def test_special_color_preserved_on_cleaning(samples, checker, analyzer, content, cleaner, cfg):
    for name in ("yellow_red_seal", "color_annotation", "few_color_elements"):
        rgb = samples[name]["image"]
        reg = content.analyze(rgb)
        cand = _clean(cleaner, analyzer, cfg, rgb, reg)
        r = checker.evaluate(rgb, cand, analyzer.analyze(_pd(rgb)), cfg, regions=reg)
        assert r["dimensions"]["color_preservation"]["chroma_retention"] >= 0.95, name
        assert r["decision"] == "accept", name


# ---- D 插图保留：灰度动态范围 ----

def test_illustration_levels_preserved(samples, checker, analyzer, content, cleaner, cfg):
    rgb = samples["grayscale_photo"]["image"]
    reg = content.analyze(rgb)
    cand = _clean(cleaner, analyzer, cfg, rgb, reg)
    r = checker.evaluate(rgb, cand, analyzer.analyze(_pd(rgb)), cfg, regions=reg)
    assert r["dimensions"]["illustration_preservation"]["gray_std_retention"] >= 0.70
    assert _check(r, "illustration")["passed"]


def test_illustration_flatten_flagged(samples, checker, analyzer, content):
    rgb = samples["grayscale_photo"]["image"]
    reg = content.analyze(rgb)
    ill = reg.illustration_mask > 0
    cand = rgb.copy()
    cand[ill] = int(_gray(rgb)[ill].mean())  # 抹平插图层次
    r = checker.evaluate(rgb, cand, analyzer.analyze(_pd(rgb)), TaskConfig(), regions=reg)
    assert r["dimensions"]["illustration_preservation"]["gray_std_retention"] < 0.70
    assert not _check(r, "illustration")["passed"]
    assert r["decision"] != "accept"


# ---- 照片页空白判定：动态范围而非暗文字前景占比 ----

def test_photo_page_no_false_fallback(checker, analyzer, content):
    ph = _photo_array(900, 1200, color=True, seed=3)
    reg = content.analyze(ph)
    r = checker.evaluate(ph, ph, analyzer.analyze(_pd(ph)), TaskConfig(), regions=reg)
    assert r["decision"] == "accept"  # 照片页对自身不应误判空白回退


def test_wiped_photo_still_fallback(checker, analyzer, content):
    ph = _photo_array(900, 1200, color=True, seed=3)
    reg = content.analyze(ph)
    for fill in (255, 128):
        cand = np.full_like(ph, fill)
        r = checker.evaluate(ph, cand, analyzer.analyze(_pd(ph)), TaskConfig(), regions=reg)
        assert not _check(r, "blank")["passed"]
        assert r["decision"] == "fallback_original"


# ---- 风险评分闸门：无严重/轻度项但累计风险超阈值仍升级重试 ----

def test_risk_gate_forces_retry(samples, checker, analyzer, content):
    rgb = samples["grayscale_photo"]["image"]
    reg = content.analyze(rgb)
    ill = reg.illustration_mask > 0
    g = _gray(rgb).astype(float)
    mean = g[ill].mean()
    cand = rgb.copy().astype(float)
    for c in range(3):
        ch = cand[:, :, c]
        ch[ill] = mean + (ch[ill] - mean) * 0.95  # 轻微降低插图对比，仍通过各项检查
    cand = np.clip(cand, 0, 255).astype(np.uint8)
    a = analyzer.analyze(_pd(rgb))
    r_default = checker.evaluate(rgb, cand, a, TaskConfig(), regions=reg)
    assert all(c["passed"] for c in r_default["checks"])
    assert r_default["risk_score"] > 0
    assert r_default["decision"] == "accept"
    r_strict = checker.evaluate(rgb, cand, a, TaskConfig(quality_risk_accept_max=0.0), regions=reg)
    assert r_strict["decision"] == "retry_lower_strength"


def test_input_not_modified(samples, checker, analyzer, content):
    rgb = samples["yellow_color_photo"]["image"]
    reg = content.analyze(rgb)
    snap = rgb.copy()
    checker.evaluate(rgb, rgb, analyzer.analyze(_pd(rgb)), TaskConfig(), regions=reg)
    assert np.array_equal(rgb, snap)