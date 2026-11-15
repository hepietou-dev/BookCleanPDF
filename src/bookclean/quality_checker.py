"""质量检查与自动回退决策（M2.3 / M3.3 分区升级）。

保留 M2 七项检查（size/blank/foreground/background/sharpness/color/skew_residual），升级为
五维分区评估：A 纸张背景 / B 文字保留 / C 彩色保留 / D 插图保留 / E 几何完整。背景指标依据
可信 paper_mask 的真实像素计算（彩色/插图/印章不再污染背景统计）；彩色与插图按各自掩膜评估；
页面类型自适应阈值集中于 config；输出 0..100 可解释风险评分（各项贡献可追踪），严重文字损伤
不可被总分抵消。指标是图像风险提示，不等于 OCR 准确率/文字完整性证明。
决策：accept / retry_lower_strength / fallback_original。
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import TaskConfig
from .content_analyzer import ContentAnalyzer, PageRegionMasks
from .models import PageData
from .page_analyzer import PageAnalyzer, PageAnalysis, _resize_max, estimate_background

_DIM_WEIGHTS = {"text": 0.40, "color": 0.20, "background": 0.15, "illustration": 0.15, "geometry": 0.10}
# 已近乎纯净纸张（std≈0）的绝对容差：去噪/增强引入的 <1 灰阶 std 属数值噪声，
# 避免乘法比例阈值在 before_std=0 时退化、对干净页触发无意义重试（不掩盖真实背景损伤）。
_BG_STD_ABS_TOL = 1.0


def _fg_mask(gray: np.ndarray) -> np.ndarray:
    bg = estimate_background(gray)
    return (bg.astype(np.int16) - gray.astype(np.int16)) > 30


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _chroma(rgb: np.ndarray) -> np.ndarray:
    a = rgb.astype(np.int16)
    return (a.max(2) - a.min(2)).astype(np.float32)


def _erode(mask: np.ndarray, k: int = 3) -> np.ndarray:
    return cv2.erode(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


def _paper_stats(gray: np.ndarray, paper_mask: np.ndarray) -> dict | None:
    """纸张背景真实像素统计：亮度用全部纸张像素，std 用腐蚀后的内部纸张（避开内容边缘渗漏）。"""
    if not paper_mask.any():
        return None
    er = _erode(paper_mask, max(3, (max(gray.shape) // 40) | 1))
    std_mask = er if er.sum() >= 0.02 * paper_mask.size else paper_mask
    return {"brightness": round(float(gray[paper_mask].mean()), 1),
            "std": round(float(gray[std_mask].std()), 2)}


def _classify_page_type(regions: PageRegionMasks) -> str:
    r = regions.metadata.get("area_ratios", {})
    flags = regions.metadata.get("flags", [])
    color, ill = r.get("color", 0.0), r.get("illustration", 0.0)
    if "photo_dominated" in flags:
        return "photo_dominated"
    if regions.confidence < 0.5 or "no_reliable_paper" in flags:
        return "uncertain"
    if color >= 0.05:
        return "color_illustration"
    if ill >= 0.05:
        return "grayscale_illustration"
    if color + ill >= 0.005 or r.get("special", 0.0) >= 0.002:
        return "mixed"
    return "pure_text"


class QualityChecker:
    def __init__(self):
        self._analyzer = PageAnalyzer()
        self._content = ContentAnalyzer()

    def evaluate(
        self,
        original: np.ndarray,
        candidate: np.ndarray,
        analysis: PageAnalysis,
        config: TaskConfig,
        geometry_applied: bool = False,
        deskew_applied: bool = False,
        regions: PageRegionMasks | None = None,
    ) -> dict:
        ao = self._analyzer.photometrics(original)
        ac = self._analyzer.photometrics(candidate)
        aligned = original.shape == candidate.shape
        if aligned:
            orig_regions = regions if regions is not None else self._content.analyze(original)
            cand_regions = orig_regions
        else:
            # 尺寸不对齐（几何裁切/旋转）：两侧各自分析，掩膜不可跨尺寸混用
            orig_regions = self._content.analyze(original)
            cand_regions = self._content.analyze(candidate)
        paper_mask = orig_regions.paper_mask > 0
        text_mask = orig_regions.text_mask > 0
        color_mask = orig_regions.color_mask > 0
        special_mask = orig_regions.special_mask > 0
        colored_mask = color_mask | special_mask  # 全部彩色内容（含印章/批注等特殊彩色）
        ill_mask = orig_regions.illustration_mask > 0
        c_text = cand_regions.text_mask > 0
        c_ill = cand_regions.illustration_mask > 0
        c_paper = cand_regions.paper_mask > 0
        go, gc = _gray(original), _gray(candidate)
        page_type = _classify_page_type(orig_regions)

        checks: list[dict] = []
        reasons: list[str] = []
        severe = False
        mild = False

        def add(name, passed, value, detail, severity="mild"):
            nonlocal severe, mild
            checks.append({"name": name, "passed": bool(passed), "value": value, "detail": detail})
            if not passed:
                reasons.append(f"{name}: {detail}")
                if severity == "severe":
                    severe = True
                else:
                    mild = True

        # 1) 页面尺寸（E 几何）
        area_ratio = (candidate.shape[0] * candidate.shape[1]) / (original.shape[0] * original.shape[1])
        min_area = 0.4 if geometry_applied else 0.9
        size_ok = area_ratio >= min_area
        add("size", size_ok, round(area_ratio, 3),
            f"面积比 {area_ratio:.2f}（阈值 {min_area}）", "severe" if area_ratio < 0.4 else "mild")

        # 2) 空白页（照片页前景占比指标不适用，改用灰度动态范围判断是否被抹空）
        if page_type == "photo_dominated":
            cand_std = float(gc.std())
            add("blank", cand_std >= config.quality_photo_blank_std_min, round(cand_std, 2),
                f"照片页候选灰度 std {cand_std:.2f}（阈值 {config.quality_photo_blank_std_min}）", "severe")
        else:
            add("blank", ac.foreground_ratio >= 0.003, round(ac.foreground_ratio, 4),
                f"候选前景占比 {ac.foreground_ratio:.4f}", "severe")

        # 3) 前景保留（B 文字）
        fg_retention = None
        if not geometry_applied and aligned:
            go_s, _ = _resize_max(go, 800)
            gc_s, _ = _resize_max(gc, 800)
            fo, fc = _fg_mask(go_s), _fg_mask(gc_s)
            fg_retention = float((fo & fc).sum() / max(1, fo.sum()))
            add("foreground", fg_retention >= 0.75, round(fg_retention, 3),
                f"前景像素保留 {fg_retention:.2f}", "severe" if fg_retention < 0.5 else "mild")
        else:
            fg_retention = ac.foreground_ratio / max(1e-6, ao.foreground_ratio)
            add("foreground", fg_retention >= 0.4, round(fg_retention, 3),
                f"几何变换后前景占比比 {fg_retention:.2f}", "severe" if fg_retention < 0.25 else "mild")

        # 4) 背景均匀度（A 纸张：依据各自可信 paper_mask 真实像素，裁切后仍可比）
        paper_before = _paper_stats(go, paper_mask)
        paper_after = _paper_stats(gc, c_paper)
        if paper_before and paper_after:
            std_limit = max(paper_before["std"] * config.quality_bg_std_ratio_max,
                            paper_before["std"] + _BG_STD_ABS_TOL)
            bg_ok = (paper_after["std"] <= std_limit
                     or paper_after["brightness"] >= paper_before["brightness"])
            add("background", bg_ok, paper_after["std"],
                f"纸张背景 std {paper_before['std']:.1f}->{paper_after['std']:.1f}, "
                f"亮度 {paper_before['brightness']:.0f}->{paper_after['brightness']:.0f}")
        else:
            bg_ok = ac.background_variance <= ao.background_variance * 1.15 or ac.background_brightness >= ao.background_brightness
            add("background", bg_ok, round(ac.background_variance, 2),
                f"背景 std {ao.background_variance:.1f}->{ac.background_variance:.1f}")

        # 5) 清晰度（二值化页面跳过）
        is_binary = len(np.unique(candidate[::4, ::4])) <= 3
        sharp_ratio = None
        if not is_binary and ao.sharpness > 1.0:
            sharp_ratio = ac.sharpness / ao.sharpness
            add("sharpness", sharp_ratio >= 0.6, round(sharp_ratio, 3), f"清晰度比 {sharp_ratio:.2f}")

        # 6) 彩色保留（C：按 color+special 掩膜色度保留率，印章/批注褪色同样视为损伤）
        chroma_ret = color_area_ret = color_shift = color_edge_ret = None
        if aligned and colored_mask.any():
            cm_c = colored_mask
            if cm_c.any():
                chroma_ret = float(_chroma(candidate)[cm_c].mean() / max(1e-6, _chroma(original)[colored_mask].mean()))
                still = (_chroma(candidate) > 30) & cm_c
                color_area_ret = float(still.sum() / max(1, cm_c.sum()))
                color_shift = float(np.abs(np.median(candidate.astype(int)[cm_c], axis=0)
                                           - np.median(original.astype(int)[colored_mask], axis=0)).mean())
                add("color", chroma_ret >= config.quality_color_chroma_min, round(chroma_ret, 3),
                    f"彩色色度保留 {chroma_ret:.2f}", "severe" if chroma_ret < 0.5 else "mild")
        elif aligned and ao.color_ratio > 0.01 and not is_binary:
            color_ratio = ac.color_ratio / ao.color_ratio
            add("color", color_ratio >= 0.3, round(color_ratio, 3), f"彩色占比比 {color_ratio:.2f}")
        # 非对齐（几何裁切）时不做跨尺寸彩色判定：残留纸色会污染候选掩膜；
        # 彩色褪色由 BackgroundCleaner 的同尺寸 color_protection_check(suggest_fallback) 把关。

        # 7) 纠斜残差（E 几何）
        skew_residual = None
        if deskew_applied:
            pd = PageData(0, candidate, "candidate", "image", candidate.shape[1], candidate.shape[0], None, "RGB")
            a2 = self._analyzer.analyze(pd)
            skew_residual = round(a2.skew_angle, 3)
            add("skew_residual", abs(a2.skew_angle) <= 1.0, skew_residual, f"残差 {a2.skew_angle:.2f}°")

        # B) 文字核心保留（严重文字损伤不可被总分抵消）
        text_core_ret = text_core_shift = None
        if text_mask.any():
            if aligned:
                core = _erode(text_mask, 3) & (go < 128)
                if core.any():
                    text_core_ret = float((gc < 160)[core].sum() / core.sum())
                    text_core_shift = float(np.abs(gc.astype(int) - go.astype(int))[core].mean())
            else:
                # 跨几何：全掩膜暗像素占比比（保守代理）。腐蚀核心口径会被残留阴影带等
                # 误判为文字的大块区域污染，故跨尺寸不用核心口径；真实笔画核心保留另由测试直接度量。
                tm_c = c_text
                dark_o = float((go < 128)[text_mask].mean())
                dark_c = float((gc < 160)[tm_c].mean()) if tm_c.any() else 0.0
                text_core_ret = min(1.0, dark_c / max(1e-6, dark_o))
            if text_core_ret is not None:
                add("text_core", text_core_ret >= config.quality_text_core_min, round(text_core_ret, 3),
                    f"文字核心保留 {text_core_ret:.2f}", "severe" if text_core_ret < 0.5 else "mild")

        # D) 插图灰度层次保留
        ill_std_ret = ill_edge_ret = None
        if ill_mask.any():
            sb = float(go[ill_mask].std())
            if sb > 1.0:
                if aligned:
                    sa = float(gc[ill_mask].std())
                else:
                    im_c = c_ill
                    sa = float(gc[im_c].std()) if im_c.any() else 0.0
                ill_std_ret = sa / sb
                add("illustration", ill_std_ret >= config.quality_illustration_std_min, round(ill_std_ret, 3),
                    f"插图灰度动态保留 {ill_std_ret:.2f}")

        # 五维分区指标
        dimensions = {
            "paper_background": {"before": paper_before, "after": paper_after,
                                 "yellow_before": round(ao.yellow_index, 3), "yellow_after": round(ac.yellow_index, 3)},
            "text_preservation": {"foreground_retention": round(fg_retention, 3) if fg_retention is not None else None,
                                  "text_core_retention": round(text_core_ret, 3) if text_core_ret is not None else None,
                                  "text_core_gray_shift": round(text_core_shift, 2) if text_core_shift is not None else None},
            "color_preservation": {"chroma_retention": round(chroma_ret, 3) if chroma_ret is not None else None,
                                   "area_retention": round(color_area_ret, 3) if color_area_ret is not None else None,
                                   "median_color_shift": round(color_shift, 2) if color_shift is not None else None},
            "illustration_preservation": {"gray_std_retention": round(ill_std_ret, 3) if ill_std_ret is not None else None},
            "geometry_integrity": {"area_ratio": round(area_ratio, 3), "skew_residual": skew_residual,
                                   "geometry_applied": geometry_applied},
        }

        # 0..100 可解释风险评分（各项贡献可追踪；严重项已由 severe 强制回退）
        text_pen = float(np.clip(1.0 - (text_core_ret if text_core_ret is not None else fg_retention), 0, 1))
        color_pen = float(np.clip(1.0 - chroma_ret, 0, 1)) if chroma_ret is not None else 0.0
        bg_pen = 0.0 if bg_ok else 0.5
        ill_pen = float(np.clip(1.0 - ill_std_ret, 0, 1)) if ill_std_ret is not None else 0.0
        geo_pen = 0.0 if size_ok else float(np.clip(1.0 - area_ratio, 0, 1))
        penalties = {"text": text_pen, "color": color_pen, "background": bg_pen,
                     "illustration": ill_pen, "geometry": geo_pen}
        contributions = [{"dimension": k, "penalty": round(p, 3), "weight": _DIM_WEIGHTS[k],
                          "contribution": round(100.0 * p * _DIM_WEIGHTS[k], 2)} for k, p in penalties.items()]
        risk_score = round(min(100.0, sum(c["contribution"] for c in contributions)), 2)

        if severe:
            decision = "fallback_original"
        elif mild:
            decision = "retry_lower_strength"
        elif risk_score > config.quality_risk_accept_max:
            decision = "retry_lower_strength"
        else:
            decision = "accept"
        return {
            "decision": decision,
            "checks": checks,
            "reasons": reasons,
            "page_type": page_type,
            "region_mask_confidence": orig_regions.confidence,
            "region_area_ratios": {k: round(float(v), 4) for k, v in orig_regions.metadata.get("area_ratios", {}).items()},
            "dimensions": dimensions,
            "risk_score": risk_score,
            "risk_contributions": contributions,
            "metrics": {
                "area_ratio": round(area_ratio, 3),
                "fg_orig": round(ao.foreground_ratio, 4),
                "fg_cand": round(ac.foreground_ratio, 4),
                "bg_std_orig": round(ao.background_variance, 2),
                "bg_std_cand": round(ac.background_variance, 2),
                "color_orig": round(ao.color_ratio, 4),
                "color_cand": round(ac.color_ratio, 4),
            },
        }