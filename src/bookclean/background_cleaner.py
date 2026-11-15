"""背景清理（M2.2 / M3.2 区域化）：背景估计、亮度均一化、Lab 去黄、轻度去噪、内容保护。

核心原则：改善背景而不损伤内容。M3.2 在 M2 已验证的除法亮度校正基础上增加区域约束：
背景统计/均一化/去黄只针对可信纸张背景（ContentAnalyzer 的 paper_mask），文字、彩色插图、
灰度插图、特殊内容与不确定区域通过软保护掩膜保持原样；纸张面积/置信度不足或照片主导时
跳过区域漂白，保留几何校正结果。处理前后检查彩色色度/面积/分布/边缘保留。
输入/输出均为 RGB / uint8 / C-contiguous。
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .config import TaskConfig
from .content_analyzer import ContentAnalyzer, PageRegionMasks
from .models import ProcessResult
from .page_analyzer import PageAnalysis, _resize_max, estimate_background

_TARGET_BG = 245.0  # 均一化目标背景亮度（接近白而非纯白，避免色阶断层）
_BG_EST_DIM = 700  # 背景估计降采样尺寸（低频场，降采样后上采样即可）
_CHROMA_KEEP = 0.95  # 彩色色度保留验收下限
_CHROMA_FALLBACK = 0.90  # 低于此值视为彩色损伤，建议回退


class BackgroundCleaner:
    def __init__(self):
        self._content = ContentAnalyzer()

    def process(self, image: np.ndarray, analysis: PageAnalysis, config: TaskConfig,
                regions: PageRegionMasks | None = None) -> ProcessResult:
        start = time.perf_counter()
        steps: list[dict] = []
        warnings: list[str] = []
        out = image
        if regions is None:
            regions = self._content.analyze(image)
        paper_mask = regions.paper_mask > 0
        paper_area = float(paper_mask.mean())
        flags = regions.metadata.get("flags", [])
        paper_ok = (regions.confidence >= config.min_paper_confidence
                    and paper_area >= config.min_paper_area_ratio
                    and "photo_dominated" not in flags
                    and "no_reliable_paper" not in flags)

        before = self._bg_stats(out, paper_mask)
        bg_full = self._estimate_bg_full(out)
        protect = self._protection_mask(out, regions)
        paper = self._paper_color(out, paper_mask)

        # 1) 亮度均一化：仅对可信纸张背景执行；干净页/照片主导/低置信度跳过
        if config.background_strength <= 0:
            steps.append({"op": "background_normalize", "executed": False, "reason": "disabled"})
        elif not paper_ok:
            reason = self._skip_reason(regions, paper_area, config)
            steps.append({"op": "background_normalize", "executed": False, "reason": reason})
            warnings.append(f"background_normalize_skipped:{reason}")
        elif before["std"] > 3.0 or before["mean"] < 248.0:
            out = self._normalize(out, bg_full, protect, config.background_strength)
            steps.append({"op": "background_normalize", "executed": True,
                          "strength": config.background_strength, "target": _TARGET_BG})
        else:
            steps.append({"op": "background_normalize", "executed": False, "reason": "background_already_clean"})

        # 2) 去黄：Lab 色度中性化，按纸色接近度加权；照片主导/低置信度时跳过以免彩色褪色
        if config.yellow_removal_strength <= 0 or analysis.yellow_index <= 0.05:
            steps.append({"op": "deyellow", "executed": False, "reason": "no_significant_yellowing"})
        elif not paper_ok:
            steps.append({"op": "deyellow", "executed": False, "reason": self._skip_reason(regions, paper_area, config)})
        else:
            out = self._deyellow(out, protect, config.yellow_removal_strength)
            steps.append({"op": "deyellow", "executed": True,
                          "strength": config.yellow_removal_strength, "yellow_index": round(analysis.yellow_index, 3)})

        # 3) 去噪：背景轻度中值 + 孤立污点 removal（保护标点/彩色/插图）
        if config.denoise_strength > 0:
            keep = (regions.color_mask > 0) | (regions.illustration_mask > 0) | (regions.special_mask > 0)
            out, removed = self._despeckle(out, bg_full, protect, config.denoise_strength,
                                           config.output_dpi, paper, keep)
            steps.append({"op": "denoise", "executed": True, "strength": config.denoise_strength,
                          "isolated_specks_removed": removed})
        else:
            steps.append({"op": "denoise", "executed": False, "reason": "disabled"})

        # 4) 彩色内容保护检查（处理前后色度/面积/分布/边缘）
        suggest_fallback = False
        color_check = self._color_preservation(image, out, regions.color_mask | regions.special_mask)
        if color_check is not None:
            steps.append({"op": "color_protection_check", "executed": True, **color_check})
            if color_check["chroma_retention"] < _CHROMA_FALLBACK:
                suggest_fallback = True
                warnings.append("color_content_faded")

        after = self._bg_stats(out, paper_mask)
        out = np.ascontiguousarray(out)
        return ProcessResult(
            image=out,
            executed=any(s["executed"] for s in steps),
            params={"operation": "background_clean", "bg_before": before, "bg_after": after,
                    "region_info": {"confidence": regions.confidence, "paper_area_ratio": round(paper_area, 4),
                                    "paper_ok": paper_ok, "flags": flags,
                                    "area_ratios": regions.metadata.get("area_ratios", {})}},
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
            warnings=warnings,
            suggest_fallback=suggest_fallback,
            steps=steps,
            regions=regions,
        )

    @staticmethod
    def _skip_reason(regions: PageRegionMasks, paper_area: float, config: TaskConfig) -> str:
        flags = regions.metadata.get("flags", [])
        if "photo_dominated" in flags:
            return "photo_dominated"
        if "no_reliable_paper" in flags:
            return "no_reliable_paper"
        if regions.confidence < config.min_paper_confidence:
            return "low_region_confidence"
        if paper_area < config.min_paper_area_ratio:
            return "insufficient_paper_area"
        return "untrusted_paper_background"

    @staticmethod
    def _estimate_bg_full(rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        small, scale = _resize_max(gray, _BG_EST_DIM)
        bg_small = estimate_background(small)
        if scale < 1.0:
            bg = cv2.resize(bg_small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            bg = bg_small
        return bg.astype(np.float32)

    def _protection_mask(self, rgb: np.ndarray, regions: PageRegionMasks) -> np.ndarray:
        """软保护掩膜 0..1：六区域内容（膨胀+高斯软化）并上梯度细笔画，避免插图周围光圈且保护细字。"""
        region_w = self._content.protection_weight(regions)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        stroke = (grad > 45).astype(np.uint8) * 255
        stroke = cv2.dilate(stroke, np.ones((3, 3), np.uint8), iterations=1)
        stroke_w = cv2.GaussianBlur(stroke.astype(np.float32), (0, 0), 1.5) / 255.0
        return np.clip(np.maximum(region_w, stroke_w), 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _paper_color(rgb: np.ndarray, paper_mask: np.ndarray) -> np.ndarray:
        region = paper_mask & (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) > 120)
        if region.any():
            return np.median(rgb[region], axis=0).astype(np.float32)
        return np.array([255.0, 255.0, 255.0], np.float32)

    @staticmethod
    def _normalize(rgb: np.ndarray, bg_full: np.ndarray, protect: np.ndarray, strength: float) -> np.ndarray:
        gain = _TARGET_BG / np.maximum(bg_full, 1.0)
        gain_eff = 1.0 + (gain - 1.0) * strength
        corrected = rgb.astype(np.float32) * gain_eff[..., None]
        p = protect[..., None]
        out = rgb.astype(np.float32) * p + corrected * (1.0 - p)
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _deyellow(rgb: np.ndarray, protect: np.ndarray, strength: float) -> np.ndarray:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).astype(np.float32)
        L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        bg_region = (protect < 0.3) & (L > 120)
        if not bg_region.any():
            return rgb
        pa, pb = float(np.median(a[bg_region])), float(np.median(b[bg_region]))
        d = np.sqrt((a - pa) ** 2 + (b - pb) ** 2)
        weight = np.exp(-((d / 25.0) ** 2)) * (1.0 - protect) * np.clip((L - 100.0) / 80.0, 0, 1)
        a2 = a + (128.0 - a) * strength * weight
        b2 = b + (128.0 - b) * strength * weight
        lab2 = np.stack([L, a2, b2], axis=2).astype(np.uint8)
        return cv2.cvtColor(lab2, cv2.COLOR_Lab2RGB)

    @staticmethod
    def _despeckle(rgb, bg_full, protect, strength, dpi, paper, keep_mask):
        out = rgb.astype(np.float32)
        med = cv2.medianBlur(rgb, 3).astype(np.float32)
        dk = (strength * (1.0 - protect))[..., None]
        out = out * (1.0 - dk) + med * dk
        out = np.clip(out, 0, 255).astype(np.uint8)

        # 孤立小污点 removal：仅移除远离任何文字、且不在彩色/插图/特殊区域内的小连通域，保护标点/脚注/彩色细节
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(np.float32)
        spots = ((bg_full - gray) > 40).astype(np.uint8)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(spots, 8)
        max_area = max(4, int((dpi / 150.0) ** 2 * 8))
        large = np.zeros(gray.shape, np.uint8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] > max_area:
                large[labels == i] = 1
        radius = max(5, int(dpi / 150.0 * 8))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        near_text = cv2.dilate(large, k)
        removed = 0
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] <= max_area:
                cx, cy = centroids[i]
                if not near_text[int(cy), int(cx)] and not keep_mask[int(cy), int(cx)]:
                    out[labels == i] = paper.astype(np.uint8)
                    removed += 1
        return np.ascontiguousarray(out), removed

    @staticmethod
    def _color_preservation(before: np.ndarray, after: np.ndarray, color_mask: np.ndarray) -> dict | None:
        cm = color_mask > 0
        if not cm.any():
            return None
        cb, ca = before.astype(np.int16), after.astype(np.int16)
        chroma_b = float((cb.max(2) - cb.min(2))[cm].mean())
        chroma_a = float((ca.max(2) - ca.min(2))[cm].mean())
        chroma_ret = chroma_a / max(1e-6, chroma_b)
        still_color = ((ca.max(2) - ca.min(2)) > 30) & cm
        area_ret = float(still_color.sum() / cm.sum())
        med_b = np.median(cb[cm], axis=0)
        med_a = np.median(ca[cm], axis=0)
        color_shift = float(np.abs(med_a - med_b).mean())

        def gmag(rgb):
            g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
            return np.sqrt(gx * gx + gy * gy)

        edge_b = float(gmag(before)[cm].mean())
        edge_a = float(gmag(after)[cm].mean())
        edge_ret = edge_a / max(1e-6, edge_b)
        return {"chroma_retention": round(chroma_ret, 4), "color_area_retention": round(area_ret, 4),
                "median_color_shift": round(color_shift, 2), "edge_retention": round(edge_ret, 4),
                "passed": bool(chroma_ret >= _CHROMA_KEEP)}

    @staticmethod
    def _bg_stats(rgb: np.ndarray, paper_mask: np.ndarray | None = None) -> dict:
        """纸张背景亮度/均匀度：优先用真实纸张像素（腐蚀掉内容边缘带，避免彩色/文字渗漏）。

        背景估计的闭运算+高斯会把彩色内容扩散到邻近纸张，故统计改用 paper_mask 内的实际像素，
        与 M3.3 质量检查的分区背景指标口径一致（std(image[paper_mask>0])）。
        """
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        small, _ = _resize_max(gray, 640)
        if paper_mask is not None and paper_mask.any():
            pm = cv2.resize(paper_mask.astype(np.uint8), (small.shape[1], small.shape[0]),
                            interpolation=cv2.INTER_NEAREST) > 0
            k = max(3, (max(small.shape) // 40) | 1)
            eroded = cv2.erode(pm.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
            if eroded.sum() >= 0.02 * pm.size:
                pm = eroded
            vals = small[pm].astype(np.float32)
        else:
            bg = estimate_background(small)
            diff = bg.astype(np.int16) - small.astype(np.int16)
            pm = (diff < 15) & (small > 120)
            if not pm.any():
                pm = diff < 30
            vals = bg[pm].astype(np.float32)
        return {"mean": round(float(vals.mean()), 1), "std": round(float(vals.std()), 2)}