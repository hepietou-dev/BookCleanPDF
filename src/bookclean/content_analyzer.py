"""内容区域识别（M3.1）：区分纸张背景 / 文字 / 彩色图像 / 灰度插图 / 特殊内容 / 不确定区域。

只测量、不修改图像。输出与原图同尺寸的 uint8 掩膜（0=非该区域，255=该区域），
附总体置信度、各区域面积比、生成参数与失败标记。传统算法（Lab/HSV/背景差/连通域/形态学），
不引入深度学习。彩色判定结合色相+饱和度+连通域面积，避免把红字/印章误判为插图、把黄纸误判为彩色。
输入统一 RGB / uint8 / C-contiguous；OpenCV 转换显式 COLOR_RGB2*。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

from .page_analyzer import estimate_background, _resize_max

_BG_EST_DIM = 700


@dataclass(frozen=True)
class RegionParams:
    """区域识别参数（启发式门槛，按页面面积缩放）。集中于此，便于校准与记录。"""
    paper_diff: int = 18          # 纸张候选：与背景估计差小于此值
    paper_min_gray: int = 110     # 纸张候选最低亮度（排除深色桌面）
    chroma_sat: int = 55          # 彩色候选最低饱和度
    chroma_dist: int = 35         # 彩色候选与纸色最小距离
    chroma_val: int = 50          # 彩色候选最低明度
    chroma_hue_dist: int = 12     # 色相远离纸色阈值
    chroma_sat_excess: int = 55   # 或饱和度显著高于纸张
    chroma_lab: float = 12.0      # Lab 色度下限（排除无彩色噪声，保留真实彩色）
    dark_diff: int = 28           # 深色墨迹（文字）背景差阈值
    mid_diff_low: int = 8         # 中等差下限（淡色文字/细线/公式 → 不确定）
    ill_texture: float = 6.0      # 灰度插图局部纹理标准差下限（排除平滑阴影）
    color_area: float = 2500.0    # 彩色连通域 >= 此面积(每百万像素) → 彩色插图，否则特殊内容
    min_comp: float = 40.0        # 小于此面积的彩色连通域视为噪声 → 不确定
    ill_area: float = 4000.0      # 灰度插图连通域最小面积(每百万像素)
    ill_min_dim: float = 60.0     # 灰度插图 bbox 最小边长(每百万像素平方根)
    ill_fill: float = 0.5         # 灰度插图实心度下限
    ill_std: float = 12.0         # 灰度插图内部灰度标准差下限
    ill_mid_fraction: float = 0.25  # 灰度插图中调像素占比下限（区分文字的双峰分布）
    close_kernel: int = 21        # 连通域合并闭运算核
    soft_dilate: int = 3          # 保护掩膜软化膨胀半径
    soft_sigma: float = 2.0       # 保护掩膜软化高斯 sigma


@dataclass
class PageRegionMasks:
    """六区域掩膜（uint8, 0/255，与原图同尺寸）+ 置信度 + 元数据。"""
    paper_mask: np.ndarray
    text_mask: np.ndarray
    color_mask: np.ndarray
    illustration_mask: np.ndarray
    special_mask: np.ndarray
    uncertain_mask: np.ndarray
    confidence: float
    metadata: dict = field(default_factory=dict)

    def content_mask(self) -> np.ndarray:
        """所有非纸张内容（需保护）的并集。"""
        return ((self.text_mask > 0) | (self.color_mask > 0) | (self.illustration_mask > 0)
                | (self.special_mask > 0) | (self.uncertain_mask > 0))


def _scale_params(p: RegionParams, h: int, w: int) -> dict:
    unit = (h * w) / 1_000_000.0
    dim_unit = np.sqrt(unit)
    return {
        "color_area": p.color_area * unit,
        "min_comp": p.min_comp * unit,
        "ill_area": p.ill_area * unit,
        "ill_min_dim": p.ill_min_dim * dim_unit,
    }


class ContentAnalyzer:
    def __init__(self, params: RegionParams | None = None):
        self.params = params or RegionParams()

    def analyze(self, rgb: np.ndarray) -> PageRegionMasks:
        p = self.params
        h, w = rgb.shape[:2]
        sp = _scale_params(p, h, w)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        bg = self._bg_full(gray)
        diff = bg.astype(np.int16) - gray.astype(np.int16)

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue, sat, val = hsv[:, :, 0].astype(np.int16), hsv[:, :, 1].astype(np.int16), hsv[:, :, 2].astype(np.int16)

        paper0 = (diff < p.paper_diff) & (gray > p.paper_min_gray)
        flags: list[str] = []
        if paper0.sum() < 0.02 * h * w:
            paper0 = diff < 30
            flags.append("low_paper_candidate")
        if paper0.any():
            paper_rgb = np.median(rgb[paper0], axis=0).astype(np.int16)
            paper_hue = float(np.median(hue[paper0]))
            paper_sat = float(np.median(sat[paper0]))
        else:
            paper_rgb = np.array([255, 255, 255], np.int16)
            paper_hue, paper_sat = 0.0, 0.0
            flags.append("no_reliable_paper")

        dist = np.abs(rgb.astype(np.int16) - paper_rgb).max(axis=2)
        hue_dist = np.abs(hue - paper_hue)
        hue_dist = np.minimum(hue_dist, 180 - hue_dist)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
        chroma_lab = np.sqrt((lab[:, :, 1].astype(np.float32) - 128.0) ** 2
                             + (lab[:, :, 2].astype(np.float32) - 128.0) ** 2)
        chromatic = (sat > p.chroma_sat) & (dist > p.chroma_dist) & (val > p.chroma_val) \
            & ((hue_dist > p.chroma_hue_dist) | (sat > paper_sat + p.chroma_sat_excess)) \
            & (chroma_lab > p.chroma_lab)
        achrom = ~chromatic

        color_mask, special_color, chrom_noise = self._split_chromatic(chromatic, hue, sp, p)

        dark = achrom & (diff > p.dark_diff)
        illustration_mask = self._gray_illustration(achrom, diff, gray, sp, p)
        # 彩色/特殊区域优先，避免灰度插图误吞彩色
        illustration_mask &= ~(color_mask | special_color)

        text_mask = dark & ~illustration_mask & ~color_mask & ~special_color
        mid = achrom & (diff > p.mid_diff_low) & (diff <= p.dark_diff)
        uncertain_mask = (mid & ~illustration_mask & ~color_mask & ~special_color) | chrom_noise

        content = text_mask | color_mask | illustration_mask | special_color | uncertain_mask
        paper_candidate = (~content) & (diff < p.paper_diff) & (gray > p.paper_min_gray)
        photo_dominated = float((color_mask | illustration_mask).mean()) > 0.5
        if photo_dominated:
            # 照片/插图占主导：亮部不是真实纸张，paper_mask 不可信，归入不确定区保持原样
            paper_mask = np.zeros((h, w), bool)
            uncertain_mask = uncertain_mask | paper_candidate
        else:
            paper_mask = paper_candidate

        special_mask = special_color
        masks = {
            "paper": paper_mask, "text": text_mask, "color": color_mask,
            "illustration": illustration_mask, "special": special_mask, "uncertain": uncertain_mask,
        }
        ratios = {k: round(float(v.mean()), 4) for k, v in masks.items()}
        if photo_dominated:
            flags.append("photo_dominated")
        if ratios["paper"] < 0.15:
            flags.append("low_paper_area")
        confidence = float(np.clip(1.0 - 1.5 * ratios["uncertain"]
                                   - (0.4 if ratios["paper"] < 0.15 else 0.0), 0.0, 1.0))
        if "no_reliable_paper" in flags:
            confidence = min(confidence, 0.2)
        if "photo_dominated" in flags:
            # 照片/插图占主导时，亮部会被误判为纸张，paper_mask 不可信，背景白化应跳过
            confidence = min(confidence, 0.5)

        meta = {
            "area_ratios": ratios,
            "params": asdict(p),
            "scaled_thresholds": {k: round(v, 1) for k, v in sp.items()},
            "flags": flags,
            "paper_rgb": paper_rgb.tolist(),
        }
        return PageRegionMasks(
            paper_mask=self._u8(paper_mask), text_mask=self._u8(text_mask),
            color_mask=self._u8(color_mask), illustration_mask=self._u8(illustration_mask),
            special_mask=self._u8(special_mask), uncertain_mask=self._u8(uncertain_mask),
            confidence=round(confidence, 3), metadata=meta,
        )

    def protection_weight(self, masks: PageRegionMasks) -> np.ndarray:
        """内容保护软掩膜（float32 0..1）：膨胀+高斯软化，避免背景处理在插图周围形成光圈。"""
        p = self.params
        content = masks.content_mask().astype(np.uint8) * 255
        k = 2 * p.soft_dilate + 1
        dilated = cv2.dilate(content, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        soft = cv2.GaussianBlur(dilated.astype(np.float32), (0, 0), p.soft_sigma) / 255.0
        return np.clip(soft, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _u8(mask: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(mask.astype(np.uint8) * 255)

    @staticmethod
    def _bg_full(gray: np.ndarray) -> np.ndarray:
        small, scale = _resize_max(gray, _BG_EST_DIM)
        bg_small = estimate_background(small)
        if scale < 1.0:
            bg = cv2.resize(bg_small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            bg = bg_small
        return bg.astype(np.float32)

    @staticmethod
    def _split_chromatic(chromatic, hue, sp, p):
        """彩色连通域：大面积实心 → 彩色插图；小/细 → 特殊内容（印章/批注/标记）；极小 → 噪声。"""
        cm = chromatic.astype(np.uint8) * 255
        closed = cv2.morphologyEx(cm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
        color = np.zeros(chromatic.shape, bool)
        special = np.zeros(chromatic.shape, bool)
        noise = np.zeros(chromatic.shape, bool)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            comp = lab == i
            if area < sp["min_comp"]:
                noise |= comp
            elif area >= sp["color_area"]:
                color |= comp
            else:
                special |= comp
        # 用原始 chromatic 约束，避免闭运算引入的非彩色像素
        return color & chromatic, special & chromatic, noise & chromatic

    def _gray_illustration(self, achrom, diff, gray, sp, p):
        """灰度插图：大面积、实心、内部灰度连续变化（中调占比高）的无彩色连通域。

        用背景差 + 局部纹理共同提取内容，使照片亮部（接近纸色但有纹理）也被纳入。
        """
        gf = gray.astype(np.float32)
        k = p.close_kernel | 1
        mean = cv2.boxFilter(gf, -1, (k, k))
        sqmean = cv2.boxFilter(gf * gf, -1, (k, k))
        local_std = np.sqrt(np.maximum(sqmean - mean * mean, 0.0))
        paper_bright = float(np.median(gray[diff < p.paper_diff])) if (diff < p.paper_diff).any() else 255.0
        # 背景差捕获暗部；全局纸色差+局部纹理捕获照片亮部（纹理门槛排除平滑阴影）
        content = (achrom & ((diff > p.mid_diff_low)
                             | ((np.abs(gf - paper_bright) > 15.0) & (local_std > p.ill_texture)))).astype(np.uint8) * 255
        closed = cv2.morphologyEx(content, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
        ill = np.zeros(gray.shape, bool)
        lo, hi = paper_bright - 120.0, paper_bright - 25.0
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < sp["ill_area"]:
                continue
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if min(bw, bh) < sp["ill_min_dim"]:
                continue
            if area / float(bw * bh) < p.ill_fill:
                continue
            comp = lab == i
            g = gray[comp].astype(np.float32)
            if g.std() < p.ill_std:
                continue
            mid_fraction = float(((g > lo) & (g < hi)).mean())
            if mid_fraction < p.ill_mid_fraction:
                continue
            if g.mean() >= paper_bright - 15.0:
                continue
            ill |= comp
        return ill