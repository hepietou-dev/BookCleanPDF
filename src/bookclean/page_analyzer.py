"""页面分析（M2.1）：只测量页面特征，不修改图像。

所有分数都是启发式指标（范围见字段注释），不是统计学置信概率。
输入统一为 RGB / uint8 / C-contiguous；OpenCV 转换显式使用 COLOR_RGB2*。
倾斜检测以文字行投影分析为主、霍夫长直线交叉验证，不依赖纸张外轮廓。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .models import PageData

PHOTO_MAX_DIM = 640  # 光度分析降采样尺寸
SKEW_MAX_DIM = 900  # 倾斜分析降采样尺寸
QUAD_MAX_DIM = 800  # 四角检测降采样尺寸


@dataclass
class PageAnalysis:
    skew_angle: float = 0.0  # 度；内容相对水平的倾斜，正值=逆时针；纠正需旋转 -skew_angle
    skew_confidence: float = 0.0  # 0..1，启发式可靠度
    page_quad: list | None = None  # 页面四角 [[x,y]*4]，原图像素坐标，顺序 tl,tr,br,bl
    perspective_confidence: float = 0.0  # 0..1，四角检测可靠度
    perspective_score: float = 0.0  # 0..1，越高透视畸变越明显
    background_brightness: float = 255.0  # 0..255，背景平均亮度
    background_variance: float = 0.0  # 背景亮度标准差（灰度级，0..~80），越高越不均
    yellow_index: float = 0.0  # 0..1，纸张泛黄程度（Lab b 通道偏移归一化）
    noise_level: float = 0.0  # 0..1，背景高频残差归一化
    sharpness: float = 0.0  # 拉普拉斯方差（未校准，越高越清晰；与分辨率相关）
    color_ratio: float = 0.0  # 0..1，彩色内容像素占比（已排除纸张底色）
    foreground_ratio: float = 0.0  # 0..1，深色前景像素占比
    dark_border: bool = False  # 是否检测到扫描仪黑边
    warnings: list[str] = field(default_factory=list)
    recommended_steps: list[str] = field(default_factory=list)


def _resize_max(img: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    scale = min(1.0, max_dim / max(img.shape[:2]))
    if scale >= 1.0:
        return img, 1.0
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return out, scale


def estimate_background(gray: np.ndarray) -> np.ndarray:
    """大尺度形态学闭运算 + 高斯平滑估计低频背景（纸张亮度场）。

    核尺寸随图像尺寸自适应；闭运算用亮背景填充深色文字笔画。
    """
    k = max(15, (max(gray.shape) // 12) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return cv2.GaussianBlur(closed, (0, 0), sigmaX=max(gray.shape) / 40.0)


def _order_points(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float64)


def _corner_angle(a, b, c) -> float:
    v1, v2 = a - b, c - b
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


class PageAnalyzer:
    def __init__(self, skew_range: float = 8.0, coarse_step: float = 0.5, fine_step: float = 0.1):
        self.skew_range = skew_range
        self.coarse_step = coarse_step
        self.fine_step = fine_step

    def analyze(self, page: PageData) -> PageAnalysis:
        rgb = page.image
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        a = PageAnalysis()
        self._photometrics(rgb, a)
        self._skew(gray, a)
        self._quad(gray, a)
        self._recommend(a)
        return a

    def photometrics(self, rgb: np.ndarray) -> PageAnalysis:
        """几何校正后重新测量光度特征（不做倾斜/四角检测）。"""
        a = PageAnalysis()
        self._photometrics(rgb, a)
        return a

    def _photometrics(self, rgb: np.ndarray, a: PageAnalysis) -> None:
        small, _ = _resize_max(rgb, PHOTO_MAX_DIM)
        g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        bg = estimate_background(g)
        diff = bg.astype(np.int16) - g.astype(np.int16)

        ring = max(2, int(0.01 * max(g.shape)))
        border = np.concatenate([g[:ring].ravel(), g[-ring:].ravel(), g[:, :ring].ravel(), g[:, -ring:].ravel()])
        a.dark_border = bool(border.mean() < 100 and float(np.median(bg)) > 150)

        # 纸张候选：接近背景估计、非深色桌面；用色度签名(R-G,G-B)修剪彩色内容块，
        # 保留亮度变化（阴影/渐变仍属于纸张，应计入背景不均）。
        paper0 = (diff < 20) & (g > 100)
        if paper0.any():
            rg = small[:, :, 0].astype(np.int16) - small[:, :, 1].astype(np.int16)
            gb = small[:, :, 1].astype(np.int16) - small[:, :, 2].astype(np.int16)
            sig_dist = np.maximum(np.abs(rg - np.median(rg[paper0])), np.abs(gb - np.median(gb[paper0])))
            paper = paper0 & (sig_dist < 25)
        else:
            paper = diff < 40
        stats_mask = paper
        if a.dark_border and paper.any():
            # 黑边页：背景估计在页面边界处有平滑凹陷，统计时腐蚀掉边界带
            eroded = cv2.erode(paper.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
            if eroded.any():
                stats_mask = eroded
        if stats_mask.any():
            a.background_brightness = float(bg[stats_mask].mean())
            a.background_variance = float(bg[stats_mask].std())
        else:
            a.background_brightness = float(bg.mean())
            a.background_variance = float(bg.std())
        a.foreground_ratio = float((diff > 30).mean())

        quiet = cv2.erode((paper & (diff < 3)).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if quiet.any():
            resid = (g.astype(np.int16) - cv2.medianBlur(g, 3).astype(np.int16))[quiet]
            a.noise_level = float(np.clip(resid.std() / 15.0, 0, 1))

        lab = cv2.cvtColor(small, cv2.COLOR_RGB2Lab)
        paper_bright = paper & (lab[:, :, 0] > 100)
        if paper_bright.any():
            b_mean = float(lab[:, :, 2][paper_bright].mean())
            a.yellow_index = float(np.clip((b_mean - 128.0) / 40.0, 0.0, 1.0))

        a.sharpness = float(cv2.Laplacian(g, cv2.CV_64F).var())

        # 彩色内容：高饱和且偏离纸色；色相远离纸色，或饱和度显著高于纸张
        # （排除泛黄纸张本身及其文字抗锯齿边缘——它们与纸色同色相）。
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        if paper.any():
            paper_hue = float(np.median(hsv[:, :, 0][paper]))
            paper_sat = float(np.median(hsv[:, :, 1][paper]))
            paper_rgb = np.median(small[paper], axis=0)
        else:
            paper_hue, paper_sat, paper_rgb = 0.0, 0.0, np.array([255, 255, 255])
        dist = np.abs(small.astype(np.int16) - paper_rgb.astype(np.int16)).max(axis=2)
        hue_dist = np.abs(hsv[:, :, 0].astype(np.int16) - paper_hue)
        hue_dist = np.minimum(hue_dist, 180 - hue_dist)
        sat = hsv[:, :, 1].astype(np.int16)
        color_mask = (sat > 60) & (dist > 40) & (hsv[:, :, 2] > 60) & ((hue_dist > 15) | (sat > paper_sat + 60))
        a.color_ratio = float(color_mask.mean())

    def _skew(self, gray: np.ndarray, a: PageAnalysis) -> None:
        g, _ = _resize_max(gray, SKEW_MAX_DIM)
        # 用背景差提取墨迹，避免阴影/泛黄干扰 Otsu 全局阈值
        bg = estimate_background(g)
        ink = ((bg.astype(np.int16) - g.astype(np.int16)) > 30).astype(np.uint8) * 255
        ratio = float((ink > 0).mean())
        if not 0.003 <= ratio <= 0.5:
            blur = cv2.GaussianBlur(g, (3, 3), 0)
            _, ink = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            ratio = float((ink > 0).mean())
        if not 0.003 <= ratio <= 0.5:
            ink = cv2.Canny(g, 50, 150)
            if float((ink > 0).mean()) < 0.002:
                a.skew_confidence = 0.0
                a.warnings.append("no_reliable_skew_reference")
                return
        ink = (ink > 0).astype(np.uint8)
        pad = int(0.15 * max(ink.shape))
        ink = cv2.copyMakeBorder(ink, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

        angles = np.arange(-self.skew_range, self.skew_range + 1e-9, self.coarse_step)
        scores = np.array([self._proj_score(ink, float(t)) for t in angles])
        best = float(angles[int(np.argmax(scores))])
        fine = np.arange(best - self.coarse_step, best + self.coarse_step + 1e-9, self.fine_step)
        fine_scores = [self._proj_score(ink, float(t)) for t in fine]
        rot = float(fine[int(np.argmax(fine_scores))])  # 使内容水平所需的旋转角
        a.skew_angle = -rot

        prominence = float((scores.max() - np.median(scores)) / (scores.max() + 1e-9))
        conf = float(np.clip(prominence * 1.6, 0.0, 1.0))

        lines = cv2.HoughLinesP(
            cv2.Canny(g, 50, 150), 1, np.pi / 360, threshold=80,
            minLineLength=int(0.25 * g.shape[1]), maxLineGap=8,
        )
        line_angles = []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                ang = math.degrees(math.atan2(-(y2 - y1), x2 - x1))
                if abs(ang) <= 12.0:
                    line_angles.append(ang)
        if len(line_angles) >= 3:
            med = float(np.median(line_angles))
            if abs(med - a.skew_angle) <= 1.5:
                conf = min(1.0, conf + 0.15)
            elif abs(med - a.skew_angle) > 3.0:
                conf *= 0.5
        a.skew_confidence = conf

    @staticmethod
    def _proj_score(ink: np.ndarray, angle: float) -> float:
        if abs(angle) < 1e-6:
            rot = ink
        else:
            h, w = ink.shape
            mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rot = cv2.warpAffine(ink, mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        rows = rot.sum(axis=1, dtype=np.float64)
        return float((rows ** 2).sum())

    def _quad(self, gray: np.ndarray, a: PageAnalysis) -> None:
        g, scale = _resize_max(gray, QUAD_MAX_DIM)
        blur = cv2.GaussianBlur(g, (5, 5), 0)
        edges = cv2.dilate(cv2.Canny(blur, 50, 150), np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            ratio = cv2.contourArea(c) / (g.shape[0] * g.shape[1])
            if not 0.25 <= ratio <= 0.98:
                continue
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            pts = _order_points(approx.reshape(4, 2).astype(np.float64))
            dev = max(abs(_corner_angle(pts[i - 1], pts[i], pts[(i + 1) % 4]) - 90.0) for i in range(4))
            if dev > 35.0:
                continue
            w1, w2 = np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3])
            h1, h2 = np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1])
            edge_dev = max(abs(w1 / (w2 + 1e-9) - 1.0), abs(h1 / (h2 + 1e-9) - 1.0))
            if edge_dev > 0.45:
                continue
            conf = float(np.clip(1.0 - dev / 35.0 * 0.5 - edge_dev / 0.45 * 0.3, 0.0, 1.0)) * min(1.0, ratio / 0.5)
            score = float(np.clip(dev / 25.0 * 0.6 + edge_dev / 0.3 * 0.4, 0.0, 1.0))
            if best is None or conf > best[0]:
                best = (conf, score, pts / scale)
        if best is not None:
            a.perspective_confidence, a.perspective_score = best[0], best[1]
            a.page_quad = best[2].round(1).tolist()

    def _recommend(self, a: PageAnalysis) -> None:
        steps = []
        if a.skew_confidence >= 0.5 and 0.15 <= abs(a.skew_angle) <= 8.0:
            steps.append("deskew")
        if a.dark_border:
            steps.append("crop")
        if a.page_quad is not None and a.perspective_confidence >= 0.7 and a.perspective_score >= 0.12:
            steps.append("perspective")
        if a.background_variance > 6.0 or a.background_brightness < 235:
            steps.append("background_normalize")
        if a.yellow_index > 0.12:
            steps.append("deyellow")
        if a.noise_level > 0.25:
            steps.append("denoise")
        if a.color_ratio > 0.003:
            steps.append("preserve_color")
        a.recommended_steps = steps
        if a.background_variance > 25.0:
            a.warnings.append("heavy_background_unevenness")
        if a.foreground_ratio < 0.005:
            a.warnings.append("very_little_content")
        if a.color_ratio > 0.02:
            a.warnings.append("significant_color_content")