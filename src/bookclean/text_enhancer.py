"""文字增强（M2.3）：Clean Gray（默认）、可选 Sauvola 自适应二值化、自动保色。

不凭空生成或改写笔画；仅做轻度锐化与（可选）局部阈值。
含彩色内容的页面自动保留 RGB；二值化仅对适合的纯文字页启用。
Sauvola 用 NumPy/OpenCV 积分图实现，不引入 scikit-image。
输入/输出均为 RGB / uint8 / C-contiguous。
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .config import TaskConfig
from .models import ProcessResult
from .page_analyzer import PageAnalysis

_COLOR_THRESHOLD = 0.005  # 彩色占比超过此值自动保色


def sauvola_threshold(gray: np.ndarray, window: int, k: float, R: float = 128.0) -> np.ndarray:
    """Sauvola 局部自适应阈值（积分图实现）。返回 uint8 二值图（前景 0 / 背景 255）。"""
    g = gray.astype(np.float64)
    h, w = g.shape
    pad = window // 2
    padded = cv2.copyMakeBorder(g, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    integral = cv2.integral(padded)
    integral_sq = cv2.integral(padded * padded)
    area = float(window * window)
    s = integral[window:, window:] - integral[:-window, window:] - integral[window:, :-window] + integral[:-window, :-window]
    s2 = integral_sq[window:, window:] - integral_sq[:-window, window:] - integral_sq[window:, :-window] + integral_sq[:-window, :-window]
    mean = s / area
    var = np.maximum(s2 / area - mean * mean, 0.0)
    std = np.sqrt(var)
    thresh = mean * (1.0 + k * (std / R - 1.0))
    return np.where(g > thresh, 255, 0).astype(np.uint8)


class TextEnhancer:
    def process(self, image: np.ndarray, analysis: PageAnalysis, config: TaskConfig) -> ProcessResult:
        start = time.perf_counter()
        has_color = analysis.color_ratio > _COLOR_THRESHOLD

        if config.output_mode == "adaptive_bw" and config.enable_binarization:
            suitable = (
                analysis.color_ratio < _COLOR_THRESHOLD
                and analysis.foreground_ratio > 0.02
                and analysis.background_variance < 15.0
            )
            if suitable:
                out = self._binarize(image, config)
                return ProcessResult(
                    image=out, executed=True,
                    params={"operation": "text_enhance", "mode": "adaptive_bw",
                            "k": config.binarization_k},
                    elapsed_ms=(time.perf_counter() - start) * 1000.0,
                )
            # 不适合二值化：回退 Clean Gray，保留灰度/彩色
            out, sharpened = self._clean_gray(image, config, has_color)
            return ProcessResult(
                image=out, executed=sharpened,
                params={"operation": "text_enhance", "mode": "clean_gray",
                        "binarization_skipped": "unsuitable_page"},
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
                warnings=["binarization_skipped_unsuitable"],
            )

        mode = "preserve_color" if has_color else "clean_gray"
        out, sharpened = self._clean_gray(image, config, has_color)
        return ProcessResult(
            image=out, executed=sharpened,
            params={"operation": "text_enhance", "mode": mode,
                    "sharpen_strength": config.sharpen_strength, "color_preserved": has_color},
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
        )

    @staticmethod
    def _clean_gray(image: np.ndarray, config: TaskConfig, has_color: bool):
        """轻度 unsharp 锐化（在 Lab 的 L 通道，保留色度与抗锯齿）。"""
        if config.sharpen_strength <= 0:
            return image, False
        amount = config.sharpen_strength * 3.0  # 0.15 -> 0.45
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        L = lab[:, :, 0].astype(np.float32)
        blur = cv2.GaussianBlur(L, (0, 0), sigmaX=1.0)
        sharp = np.clip(L + amount * (L - blur), 0, 255)
        lab[:, :, 0] = sharp.astype(np.uint8)
        out = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
        return np.ascontiguousarray(out), True

    @staticmethod
    def _binarize(image: np.ndarray, config: TaskConfig) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        window = max(15, (min(gray.shape) // 30) | 1)
        bw = sauvola_threshold(gray, window, config.binarization_k)
        return np.ascontiguousarray(cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB))