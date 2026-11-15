"""几何校正（M2.1）：置信度门控纠斜、保守裁边、高置信度透视校正。

输入/输出均为 RGB / uint8 / C-contiguous。每个能力独立可跳过；
跳过或回退时返回原图（不修改输入数组），并在 ProcessResult 中记录原因。
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .config import TaskConfig
from .models import ProcessResult
from .page_analyzer import PageAnalysis

_WHITE = (255, 255, 255)


def _skip(image: np.ndarray, op: str, reason: str, start: float, **params) -> ProcessResult:
    return ProcessResult(
        image=image,
        executed=False,
        params={"operation": op, "reason": reason, **params},
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
    )


def rotate_expand(image: np.ndarray, angle: float) -> np.ndarray:
    """绕中心旋转 angle 度（正值=逆时针），扩展画布避免裁切，白色填充新边缘。"""
    h, w = image.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(mat[0, 0]), abs(mat[0, 1])
    new_w = int(round(h * sin + w * cos))
    new_h = int(round(h * cos + w * sin))
    mat[0, 2] += new_w / 2 - w / 2
    mat[1, 2] += new_h / 2 - h / 2
    out = cv2.warpAffine(
        image, mat, (new_w, new_h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=_WHITE,
    )
    return np.ascontiguousarray(out)


class GeometryCorrector:
    def deskew(self, image: np.ndarray, analysis: PageAnalysis, config: TaskConfig) -> ProcessResult:
        start = time.perf_counter()
        angle = analysis.skew_angle
        info = {"angle": round(angle, 3), "confidence": round(analysis.skew_confidence, 3)}
        if not config.enable_deskew:
            return _skip(image, "deskew", "disabled", start, **info)
        if analysis.skew_confidence < config.min_skew_confidence:
            return _skip(image, "deskew", "low_confidence", start, **info)
        if abs(angle) < 0.15:
            return _skip(image, "deskew", "already_level", start, **info)
        if abs(angle) > config.deskew_max_angle:
            return _skip(image, "deskew", "angle_out_of_range", start, **info)
        out = rotate_expand(image, -angle)
        return ProcessResult(
            image=out,
            executed=True,
            params={"operation": "deskew", "rotation": round(-angle, 3),
                    "new_size": [out.shape[1], out.shape[0]], **info},
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
        )

    def crop(self, image: np.ndarray, analysis: PageAnalysis, config: TaskConfig) -> ProcessResult:
        start = time.perf_counter()
        if not config.enable_crop:
            return _skip(image, "crop", "disabled", start)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        if analysis.dark_border:
            box = self._dark_border_box(gray, config)
            mode = "dark_border"
        else:
            box = self._content_margin_box(gray, analysis, config)
            mode = "content_margin"
        if box is None:
            return _skip(image, "crop", "no_safe_crop_found", start, mode=mode)
        x0, y0, x1, y1 = box
        if (x1 - x0) * (y1 - y0) >= 0.99 * w * h:
            return _skip(image, "crop", "margin_too_small", start, mode=mode)
        out = np.ascontiguousarray(image[y0:y1, x0:x1])
        return ProcessResult(
            image=out,
            executed=True,
            params={"operation": "crop", "mode": mode, "crop_box": [x0, y0, x1, y1],
                    "original_size": [w, h], "new_size": [x1 - x0, y1 - y0],
                    "removed_fraction": round(1.0 - (x1 - x0) * (y1 - y0) / (w * h), 4)},
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
        )

    @staticmethod
    def _dark_border_box(gray: np.ndarray, config: TaskConfig):
        """黑边路径：取所有大面积亮连通域的外接框（页面可能被深色图带分割）。"""
        h, w = gray.shape
        bright = (gray > 120).astype(np.uint8)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
        if n <= 1:
            return None
        keep = np.zeros((h, w), bool)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= 0.01 * h * w:
                x, y, bw, bh, _ = stats[i]
                keep[y:y + bh, x:x + bw] = True
        if not keep.any():
            return None
        ys, xs = np.where(keep)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        if (x1 - x0) * (y1 - y0) < 0.3 * h * w:
            return None
        mx = int(round(config.crop_safety_margin * w))
        my = int(round(config.crop_safety_margin * h))
        return max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)

    @staticmethod
    def _content_margin_box(gray: np.ndarray, analysis: PageAnalysis, config: TaskConfig):
        """内容边距路径：前景外接框 + 安全边距；裁掉的条带必须不含任何墨迹。"""
        h, w = gray.shape
        ink = gray < (analysis.background_brightness - 35)
        if not ink.any():
            return None
        ys, xs = np.where(ink)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        mx = max(8, int(round(0.015 * w)))
        my = max(8, int(round(0.015 * h)))
        x0, y0 = max(0, x0 - mx), max(0, y0 - my)
        x1, y1 = min(w, x1 + mx), min(h, y1 + my)
        if (x1 - x0) < 0.5 * w or (y1 - y0) < 0.5 * h:
            return None
        return x0, y0, x1, y1

    def correct_perspective(self, image: np.ndarray, analysis: PageAnalysis, config: TaskConfig) -> ProcessResult:
        start = time.perf_counter()
        if not config.enable_perspective:
            return _skip(image, "perspective", "disabled", start)
        if analysis.page_quad is None or analysis.perspective_confidence < config.min_perspective_confidence:
            return _skip(image, "perspective", "low_confidence", start,
                         confidence=round(analysis.perspective_confidence, 3))
        if analysis.perspective_score < config.min_perspective_score:
            return _skip(image, "perspective", "no_significant_perspective", start,
                         score=round(analysis.perspective_score, 3))
        pts = np.asarray(analysis.page_quad, np.float32)
        w_t = int(round(max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3]))))
        h_t = int(round(max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1]))))
        if w_t < 64 or h_t < 64 or not 0.3 <= w_t / h_t <= 3.0:
            return _skip(image, "perspective", "implausible_target_size", start)
        dst = np.array([[0, 0], [w_t - 1, 0], [w_t - 1, h_t - 1], [0, h_t - 1]], np.float32)
        mat = cv2.getPerspectiveTransform(pts, dst)
        out = cv2.warpPerspective(
            image, mat, (w_t, h_t), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=_WHITE,
        )
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        quad_mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(quad_mask, pts.astype(np.int32), 1)
        before = float((gray < 180)[quad_mask > 0].mean())
        after = float((cv2.cvtColor(out, cv2.COLOR_RGB2GRAY) < 180).mean())
        if before > 0.005 and not 0.5 <= after / before <= 2.5:
            return ProcessResult(
                image=image, executed=False, suggest_fallback=True,
                params={"operation": "perspective", "reason": "content_area_changed",
                        "ink_before": round(before, 4), "ink_after": round(after, 4)},
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
                warnings=["perspective_content_change"],
            )
        return ProcessResult(
            image=np.ascontiguousarray(out),
            executed=True,
            params={"operation": "perspective", "quad": analysis.page_quad,
                    "target_size": [w_t, h_t], "matrix": mat.tolist(),
                    "score": round(analysis.perspective_score, 3),
                    "confidence": round(analysis.perspective_confidence, 3)},
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
        )