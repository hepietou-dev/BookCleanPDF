"""M4.7.1 灰度转换：RGB/RGBA/单通道 -> 8-bit 单通道灰度。

导出级色彩转换，发生在 M3 内容识别与 FinalImageResolver 之后、ImageEncoder 之前。
不修改 original_pages / processed_pages / Manifest；纯函数，不改输入数组。

算法：OpenCV cv2.COLOR_RGB2GRAY（标准亮度加权 0.299R+0.587G+0.114B，非算术平均，非 BGR）。
输出：单通道二维 uint8 C-contiguous，0–255，保留中间灰阶；不二值化、不复制颜色通道。
RGBA：先合成白底再转灰度，避免丢弃 Alpha 造成黑底。
已灰度（单通道，或 RGB 三通道全等）：跳过实质转换，标记 already_grayscale。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GrayscaleResult:
    gray: np.ndarray            # 2D uint8 C-contiguous
    source_mode: str            # "L" / "RGB" / "RGBA" / "1"
    already_grayscale: bool     # 输入本就单通道或 RGB 三通道全等
    converted: bool             # 是否做了实质 RGB->Gray 亮度加权转换
    alpha_composited: bool      # 是否做了 RGBA 白底合成


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 (H,W,3) -> 单通道 uint8 (H,W)，亮度加权，C-contiguous，不改输入。"""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb_to_gray 需要 (H,W,3) RGB 数组，收到 shape={rgb.shape}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"rgb_to_gray 需要 uint8，收到 dtype={rgb.dtype}")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return np.ascontiguousarray(gray, dtype=np.uint8)


def composite_alpha_white(rgba: np.ndarray) -> np.ndarray:
    """RGBA uint8 (H,W,4) -> RGB uint8 (H,W,3)，透明区合成到白底。"""
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"composite_alpha_white 需要 (H,W,4) RGBA 数组，收到 shape={rgba.shape}")
    if rgba.dtype != np.uint8:
        raise ValueError(f"composite_alpha_white 需要 uint8，收到 dtype={rgba.dtype}")
    fg = rgba[:, :, :3].astype(np.float64)
    alpha = rgba[:, :, 3:4].astype(np.float64) / 255.0
    out = fg * alpha + 255.0 * (1.0 - alpha)
    return np.ascontiguousarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))


def _rgb_equal_channels(rgb: np.ndarray) -> bool:
    return bool(np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
                and np.array_equal(rgb[:, :, 1], rgb[:, :, 2]))


def to_grayscale(arr: np.ndarray) -> GrayscaleResult:
    """任意受支持页面数组 -> 8-bit 单通道灰度。

    支持：2D 单通道（L/1）、(H,W,3) RGB、(H,W,4) RGBA、(H,W,1)。
    不修改输入；输出恒为 2D uint8 C-contiguous。
    """
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8) * 255
        source_mode = "1"
        return GrayscaleResult(np.ascontiguousarray(arr, dtype=np.uint8),
                               source_mode, True, False, False)
    if arr.dtype != np.uint8:
        raise ValueError(f"to_grayscale 需要 uint8（或 bool）数组，收到 dtype={arr.dtype}")

    if arr.ndim == 2:
        return GrayscaleResult(np.ascontiguousarray(arr, dtype=np.uint8),
                               "L", True, False, False)

    if arr.ndim == 3:
        ch = arr.shape[2]
        if ch == 1:
            return GrayscaleResult(np.ascontiguousarray(arr[:, :, 0], dtype=np.uint8),
                                   "L", True, False, False)
        if ch == 3:
            if _rgb_equal_channels(arr):
                # 三通道全等：直接取单通道，避免重复实质转换
                return GrayscaleResult(np.ascontiguousarray(arr[:, :, 0], dtype=np.uint8),
                                       "RGB", True, False, False)
            return GrayscaleResult(rgb_to_gray(arr), "RGB", False, True, False)
        if ch == 4:
            rgb = composite_alpha_white(arr)
            return GrayscaleResult(rgb_to_gray(rgb), "RGBA", False, True, True)
        raise ValueError(f"to_grayscale 不支持 {ch} 通道数组")

    raise ValueError(f"to_grayscale 需要 2D 或 3D 数组，收到 ndim={arr.ndim}")