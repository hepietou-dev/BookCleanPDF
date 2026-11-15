"""图像标准化：统一 RGB / uint8 / C 连续数组；Alpha 合成白底；ICC 尽力处理。

本模块只做输入标准化，不做任何增强（去黄/纠斜/锐化等属于 M2+）。
"""

import io

import numpy as np
from PIL import Image, ImageCms

_SRGB = ImageCms.createProfile("sRGB")


def _apply_icc(im: Image.Image, warnings: list[str]) -> Image.Image:
    icc = im.info.get("icc_profile")
    if not icc:
        return im
    try:
        profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        return ImageCms.profileToProfile(im, profile, _SRGB, outputMode="RGB")
    except Exception as exc:
        warnings.append(f"ICC 色彩配置处理失败，按默认转换: {type(exc).__name__}: {exc}")
        return im


def _composite_white(im: Image.Image) -> Image.Image:
    rgba = im.convert("RGBA")
    base = Image.new("RGB", rgba.size, (255, 255, 255))
    base.paste(rgba, mask=rgba.getchannel("A"))
    return base


def normalize_to_rgb(im: Image.Image) -> tuple[Image.Image, list[str]]:
    """把任意常见色彩模式的 Pillow 图像转为 RGB，返回 (图像, 警告列表)。"""
    warnings: list[str] = []
    mode = im.mode

    if mode in ("RGBA", "LA") or (mode == "P" and "transparency" in im.info):
        return _composite_white(im), warnings
    if mode == "CMYK":
        converted = _apply_icc(im, warnings)
        if converted.mode != "RGB":
            warnings.append("CMYK 无可用 ICC 配置，使用近似转换，颜色可能有偏差")
            converted = converted.convert("RGB")
        return converted, warnings
    if mode == "RGB":
        return _apply_icc(im, warnings), warnings
    if mode in ("L", "P", "1", "I", "F", "I;16", "I;16B"):
        try:
            return im.convert("RGB"), warnings
        except Exception as exc:
            warnings.append(f"{mode} 模式直接转 RGB 失败，经灰度中转: {type(exc).__name__}")
            return im.convert("L").convert("RGB"), warnings
    try:
        return im.convert("RGB"), warnings
    except Exception as exc:
        warnings.append(f"未知模式 {mode} 转 RGB 失败，经灰度中转: {type(exc).__name__}")
        return im.convert("L").convert("RGB"), warnings


def to_rgb_array(im: Image.Image) -> np.ndarray:
    """Pillow 图像 -> RGB / uint8 / C-contiguous NumPy 数组。"""
    arr = np.asarray(im, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"标准化结果不是 RGB 图像: shape={arr.shape}")
    return np.ascontiguousarray(arr)


def make_thumbnail(im: Image.Image, max_edge: int = 300) -> Image.Image:
    """生成最长边 max_edge 的缩略图；resize 返回新图像，不改原图对象与像素。"""
    w, h = im.size
    scale = min(1.0, max_edge / max(w, h))
    if scale >= 1.0:
        return im.copy()
    return im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)