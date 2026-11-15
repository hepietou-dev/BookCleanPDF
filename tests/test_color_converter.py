"""M4.7.1 灰度转换测试：亮度加权 / 单通道 uint8 C-contiguous / 不改输入 / 不二值化 / RGBA 白底。"""

import cv2
import numpy as np
import pytest

from bookclean.color_converter import (composite_alpha_white, rgb_to_gray,
                                       to_grayscale)


def _solid(rgb, h=64, w=64):
    return np.tile(np.array(rgb, dtype=np.uint8), (h, w, 1))


# ---------- 10.1 颜色区域与亮度关系 ----------

@pytest.mark.parametrize("rgb,expected", [
    ((255, 0, 0), 76),    # 红
    ((0, 255, 0), 150),   # 绿
    ((0, 0, 255), 29),    # 蓝
    ((255, 255, 0), 226), # 黄
    ((255, 255, 255), 255),  # 白
    ((0, 0, 0), 0),       # 黑
    ((128, 128, 128), 128),  # 灰
])
def test_luminance_values(rgb, expected):
    g = rgb_to_gray(_solid(rgb))
    assert g.shape == (64, 64)
    assert int(g[0, 0]) == expected


def test_rgb_not_mapped_to_same_gray():
    r = int(rgb_to_gray(_solid((255, 0, 0)))[0, 0])
    g = int(rgb_to_gray(_solid((0, 255, 0)))[0, 0])
    b = int(rgb_to_gray(_solid((0, 0, 255)))[0, 0])
    assert len({r, g, b}) == 3          # 红绿蓝不得同灰度值
    assert g > r > b                    # 亮度加权：绿>红>蓝


def test_not_arithmetic_mean():
    rgb = _solid((255, 0, 0))
    g = int(rgb_to_gray(rgb)[0, 0])
    assert g != int(np.mean([255, 0, 0]))  # 76 != 85，证明非算术平均


def test_matches_opencv_rgb2gray():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    assert np.array_equal(rgb_to_gray(rgb), cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))


# ---------- 输出契约 ----------

def test_output_contract_random_color():
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (100, 120, 3), dtype=np.uint8)
    res = to_grayscale(rgb)
    g = res.gray
    assert g.ndim == 2
    assert g.dtype == np.uint8
    assert g.flags["C_CONTIGUOUS"]
    assert g.shape == (100, 120)
    assert g.min() >= 0 and g.max() <= 255
    assert res.source_mode == "RGB"
    assert res.converted and not res.already_grayscale


def test_input_not_mutated():
    rng = np.random.default_rng(2)
    rgb = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    before = rgb.copy()
    to_grayscale(rgb)
    assert np.array_equal(rgb, before)


def test_gradient_keeps_midtones():
    h, w = 64, 256
    ramp = np.linspace(0, 255, w).astype(np.uint8)
    rgb = np.stack([np.tile(ramp, (h, 1))] * 3, axis=2)
    g = to_grayscale(rgb).gray
    uniq = np.unique(g)
    assert len(uniq) > 100                 # 充足中间灰阶
    assert not set(uniq).issubset({0, 255})  # 非二值


def test_no_binarization_on_photo():
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    g = to_grayscale(rgb).gray
    vals = np.unique(g)
    assert len(vals) > 50
    assert not np.all((g == 0) | (g == 255))


# ---------- 已灰度 / RGB 三通道相等 ----------

def test_already_single_channel_L():
    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    res = to_grayscale(arr)
    assert res.source_mode == "L"
    assert res.already_grayscale and not res.converted
    assert np.array_equal(res.gray, arr)
    assert res.gray.flags["C_CONTIGUOUS"]


def test_rgb_equal_channels_skips_conversion():
    ramp = np.linspace(0, 255, 64).astype(np.uint8)
    plane = np.tile(ramp, (64, 1))
    rgb = np.stack([plane] * 3, axis=2)
    res = to_grayscale(rgb)
    assert res.source_mode == "RGB"
    assert res.already_grayscale and not res.converted
    assert np.array_equal(res.gray, plane)


def test_single_channel_1bit_bool():
    arr = np.zeros((16, 16), dtype=bool)
    arr[::2, ::2] = True
    res = to_grayscale(arr)
    assert res.source_mode == "1"
    assert res.gray.dtype == np.uint8
    assert set(np.unique(res.gray)) == {0, 255}
    assert res.already_grayscale


def test_channel_axis_one():
    plane = np.arange(64, dtype=np.uint8).reshape(8, 8)
    arr = plane[:, :, None]
    res = to_grayscale(arr)
    assert res.source_mode == "L"
    assert np.array_equal(res.gray, plane)


# ---------- RGBA 白底合成 ----------

def test_rgba_transparent_becomes_white():
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)  # 全透明黑
    res = to_grayscale(rgba)
    assert res.alpha_composited
    assert res.source_mode == "RGBA"
    assert int(res.gray[0, 0]) == 255            # 透明->白，而非黑


def test_rgba_opaque_preserves_color():
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[:, :, 0] = 255  # 不透明红
    rgba[:, :, 3] = 255
    res = to_grayscale(rgba)
    assert int(res.gray[0, 0]) == 76


def test_rgba_half_alpha():
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[:, :, 3] = 128  # 半透明黑 -> 约 128 灰
    rgb = composite_alpha_white(rgba)
    assert rgb.shape == (16, 16, 3)
    assert abs(int(rgb[0, 0, 0]) - 128) <= 1


def test_composite_does_not_mutate():
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[:, :, 3] = 100
    before = rgba.copy()
    composite_alpha_white(rgba)
    assert np.array_equal(rgba, before)


# ---------- 错误输入 ----------

def test_reject_wrong_shape():
    with pytest.raises(ValueError):
        rgb_to_gray(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError):
        rgb_to_gray(np.zeros((8, 8, 4), dtype=np.uint8))


def test_reject_wrong_dtype():
    with pytest.raises(ValueError):
        rgb_to_gray(np.zeros((8, 8, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        to_grayscale(np.zeros((8, 8, 3), dtype=np.int16))


def test_reject_unsupported_channels():
    with pytest.raises(ValueError):
        to_grayscale(np.zeros((8, 8, 5), dtype=np.uint8))