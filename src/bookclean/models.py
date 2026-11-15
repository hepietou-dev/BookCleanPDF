"""核心数据结构。分数范围含义见各字段注释；启发式分数不是统计置信概率。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .content_analyzer import PageRegionMasks


@dataclass
class PageData:
    page_index: int
    image: np.ndarray
    source_path: str
    source_kind: str  # "pdf" / "image"
    original_width: int
    original_height: int
    source_dpi: float | None
    color_mode: str
    metadata: dict = field(default_factory=dict)


@dataclass
class PageFeatures:
    orientation: int | None = None
    skew_angle: float | None = None
    skew_confidence: float = 0.0  # 0..1，启发式可靠度
    page_boundary: list | None = None
    perspective_score: float = 0.0  # 0..1，越高透视畸变越明显
    background_unevenness: float = 0.0  # 0..1，越高背景亮度越不均
    yellowing_score: float = 0.0  # 0..1，越高泛黄越明显
    blur_score: float = 0.0  # 拉普拉斯方差类指标，越高越清晰（未校准）
    noise_score: float = 0.0  # 0..1，越高噪点越多
    color_content_score: float = 0.0  # 0..1，彩色内容占比估计
    text_density: float = 0.0  # 0..1，深色墨迹像素占比估计
    is_spread_candidate: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class ProcessResult:
    """处理模块统一返回：保留过程信息，不只返回图像。"""

    image: np.ndarray
    executed: bool
    params: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    suggest_fallback: bool = False
    steps: list = field(default_factory=list)  # M2：各处理步骤记录（名称/参数/是否执行/回退原因）
    quality: dict | None = None  # M2：质量检查结果（QualityChecker 输出）
    regions: PageRegionMasks | None = None  # M3：区域掩膜（背景清理产出，供质量检查/回退复用）