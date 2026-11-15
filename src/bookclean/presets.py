"""统一预设配置（M2.4）。所有参数集中于此，算法文件不写死默认值。

预设键沿用 M0/M1 的 TaskConfig.preset 取值：
  auto        -> Auto Clean（默认，通用扫描/手机）
  old_book    -> Old Book（泛黄发灰旧书）
  phone_photo -> Photo（手机拍摄）
  faithful    -> Preserve（轻度修复、最大保真）
"""

from __future__ import annotations

from .config import TaskConfig

_PRESET_DEFS: dict[str, dict] = {
    "auto": dict(
        background_strength=0.7, yellow_removal_strength=0.6, denoise_strength=0.25,
        sharpen_strength=0.15, enable_deskew=True, enable_crop=True, enable_perspective=True,
        output_mode="clean_gray",
    ),
    "old_book": dict(
        background_strength=0.85, yellow_removal_strength=0.8, denoise_strength=0.35,
        sharpen_strength=0.12, enable_deskew=True, enable_crop=True, enable_perspective=False,
        output_mode="clean_gray",
    ),
    "phone_photo": dict(
        background_strength=0.8, yellow_removal_strength=0.5, denoise_strength=0.3,
        sharpen_strength=0.15, enable_deskew=True, enable_crop=True, enable_perspective=True,
        output_mode="clean_gray",
    ),
    "faithful": dict(
        background_strength=0.3, yellow_removal_strength=0.2, denoise_strength=0.1,
        sharpen_strength=0.0, enable_deskew=True, enable_crop=True, enable_perspective=False,
        min_skew_confidence=0.7, output_mode="clean_gray",
    ),
}

PRESET_LABELS = {
    "auto": "Auto Clean", "old_book": "Old Book", "phone_photo": "Photo", "faithful": "Preserve",
}


def get_preset(name: str) -> TaskConfig:
    if name not in _PRESET_DEFS:
        raise ValueError(f"未知预设 {name!r}，可选: {tuple(_PRESET_DEFS)}")
    cfg = TaskConfig(preset=name, **_PRESET_DEFS[name])
    cfg.validate()
    return cfg


def list_presets() -> list[str]:
    return list(_PRESET_DEFS)