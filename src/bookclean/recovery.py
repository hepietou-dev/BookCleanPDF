"""M4.8.4 处理结果完整性校验与断点续跑支持。

复用 processed_pages 前必须多维校验，而非仅看文件存在：
  存在 / 大小>0 / 可完整解码 / 格式 PNG / 尺寸 / 色彩模式 / SHA256 / 处理配置匹配。
任一不符即判定不可复用，续跑时重新处理该页（绝不重跑 M1，绝不改动 original_pages）。

SHA256 与尺寸/色彩/格式在处理完成时写入 Manifest（只增不改 M1/M2 字段）。
所有校验为文件/像素口径，非 OCR。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

# 续跑复用需匹配的处理配置键（与 ProcessingRunner._update_entry.processing_parameters 一致）
_CONFIG_KEYS = ("preset", "output_mode", "background_strength",
                "yellow_removal_strength", "denoise_strength", "sharpen_strength")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_processed_image(path: str | Path, expected: dict | None = None) -> tuple[bool, list[str]]:
    """多维校验处理结果图像。expected 可含 sha256/width_px/height_px/color_mode/format。

    返回 (是否可复用, 不符原因列表)。文件缺失/损坏/被篡改/尺寸或色彩不符均判失败。
    """
    path = Path(path)
    reasons: list[str] = []
    if not path.exists():
        return False, ["文件不存在"]
    if path.stat().st_size <= 0:
        return False, ["文件大小为 0"]
    try:
        with Image.open(path) as im:
            im.load()  # 完整解码，截断/损坏在此暴露
            fmt = (im.format or "").upper()
            mode = im.mode
            w, h = im.size
    except Exception as exc:
        return False, [f"图像无法完整解码: {type(exc).__name__}: {exc}"]

    expected = expected or {}
    exp_fmt = expected.get("format")
    if exp_fmt and fmt != exp_fmt.upper():
        reasons.append(f"格式 {fmt} != 期望 {exp_fmt}")
    elif fmt != "PNG":
        reasons.append(f"格式 {fmt} != PNG")
    exp_mode = expected.get("color_mode")
    if exp_mode and mode != exp_mode:
        reasons.append(f"色彩模式 {mode} != 期望 {exp_mode}")
    exp_w, exp_h = expected.get("width_px"), expected.get("height_px")
    if exp_w and exp_h and (w, h) != (exp_w, exp_h):
        reasons.append(f"尺寸 {w}x{h} != 期望 {exp_w}x{exp_h}")
    exp_sha = expected.get("sha256")
    if exp_sha:
        actual = sha256_file(path)
        if actual != exp_sha:
            reasons.append("SHA256 与记录不符（文件被篡改或损坏）")
    return (not reasons), reasons


def processing_config_matches(entry: dict, config) -> bool:
    """处理配置是否与当前一致（预设/输出模式/各强度）。缺失记录视为不匹配（保守重处理）。"""
    recorded = entry.get("processing_parameters") or {}
    if not recorded:
        return False
    for key in _CONFIG_KEYS:
        if recorded.get(key) != getattr(config, key, None):
            return False
    return True


def can_reuse_processed(task_dir: str | Path, entry: dict, config) -> tuple[bool, list[str]]:
    """续跑时判断某页处理结果可否直接复用（不重跑 M1–M3）。"""
    if entry.get("processing_status") != "completed":
        return False, [f"处理状态 {entry.get('processing_status')} 非 completed"]
    rel = entry.get("processed_image_path")
    if not rel:
        return False, ["缺少 processed_image_path"]
    expected = {
        "sha256": entry.get("processed_sha256"),
        "width_px": entry.get("processed_width_px"),
        "height_px": entry.get("processed_height_px"),
        "color_mode": entry.get("processed_color_mode"),
        "format": entry.get("processed_format"),
    }
    ok, reasons = validate_processed_image(Path(task_dir) / rel, expected)
    if not ok:
        return False, reasons
    if not processing_config_matches(entry, config):
        return False, ["处理配置与当前不一致"]
    return True, []