"""M4.8.1 扫描黑边检测与保守清除。

独立于 M1–M3 流水线：对任务目录的"最终图像"（processed 或 original 回退）做
四边暗带检测，仅在多重守卫全部通过且置信度达标时裁切，并保留安全薄边。
不修改原始页面与 processed_pages；输出 border_pages/ + border_masks/ + Manifest 扩展字段。

检测流程（降采样灰度图）：
  1. 中央区域估计纸张亮度；亮像素占比过低 -> 照片/暗页，整页跳过。
  2. 每条边从边缘向内扫描连续"暗覆盖率 >= min_coverage"的列/行，得到黑边宽度。
  3. 五重守卫：宽度上限 / 局部纹理（文字）/ 亮内容（黑底白字）/
     裁切线内侧连通域跨越（插图压边）/ 内侧纸面确认。
  4. 置信度 = 0.4*覆盖率 + 0.35*暗度 + 0.25*均匀度；低于门槛的边保留不裁。
  5. 裁切线 = 黑边宽度 - 安全薄边；映射回全分辨率，纯像素切片（内部逐像素不变）。

传统 OpenCV/NumPy 实现，无 AI 模型、无网络依赖。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import BorderCleanConfig
from .utils.atomic_file import atomic_save_png, atomic_write_json

logger = logging.getLogger("bookclean")

_EDGES = ("left", "right", "top", "bottom")


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _profile(gray: np.ndarray, edge: str, dark_thr: float) -> tuple[np.ndarray, np.ndarray]:
    dark = gray < dark_thr
    if edge in ("left", "right"):
        cov = dark.mean(axis=0)
        mean = gray.mean(axis=0)
        if edge == "right":
            cov, mean = cov[::-1], mean[::-1]
    else:
        cov = dark.mean(axis=1)
        mean = gray.mean(axis=1)
        if edge == "bottom":
            cov, mean = cov[::-1], mean[::-1]
    return cov, mean


def _band(gray: np.ndarray, edge: str, width: int) -> np.ndarray | None:
    if width <= 0:
        return None
    h, w = gray.shape
    if edge == "left":
        return gray[:, :width]
    if edge == "right":
        return gray[:, w - width:]
    if edge == "top":
        return gray[:width, :]
    return gray[h - width:, :]


def _interior_strip(gray: np.ndarray, edge: str, width: int, depth: int,
                    widths: dict[str, int]) -> np.ndarray:
    h, w = gray.shape
    tw, bw, lw, rw = widths["top"], widths["bottom"], widths["left"], widths["right"]
    if edge == "left":
        return gray[tw:h - bw, width:min(width + depth, w - rw)]
    if edge == "right":
        return gray[tw:h - bw, max(w - width - depth, lw):w - width]
    if edge == "top":
        return gray[width:min(width + depth, h - bw), lw:w - rw]
    return gray[max(h - width - depth, tw):h - width, lw:w - rw]


def _local_std(band: np.ndarray, k: int = 5) -> np.ndarray:
    bf = band.astype(np.float32)
    m = cv2.boxFilter(bf, -1, (k, k))
    sq = cv2.boxFilter(bf * bf, -1, (k, k))
    return np.sqrt(np.maximum(sq - m * m, 0.0))


def _inner_trim(band: np.ndarray, edge: str, n: int = 2) -> np.ndarray:
    # 排除边带内缘的降采样过渡像素（黑边/纸面混合），避免误判纹理；裁切线本就保留安全薄边
    if edge == "left":
        return band[:, :-n] if band.shape[1] > n else band
    if edge == "right":
        return band[:, n:] if band.shape[1] > n else band
    if edge == "top":
        return band[:-n, :] if band.shape[0] > n else band
    return band[n:, :] if band.shape[0] > n else band


def detect_borders(image: np.ndarray, config: BorderCleanConfig | None = None) -> tuple[dict, np.ndarray | None]:
    """检测黑边并规划保守裁切。返回 (检测信息, 全分辨率黑边掩膜或 None)。

    掩膜仅在决定裁切时生成（255=被清除的黑边区域，0=保留区域）。
    检测信息中的 width_px / crop_box 均为全分辨率坐标。
    """
    config = config or BorderCleanConfig()
    config.validate()
    gray_full = _to_gray(image)
    H, W = gray_full.shape
    info: dict = {
        "executed": False, "reason": "", "edges": {},
        "analysis_size": [W, H], "crop_box": None, "final_size": None,
        "paper_brightness": None, "dark_threshold": None,
    }

    scale = min(1.0, config.analysis_max_dim / max(H, W))
    if scale < 1.0:
        aw, ah = max(1, round(W * scale)), max(1, round(H * scale))
        gray = cv2.resize(gray_full, (aw, ah), interpolation=cv2.INTER_AREA)
    else:
        gray, aw, ah = gray_full, W, H
    info["analysis_size"] = [aw, ah]
    sx, sy = W / aw, H / ah

    cy0, cy1 = ah // 5, ah - ah // 5
    cx0, cx1 = aw // 5, aw - aw // 5
    central = gray[cy0:cy1, cx0:cx1]
    bright = central[central > 160]
    if bright.size < config.min_paper_fraction * central.size:
        info["reason"] = "insufficient_paper"
        return info, None
    paper = float(np.median(bright))
    dark_thr = min(config.dark_abs_max, paper - config.dark_delta)
    info["paper_brightness"] = round(paper, 1)
    info["dark_threshold"] = round(dark_thr, 1)
    if dark_thr <= 0:
        info["reason"] = "no_dark_threshold"
        return info, None

    widths: dict[str, int] = {}
    profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for edge in _EDGES:
        cov, mean = _profile(gray, edge, dark_thr)
        profiles[edge] = (cov, mean)
        ok = cov >= config.min_coverage
        if not ok[0]:
            widths[edge] = 0
        elif ok.all():
            widths[edge] = len(ok)
        else:
            widths[edge] = int(np.argmin(ok))

    lines: dict[str, int] = {}
    for edge in _EDGES:
        width = widths[edge]
        dim = aw if edge in ("left", "right") else ah
        e: dict = {"width_analysis": width, "width_px": round(width * (sx if edge in ("left", "right") else sy)),
                   "executed": False, "reason": "", "confidence": 0.0,
                   "coverage": 0.0, "mean": 0.0, "std": 0.0}
        info["edges"][edge] = e
        if width <= 0:
            e["reason"] = "no_border"
            lines[edge] = 0
            continue
        if width > config.max_edge_fraction * dim:
            e["reason"] = "width_exceeds_limit"
            lines[edge] = 0
            continue
        band = _inner_trim(_band(gray, edge, width), edge)
        cov, _ = profiles[edge]
        band_mean = float(band.mean())
        band_std = float(band.std())
        e["coverage"] = round(float(cov[:width].mean()), 3)
        e["mean"] = round(band_mean, 1)
        e["std"] = round(band_std, 1)

        tex_frac = float((_local_std(band) > config.texture_std_max).mean())
        if tex_frac > config.max_texture_fraction:
            e["reason"] = "texture_in_border"
            lines[edge] = 0
            continue
        bright_frac = float((band > dark_thr).mean())
        if bright_frac > config.max_bright_fraction:
            e["reason"] = "bright_content_in_border"
            lines[edge] = 0
            continue
        # 全分辨率亮连通域守卫：小字号白字的全局占比可能低于噪底，但字形团块面积可辨（灰尘小点放行）
        fband = _inner_trim(_band(gray_full, edge, e["width_px"]), edge)
        if fband.size:
            nb, _, st, _ = cv2.connectedComponentsWithStats((fband > dark_thr).astype(np.uint8), connectivity=8)
            largest_bright = int(st[1:, cv2.CC_STAT_AREA].max()) if nb > 1 else 0
            if largest_bright > config.bright_cc_min_area:
                e["reason"] = "bright_content_in_border"
                lines[edge] = 0
                continue

        depth = max(2, int(config.straddle_guard_fraction * dim))
        strip = _interior_strip(gray, edge, width, depth, widths)
        if strip.size == 0:
            e["reason"] = "no_room_inside"
            lines[edge] = 0
            continue
        if float(np.median(strip)) < dark_thr:
            e["reason"] = "no_paper_inside"
            lines[edge] = 0
            continue
        dark_strip = (strip < dark_thr).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(dark_strip, connectivity=8)
        largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
        if largest > config.straddle_cc_fraction * strip.size:
            e["reason"] = "content_straddles_border"
            lines[edge] = 0
            continue

        confidence = (0.4 * e["coverage"]
                      + 0.35 * max(0.0, 1.0 - band_mean / dark_thr)
                      + 0.25 * max(0.0, 1.0 - band_std / 40.0))
        e["confidence"] = round(confidence, 3)
        if confidence < config.min_confidence:
            e["reason"] = "low_confidence"
            lines[edge] = 0
            continue
        sliver = max(1, int(config.safety_margin_fraction * dim))
        line = width - sliver
        if line <= 0:
            e["reason"] = "border_too_thin"
            lines[edge] = 0
            continue
        e["executed"] = True
        e["reason"] = "cropped"
        lines[edge] = line

    if not any(lines.values()):
        info["reason"] = "no_border" if not any(widths.values()) else "all_edges_skipped"
        return info, None

    x0 = round(lines["left"] * sx)
    x1 = W - round(lines["right"] * sx)
    y0 = round(lines["top"] * sy)
    y1 = H - round(lines["bottom"] * sy)
    if x1 - x0 < 10 or y1 - y0 < 10 or (x1 - x0) * (y1 - y0) < config.min_crop_area_fraction * W * H:
        info["reason"] = "crop_too_small"
        for e in info["edges"].values():
            e["executed"] = False
        return info, None

    info["executed"] = True
    info["reason"] = "cropped"
    info["crop_box"] = [int(x0), int(y0), int(x1), int(y1)]
    info["final_size"] = [int(x1 - x0), int(y1 - y0)]
    mask = np.zeros((H, W), np.uint8)
    mask[:y0, :] = 255
    mask[y1:, :] = 255
    mask[:, :x0] = 255
    mask[:, x1:] = 255
    return info, mask


class BorderCleaningRunner:
    """读取任务目录，对每页最终图像做黑边清除，兼容扩展 Manifest（只增不改）。

    源图像经 FinalImageResolver 解析（processed 优先，回退 original），
    解析时忽略已有 border_* 字段，保证重复运行幂等。
    原始页面与 processed_pages 只读；dry_run 模式不写任何文件。
    """

    def run(self, task_dir: str | Path, config: BorderCleanConfig | None = None,
            progress_cb: Callable[[dict], None] | None = None,
            cancel_event=None, dry_run: bool = False) -> dict:
        from .pdf_exporter import FinalImageResolver

        config = config or BorderCleanConfig()
        config.validate()
        task_dir = Path(task_dir)
        manifest_path = task_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        resolver = FinalImageResolver(task_dir)
        if not dry_run:
            (task_dir / "border_pages").mkdir(exist_ok=True)
            (task_dir / "border_masks").mkdir(exist_ok=True)
        started = time.monotonic()
        timings: list[float] = []
        reasons: dict[str, int] = {}
        cleaned = skipped = failed = 0
        pages = sorted(manifest["pages"], key=lambda p: p["page_index"])

        for entry in pages:
            if cancel_event is not None and cancel_event.is_set():
                entry["border_status"] = "cancelled"
                break
            t0 = time.perf_counter()
            try:
                base_page = {k: v for k, v in entry.items() if not k.startswith("border_")}
                fi = resolver.resolve(manifest, base_page)
                if fi.path is None:
                    entry["border_status"] = "skipped"
                    entry["border_detection"] = {"executed": False, "reason": "missing_image"}
                    skipped += 1
                    reasons["missing_image"] = reasons.get("missing_image", 0) + 1
                else:
                    with Image.open(fi.path) as im:
                        mode = im.mode if im.mode in ("RGB", "L") else "RGB"
                        arr = np.ascontiguousarray(np.asarray(im.convert(mode), dtype=np.uint8))
                        dpi = im.info.get("dpi")
                    info, mask = detect_borders(arr, config)
                    entry["border_source_path"] = fi.path.relative_to(task_dir).as_posix()
                    entry["border_detection"] = info
                    if dry_run:
                        entry["border_status"] = "dry_run"
                        if info["executed"]:
                            cleaned += 1
                        else:
                            skipped += 1
                    elif info["executed"]:
                        x0, y0, x1, y1 = info["crop_box"]
                        name = f"page_{entry['display_index']:06d}.png"
                        rel = f"border_pages/{name}"
                        out = Image.fromarray(arr[y0:y1, x0:x1])
                        d = entry.get("dpi") or (dpi[0] if isinstance(dpi, tuple) else dpi) or 300
                        atomic_save_png(out, task_dir / rel, dpi=d)
                        mask_rel = f"border_masks/{name}"
                        atomic_save_png(Image.fromarray(mask), task_dir / mask_rel)
                        entry["border_status"] = "completed"
                        entry["border_image_path"] = rel
                        entry["border_mask_path"] = mask_rel
                        cleaned += 1
                    else:
                        entry["border_status"] = "skipped"
                        skipped += 1
                    r = info.get("reason") or "unknown"
                    reasons[r] = reasons.get(r, 0) + 1
                    del arr, mask
            except Exception as exc:
                entry["border_status"] = "failed"
                entry["border_error"] = f"{type(exc).__name__}: {exc}"
                failed += 1
                logger.error("黑边清除失败 第 %d 页: %s", entry["display_index"], exc)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            entry["border_ms"] = round(elapsed_ms, 1)
            timings.append(elapsed_ms)
            if not dry_run:
                atomic_write_json(manifest_path, manifest)
            if progress_cb is not None:
                progress_cb(entry)

        summary = {
            "total_pages": len(pages),
            "cleaned": cleaned, "skipped": skipped, "failed": failed,
            "dry_run": dry_run,
            "reasons": reasons,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "avg_page_ms": round(sum(timings) / len(timings), 1) if timings else 0.0,
        }
        result = {"task_dir": task_dir, "summary": summary}
        if not dry_run:
            manifest["border_status"] = "completed" if failed == 0 else "completed_with_errors"
            manifest["border_summary"] = summary
            if cleaned:
                sheet = self._comparison_sheet(task_dir, pages)
                manifest["border_contact_sheet"] = sheet
            atomic_write_json(manifest_path, manifest)
            atomic_write_json(task_dir / "border_report.json", summary)
            result["manifest"] = manifest
        else:
            result["pages"] = pages
        return result

    @staticmethod
    def _comparison_sheet(task_dir: Path, pages: list[dict]) -> str:
        done = [p for p in pages if p.get("border_status") == "completed" and p.get("border_image_path")]
        cols, cell_w, cell_h, label_h, pad = 4, 200, 280, 22, 8
        rows = max(1, (len(done) + cols - 1) // cols)
        sheet = Image.new("RGB", (cols * (2 * cell_w + pad) + pad, rows * (cell_h + label_h + pad) + pad), (245, 245, 245))
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype("msyh.ttc", 13)
            label = "第 {} 页  清除前 | 清除后"
        except OSError:
            font = ImageFont.load_default(size=13)
            label = "Page {}  before | after"
        for i, entry in enumerate(done):
            x = pad + (i % cols) * (2 * cell_w + pad)
            y = pad + (i // cols) * (cell_h + label_h + pad)
            for j, key in enumerate(("border_source_path", "border_image_path")):
                with Image.open(task_dir / entry[key]) as thumb:
                    thumb.thumbnail((cell_w, cell_h), Image.LANCZOS)
                    sheet.paste(thumb, (x + j * cell_w + (cell_w - thumb.width) // 2, y))
            draw.text((x, y + cell_h + 3), label.format(entry["display_index"]), fill=(60, 60, 60), font=font)
        rel = "border_contact_sheet.jpg"
        sheet.save(task_dir / rel, format="JPEG", quality=85)
        return rel