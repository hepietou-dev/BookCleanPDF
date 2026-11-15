"""M4.8.3 逐页自适应编码研究（实测，非 OCR）。

在用户选定的压缩档位（balanced/compact）内，研究逐页自适应编码是否稳定优于现有固定策略。
固定策略（image_encoder.py）：
- balanced：PNG 与 JPEG q95 4:4:4 比体积取小 + 守卫（mean-diff>2.5 回退 PNG）。
- compact：文字/灰度 q90、彩色 q88；同样 PNG↔JPEG 取小 + 守卫。灰度 compact 恒 q90（研究点）。

自适应候选（仅 OpenCV/像素辅助特征，不引入 OCR、不引入大型模型）：
- 质量阶梯：在档位内逐页尝试更低 JPEG 质量，看守卫与文字锐度是否允许。
- 色度子采样：彩色照片页 4:4:4 -> 4:2:0（仅辅助评估，文字/地图/批注页禁用）。

多维质量口径（启发式像素，非 OCR/文本完整性证明）：
  global mean-abs-diff、max-abs-diff、文字锐度保留率（Laplacian）、彩色区域均差。
决策规则：自适应只有在「体积稳定下降 且 文字锐度不降 且 彩色内容不劣化 且 守卫不超阈」时才采纳。

样本：真实 120 页子集（FinalImageResolver 选最终图，含黑边清除结果）+ 10 类合成页。
用法: python scripts/m4_8_3_encoding_research.py [输出目录]
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from PIL import Image

from bookclean.config import PDFExportConfig
from bookclean.color_converter import to_grayscale
from bookclean.image_encoder import (
    ImageEncoder, _jpeg_bytes, _png_bytes, _is_exactly_gray,
    JPEG_GUARD_MEAN_DIFF_MAX, BALANCED_JPEG_QUALITY,
    COMPACT_JPEG_QUALITY_COLOR, COMPACT_JPEG_QUALITY_TEXT,
)
from bookclean.page_layout import PageLayoutResolver
from bookclean.pdf_exporter import FinalImageResolver
from make_m2_samples import table_page, text_page  # noqa: E402
from m4_2_compare import build_pages  # noqa: E402

LADDER = (82, 85, 88, 90, 92, 95)
SUBSAMPLE_QUALITIES = (85, 88, 90)
COLOR_DIFF_THRESHOLD = 20  # 通道极差超过此值视为彩色像素
REAL_TASK = Path("temp/m4_6_subset/tasks/task_20261112_140743_86d276")


def _sharpness(gray: np.ndarray) -> float:
    g = gray.astype(np.float64)
    lap = (g[2:, 1:-1] + g[:-2, 1:-1] + g[1:-1, 2:] + g[1:-1, :-2] - 4 * g[1:-1, 1:-1])
    return float(np.abs(lap).mean())


def _features(arr_rgb: np.ndarray, page_type: str | None) -> dict:
    gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(small, 80, 180)
    edge_density = float((edges > 0).mean())
    hist = cv2.calcHist([small], [0], None, [64], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())
    chroma = arr_rgb.max(axis=2).astype(np.int16) - arr_rgb.min(axis=2).astype(np.int16)
    color_frac = float((chroma > COLOR_DIFF_THRESHOLD).mean())
    return {"page_type": page_type, "edge_density": round(edge_density, 5),
            "entropy": round(entropy, 4), "color_pixel_frac": round(color_frac, 5)}


def _jpeg_candidate(work_im: Image.Image, src: np.ndarray, q: int, subsampling: int,
                    sharp_src: float, color_mask: np.ndarray | None) -> dict:
    data = _jpeg_bytes_q(work_im, q, subsampling)
    dec_im = Image.open(io.BytesIO(data)).convert(work_im.mode)
    dec = np.asarray(dec_im, dtype=np.int16)
    diff = np.abs(dec - src)
    mean_diff = float(diff.mean())
    gray_dec = np.asarray(dec_im.convert("L"), dtype=np.float64)
    sharp = _sharpness(gray_dec)
    rec = {"q": q, "subsampling": subsampling, "bytes": len(data),
           "mean_diff": round(mean_diff, 4), "max_diff": int(diff.max()),
           "sharp_retention": round(sharp / sharp_src, 5) if sharp_src > 0 else 1.0,
           "guard_pass": mean_diff <= JPEG_GUARD_MEAN_DIFF_MAX}
    if color_mask is not None and color_mask.any():
        rec["color_diff"] = round(float(diff.mean(axis=2)[color_mask].mean()), 4)
    else:
        rec["color_diff"] = 0.0
    return rec


def _jpeg_bytes_q(im: Image.Image, quality: int, subsampling: int) -> bytes:
    buf = io.BytesIO()
    try:
        im.save(buf, format="JPEG", quality=quality, subsampling=subsampling, optimize=True)
    except OSError:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, subsampling=subsampling, optimize=False)
    return buf.getvalue()


def _prepare(im: Image.Image, color_mode: str):
    """返回 (work_im, src_int16, gray_bool, color_mask|None)，与编码器口径一致。"""
    if color_mode == "grayscale":
        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
            arr = np.asarray(im.convert("RGBA"))
        elif im.mode == "1":
            arr = np.asarray(im.convert("L"))
        elif im.mode == "L":
            arr = np.asarray(im)
        else:
            arr = np.asarray(im.convert("RGB"))
        res = to_grayscale(arr)
        work = Image.fromarray(res.gray)
        src = np.asarray(work, dtype=np.int16)
        return work, src, True, None
    rgb = im.convert("RGB")
    arr = np.asarray(rgb)
    gray = _is_exactly_gray(arr)
    if gray:
        work = rgb.convert("L")
        src = np.asarray(work, dtype=np.int16)
        return work, src, True, None
    work = rgb
    src = np.asarray(work, dtype=np.int16)
    chroma = arr.max(axis=2).astype(np.int16) - arr.min(axis=2).astype(np.int16)
    color_mask = chroma > COLOR_DIFF_THRESHOLD
    return work, src, False, color_mask


def _fixed_baselines(image_path: Path, layout, page_type: str | None) -> dict:
    out = {}
    for color_mode in ("preserve", "grayscale"):
        for level in ("balanced", "compact"):
            cfg = PDFExportConfig(compression=level, color_mode=color_mode)
            enc = ImageEncoder(cfg).encode(image_path, layout, page_type=page_type)
            dec = np.asarray(Image.open(io.BytesIO(enc.data)).convert(
                "L" if enc.color_mode == "L" else "RGB"), dtype=np.int16)
            out[f"{color_mode}/{level}"] = {
                "format": enc.format, "method": enc.compression_method,
                "jpeg_quality": enc.jpeg_quality, "color_mode": enc.color_mode,
                "bytes": enc.encoded_size_bytes, "warnings": enc.warnings,
            }
    return out


def measure_page(image_path: Path, layout, page_type: str | None, label: str) -> dict:
    with Image.open(image_path) as im:
        im.load()
        rgb_arr = np.asarray(im.convert("RGB"))
        feats = _features(rgb_arr, page_type)
        fixed = _fixed_baselines(image_path, layout, page_type)
        modes = {}
        for color_mode in ("preserve", "grayscale"):
            work, src, gray, color_mask = _prepare(im, color_mode)
            sharp_src = _sharpness(np.asarray(work.convert("L"), dtype=np.float64))
            png_data = _png_bytes(work, 150.0)
            ladder = [_jpeg_candidate(work, src, q, 0, sharp_src, color_mask) for q in LADDER]
            sub = []
            if not gray:
                sub = [_jpeg_candidate(work, src, q, 2, sharp_src, color_mask) for q in SUBSAMPLE_QUALITIES]
            modes[color_mode] = {"gray": gray, "png_bytes": len(png_data),
                                 "ladder": ladder, "subsample": sub}
    return {"label": label, "features": feats, "fixed": fixed, "modes": modes}


def _policy_choice(rec: dict, color_mode: str, level: str) -> dict:
    """从实测网格推导各策略选择，返回 {policy: {bytes, mean_diff, sharp_retention, color_diff, q, sub, fmt}}。"""
    mode = rec["modes"][color_mode]
    gray = mode["gray"]
    pt = rec["features"]["page_type"]
    fixed_key = f"{color_mode}/{level}"
    fx = rec["fixed"][fixed_key]
    png = mode["png_bytes"]
    ladder = mode["ladder"]
    sub = mode["subsample"]
    by_q = {c["q"]: c for c in ladder}

    def png_pick():
        return {"bytes": png, "mean_diff": 0.0, "sharp_retention": 1.0,
                "color_diff": 0.0, "q": None, "sub": None, "fmt": "png"}

    def jpeg_pick(c):
        return {"bytes": c["bytes"], "mean_diff": c["mean_diff"],
                "sharp_retention": c["sharp_retention"], "color_diff": c["color_diff"],
                "q": c["q"], "sub": c["subsampling"], "fmt": "jpeg"}

    # 固定基线（与编码器一致：取小 + 守卫）
    fixed_q = fx["jpeg_quality"]
    if fx["format"] == "png":
        fixed = png_pick()
    else:
        c = by_q.get(fixed_q)
        fixed = jpeg_pick(c) if c else png_pick()

    # 策略 A guard_min：守卫内最低质量（最激进），与 PNG 取小
    safe = [c for c in ladder if c["guard_pass"]]
    a = png_pick()
    if safe:
        best = min(safe, key=lambda c: c["bytes"])
        if best["bytes"] < a["bytes"]:
            a = jpeg_pick(best)

    # 策略 B text_safe：文字/灰度页保持固定；彩色照片/插图页在「守卫 + 锐度不劣于固定 + 彩色不劣化」内取最小
    # 锐度判据为相对固定（照片本身 JPEG 锐度保留率约 0.87，绝对阈值会误杀所有照片自适应）
    b = dict(fixed)
    is_text = gray or pt == "pure_text"
    if not is_text:
        cand = [c for c in (ladder + sub)
                if c["guard_pass"] and c["sharp_retention"] >= fixed["sharp_retention"] - 0.01
                and c["color_diff"] <= fixed["color_diff"] + 0.5]
        if cand:
            best = min(cand, key=lambda c: c["bytes"])
            if best["bytes"] < b["bytes"]:
                b = jpeg_pick(best)

    # 策略 C subsample_photo：仅 photo_dominated 彩色页用 4:2:0（固定质量），其余固定
    c_pol = dict(fixed)
    if pt == "photo_dominated" and not gray and sub:
        match = [s for s in sub if s["q"] == fixed_q and s["guard_pass"]]
        if match and match[0]["bytes"] < c_pol["bytes"]:
            c_pol = jpeg_pick(match[0])

    return {"fixed": fixed, "guard_min": a, "text_safe": b, "subsample_photo": c_pol}


def aggregate(records: list[dict]) -> dict:
    summary = {}
    for color_mode in ("preserve", "grayscale"):
        for level in ("balanced", "compact"):
            totals = {p: 0 for p in ("fixed", "guard_min", "text_safe", "subsample_photo")}
            guard_violations = {p: 0 for p in totals}
            color_worse = {p: 0 for p in totals}
            text_sharp_drop = {p: 0 for p in totals}
            per_type = {}
            for rec in records:
                ch = _policy_choice(rec, color_mode, level)
                pt = rec["features"]["page_type"] or "unknown"
                gray = rec["modes"][color_mode]["gray"]
                is_text = gray or pt == "pure_text"
                pt_acc = per_type.setdefault(pt, {p: 0 for p in totals})
                for p, choice in ch.items():
                    totals[p] += choice["bytes"]
                    pt_acc[p] += choice["bytes"]
                    if choice["mean_diff"] > JPEG_GUARD_MEAN_DIFF_MAX:
                        guard_violations[p] += 1
                    if choice["color_diff"] > ch["fixed"]["color_diff"] + 0.5:
                        color_worse[p] += 1
                    if is_text and choice["sharp_retention"] < ch["fixed"]["sharp_retention"] - 0.005:
                        text_sharp_drop[p] += 1
            base = totals["fixed"]
            summary[f"{color_mode}/{level}"] = {
                "total_bytes": totals,
                "saving_vs_fixed_pct": {p: round((base - totals[p]) / base * 100, 3) if base else 0.0
                                        for p in totals},
                "guard_violations": guard_violations,
                "color_worse_pages": color_worse,
                "text_sharp_drop_pages": text_sharp_drop,
                "per_page_type_bytes": per_type,
            }
    return summary


def run(out_dir="temp/m4_8_review"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    resolver = PageLayoutResolver(PDFExportConfig())
    records = []
    t0 = time.monotonic()

    # 真实 120 页
    manifest = json.load(open(REAL_TASK / "manifest.json", encoding="utf-8"))
    fir = FinalImageResolver(REAL_TASK)
    pages = sorted(manifest["pages"], key=lambda p: p["page_index"])
    print(f"真实样本 {len(pages)} 页 -> {REAL_TASK}", flush=True)
    for i, page in enumerate(pages):
        fi = fir.resolve(manifest, page)
        if fi.path is None:
            continue
        with Image.open(fi.path) as im:
            w, h = im.size
        layout = resolver.resolve(page, w, h)
        rec = measure_page(fi.path, layout, page.get("page_type"),
                           f"real_{page['page_index'] + 1:03d}")
        rec["origin"] = "real"
        rec["final_image_source"] = fi.source
        records.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  real {i + 1}/{len(pages)}  elapsed {time.monotonic() - t0:.0f}s", flush=True)

    # 合成 10 类
    synth = build_pages()
    synth_dir = out / "encoding_synth_sources"
    synth_dir.mkdir(parents=True, exist_ok=True)
    print(f"合成样本 {len(synth)} 页", flush=True)
    for name, rgb in synth.items():
        sp = synth_dir / f"{name}.png"
        Image.fromarray(rgb).save(sp, dpi=(150, 150))
        pt = ("pure_text" if name.startswith(("01", "02", "03", "04", "09"))
              else "photo_dominated" if name in ("06_color_photo", "07_gray_illustration")
              else "grayscale_illustration" if name == "08_table_lines"
              else "pure_text")
        layout = resolver.resolve({"page_index": 0, "source_kind": "image", "source_path": str(sp)},
                                  rgb.shape[1], rgb.shape[0])
        rec = measure_page(sp, layout, pt, f"synth_{name}")
        rec["origin"] = "synthetic"
        records.append(rec)

    summary = aggregate(records)
    result = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "real_task": str(REAL_TASK), "n_real": sum(r["origin"] == "real" for r in records),
              "n_synthetic": sum(r["origin"] == "synthetic" for r in records),
              "ladder": list(LADDER), "guard_mean_diff_max": JPEG_GUARD_MEAN_DIFF_MAX,
              "summary": summary, "records": records}
    (out / "encoding_research.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(f"\n总耗时 {time.monotonic() - t0:.0f}s", flush=True)
    for key, s in summary.items():
        print(f"\n[{key}] fixed={s['total_bytes']['fixed']:,}B")
        for p in ("guard_min", "text_safe", "subsample_photo"):
            print(f"  {p:16s} {s['total_bytes'][p]:,}B  "
                  f"save {s['saving_vs_fixed_pct'][p]:+.2f}%  "
                  f"guard_viol {s['guard_violations'][p]}  "
                  f"color_worse {s['color_worse_pages'][p]}  "
                  f"text_sharp_drop {s['text_sharp_drop_pages'][p]}")
    print(f"\n证据 -> {out / 'encoding_research.json'}", flush=True)
    return result


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "temp/m4_8_review")