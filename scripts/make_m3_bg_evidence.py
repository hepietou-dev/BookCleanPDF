"""M3.2 区域化背景清理可视化证据：原图 | 清理后 对比联系表 + 量化指标。

对带真值掩膜的合成样本运行 BackgroundCleaner，展示纸张背景改善而彩色/灰度/特殊内容保真。
运行: python scripts/make_m3_bg_evidence.py [输出目录]，默认 temp/m3_bg_evidence/。本地合成，不联网。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_m2_samples import build_region_samples, _photo_array  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bookclean.background_cleaner import BackgroundCleaner  # noqa: E402
from bookclean.config import TaskConfig  # noqa: E402
from bookclean.models import PageData  # noqa: E402
from bookclean.page_analyzer import PageAnalyzer  # noqa: E402


def _font(size):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)
    except OSError:
        return ImageFont.load_default(size=size)


def _pd(rgb):
    return PageData(0, rgb, "synthetic", "image", rgb.shape[1], rgb.shape[0], 150.0, "RGB")


def _chroma_ret(before, after, mask):
    if not mask.any():
        return 1.0
    cb = before.astype(int).max(2) - before.astype(int).min(2)
    ca = after.astype(int).max(2) - after.astype(int).min(2)
    return float(ca[mask].mean() / max(1e-6, cb[mask].mean()))


def _panel(rgb, title, sub, width=440):
    h, w = rgb.shape[:2]
    im = Image.fromarray(rgb).resize((width, int(h * width / w)))
    bar = 54
    panel = Image.new("RGB", (width, im.height + bar), (250, 250, 250))
    panel.paste(im, (0, bar))
    d = ImageDraw.Draw(panel)
    d.text((8, 5), title, font=_font(19), fill=(20, 20, 20))
    d.text((8, 30), sub, font=_font(14), fill=(90, 90, 90))
    return panel


def main(out_dir="temp/m3_bg_evidence"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cleaner, analyzer = BackgroundCleaner(), PageAnalyzer()
    cfg = TaskConfig(background_strength=0.8, yellow_removal_strength=0.8, denoise_strength=0.3)
    samples = build_region_samples()
    cases = ["yellow_black_text", "yellow_color_map", "yellow_color_photo", "yellow_red_seal",
             "color_annotation", "grayscale_photo", "color_chart", "large_color_illustration"]
    metrics, panels = {}, []
    for name in cases:
        s = samples[name]
        rgb = s["image"]
        res = cleaner.process(rgb, analyzer.analyze(_pd(rgb)), cfg)
        b, a = res.params["bg_before"], res.params["bg_after"]
        cr = _chroma_ret(rgb, res.image, s["regions"]["color"])
        sr = _chroma_ret(rgb, res.image, s["regions"]["special"])
        metrics[name] = {"bg_before": b, "bg_after": a, "color_chroma_retention": round(cr, 4),
                         "special_chroma_retention": round(sr, 4),
                         "normalize_executed": next(x["executed"] for x in res.steps if x["op"] == "background_normalize")}
        d = out / name
        d.mkdir(exist_ok=True)
        Image.fromarray(rgb).save(d / "before.png")
        Image.fromarray(res.image).save(d / "after.png")
        sub = f"bg {b['mean']:.0f}/{b['std']:.1f} -> {a['mean']:.0f}/{a['std']:.1f}  chroma {cr:.2f}"
        panels.append((_panel(rgb, name, "original"), _panel(res.image, name, sub)))
        print(f"{name:26s} bg {b['mean']:.0f}/{b['std']:.1f} -> {a['mean']:.0f}/{a['std']:.1f}  colorChroma={cr:.3f}")

    # 照片主导页：应跳过白化，保持原样
    photo = _photo_array(900, 1200, color=True, seed=3)
    res = cleaner.process(photo, analyzer.analyze(_pd(photo)), cfg)
    norm = next(x for x in res.steps if x["op"] == "background_normalize")
    metrics["photo_dominated"] = {"normalize_executed": norm["executed"], "reason": norm.get("reason"),
                                  "mean_abs_change": round(float(np.abs(res.image.astype(int) - photo.astype(int)).mean()), 3)}
    panels.append((_panel(photo, "photo_dominated", "original"),
                   _panel(res.image, "photo_dominated", f"skipped: {norm.get('reason')}")))
    print(f"photo_dominated            skipped={not norm['executed']} reason={norm.get('reason')}")

    pw = panels[0][0].width
    ph = max(p.height for pair in panels for p in pair)
    gap = 12
    sheet = Image.new("RGB", (pw * 2 + gap * 3, (ph + gap) * len(panels) + gap), (255, 255, 255))
    y = gap
    for orig, aft in panels:
        sheet.paste(orig, (gap, y))
        sheet.paste(aft, (pw + gap * 2, y))
        y += ph + gap
    sheet.save(out / "bg_contact_sheet.jpg", quality=88)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n联系表 -> {out / 'bg_contact_sheet.jpg'}  ({sheet.width}x{sheet.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "temp/m3_bg_evidence"))