"""M3.1 区域掩膜可视化与量化证据生成器。

对 10 类带真值掩膜的合成样本运行 ContentAnalyzer，输出：
  - 每样本每区域的 IoU / 召回 / 精度 + 文字核心保护率（metrics.json）
  - 原图 + 六区域彩色叠加联系表（region_contact_sheet.jpg）
  - 每样本单独的原图/叠加对比图（便于逐页核验）

运行: python scripts/make_m3_region_evidence.py [输出目录]
默认输出 temp/m3_region_evidence/。全部本地合成，不联网。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_m2_samples import build_region_samples  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bookclean.content_analyzer import ContentAnalyzer  # noqa: E402

REGIONS = ["paper", "text", "color", "illustration", "special", "uncertain"]
OVERLAY = {
    "text": (40, 90, 255),
    "color": (255, 0, 255),
    "illustration": (0, 200, 0),
    "special": (255, 40, 40),
    "uncertain": (255, 170, 0),
}


def _font(size):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)
    except OSError:
        return ImageFont.load_default(size=size)


def _pred(m, r):
    return getattr(m, r + "_mask") > 0


def _metrics(m, s):
    out = {}
    for r in REGIONS:
        p, t = _pred(m, r), s["regions"].get(r)
        if t is None:
            out[r] = {"recall": None, "precision": None, "iou": None}
            continue
        rec = float((p & t).sum() / t.sum()) if t.sum() else 1.0
        prec = float((p & t).sum() / p.sum()) if p.sum() else 1.0
        u = (p | t).sum()
        iou = float((p & t).sum() / u) if u else 1.0
        out[r] = {"recall": round(rec, 4), "precision": round(prec, 4), "iou": round(iou, 4)}
    protected = (_pred(m, "text") | _pred(m, "color") | _pred(m, "illustration")
                 | _pred(m, "special") | _pred(m, "uncertain"))
    core = cv2.erode(s["regions"]["text"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out["text_core_protection"] = round(float((protected & core).sum() / core.sum()), 4) if core.sum() else 1.0
    return out


def _overlay(rgb, m):
    out = rgb.astype(np.float32).copy()
    for r, col in OVERLAY.items():
        mask = _pred(m, r)
        if mask.any():
            out[mask] = out[mask] * 0.45 + np.asarray(col, np.float32) * 0.55
    return out.clip(0, 255).astype(np.uint8)


def _label_panel(rgb, title, sub, width=460):
    h, w = rgb.shape[:2]
    scale = width / w
    im = Image.fromarray(rgb).resize((width, int(h * scale)))
    bar = 56
    panel = Image.new("RGB", (width, im.height + bar), (250, 250, 250))
    panel.paste(im, (0, bar))
    d = ImageDraw.Draw(panel)
    d.text((8, 6), title, font=_font(20), fill=(20, 20, 20))
    d.text((8, 32), sub, font=_font(15), fill=(90, 90, 90))
    return panel


def main(out_dir="temp/m3_region_evidence"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = build_region_samples()
    ca = ContentAnalyzer()
    all_metrics = {}
    panels = []
    for name, s in samples.items():
        m = ca.analyze(s["image"])
        met = _metrics(m, s)
        all_metrics[name] = {"confidence": m.confidence, "flags": m.metadata["flags"],
                             "area_ratios": m.metadata["area_ratios"], "regions": met}
        ov = _overlay(s["image"], m)
        d = out / name
        d.mkdir(exist_ok=True)
        Image.fromarray(s["image"]).save(d / "original.png")
        Image.fromarray(ov).save(d / "overlay.png")
        sub = f"conf={m.confidence:.2f}  textCore={met['text_core_protection']:.3f}"
        panels.append((_label_panel(s["image"], name, "original"),
                       _label_panel(ov, name, sub)))
        print(f"{name:28s} conf={m.confidence:.3f} textCore={met['text_core_protection']:.4f} "
              f"paperR={met['paper']['recall']:.3f} colorR={met['color']['recall']}")

    # 联系表：每行一个样本（原图 | 叠加），统一宽度
    pw = panels[0][0].width
    ph = max(p.height for pair in panels for p in pair)
    gap = 12
    sheet = Image.new("RGB", (pw * 2 + gap * 3, (ph + gap) * len(panels) + gap), (255, 255, 255))
    y = gap
    for orig, ov in panels:
        sheet.paste(orig, (gap, y))
        sheet.paste(ov, (pw + gap * 2, y))
        y += ph + gap
    sheet.save(out / "region_contact_sheet.jpg", quality=88)
    (out / "metrics.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n联系表 -> {out / 'region_contact_sheet.jpg'}  ({sheet.width}x{sheet.height})")
    print(f"指标 -> {out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "temp/m3_region_evidence"))