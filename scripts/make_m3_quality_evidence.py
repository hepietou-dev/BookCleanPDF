"""M3.3 分区质量检查可视化证据：原图 | 候选 对比联系表 + 五维分区指标 + 可解释风险评分。

对带真值掩膜的合成样本运行 BackgroundCleaner 生成候选，再用 QualityChecker 做五维分区评估；
另含损伤场景（抹文字/彩色褪色/抹平插图/照片页）验证严重文字损伤不可抵消、照片页不误回退。
运行: python scripts/make_m3_quality_evidence.py [输出目录]，默认 temp/m3_quality_evidence/。本地合成，不联网。
指标是图像风险提示，不等于 OCR 准确率/文字完整性证明。
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
from bookclean.content_analyzer import ContentAnalyzer  # noqa: E402
from bookclean.models import PageData  # noqa: E402
from bookclean.page_analyzer import PageAnalyzer  # noqa: E402
from bookclean.quality_checker import QualityChecker  # noqa: E402


def _font(size):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)
    except OSError:
        return ImageFont.load_default(size=size)


def _pd(rgb):
    return PageData(0, rgb, "synthetic", "image", rgb.shape[1], rgb.shape[0], 150.0, "RGB")


def _gray(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)


def _panel(rgb, title, lines, width=440):
    h, w = rgb.shape[:2]
    im = Image.fromarray(rgb).resize((width, int(h * width / w)))
    bar = 24 + 18 * len(lines)
    panel = Image.new("RGB", (width, im.height + bar), (250, 250, 250))
    panel.paste(im, (0, bar))
    d = ImageDraw.Draw(panel)
    d.text((8, 4), title, font=_font(18), fill=(20, 20, 20))
    for i, ln in enumerate(lines):
        d.text((8, 26 + 18 * i), ln, font=_font(13), fill=(70, 70, 70))
    return panel


def _summary(r):
    dim = r["dimensions"]
    pb = dim["paper_background"]
    pstd = f"{pb['before']['std']:.1f}->{pb['after']['std']:.1f}" if pb["before"] and pb["after"] else "n/a"
    tcr = dim["text_preservation"]["text_core_retention"]
    chr_ = dim["color_preservation"]["chroma_retention"]
    ill = dim["illustration_preservation"]["gray_std_retention"]
    return [
        f"type={r['page_type']}  decision={r['decision']}",
        f"risk={r['risk_score']:.1f}/100  conf={r['region_mask_confidence']:.2f}",
        f"paperStd {pstd}  textCore={tcr if tcr is not None else 'n/a'}",
        f"chroma={chr_ if chr_ is not None else 'n/a'}  illStd={ill if ill is not None else 'n/a'}",
    ]


def main(out_dir="temp/m3_quality_evidence"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cleaner, analyzer, content, checker = BackgroundCleaner(), PageAnalyzer(), ContentAnalyzer(), QualityChecker()
    cfg = TaskConfig(background_strength=0.8, yellow_removal_strength=0.8, denoise_strength=0.3)
    samples = build_region_samples()
    cases = ["yellow_black_text", "yellow_color_map", "yellow_color_photo", "yellow_red_seal",
             "color_annotation", "grayscale_photo", "color_chart", "large_color_illustration"]
    reports, panels = {}, []

    def record(name, rgb, cand, reg, tag):
        r = checker.evaluate(rgb, cand, analyzer.analyze(_pd(rgb)), cfg, regions=reg)
        reports[name] = r
        d = out / name
        d.mkdir(exist_ok=True)
        Image.fromarray(rgb).save(d / "original.png")
        Image.fromarray(cand).save(d / "candidate.png")
        panels.append((_panel(rgb, f"{name} ({tag})", ["original"]),
                       _panel(cand, f"{name} ({tag})", _summary(r))))
        print(f"{name:26s} type={r['page_type']:22s} dec={r['decision']:20s} risk={r['risk_score']:5.1f}")
        return r

    for name in cases:
        rgb = samples[name]["image"]
        reg = content.analyze(rgb)
        cand = cleaner.process(rgb, analyzer.analyze(_pd(rgb)), cfg, regions=reg).image
        record(name, rgb, cand, reg, "cleaned")

    # 损伤场景 1：干净白纸抹掉文字（背景良好，文字核心严重损伤 -> 强制回退）
    rgb = samples["pure_bw_text"]["image"]
    reg = content.analyze(rgb)
    cand = rgb.copy()
    cand[reg.text_mask > 0] = 255
    record("damage_text_erase", rgb, cand, reg, "text wiped")

    # 损伤场景 2：彩色地图褪色（color_mask 强制灰度）
    rgb = samples["yellow_color_map"]["image"]
    reg = content.analyze(rgb)
    cand = rgb.copy()
    cm = reg.color_mask > 0
    cand[cm] = _gray(rgb)[cm][:, None]
    record("damage_color_fade", rgb, cand, reg, "color faded")

    # 损伤场景 3：抹平灰度插图层次
    rgb = samples["grayscale_photo"]["image"]
    reg = content.analyze(rgb)
    cand = rgb.copy()
    ill = reg.illustration_mask > 0
    cand[ill] = int(_gray(rgb)[ill].mean())
    record("damage_ill_flatten", rgb, cand, reg, "illustration flat")

    # 照片页：对自身不应误判空白回退
    photo = _photo_array(900, 1200, color=True, seed=3)
    reg = content.analyze(photo)
    record("photo_identical", photo, photo.copy(), reg, "photo self")

    pw = panels[0][0].width
    ph = max(p.height for pair in panels for p in pair)
    gap = 12
    sheet = Image.new("RGB", (pw * 2 + gap * 3, (ph + gap) * len(panels) + gap), (255, 255, 255))
    y = gap
    for orig, cand in panels:
        sheet.paste(orig, (gap, y))
        sheet.paste(cand, (pw + gap * 2, y))
        y += ph + gap
    sheet.save(out / "quality_contact_sheet.jpg", quality=88)
    (out / "quality_report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n联系表 -> {out / 'quality_contact_sheet.jpg'}  ({sheet.width}x{sheet.height})")
    print(f"完整分区报告 -> {out / 'quality_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "temp/m3_quality_evidence"))