"""M4.2 编码对比证据：10 类页面 × 3 模式（lossless/balanced/compact）。

比较维度（不止体积）：输出文件大小、嵌入 PDF 后的渲染效果、文字边缘质量、
彩色内容保留、与原始处理结果（源图）的像素差异。所有指标为启发式像素口径，非 OCR。

样本全部本地合成（复用 make_m2_samples 生成器），不联网。源图代表“已通过 M3 的图像”。
用法: python scripts/m4_2_compare.py [输出目录]
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import img2pdf
import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

from bookclean.config import PDFExportConfig
from bookclean.image_encoder import ImageEncoder
from bookclean.page_layout import PageLayoutResolver
from make_m2_samples import (  # noqa: E402
    _Canvas, _font, _paste_photo, table_page, text_page,
)

W, H = 1000, 1400
MODES = ("lossless", "balanced", "compact")


def _base(lang="en", **kw):
    rgb, m = text_page(W, H, lang=lang, **kw)
    return rgb, m


def _map_page():
    rgb, m = _base()
    c = _Canvas(W, H)
    c.rgb.paste(Image.fromarray(rgb))
    d = ImageDraw.Draw(c.rgb)
    poly = [(int(W * 0.12), int(H * 0.6)), (int(W * 0.4), int(H * 0.56)),
            (int(W * 0.52), int(H * 0.68)), (int(W * 0.42), int(H * 0.82)),
            (int(W * 0.16), int(H * 0.8))]
    d.polygon(poly, fill=(120, 180, 230), outline=(40, 90, 160))
    d.polygon([(int(W * 0.22), int(H * 0.66)), (int(W * 0.34), int(H * 0.64)),
               (int(W * 0.36), int(H * 0.74))], fill=(210, 150, 90))
    d.line((int(W * 0.14), int(H * 0.72), int(W * 0.5), int(H * 0.7)), fill=(200, 60, 60), width=3)
    f = _font("times.ttf", 22)
    d.text((int(W * 0.16), int(H * 0.84)), "Fig. 3  Regional map (color)", font=f, fill=(30, 30, 30))
    return np.ascontiguousarray(np.asarray(c.rgb, dtype=np.uint8))


def _photo_page(color=True):
    rgb, m = _base()
    cm = np.zeros((H, W), bool)
    rgb = _paste_photo(rgb, cm, (int(W * 0.12), int(H * 0.56), int(W * 0.88), int(H * 0.86)),
                       color=color, seed=11 if color else 9)
    if not color:
        # 精确灰度（R==G==B）：单通道亮度复制，演示灰度 JPEG 路径
        lum = np.round(rgb.astype(np.float64).mean(axis=2)).astype(np.uint8)
        rgb = np.stack([lum, lum, lum], axis=-1)
    return rgb


def _formula_page():
    c = _Canvas(W, H)
    title = _font("timesbd.ttf", 36)
    f = _font("times.ttf", 30)
    small = _font("times.ttf", 24)
    c.text((90, 90), "Mathematical Formulas", title)
    eqs = ["E = mc^2", "a^2 + b^2 = c^2", "lim_{x->0} sin(x)/x = 1",
           "f'(x) = lim_{h->0} [f(x+h) - f(x)] / h"]
    y = 200
    for e in eqs:
        c.text((90, y), e, f)
        y += 70
    # 手绘分数（细横线 + 小字），考察细线与小glyph边缘
    y += 30
    c.text((90, y), "sum_{i=1}^{n} i  =", f)
    c.text((360, y - 18), "n(n+1)", small)
    c.line((360, y + 12, 470, y + 12), (25, 25, 25), 2)
    c.text((390, y + 18), "2", small)
    # 根号
    y += 110
    c.text((90, y), "sqrt(2) = 1.41421356...", f)
    c.line((300, y + 30, 310, y + 40), (25, 25, 25), 2)
    c.line((310, y + 40, 322, y - 6), (25, 25, 25), 2)
    c.line((322, y - 6, 470, y - 6), (25, 25, 25), 2)
    return np.ascontiguousarray(np.asarray(c.rgb, dtype=np.uint8))


def _annotation_page():
    rgb, m = _base()
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    d.line((int(W * 0.1), int(H * 0.34), int(W * 0.62), int(H * 0.34)), fill=(200, 40, 40), width=3)
    f = _font("simsun.ttc", 26)
    d.text((int(W * 0.64), int(H * 0.3)), "此处存疑", font=f, fill=(40, 80, 200))
    cx, cy, r = int(W * 0.78), int(H * 0.86), 55
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(200, 30, 30), width=5)
    fs = _font("simsun.ttc", 22)
    d.text((cx - 44, cy - 12), "审阅之章", font=fs, fill=(200, 30, 30))
    return np.ascontiguousarray(np.asarray(im, dtype=np.uint8))


def build_pages():
    return {
        "01_pure_text_en": _base(title=False)[0],
        "02_chinese_song": _base(lang="cn")[0],
        "03_english_serif": _base(lang="en")[0],
        "04_small_footnote": _base(footnotes=True)[0],
        "05_color_map": _map_page(),
        "06_color_photo": _photo_page(color=True),
        "07_gray_illustration": _photo_page(color=False),
        "08_table_lines": table_page(W, H)[0],
        "09_math_formula": _formula_page(),
        "10_red_annotation": _annotation_page(),
    }


def _sharpness(arr):
    """文字边缘锐度代理：灰度 Laplacian 绝对值均值（越高越锐利）。启发式，非 OCR。"""
    g = arr.astype(np.float64).mean(axis=2)
    lap = (g[2:, 1:-1] + g[:-2, 1:-1] + g[1:-1, 2:] + g[1:-1, :-2] - 4 * g[1:-1, 1:-1])
    return float(np.abs(lap).mean())


def _embed_and_render(enc, layout, src_dpi):
    pw, ph = layout.page_width_pt, layout.page_height_pt
    iw, ih = layout.image_display_width_pt, layout.image_display_height_pt
    lf = lambda w, h, dpi: (pw, ph, iw, ih)
    pdf_bytes = img2pdf.convert(enc.data, layout_fun=lf)
    doc = pdfium.PdfDocument(pdf_bytes)
    pil = doc[0].render(scale=src_dpi / 72.0).to_pil().convert("RGB")
    doc.close()
    return pil, len(pdf_bytes)


def run(out_dir="temp/m4_2_compare"):
    out = Path(out_dir)
    (out / "sources").mkdir(parents=True, exist_ok=True)
    pages = build_pages()
    resolver = PageLayoutResolver(PDFExportConfig())
    records = []
    decoded = {}  # name -> {mode: PIL image}

    for name, rgb in pages.items():
        src_path = out / "sources" / f"{name}.png"
        Image.fromarray(rgb).save(src_path, dpi=(150, 150))
        src = np.asarray(Image.open(src_path).convert("RGB")).astype(np.int16)
        layout = resolver.resolve(
            {"page_index": 1, "source_kind": "image", "source_path": str(src_path)},
            W, H,
        )
        decoded[name] = {}
        row = {"name": name, "source_png_bytes": src_path.stat().st_size,
               "source_sharpness": round(_sharpness(src), 3), "modes": {}}
        for mode in MODES:
            enc = ImageEncoder(PDFExportConfig(compression=mode)).encode(src_path, layout)
            dec = np.asarray(Image.open(io.BytesIO(enc.data)).convert("RGB")).astype(np.int16)
            diff = np.abs(dec - src)
            render, pdf_bytes = _embed_and_render(enc, layout, 150)
            decoded[name][mode] = render
            rarr = np.asarray(render).astype(np.int16)
            row["modes"][mode] = {
                "format": enc.format,
                "method": enc.compression_method,
                "jpeg_quality": enc.jpeg_quality,
                "color_mode": enc.color_mode,
                "encoded_bytes": enc.encoded_size_bytes,
                "single_page_pdf_bytes": pdf_bytes,
                "downsampled": enc.downsampled,
                "mean_abs_diff": round(float(diff.mean()), 4),
                "max_abs_diff": int(diff.max()),
                "decoded_sharpness": round(_sharpness(dec), 3),
                "render_mean_abs_diff": round(float(np.abs(rarr - src).mean()), 4),
                "warnings": enc.warnings,
            }
        records.append(row)
        print(f"{name}: " + "  ".join(
            f"{m}={row['modes'][m]['encoded_bytes']}B({row['modes'][m]['method']})" for m in MODES))

    (out / "metrics.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    _contact_sheet(out, pages, decoded)
    _zoom_strips(out, pages, decoded)
    _write_table(out, records)
    print(f"\n证据输出 -> {out}")
    return records


def _thumb(im, tw=220):
    th = int(im.height * tw / im.width)
    return im.resize((tw, th), Image.LANCZOS)


def _contact_sheet(out, pages, decoded):
    tw = 220
    th = int(H * tw / W)
    pad, label_h = 12, 26
    cols = ["source"] + list(MODES)
    cw = tw * len(cols) + pad * (len(cols) + 1)
    ch = (th + label_h + pad) * len(pages) + pad + 30
    sheet = Image.new("RGB", (cw, ch), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    f = _font("times.ttf", 18)
    for j, c in enumerate(cols):
        d.text((pad + j * (tw + pad) + tw // 2 - 30, 6), c, font=f, fill=(0, 0, 0))
    for i, name in enumerate(pages):
        y = 30 + pad + i * (th + label_h + pad)
        d.text((pad, y + th + 4), name, font=f, fill=(0, 0, 0))
        src_thumb = _thumb(Image.fromarray(pages[name]), tw)
        sheet.paste(src_thumb, (pad, y))
        for j, mode in enumerate(MODES):
            sheet.paste(_thumb(decoded[name][mode], tw), (pad + (j + 1) * (tw + pad), y))
    sheet.save(out / "overview_contact_sheet.png")


def _zoom_strips(out, pages, decoded):
    # 文字边缘 / 彩色 / 细线 局部放大对比（源 + 3 模式）
    crops = {
        "03_english_serif": (90, 200, 620, 360),    # 正文文字边缘
        "02_chinese_song": (90, 200, 620, 360),      # 宋体文字边缘
        "09_math_formula": (80, 430, 560, 620),      # 分数/根号细线
        "05_color_map": (100, 780, 560, 1180),       # 彩色地图
        "06_color_photo": (120, 780, 880, 1200),     # 彩色照片
        "10_red_annotation": (600, 1140, 900, 1300),  # 红色印章
    }
    f = _font("times.ttf", 20)
    for name, box in crops.items():
        src = Image.fromarray(pages[name]).crop(box)
        scale = 2
        src = src.resize((src.width * scale, src.height * scale), Image.NEAREST)
        zw, zh = src.size
        pad, label_h = 10, 28
        cols = ["source"] + list(MODES)
        canvas = Image.new("RGB", (zw * len(cols) + pad * (len(cols) + 1), zh + label_h + pad * 2 + 24),
                           (245, 245, 245))
        d = ImageDraw.Draw(canvas)
        d.text((pad, 4), f"{name}  zoom x{scale}  box={box}", font=f, fill=(0, 0, 0))
        for j, c in enumerate(cols):
            x = pad + j * (zw + pad)
            d.text((x, 24 + 4), c, font=f, fill=(0, 0, 0))
            im = src if c == "source" else decoded[name][c].crop(box).resize((zw, zh), Image.NEAREST)
            canvas.paste(im, (x, 24 + label_h))
        canvas.save(out / f"zoom_{name}.jpg", quality=95)


def _write_table(out, records):
    lines = ["# M4.2 编码对比（10 类页面 × 3 模式）", "",
             "指标为启发式像素口径，非 OCR / 文本完整性证明。源图代表已通过 M3 的图像。", "",
             "| 页面 | 源PNG | Lossless | Balanced | Compact | B方法 | C方法 | B均差 | C均差 | B锐度 | C锐度 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        m = r["modes"]
        lines.append(
            f"| {r['name']} | {r['source_png_bytes']} | {m['lossless']['encoded_bytes']} | "
            f"{m['balanced']['encoded_bytes']} | {m['compact']['encoded_bytes']} | "
            f"{m['balanced']['method']} | {m['compact']['method']} | "
            f"{m['balanced']['mean_abs_diff']} | {m['compact']['mean_abs_diff']} | "
            f"{m['balanced']['decoded_sharpness']} | {m['compact']['decoded_sharpness']} |")
    lines += ["", "源锐度（Laplacian 均值）见 metrics.json；渲染差异 render_mean_abs_diff 见 metrics.json。"]
    (out / "comparison_table.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "temp/m4_2_compare")