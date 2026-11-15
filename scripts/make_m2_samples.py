"""M2 离线合成样本生成器。

全部样本由 Pillow / OpenCV 本地生成，不联网、不依赖真实书籍数据。
每个样本同时生成真实前景掩膜（与文字逐像素对齐），用于内容保真度测试。

运行: python scripts/make_m2_samples.py [输出目录]
默认输出到 temp/m2_samples/，每个样本一个子目录（image.png / fg_mask.png / meta.json）。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path("C:/Windows/Fonts")

EN_PARAS = [
    "The war propelled the Orang Asli into the modern world with sudden violence. "
    "In Perak, in particular, they were under pressure from the Chinese agriculturalists "
    "who were moving into the forests to escape the Japanese.",
    "They traded with them, and this included the supply of food, particularly meat from "
    "hunting, to guerrilla camps. But towards the end of November an event occurred that "
    "was to have long-lasting repercussions for the Orang Asli.",
    "On the British retreat, Noone, trusting his local knowledge and connections, had "
    "taken to the forest with the Temiar communities. He had been in isolation from the "
    "other Europeans, and from the war, but he had been in contact with British officers.",
    "He applied the principles of the British school of structural-functionalist "
    "anthropology to the understanding of Orang Asli social organization and leadership. "
    "He imparted this to the Malayan Communist Party and they applied it to their revolution.",
]

CN_PARAS = [
    "战争以突如其来的暴力把奥朗阿斯利人推进了现代世界。特别是在霹雳州，他们受到为躲避"
    "日本人而迁入森林的华人垦殖者的压力。",
    "他与马来亚共产党人交往，并把英国结构功能主义人类学派的原理传授给他们。共产党人则"
    "把这些原理应用于自己的革命实践。",
    "英国撤退时，努恩凭借自己对当地的了解与人脉，与特米亚尔社群一起进入森林。他与其他"
    "欧洲人失去了联系，但仍与一些英国军官保持接触。",
]

FOOTNOTES_EN = [
    "143. Pat Noone, 'Report on the Orang Asli', Federated Malay States Archives, 1938, p. 12.",
    "144. See C. A. Bayly & T. N. Harper, 'Forgotten Armies', ch. 7, nn. 44-47, pp. 311-313.",
    "145. Interview with Temiar headman, Kuala Betis, 12 December 1945, WO 203/4471.",
]


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default(size=size)


class _Canvas:
    """同步绘制 RGB 图与前景掩膜，保证掩膜与文字逐像素对齐。"""

    def __init__(self, w: int, h: int, bg=(255, 255, 255)):
        self.rgb = Image.new("RGB", (w, h), bg)
        self.mask = Image.new("L", (w, h), 0)
        self.d = ImageDraw.Draw(self.rgb)
        self.dm = ImageDraw.Draw(self.mask)

    def text(self, xy, s, font, fill=(25, 25, 25)):
        self.d.text(xy, s, font=font, fill=fill)
        self.dm.text(xy, s, font=font, fill=255)

    def line(self, box, fill, width):
        self.d.line(box, fill=fill, width=width)
        self.dm.line(box, fill=255, width=width)

    def rectangle(self, box, fill):
        self.d.rectangle(box, fill=fill)
        self.dm.rectangle(box, fill=255)

    def arrays(self):
        rgb = np.ascontiguousarray(np.asarray(self.rgb, dtype=np.uint8))
        mask = np.asarray(self.mask) > 0
        return rgb, mask


def _wrap(draw, text, font, max_w, cjk=False):
    units = list(text) if cjk else text.split(" ")
    sep = "" if cjk else " "
    lines, cur = [], ""
    for u in units:
        trial = cur + sep + u if cur else u
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = u
    if cur:
        lines.append(cur)
    return lines


def text_page(
    width=1000,
    height=1400,
    bg=(255, 255, 255),
    margin=90,
    lang="en",
    footnotes=False,
    page_number="267",
    title=True,
    body_size=30,
):
    """生成标准文字页，返回 (rgb, fg_mask)。"""
    c = _Canvas(width, height, bg)
    cjk = lang == "cn"
    body = _font("simsun.ttc" if cjk else "times.ttf", body_size)
    small = _font("times.ttf", max(14, body_size - 12))
    title_f = _font("timesbd.ttf" if not cjk else "msyhbd.ttc", body_size + 6)
    paras = CN_PARAS if cjk else EN_PARAS
    max_w = width - 2 * margin
    y = margin
    if title:
        t = "1942: THE ABYSS AND THE WAY BACK" if not cjk else "一九四二：深渊与归途"
        tw = c.d.textlength(t, font=title_f)
        c.text(((width - tw) / 2, y), t, title_f)
        y += int(body_size * 2.2)
    lh = int(body_size * 1.5)
    for para in paras:
        for ln in _wrap(c.d, para, body, max_w, cjk):
            c.text((margin, y), ln, body)
            y += lh
        y += int(lh * 0.5)
    if footnotes:
        fy = height - margin - 3 * 26 - 30
        c.line((margin, fy - 12, margin + max_w // 3, fy - 12), (60, 60, 60), 1)
        for i, fn in enumerate(FOOTNOTES_EN):
            c.text((margin, fy + i * 26), fn, small)
    if page_number:
        nw = c.d.textlength(page_number, font=body)
        c.text(((width - nw) / 2, height - margin + 10), page_number, body)
    return c.arrays()


def rotate(rgb, mask, angle):
    """按已知角度旋转（正值=逆时针），扩展画布，白底；掩膜同步旋转。"""
    im = Image.fromarray(rgb).rotate(angle, expand=True, fillcolor=(255, 255, 255), resample=Image.BICUBIC)
    m = Image.fromarray(mask.astype(np.uint8) * 255).rotate(angle, expand=True, fillcolor=0, resample=Image.NEAREST)
    return np.ascontiguousarray(np.asarray(im, dtype=np.uint8)), np.asarray(m) > 127


def yellow(rgb, mask, paper=(238, 222, 165), noise=2.0, gradient=0.0, seed=3):
    """纸张泛黄：整页乘以纸张色（文字保持深色），可叠加渐变与噪声。"""
    out = rgb.astype(np.float32) * (np.asarray(paper, np.float32) / 255.0)
    if gradient:
        h, w = out.shape[:2]
        g = 1.0 - gradient * np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
        out = out * g
    if noise:
        rng = np.random.default_rng(seed)
        out = out + rng.normal(0, noise, out.shape).astype(np.float32)
    return np.ascontiguousarray(out.clip(0, 255).astype(np.uint8)), mask


def shadow(rgb, mask, kind="left", strength=0.35):
    """局部阴影：left=左侧发暗，spine=中间书脊阴影，corner=角落径向阴影。"""
    h, w = rgb.shape[:2]
    X = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    Y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    if kind == "left":
        field = 1.0 - strength * (1.0 - X)
    elif kind == "spine":
        field = 1.0 - strength * np.exp(-((X - 0.5) ** 2) / 0.015)
    else:
        field = 1.0 - strength * np.exp(-(((X - 0.1) ** 2) + ((Y - 0.1) ** 2)) / 0.05)
    out = rgb.astype(np.float32) * field[..., None]
    return np.ascontiguousarray(out.clip(0, 255).astype(np.uint8)), mask


def black_border(rgb, mask, border=70, desk=(15, 15, 15)):
    """扫描仪黑边：页面放到深色桌面上。"""
    h, w = rgb.shape[:2]
    canvas = np.full((h + 2 * border, w + 2 * border, 3), desk, np.uint8)
    canvas[border:border + h, border:border + w] = rgb
    m = np.zeros((h + 2 * border, w + 2 * border), bool)
    m[border:border + h, border:border + w] = mask
    return np.ascontiguousarray(canvas), m


def perspective(rgb, mask, desk=(70, 70, 70)):
    """手机透视：把页面 warp 成不规则四边形放到深色桌面上。"""
    h, w = rgb.shape[:2]
    W, H = w + 200, h + 200
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    dst = np.array(
        [[230, 120], [W - 190, 80], [W - 60, H - 80], [100, H - 140]], np.float32
    )
    mat = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(rgb, mat, (W, H), flags=cv2.INTER_LINEAR, borderValue=desk)
    m = cv2.warpPerspective(mask.astype(np.uint8) * 255, mat, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    return np.ascontiguousarray(warped), m > 127, mat


def color_content(rgb, mask):
    """在文字页上添加彩色图表、地图状色块与红色印章；返回 (rgb, fg_mask, color_mask)。"""
    im = Image.fromarray(rgb)
    cm = Image.new("L", im.size, 0)
    d = ImageDraw.Draw(im)
    dc = ImageDraw.Draw(cm)
    w, h = im.size
    # 彩色柱状图
    x0, y0, bw, bh = int(w * 0.55), int(h * 0.62), int(w * 0.34), int(h * 0.2)
    colors = [(210, 60, 50), (50, 120, 210), (60, 170, 90), (230, 180, 40)]
    for i, col in enumerate(colors):
        bx = x0 + 14 + i * (bw // 4)
        bh_i = int(bh * (0.4 + 0.15 * i))
        box = (bx, y0 + bh - bh_i, bx + bw // 4 - 18, y0 + bh)
        d.rectangle(box, fill=col)
        dc.rectangle(box, fill=255)
    d.line((x0, y0, x0, y0 + bh, x0 + bw, y0 + bh), fill=(40, 40, 40), width=2)
    # 地图状多边形
    poly = [(int(w * 0.12), int(h * 0.66)), (int(w * 0.3), int(h * 0.63)),
            (int(w * 0.42), int(h * 0.72)), (int(w * 0.33), int(h * 0.82)),
            (int(w * 0.15), int(h * 0.8))]
    d.polygon(poly, fill=(120, 180, 230), outline=(40, 90, 160))
    dc.polygon(poly, fill=255)
    # 红色印章
    cx, cy, r = int(w * 0.8), int(h * 0.88), 52
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(200, 30, 30), width=5)
    dc.ellipse((cx - r, cy - r, cx + r, cy + r), outline=255, width=5)
    f = _font("simsun.ttc", 22)
    d.text((cx - 44, cy - 12), "审阅之章", font=f, fill=(200, 30, 30))
    dc.text((cx - 44, cy - 12), "审阅之章", font=f, fill=255)
    out = np.ascontiguousarray(np.asarray(im, dtype=np.uint8))
    return out, mask, np.asarray(cm) > 0


def specks(rgb, mask, n=140, seed=7, max_r=2):
    """在背景区域撒孤立小污点（避开文字），返回 (rgb, fg_mask, speck_mask)。"""
    rng = np.random.default_rng(seed)
    h, w = rgb.shape[:2]
    out = rgb.copy()
    sm = np.zeros((h, w), np.uint8)
    placed = 0
    while placed < n:
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(1, max_r + 1))
        y0, y1 = max(0, y - r - 3), min(h, y + r + 4)
        x0, x1 = max(0, x - r - 3), min(w, x + r + 4)
        if mask[y0:y1, x0:x1].any():
            continue
        cv2.circle(out, (x, y), r, (40, 40, 40), -1)
        cv2.circle(sm, (x, y), r, 1, -1)
        placed += 1
    return np.ascontiguousarray(out), mask, sm > 0


def table_page(width=1000, height=1400):
    """含细线表格与公式样文本的页面。"""
    rgb, mask = text_page(width, height, title=True)
    c = _Canvas(width, height)
    c.rgb.paste(Image.fromarray(rgb))
    c.mask.paste(Image.fromarray(mask.astype(np.uint8) * 255))
    f = _font("times.ttf", 24)
    x0, y0, tw, th = 90, int(height * 0.58), width - 180, 220
    rows, cols = 4, 4
    for i in range(rows + 1):
        c.line((x0, y0 + i * th // rows, x0 + tw, y0 + i * th // rows), (30, 30, 30), 1)
    for j in range(cols + 1):
        c.line((x0 + j * tw // cols, y0, x0 + j * tw // cols, y0 + th), (30, 30, 30), 1)
    for i in range(rows):
        for j in range(cols):
            c.text((x0 + j * tw // cols + 12, y0 + i * th // rows + 12),
                   f"x{i}{j} = {i * cols + j}", f)
    c.text((90, y0 + th + 40), "E = mc^2;  a^2 + b^2 = c^2;  sum_{i=1}^{n} i = n(n+1)/2", f)
    return c.arrays()


def _photo_array(w: int, h: int, color: bool = True, seed: int = 5) -> np.ndarray:
    """合成照片状纹理块：平滑渐变 + 形状 + 细噪声，宽动态范围。"""
    rng = np.random.default_rng(seed)
    Y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    X = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    if color:
        r = 90 + 120 * (0.5 + 0.5 * np.sin(2.0 * np.pi * X)) * (1.0 - 0.4 * Y)
        g = 110 + 90 * (0.5 + 0.5 * np.cos(1.5 * np.pi * Y)) * (0.7 + 0.3 * X)
        b = 150 + 80 * (0.5 + 0.5 * np.sin(2.0 * np.pi * (X + Y)))
        arr = np.stack([r, g, b], axis=2)
        cv2.circle(arr, (int(w * 0.7), int(h * 0.25)), int(min(w, h) * 0.12), (240, 220, 90), -1)
        cv2.ellipse(arr, (int(w * 0.35), int(h * 0.8)), (int(w * 0.4), int(h * 0.25)), 0, 0, 360, (60, 130, 70), -1)
    else:
        base = 70 + 130 * (0.5 + 0.5 * np.sin(2.0 * np.pi * X) * np.cos(np.pi * Y))
        arr = np.stack([base, base, base], axis=2)
        cv2.circle(arr, (int(w * 0.65), int(h * 0.3)), int(min(w, h) * 0.13), (210, 210, 210), -1)
        cv2.rectangle(arr, (int(w * 0.15), int(h * 0.6)), (int(w * 0.5), int(h * 0.9)), (45, 45, 45), -1)
    arr = arr + rng.normal(0, 5, arr.shape).astype(np.float32)
    return np.ascontiguousarray(arr.clip(0, 255).astype(np.uint8))


def _paste_photo(rgb, mask_out, box, color, seed=5):
    x0, y0, x1, y1 = box
    rgb = rgb.copy()
    photo = _photo_array(x1 - x0, y1 - y0, color=color, seed=seed)
    rgb[y0:y1, x0:x1] = photo
    mask_out[y0:y1, x0:x1] = True
    return rgb


def build_region_samples():
    """M3.1 区域识别样本：10 类，每类带 6 区域真值掩膜（paper/text/color/illustration/special）。

    返回 {name: {"image", "regions": {paper,text,color,illustration,special}, "meta"}}。
    真值掩膜与绘制逐像素对齐；paper 为所有内容之外的背景区域。
    """
    W, H = 900, 1200
    samples = {}

    def _base(footnotes=False):
        rgb, m = text_page(W, H, footnotes=footnotes)
        return rgb, m

    def _finish(name, rgb, text, color=None, illustration=None, special=None, meta=None):
        color = np.zeros((H, W), bool) if color is None else color
        illustration = np.zeros((H, W), bool) if illustration is None else illustration
        special = np.zeros((H, W), bool) if special is None else special
        # 其他内容绘制时会覆盖文字像素，真值文字需排除被覆盖区域
        text = text & ~(color | illustration | special)
        content = text | color | illustration | special
        paper = ~content
        samples[name] = {
            "image": np.ascontiguousarray(rgb),
            "regions": {"paper": paper, "text": text, "color": color,
                        "illustration": illustration, "special": special},
            "meta": {"type": name, **(meta or {})},
        }

    # 1 黄纸黑字
    rgb, m = _base()
    rgb, m = yellow(rgb, m, paper=(234, 214, 158), noise=2.0)
    _finish("yellow_black_text", rgb, m)

    # 2 黄纸红色印章
    rgb, m = _base()
    rgb, m = yellow(rgb, m, paper=(234, 214, 158), noise=2.0)
    sp = np.zeros((H, W), bool)
    im = Image.fromarray(rgb); d = ImageDraw.Draw(im)
    sm = Image.new("L", (W, H), 0); ds = ImageDraw.Draw(sm)
    cx, cy, r = int(W * 0.78), int(H * 0.86), 55
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(200, 30, 30), width=5)
    ds.ellipse((cx - r, cy - r, cx + r, cy + r), outline=255, width=5)
    f = _font("simsun.ttc", 22)
    d.text((cx - 44, cy - 12), "审阅之章", font=f, fill=(200, 30, 30))
    ds.text((cx - 44, cy - 12), "审阅之章", font=f, fill=255)
    rgb = np.ascontiguousarray(np.asarray(im, dtype=np.uint8)); sp = np.asarray(sm) > 0
    _finish("yellow_red_seal", rgb, m, special=sp)

    # 3 黄纸彩色地图
    rgb, m = _base()
    rgb, m = yellow(rgb, m, paper=(234, 214, 158), noise=2.0)
    cm = np.zeros((H, W), bool)
    im = Image.fromarray(rgb); d = ImageDraw.Draw(im)
    cmim = Image.new("L", (W, H), 0); dc = ImageDraw.Draw(cmim)
    poly = [(int(W * 0.12), int(H * 0.66)), (int(W * 0.34), int(H * 0.62)),
            (int(W * 0.46), int(H * 0.72)), (int(W * 0.36), int(H * 0.84)), (int(W * 0.14), int(H * 0.82))]
    d.polygon(poly, fill=(120, 180, 230), outline=(40, 90, 160))
    dc.polygon(poly, fill=255)
    d.polygon([(int(W * 0.2), int(H * 0.7)), (int(W * 0.3), int(H * 0.68)), (int(W * 0.32), int(H * 0.76))], fill=(210, 150, 90))
    dc.polygon([(int(W * 0.2), int(H * 0.7)), (int(W * 0.3), int(H * 0.68)), (int(W * 0.32), int(H * 0.76))], fill=255)
    rgb = np.ascontiguousarray(np.asarray(im, dtype=np.uint8)); cm = np.asarray(cmim) > 0
    _finish("yellow_color_map", rgb, m, color=cm)

    # 4 黄纸彩色照片
    rgb, m = _base()
    rgb, m = yellow(rgb, m, paper=(234, 214, 158), noise=2.0)
    cm = np.zeros((H, W), bool)
    rgb = _paste_photo(rgb, cm, (int(W * 0.5), int(H * 0.6), int(W * 0.9), int(H * 0.85)), color=True)
    _finish("yellow_color_photo", rgb, m, color=cm)

    # 5 纯黑白文字
    rgb, m = _base()
    _finish("pure_bw_text", rgb, m)

    # 6 文字旁彩色批注
    rgb, m = _base()
    sp = np.zeros((H, W), bool)
    im = Image.fromarray(rgb); d = ImageDraw.Draw(im)
    sm = Image.new("L", (W, H), 0); ds = ImageDraw.Draw(sm)
    d.line((int(W * 0.1), int(H * 0.34), int(W * 0.6), int(H * 0.34)), fill=(200, 40, 40), width=3)
    ds.line((int(W * 0.1), int(H * 0.34), int(W * 0.6), int(H * 0.34)), fill=255, width=3)
    f = _font("simsun.ttc", 24)
    d.text((int(W * 0.62), int(H * 0.3)), "此处存疑", font=f, fill=(40, 80, 200))
    ds.text((int(W * 0.62), int(H * 0.3)), "此处存疑", font=f, fill=255)
    rgb = np.ascontiguousarray(np.asarray(im, dtype=np.uint8)); sp = np.asarray(sm) > 0
    _finish("color_annotation", rgb, m, special=sp)

    # 7 灰度照片
    rgb, m = _base()
    im = np.zeros((H, W), bool)
    rgb = _paste_photo(rgb, im, (int(W * 0.15), int(H * 0.6), int(W * 0.6), int(H * 0.86)), color=False, seed=9)
    _finish("grayscale_photo", rgb, m, illustration=im)

    # 8 彩色图表
    rgb, m = _base()
    cm = np.zeros((H, W), bool)
    im = Image.fromarray(rgb); d = ImageDraw.Draw(im)
    cmim = Image.new("L", (W, H), 0); dc = ImageDraw.Draw(cmim)
    x0, y0, bw, bh = int(W * 0.5), int(H * 0.62), int(W * 0.4), int(H * 0.22)
    for i, col in enumerate([(210, 60, 50), (50, 120, 210), (60, 170, 90), (230, 180, 40)]):
        bx = x0 + 14 + i * (bw // 4); bhi = int(bh * (0.4 + 0.15 * i))
        box = (bx, y0 + bh - bhi, bx + bw // 4 - 18, y0 + bh)
        d.rectangle(box, fill=col); dc.rectangle(box, fill=255)
    rgb = np.ascontiguousarray(np.asarray(im, dtype=np.uint8)); cm = np.asarray(cmim) > 0
    _finish("color_chart", rgb, m, color=cm)

    # 9 少量彩色元素
    rgb, m = _base()
    sp = np.zeros((H, W), bool)
    im = Image.fromarray(rgb); d = ImageDraw.Draw(im)
    sm = Image.new("L", (W, H), 0); ds = ImageDraw.Draw(sm)
    for (cx, cy, col) in [(int(W * 0.7), int(H * 0.2), (220, 60, 60)),
                          (int(W * 0.75), int(H * 0.45), (60, 160, 90)),
                          (int(W * 0.2), int(H * 0.9), (60, 90, 210))]:
        d.ellipse((cx - 12, cy - 12, cx + 12, cy + 12), fill=col)
        ds.ellipse((cx - 12, cy - 12, cx + 12, cy + 12), fill=255)
    rgb = np.ascontiguousarray(np.asarray(im, dtype=np.uint8)); sp = np.asarray(sm) > 0
    _finish("few_color_elements", rgb, m, special=sp)

    # 10 大面积彩色插图
    rgb, m = text_page(W, H, title=True, body_size=26)
    cm = np.zeros((H, W), bool)
    rgb = _paste_photo(rgb, cm, (int(W * 0.1), int(H * 0.42), int(W * 0.9), int(H * 0.9)), color=True, seed=11)
    _finish("large_color_illustration", rgb, m, color=cm)

    return samples


def build_samples():
    """12 类典型样本。返回 {name: {"image", "fg_mask", "meta"}}。"""
    samples = {}

    rgb, m = text_page()
    samples["clean_scan"] = {"image": rgb, "fg_mask": m, "meta": {"type": "clean_scan"}}

    rgb, m = text_page()
    rgb, m = yellow(rgb, m, paper=(248, 240, 214), noise=1.5)
    samples["light_yellow"] = {"image": rgb, "fg_mask": m, "meta": {"type": "light_yellow"}}

    rgb, m = text_page()
    rgb, m = yellow(rgb, m, paper=(232, 210, 150), noise=2.5, gradient=0.12)
    samples["heavy_yellow"] = {"image": rgb, "fg_mask": m, "meta": {"type": "heavy_yellow"}}

    rgb, m = text_page()
    rgb, m = shadow(rgb, m, kind="spine", strength=0.4)
    samples["local_shadow"] = {"image": rgb, "fg_mask": m, "meta": {"type": "local_shadow"}}

    rgb, m = text_page()
    rgb, m = black_border(rgb, m, border=70)
    samples["black_border"] = {"image": rgb, "fg_mask": m, "meta": {"type": "black_border", "border": 70}}

    rgb, m = text_page()
    rgb, m = rotate(rgb, m, 2.0)
    samples["skew_light"] = {"image": rgb, "fg_mask": m, "meta": {"type": "skew_light", "angle": 2.0}}

    rgb, m = text_page()
    rgb, m = rotate(rgb, m, 5.0)
    samples["skew_moderate"] = {"image": rgb, "fg_mask": m, "meta": {"type": "skew_moderate", "angle": 5.0}}

    rgb, m = text_page()
    rgb, m, mat = perspective(rgb, m)
    samples["perspective"] = {"image": rgb, "fg_mask": m,
                              "meta": {"type": "perspective", "homography": mat.tolist()}}

    rgb, m = text_page()
    rgb, m, cm = color_content(rgb, m)
    samples["color_illustration"] = {"image": rgb, "fg_mask": m, "meta": {"type": "color_illustration"},
                                     "color_mask": cm}

    rgb, m = text_page(footnotes=True)
    samples["footnotes"] = {"image": rgb, "fg_mask": m, "meta": {"type": "footnotes"}}

    rgb, m = table_page()
    samples["formula_table"] = {"image": rgb, "fg_mask": m, "meta": {"type": "formula_table"}}

    rgb, m = text_page(bg=(252, 252, 252))
    samples["already_clean"] = {"image": rgb, "fg_mask": m, "meta": {"type": "already_clean"}}

    return samples


def main(out_dir: str = "temp/m2_samples") -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = build_samples()
    for name, s in samples.items():
        d = out / name
        d.mkdir(exist_ok=True)
        Image.fromarray(s["image"]).save(d / "image.png", dpi=(150, 150))
        Image.fromarray(s["fg_mask"].astype(np.uint8) * 255).save(d / "fg_mask.png")
        if "color_mask" in s:
            Image.fromarray(s["color_mask"].astype(np.uint8) * 255).save(d / "color_mask.png")
        (d / "meta.json").write_text(json.dumps(s["meta"], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{name}: {s['image'].shape[1]}x{s['image'].shape[0]}")
    print(f"共 {len(samples)} 个样本 -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "temp/m2_samples"))