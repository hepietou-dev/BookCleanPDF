"""M6.2 可搜索文字层写入器（W2：pikepdf 合成 CID 字体，不嵌字体）。

中性模块：仅依赖 pikepdf（已在运行时锁内）与 page_layout 的布局数据，
**不导入 bookclean.ocr / 任何引擎代码**。OCR 关闭时本模块不被调用，导出与 M5 字节级一致。

原理（docs/m6_0_ocr_architecture.md §7–§8，PoC 已验证像素差 0）：
- 字体对象 Type0 / Identity-H，descendant CIDFontType0，DW 1000，不嵌字体文件；
  FontDescriptor Ascent 900 / Descent -200。字形永不渲染（render_mode 3），故不嵌字体不影响任何像素。
- 每行一个 BT…ET：/F1 <size> Tf、3 Tr（不可见）、1 0 0 1 X Y Tm 定位基线、
  Tz 水平标度使整行 advance 匹配检测框宽、<hex> Tj 输出 CID。
- ToUnicode CMap（bfchar）保证提取/搜索/复制得到正确 Unicode（含中英标点、补充平面代理对）。
- 坐标变换：图像像素（y 向下）→ PDF 点（y 向上），居中偏移 + 等比缩放 + y 翻转；
  contain 与 cover 统一公式，cover 负偏移自然成立，页外行裁剪/跳过。

已知近似（如实记录）：DW 1000 均匀字宽 + Tz 整行标度 ⇒ 行内逐字位置为近似，
对搜索/复制无影响，选择高亮框略有偏差。不嵌字体 ⇒ 不满足 PDF/A（BookClean 无此目标）。
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field

import pikepdf

from .page_layout import PageLayout

# 与 M6.0 PoC 一致的几何系数（保证坐标精度 ≤3pt 的已验证链路）。
SIZE_FACTOR = 1.1          # 字号 = 框高 / 1.1
BASELINE_DESCENT = 0.2     # 基线 = 框底 + 0.2 * 字号（descent 余量）
MIN_FONT_SIZE = 1.0        # 钳制下限（防退化框产生 0/负字号）
MAX_FONT_SIZE = 1000.0     # 钳制上限（防异常大框）
_EPS = 1e-6


@dataclass(frozen=True)
class TextLine:
    """一行 OCR 文字（图像像素坐标，轴对齐 bbox）。引擎原文，不做美化。"""

    text: str
    bbox_px: tuple[float, float, float, float]  # x0, y0, x1, y1（图像像素，y 向下）


@dataclass
class PageText:
    """一页的文字层数据。image_*_px 为 OCR 输入图像（最终解析图像）的像素尺寸。"""

    lines: list[TextLine] = field(default_factory=list)
    image_width_px: int = 0
    image_height_px: int = 0


@dataclass(frozen=True)
class PlacedLine:
    """变换后的一行（PDF 点坐标）。"""

    text: str
    x0_pt: float
    baseline_pt: float
    size_pt: float
    tz_pct: float


def _clean_text(text: str) -> str:
    """剔除控制字符（Cc）与代理项；保留可见文字与常规空白。防御非法字符。"""
    out = []
    for ch in text:
        cp = ord(ch)
        if 0xD800 <= cp <= 0xDFFF:  # 孤立代理项（非法 Unicode）
            continue
        if unicodedata.category(ch) == "Cc":  # 控制字符（含 \x00/\t/\n 等）
            continue
        out.append(ch)
    return "".join(out)


def _finite_box(bbox_px) -> tuple[float, float, float, float] | None:
    """校验 bbox 为 4 个有限实数且 x1>x0 / y1>y0；否则 None（防御非法/退化框）。"""
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox_px)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def transform_line(
    line: TextLine,
    image_width_px: int,
    image_height_px: int,
    layout: PageLayout,
) -> PlacedLine | None:
    """图像像素 bbox → PDF 点放置参数（contain/cover 统一）。

    完全在页外的行返回 None（跳过注入）；部分越界（cover）按页矩形裁剪框后再注入。
    空行/非法字符/退化框返回 None。
    """
    text = _clean_text(line.text)
    if not text.strip():
        return None
    box = _finite_box(line.bbox_px)
    if box is None or image_width_px <= 0 or image_height_px <= 0:
        return None
    x0, y0, x1, y1 = box

    pw = layout.page_width_pt
    ph = layout.page_height_pt
    iw = layout.image_display_width_pt
    ih = layout.image_display_height_pt
    if iw <= 0 or ih <= 0 or pw <= 0 or ph <= 0:
        return None
    sx = iw / image_width_px
    sy = ih / image_height_px
    ox = (pw - iw) / 2.0
    oy = (ph - ih) / 2.0

    # 图像 y 向下 → PDF y 向上翻转；含居中偏移（cover 时 ox/oy 可为负）。
    X0 = ox + x0 * sx
    X1 = ox + x1 * sx
    Y0 = oy + (image_height_px - y1) * sy  # 框底（PDF 坐标）
    Y1 = oy + (image_height_px - y0) * sy  # 框顶

    # 完全在页外 → 跳过。
    if X1 <= 0 or X0 >= pw or Y1 <= 0 or Y0 >= ph:
        return None
    # 部分越界（cover）→ 裁剪到页矩形。
    X0 = max(0.0, min(X0, pw))
    X1 = max(0.0, min(X1, pw))
    Y0 = max(0.0, min(Y0, ph))
    Y1 = max(0.0, min(Y1, ph))

    tw = max(_EPS, X1 - X0)
    th = max(_EPS, Y1 - Y0)
    size = th / SIZE_FACTOR
    size = max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, size))
    n = len(text)
    tz = tw / (n * size) * 100.0
    baseline = Y0 + BASELINE_DESCENT * size
    return PlacedLine(text=text, x0_pt=X0, baseline_pt=baseline, size_pt=size, tz_pct=tz)


def _unicode_hex(ch: str) -> str:
    """字符 → ToUnicode 目标（UTF-16BE hex）。补充平面用代理对。"""
    cp = ord(ch)
    if cp <= 0xFFFF:
        return f"{cp:04X}"
    cp -= 0x10000
    hi = 0xD800 + (cp >> 10)
    lo = 0xDC00 + (cp & 0x3FF)
    return f"{hi:04X}{lo:04X}"


def _build_cmap(bfchar: list[str]) -> str:
    # PDF CMap 规范（Adobe TN #5014）：每个 beginbfchar 块至多 100 条，超出须分块，
    # 否则阅读器（PDFium 等）会误解析导致字符丢失/错映射。多页共享字体时字符数可远超 100。
    blocks = []
    for i in range(0, len(bfchar), 100):
        chunk = bfchar[i:i + 100]
        blocks.append(f"{len(chunk)} beginbfchar\n" + "\n".join(chunk) + "\nendbfchar\n")
    return (
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        "/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n"
        "1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
        + "".join(blocks) +
        "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n"
    )


class TextLayerWriter:
    """向一个 pikepdf.Pdf 的多个页面注入不可见文字层，共享同一字体对象。

    字符 → CID 全局去重；ToUnicode CMap 随新字符增量刷新。多页复用同一 Type0 字体，
    每页 Resources/Font/F1 指向同一间接对象。线程不安全（导出在单线程内逐页调用）。
    """

    def __init__(self, pdf: pikepdf.Pdf):
        self._pdf = pdf
        self._gids: dict[str, int] = {}
        self._bfchar: list[str] = []
        self._font: pikepdf.Object | None = None
        self._cmap_stream: pikepdf.Stream | None = None
        self._dirty = False

    def _ensure_font(self) -> None:
        if self._font is not None:
            return
        pdf = self._pdf
        fd = pdf.make_indirect(pikepdf.Dictionary(
            Type=pikepdf.Name("/FontDescriptor"), FontName=pikepdf.Name("/BCInvisible"),
            Flags=4, FontBBox=[0, -200, 1000, 900], ItalicAngle=0, Ascent=900, Descent=-200,
            CapHeight=700, StemV=80))
        cid = pdf.make_indirect(pikepdf.Dictionary(
            Type=pikepdf.Name("/Font"), Subtype=pikepdf.Name("/CIDFontType0"),
            BaseFont=pikepdf.Name("/BCInvisible"),
            CIDSystemInfo=pikepdf.Dictionary(Registry=pikepdf.String("Adobe"),
                                             Ordering=pikepdf.String("Identity"), Supplement=0),
            FontDescriptor=fd, DW=1000))
        self._cmap_stream = pikepdf.Stream(pdf, b"")
        self._font = pdf.make_indirect(pikepdf.Dictionary(
            Type=pikepdf.Name("/Font"), Subtype=pikepdf.Name("/Type0"),
            BaseFont=pikepdf.Name("/BCInvisible"), Encoding=pikepdf.Name("/Identity-H"),
            DescendantFonts=[cid], ToUnicode=self._cmap_stream))

    def _gid(self, ch: str) -> int:
        g = self._gids.get(ch)
        if g is None:
            g = len(self._gids) + 1  # CID 0 保留
            self._gids[ch] = g
            self._bfchar.append(f"<{g:04X}> <{_unicode_hex(ch)}>")
            self._dirty = True
        return g

    def _flush_cmap(self) -> None:
        if self._dirty and self._cmap_stream is not None:
            self._cmap_stream.write(_build_cmap(self._bfchar).encode("ascii"))
            self._dirty = False

    def inject_page(self, page: pikepdf.Page, page_text: PageText, layout: PageLayout) -> int:
        """向单页注入文字层，返回注入字符数（0 表示无可注入文字）。"""
        if not page_text.lines:
            return 0
        placed: list[PlacedLine] = []
        for line in page_text.lines:
            p = transform_line(line, page_text.image_width_px, page_text.image_height_px, layout)
            if p is not None:
                placed.append(p)
        if not placed:
            return 0

        self._ensure_font()
        ops: list[str] = []
        nchars = 0
        for p in placed:
            emit = p.text
            # PDFium（Chrome/Edge 同源）会丢弃孤立的单个非 ASCII 字形（实测单字"男"不可提取/搜索，
            # 而"男男"/"男 "正常）。补一个不可见空格使该字形可提取；Tz 仍按原字数计算，
            # 故原字保持整框宽度，空格仅向框外延伸（Tr 3 不可见）。不改变有意义文字内容。
            if len(emit) == 1 and ord(emit) > 0x7F:
                emit = emit + " "
            # PDFium 会对"以 ASCII 连字符结尾、且后面还有文本对象"的行做去连字符（de-hyphenation），
            # 把 '-' 误提取为 \x02 并吞掉换行（实测 en-dash/句号/逗号无此问题；普通空格会被裁剪掉，
            # 无法阻止）。补一个不可见 NBSP(U+00A0) 使该行不再以连字符结尾即可避免；NBSP 属空白，
            # 提取后按空白归一化即与引擎原文一致。Tz 仍按原字数计算，NBSP 仅向框外延伸（Tr 3 不可见）。
            if emit.endswith("-"):
                emit = emit + "\u00a0"
            hexstr = "".join(f"{self._gid(ch):04X}" for ch in emit)
            ops.append(
                f"BT /F1 {p.size_pt:.2f} Tf 3 Tr 1 0 0 1 {p.x0_pt:.2f} {p.baseline_pt:.2f} "
                f"Tm {p.tz_pct:.2f} Tz <{hexstr}> Tj ET"
            )
            nchars += len(p.text)

        res = page.obj.get("/Resources", pikepdf.Dictionary())
        fonts = res.get("/Font", pikepdf.Dictionary())
        fonts["/F1"] = self._font
        res["/Font"] = fonts
        page.obj["/Resources"] = res

        content = pikepdf.Stream(self._pdf, ("\n".join(ops)).encode("ascii"))
        old = page.obj.get("/Contents")
        if old is None:
            page.obj["/Contents"] = content
        elif isinstance(old, pikepdf.Array):
            old.append(content)
        else:
            page.obj["/Contents"] = pikepdf.Array([old, content])

        self._flush_cmap()
        return nchars