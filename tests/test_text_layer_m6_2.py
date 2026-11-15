# -*- coding: utf-8 -*-
"""M6.2 验收测试：可搜索文字层（W2）。

覆盖：坐标变换（contain/cover/缩放/越界/非法防御/PoC 精度）、写入器（ToUnicode 往返、
Tr3 像素差 0、多页共享字体与 CMap 分块、空/超长/非法防御、PDFium 单字与行尾连字符 quirks）、
OCR 适配器、导出集成（OCR 关字节级一致 / OCR 开 completed 有文字 failed 无文字 / 验证链保留）。
全部离线：合成图像与合成 OCR 结果，不依赖真实插件。
"""

from __future__ import annotations

import json
import re
import sys

import numpy as np
import pikepdf
import pypdfium2 as pdfium
import pytest
from PIL import Image

from bookclean.config import PDFExportConfig
from bookclean.page_layout import PageLayout
from bookclean.pdf_exporter import PDFExporter
from bookclean.text_layer import (
    PageText,
    TextLayerWriter,
    TextLine,
    transform_line,
)
from bookclean.ocr.provider import OcrLine, PageOcrResult, STATUS_COMPLETED, STATUS_FAILED
from bookclean.ocr.text_source import OcrTextSource, page_text_from_result


# ---------------------------------------------------------------- 工具

def layout(pw=500.0, ph=1000.0, iw=500.0, ih=1000.0, wpx=1000, hpx=2000, fit="contain"):
    return PageLayout(page_index=0, page_width_pt=pw, page_height_pt=ph,
                      image_width_px=wpx, image_height_px=hpx,
                      image_display_width_pt=iw, image_display_height_pt=ih,
                      effective_dpi_x=144, effective_dpi_y=144, fit_mode=fit,
                      source_page_size="test", estimated_size=False)


def solid(color=(255, 255, 255), size=(400, 560)):
    return Image.new("RGB", size, color)


def make_task(tmp_path, n=2, size=(400, 560)):
    task = tmp_path / "task"
    (task / "original_pages").mkdir(parents=True)
    manifest = {"task_id": "t_m62", "pages": [], "inputs": []}
    for i in range(n):
        rel = f"original_pages/page_{i + 1:06d}.png"
        solid(size=size).save(task / rel, dpi=(150, 150))
        manifest["pages"].append({
            "page_index": i, "display_index": i + 1, "source_kind": "image",
            "source_path": str(tmp_path / f"input_{i}.png"),
            "status": "completed", "image_path": rel, "processed_image_path": None,
            "processing_status": "completed", "page_type": None, "dpi": 150.0,
            "metadata": {}, "final_image_source": "enhanced",
            "fallback_level": "none", "processing_decision": "accept",
        })
        manifest["inputs"].append({"path": str(tmp_path / f"input_{i}.png"), "kind": "image"})
    (task / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return task


def export(task, tmp_path, name, text_source=None, **cfg):
    out = tmp_path / name
    cfg.setdefault("overwrite", True)
    info = PDFExporter(PDFExportConfig(**cfg)).export(task, out, text_source=text_source)
    return out, info


def render_arrays(path, dpi=100):
    doc = pdfium.PdfDocument(str(path))
    arrs = [np.asarray(doc[i].render(scale=dpi / 72.0).to_pil().convert("RGB"), dtype=np.int16)
            for i in range(len(doc))]
    doc.close()
    return arrs


def extract_pages(path):
    doc = pdfium.PdfDocument(str(path))
    texts = [doc[i].get_textpage().get_text_bounded() for i in range(len(doc))]
    doc.close()
    return texts


def norm(t):
    return "".join(t.split())


def inject_pdf(tmp_path, page_texts, name="w.pdf", size=(500, 1000)):
    """把若干 PageText 注入空白页（writer 级测试），返回路径。"""
    pdf = pikepdf.new()
    w = TextLayerWriter(pdf)
    for pt in page_texts:
        page = pdf.add_blank_page(page_size=size)
        w.inject_page(page, pt, layout(pw=size[0], ph=size[1], iw=size[0], ih=size[1],
                                       wpx=pt.image_width_px, hpx=pt.image_height_px))
    p = tmp_path / name
    pdf.save(p)
    pdf.close()
    return p


def ocr_result(page_no, lines, w=1000, h=2000, status=STATUS_COMPLETED):
    return PageOcrResult(
        page_no=page_no, status=status,
        lines=tuple(OcrLine(line_id=i, text=t, score=0.9, bbox_px=b)
                    for i, (t, b) in enumerate(lines)),
        image_width_px=w, image_height_px=h, image_sha256="x")


# ---------------------------------------------------------------- 坐标变换

def test_transform_contain_exact_fit():
    # image 1000x2000 -> display 500x1000, sx=sy=0.5, ox=oy=0
    p = transform_line(TextLine("abcd", (100, 200, 500, 280)), 1000, 2000, layout())
    assert p is not None
    assert p.x0_pt == pytest.approx(50.0)
    th = (2000 - 200) * 0.5 - (2000 - 280) * 0.5  # 900-860=40
    size = th / 1.1
    assert p.size_pt == pytest.approx(size)
    assert p.baseline_pt == pytest.approx(860.0 + 0.2 * size)
    assert p.tz_pct == pytest.approx(200.0 / (4 * size) * 100.0)


def test_transform_contain_letterbox_offset():
    # 页 500x1000，显示 400x800 居中 ⇒ ox=50, oy=100
    p = transform_line(TextLine("ab", (0, 0, 1000, 200)), 1000, 2000,
                       layout(iw=400, ih=800))
    assert p.x0_pt == pytest.approx(50.0)
    # Y0 = oy + (2000-200)*0.4 = 100+720=820
    size = (200 * 0.4) / 1.1
    assert p.baseline_pt == pytest.approx(820.0 + 0.2 * size)


def test_transform_downsample_scale():
    # 显示尺寸为图像 1/4（sx=sy=0.25），页=显示尺寸 ⇒ 无居中偏移
    p = transform_line(TextLine("x", (400, 800, 800, 1000)), 1000, 2000,
                       layout(pw=250, ph=500, iw=250, ih=500))
    assert p.x0_pt == pytest.approx(100.0)
    assert p.size_pt == pytest.approx((200 * 0.25) / 1.1)


def test_transform_cover_negative_offset_and_clip():
    # cover：显示 600x1200 > 页 500x1000 ⇒ ox=-50, oy=-100
    lay = layout(iw=600, ih=1200, fit="cover")
    # 完全在页外（顶部上方）⇒ None
    off = transform_line(TextLine("off", (0, 0, 100, 100)), 1000, 2000, lay)
    assert off is None
    # 部分越界（跨底边）⇒ 裁剪到页矩形
    part = transform_line(TextLine("part", (0, 1800, 100, 2000)), 1000, 2000, lay)
    assert part is not None
    assert part.x0_pt == pytest.approx(0.0)  # -50 裁剪到 0


def test_transform_empty_illegal_degenerate():
    lay = layout()
    assert transform_line(TextLine("", (0, 0, 100, 100)), 1000, 2000, lay) is None
    assert transform_line(TextLine("   ", (0, 0, 100, 100)), 1000, 2000, lay) is None
    assert transform_line(TextLine("a", (100, 100, 100, 200)), 1000, 2000, lay) is None  # x1==x0
    assert transform_line(TextLine("a", (0, 0, 100, 100)), 0, 2000, lay) is None  # 非法图像尺寸
    assert transform_line(TextLine("a", (float("nan"), 0, 100, 100)), 1000, 2000, lay) is None
    assert transform_line(TextLine("a", (0, 0, float("inf"), 100)), 1000, 2000, lay) is None


def test_transform_filters_control_and_surrogates():
    p = transform_line(TextLine("a\x00b\tc\ud800d", (0, 0, 400, 100)), 1000, 2000, layout())
    assert p.text == "abcd"


def test_transform_font_size_clamp():
    # 极小框 ⇒ 字号钳制到下限 1.0
    tiny = transform_line(TextLine("a", (0, 0, 10, 1)), 1000, 2000, layout())
    assert tiny.size_pt >= 1.0
    # 极大框 ⇒ 上限 1000
    huge = transform_line(TextLine("a", (0, 0, 1000, 2000)), 1000, 2000,
                          layout(pw=10000, ph=20000, iw=10000, ih=20000))
    assert huge.size_pt <= 1000.0


def test_transform_poc_precision():
    # PoC 已验证链路：contain 居中，坐标偏差应 ≤3pt（此处为解析精确值，偏差≈0）
    lay = layout(pw=595, ph=842, iw=595, ih=842, wpx=1240, hpx=1754)
    p = transform_line(TextLine("The quick brown fox", (100, 200, 1100, 260)), 1240, 1754, lay)
    sx = 595 / 1240
    assert p.x0_pt == pytest.approx(100 * sx, abs=0.01)
    assert p.baseline_pt == pytest.approx((1754 - 260) * (842 / 1754) + 0.2 * p.size_pt, abs=0.01)


# ---------------------------------------------------------------- 写入器

def test_writer_tounicode_roundtrip_cjk_latin_punct(tmp_path):
    text = "中文与 English 混排：标点，。！？（）《》— test 0123456789."
    pt = PageText(lines=[TextLine(text, (50, 50, 950, 150))], image_width_px=1000, image_height_px=2000)
    p = inject_pdf(tmp_path, [pt])
    got = extract_pages(p)[0]
    assert norm(got) == norm(text)


def test_writer_supplementary_plane_cmap(tmp_path):
    # 补充平面字符（U+20000 CJK 扩展 B）：ToUnicode 必须用代理对编码
    ch = "\U00020000"
    pt = PageText(lines=[TextLine("A" + ch + "B", (50, 50, 950, 150))],
                  image_width_px=1000, image_height_px=2000)
    p = inject_pdf(tmp_path, [pt])
    with pikepdf.open(p) as pdf:
        cmap = pdf.pages[0].obj["/Resources"]["/Font"]["/F1"]["/ToUnicode"].read_bytes().decode("ascii")
    assert "D840DC00" in cmap  # U+20000 的 UTF-16BE 代理对
    got = extract_pages(p)[0]
    assert "A" in got and "B" in got


def test_writer_tr3_pixel_diff_zero(tmp_path):
    # 图像底 + 注入文字层 ⇒ 渲染像素与未注入完全一致（字形永不渲染）
    task = make_task(tmp_path, n=1)
    base, _ = export(task, tmp_path, "base.pdf")
    pt = PageText(lines=[TextLine("不可见文字 invisible", (50, 50, 350, 150))],
                  image_width_px=400, image_height_px=560)
    src = OcrTextSource({0: pt})
    search, _ = export(task, tmp_path, "search.pdf", text_source=src)
    ab, as_ = render_arrays(base), render_arrays(search)
    assert int(np.abs(ab[0] - as_[0]).max()) == 0
    assert norm(extract_pages(search)[0]) == norm("不可见文字 invisible")
    assert norm(extract_pages(base)[0]) == ""


def test_writer_multipage_shared_font_and_cmap_chunks(tmp_path):
    # 跨页累积 >100 唯一字符 ⇒ 共享同一字体对象，CMap 分块且每块 ≤100
    lines = []
    chars = [chr(c) for c in range(0x4E00, 0x4E00 + 150)]  # 150 个唯一 CJK
    pt1 = PageText(lines=[TextLine("".join(chars[:80]), (50, 50, 950, 150))],
                   image_width_px=1000, image_height_px=2000)
    pt2 = PageText(lines=[TextLine("".join(chars[80:]), (50, 200, 950, 300))],
                   image_width_px=1000, image_height_px=2000)
    p = inject_pdf(tmp_path, [pt1, pt2])
    with pikepdf.open(p) as pdf:
        f1 = pdf.pages[0].obj["/Resources"]["/Font"]["/F1"]
        f2 = pdf.pages[1].obj["/Resources"]["/Font"]["/F1"]
        assert f1.objgen == f2.objgen  # 同一间接对象
        cmap = f1["/ToUnicode"].read_bytes().decode("ascii")
    blocks = re.findall(r"(\d+) beginbfchar\n(.*?)endbfchar", cmap, re.S)
    assert len(blocks) >= 2
    for count_str, body in blocks:
        n = int(count_str)
        assert 1 <= n <= 100
        entries = re.findall(r"<[0-9A-Fa-f]{4}> <[0-9A-Fa-f]+>", body)
        assert len(entries) == n
    texts = extract_pages(p)
    assert norm(texts[0]) == norm("".join(chars[:80]))
    assert norm(texts[1]) == norm("".join(chars[80:]))


def test_writer_empty_and_no_lines(tmp_path):
    pdf = pikepdf.new()
    w = TextLayerWriter(pdf)
    page = pdf.add_blank_page(page_size=(500, 1000))
    assert w.inject_page(page, PageText(lines=[]), layout()) == 0
    # 全为非法框 ⇒ 0，且不创建字体
    pt = PageText(lines=[TextLine("a", (0, 0, 0, 0))], image_width_px=1000, image_height_px=2000)
    assert w.inject_page(page, pt, layout()) == 0
    assert "/Font" not in page.obj.get("/Resources", {})


def test_writer_single_non_ascii_glyph_extractable(tmp_path):
    # PDFium 丢弃孤立单个非 ASCII 字形 ⇒ 补不可见空格后可提取
    pt = PageText(lines=[TextLine("男", (50, 50, 150, 150))], image_width_px=1000, image_height_px=2000)
    p = inject_pdf(tmp_path, [pt])
    assert "男" in extract_pages(p)[0]


def test_writer_line_final_hyphen_not_dehyphenated(tmp_path):
    # PDFium 对行尾 ASCII 连字符去连字符（误提取 \x02）⇒ 补 NBSP 后保留
    pt = PageText(lines=[TextLine("Singa-", (50, 50, 400, 150)),
                         TextLine("pore", (50, 200, 400, 300))],
                  image_width_px=1000, image_height_px=2000)
    p = inject_pdf(tmp_path, [pt])
    got = extract_pages(p)[0]
    assert "\x02" not in got
    assert norm(got) == "Singa-pore"


def test_writer_mid_hyphen_and_endash_unchanged(tmp_path):
    pt = PageText(lines=[TextLine("cost-of-living", (50, 50, 600, 150)),
                         TextLine("abc–", (50, 200, 400, 300)),
                         TextLine("def", (50, 350, 400, 450))],
                  image_width_px=1000, image_height_px=2000)
    p = inject_pdf(tmp_path, [pt])
    got = extract_pages(p)[0]
    assert "\x02" not in got
    assert norm(got) == "cost-of-livingabc–def"


def test_writer_long_text_no_crash(tmp_path):
    long = "字" * 5000
    pt = PageText(lines=[TextLine(long, (50, 50, 950, 150))], image_width_px=1000, image_height_px=2000)
    p = inject_pdf(tmp_path, [pt])
    assert norm(extract_pages(p)[0]) == long


# ---------------------------------------------------------------- OCR 适配器

def test_adapter_completed_with_lines():
    r = ocr_result(1, [("你好", (0, 0, 100, 50))])
    pt = page_text_from_result(r)
    assert pt is not None and pt.lines[0].text == "你好"
    assert pt.image_width_px == 1000


def test_adapter_failed_or_empty_returns_none():
    assert page_text_from_result(ocr_result(1, [], status=STATUS_FAILED)) is None
    assert page_text_from_result(ocr_result(1, [], status=STATUS_COMPLETED)) is None


def test_adapter_from_results_keyed_by_page_index():
    results = {0: ocr_result(1, [("a", (0, 0, 100, 50))]),
               2: ocr_result(3, [("b", (0, 0, 100, 50))]),
               1: ocr_result(2, [], status=STATUS_FAILED)}
    src = OcrTextSource.from_ocr_results(results)
    assert len(src) == 2
    assert src(0, None, None) is not None
    assert src(2, None, None) is not None
    assert src(1, None, None) is None  # failed
    assert src(99, None, None) is None  # 缺失


# ---------------------------------------------------------------- 导出集成

def pdf_content_bytes(path):
    """提取 PDF 实质内容字节：逐页内容流 + 图像 XObject 数据 + 资源键。

    不含文档元数据（CreationDate/ModDate/XMP）与 trailer /ID——这些是 M5 起就逐次导出
    不同的非确定字段，与 OCR 无关。OCR 若改动导出，必然体现在内容流/字体/图像上。
    """
    sig = []
    with pikepdf.open(path) as pdf:
        for page in pdf.pages:
            c = page.obj.get("/Contents")
            if isinstance(c, pikepdf.Array):
                data = b"".join(s.read_bytes() for s in c)
            elif c is not None:
                data = c.read_bytes()
            else:
                data = b""
            res = page.obj.get("/Resources", {})
            res_keys = sorted(str(k) for k in res.keys())
            imgs = []
            for name, xo in res.get("/XObject", {}).items():
                if xo.get("/Subtype") == pikepdf.Name("/Image"):
                    imgs.append((str(name), xo.read_raw_bytes()))
            sig.append((data, res_keys, sorted(imgs)))
    return sig


def test_export_ocr_off_content_byte_identical_and_no_text(tmp_path):
    # OCR 关 ⇒ 导出内容与 M5 一致（代码路径全守卫）；此处以"无钩子 vs 空钩子"的
    # 内容字节级一致实证 OCR 纯增量。整文件字节因 M5 起即含逐次不同的时间戳/ID 不可比。
    task = make_task(tmp_path, n=2)
    a, _ = export(task, tmp_path, "a.pdf")
    empty = OcrTextSource({})
    c, _ = export(task, tmp_path, "c.pdf", text_source=empty)
    assert pdf_content_bytes(a) == pdf_content_bytes(c)  # 内容字节级一致
    for t in extract_pages(a):
        assert norm(t) == ""  # 无文字层
    with pikepdf.open(a) as pdf:
        for page in pdf.pages:
            assert "/Font" not in page.obj.get("/Resources", {})  # 无字体资源


def test_export_ocr_on_completed_has_text_failed_no_text(tmp_path):
    task = make_task(tmp_path, n=2)
    pt0 = PageText(lines=[TextLine("第一页文字 page one", (50, 50, 350, 150))],
                   image_width_px=400, image_height_px=560)
    results = {0: ocr_result(1, [("第一页文字 page one", (50, 50, 350, 150))], w=400, h=560),
               1: ocr_result(2, [], w=400, h=560, status=STATUS_FAILED)}
    src = OcrTextSource.from_ocr_results(results)
    out, info = export(task, tmp_path, "s.pdf", text_source=src)
    texts = extract_pages(out)
    assert norm(texts[0]) == norm("第一页文字 page one")
    assert norm(texts[1]) == ""  # failed 页无文字，但导出成功
    assert info["export_status"] == "completed"
    assert info["ocr"]["pages_with_text"] == 1
    assert info["ocr"]["pages_without_text"] == 1
    assert info["validation"]["passed"] is True
    assert info["validation"]["ocr_text_check"]["pages_with_text"] == 1


def test_export_ocr_on_pixel_identical_to_off(tmp_path):
    task = make_task(tmp_path, n=2)
    base, _ = export(task, tmp_path, "base.pdf")
    src = OcrTextSource({i: PageText(lines=[TextLine(f"页{i}文本 text", (50, 50, 350, 150))],
                                     image_width_px=400, image_height_px=560) for i in range(2)})
    search, info = export(task, tmp_path, "search.pdf", text_source=src)
    ab, as_ = render_arrays(base), render_arrays(search)
    assert all(int(np.abs(x - y).max()) == 0 for x, y in zip(ab, as_))
    assert info["validation"]["passed"] is True
    assert info["ocr"]["coverage"] == 1.0


def test_export_validation_chain_preserved_with_ocr(tmp_path):
    # OCR 开启不削弱既有验证：页数/尺寸/内容检查仍执行且通过
    task = make_task(tmp_path, n=3)
    src = OcrTextSource({i: PageText(lines=[TextLine(f"文字{i}", (50, 50, 300, 120))],
                                     image_width_px=400, image_height_px=560) for i in range(3)})
    out, info = export(task, tmp_path, "v.pdf", text_source=src)
    v = info["validation"]
    assert v["passed"] is True and v["page_count"] == 3
    assert all(p["size_ok"] and p["blank_ok"] for p in v["per_page"])
    assert v["ocr_text_check"]["pages_expected"] == 3


def test_export_isolation_no_ocr_import():
    # 导出内核 + 中性写入器不得导入任何 bookclean.ocr 模块（干净子进程验证）
    import os
    import subprocess
    from pathlib import Path
    src = str(Path(__file__).resolve().parent.parent / "src")
    code = ("import sys; import bookclean.pdf_exporter, bookclean.text_layer; "
            "ocr=[m for m in sys.modules if m.startswith('bookclean.ocr')]; "
            "print(','.join(ocr)); sys.exit(1 if ocr else 0)")
    env = {**os.environ, "PYTHONPATH": src}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"导出内核导入了 OCR 模块: {r.stdout} {r.stderr}"