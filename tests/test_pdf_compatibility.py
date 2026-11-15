"""M4.8.5 PDF 结构与阅读器兼容性测试。

锁定导出 PDF 的兼容性属性（PDFium=Chrome/Edge 同源引擎逐页渲染）：
版本/未加密/标准色空间(DeviceRGB/DeviceGray 8-bit)/标准过滤器(DCTDecode/FlateDecode)/
无 OCR 文本层(0 字符)/元数据可读/混合尺寸方向可渲染/线性化选项有效。
全部为文件/像素口径，非 OCR。合成样本自包含，不依赖 temp 真实产物。
"""

import numpy as np
import pikepdf
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

from bookclean.pdf_verifier import _classify_colorspace, _iter_image_xobjects
from test_pdf_exporter import export, make_task, solid


def _content_image(size=(200, 280), color=(250, 250, 245)):
    im = Image.new("RGB", size, color)
    d = ImageDraw.Draw(im)
    for y in range(30, size[1] - 20, 18):
        d.line([(20, y), (size[0] - 20, y)], fill=(30, 30, 30), width=2)
    return im


def _images(pdf_path):
    with pikepdf.open(pdf_path) as pk:
        return [(pno, name, _classify_colorspace(obj.get("/ColorSpace")),
                 int(obj.get("/BitsPerComponent", 0)), str(obj.get("/Filter")))
                for pno, name, obj in _iter_image_xobjects(pk)]


def test_pdf_version_modern_and_unencrypted(tmp_path):
    task, _ = make_task(tmp_path, [{"orig": solid((10, 10, 10)), "proc": _content_image()}])
    out, _ = export(task, tmp_path)
    with pikepdf.open(out) as pk:
        assert pk.pdf_version in ("1.4", "1.5", "1.6", "1.7", "2.0")
        assert not pk.is_encrypted


def test_no_ocr_text_layer(tmp_path):
    task, _ = make_task(tmp_path, [{"orig": solid((10, 10, 10)), "proc": _content_image()},
                                   {"orig": solid((20, 20, 20)), "proc": _content_image()}])
    out, _ = export(task, tmp_path)
    doc = pdfium.PdfDocument(str(out))
    chars = sum(doc[i].get_textpage().count_chars() for i in range(len(doc)))
    doc.close()
    assert chars == 0  # 图像型 PDF，无 OCR 文本层（设计而非缺陷）


def test_pdfium_renders_every_page(tmp_path):
    task, _ = make_task(tmp_path, [{"orig": solid((10, 10, 10)), "proc": _content_image()}
                                   for _ in range(4)])
    out, _ = export(task, tmp_path)
    doc = pdfium.PdfDocument(str(out))
    for i in range(len(doc)):
        arr = np.asarray(doc[i].render(scale=0.5).to_pil().convert("RGB"))
        assert arr.size > 0
    assert len(doc) == 4
    doc.close()


def test_preserve_colorspace_device_rgb_or_gray_8bit(tmp_path):
    gray = _content_image().convert("L").convert("RGB")
    task, _ = make_task(tmp_path, [{"orig": solid((10, 10, 10)), "proc": _content_image()},
                                   {"orig": solid((20, 20, 20)), "proc": gray}])
    out, _ = export(task, tmp_path, color_mode="preserve")
    for pno, name, (label, ncomp, is_gray), bits, filt in _images(out):
        assert label in ("/DeviceRGB", "/DeviceGray"), (pno, label)
        assert bits == 8


def test_grayscale_colorspace_all_device_gray_8bit(tmp_path):
    task, _ = make_task(tmp_path, [{"orig": solid((10, 10, 10)), "proc": _content_image()},
                                   {"orig": solid((20, 20, 20)), "proc": _content_image()}])
    out, _ = export(task, tmp_path, color_mode="grayscale")
    imgs = _images(out)
    assert imgs
    for pno, name, (label, ncomp, is_gray), bits, filt in imgs:
        assert is_gray and label in ("/DeviceGray", "/CalGray"), (pno, label)
        assert bits == 8


def test_filters_standard_only_no_jbig2_jpx(tmp_path):
    task, _ = make_task(tmp_path, [{"orig": solid((10, 10, 10)), "proc": _content_image()},
                                   {"orig": solid((20, 20, 20)), "proc": _content_image()}])
    out, _ = export(task, tmp_path, compression="compact")
    for pno, name, cs, bits, filt in _images(out):
        for f in filt.replace("/", "").split("+"):
            assert f in ("DCTDecode", "FlateDecode"), (pno, filt)


def test_catalog_image_based_no_form(tmp_path):
    task, _ = make_task(tmp_path, [{"orig": solid((10, 10, 10)), "proc": _content_image()}])
    out, _ = export(task, tmp_path)
    with pikepdf.open(out) as pk:
        assert "/Pages" in pk.Root
        assert "/AcroForm" not in pk.Root  # 无表单
        assert len(pk.pages) == 1


def test_metadata_docinfo_and_xmp_readable(tmp_path):
    task, _ = make_task(tmp_path, [{"orig": solid((10, 10, 10)), "proc": _content_image()}])
    out, _ = export(task, tmp_path, title="兼容测试")
    with pikepdf.open(out) as pk:
        assert str(pk.docinfo.get("/Title", "")) == "兼容测试"
        assert "BookClean" in str(pk.docinfo.get("/Producer", ""))
        assert "/Metadata" in pk.Root
        with pk.open_metadata() as meta:
            assert meta.get("dc:title") == "兼容测试"


def test_mixed_orientation_and_size_render(tmp_path):
    specs = [
        {"orig": solid((10, 10, 10)), "proc": _content_image((200, 280))},   # 竖
        {"orig": solid((10, 10, 10)), "proc": _content_image((280, 200))},   # 横
        {"orig": solid((10, 10, 10)), "proc": _content_image((120, 160))},   # 小
    ]
    task, _ = make_task(tmp_path, specs)
    out, info = export(task, tmp_path)
    doc = pdfium.PdfDocument(str(out))
    sizes = [doc[i].get_size() for i in range(len(doc))]
    doc.close()
    assert sizes[0][1] > sizes[0][0]  # 竖
    assert sizes[1][0] > sizes[1][1]  # 横
    assert len(sizes) == 3


def test_linearized_option_valid_and_renders(tmp_path):
    task, _ = make_task(tmp_path, [{"orig": solid((10, 10, 10)), "proc": _content_image()}
                                   for _ in range(3)])
    out, _ = export(task, tmp_path, linearize=True)
    with pikepdf.open(out) as pk:
        assert pk.is_linearized
    doc = pdfium.PdfDocument(str(out))
    for i in range(len(doc)):
        doc[i].render(scale=0.5)
    assert len(doc) == 3
    doc.close()


def test_object_streams_pdf15(tmp_path):
    task, _ = make_task(tmp_path, [{"orig": solid((10, 10, 10)), "proc": _content_image()}])
    out, _ = export(task, tmp_path)
    with pikepdf.open(out) as pk:
        objstm = sum(1 for o in pk.objects
                     if _is_objstm(o))
        assert pk.pdf_version == "1.5"
        assert objstm >= 1  # ObjectStreamMode.generate


def _is_objstm(o):
    try:
        return o.get("/Type") == pikepdf.Name("/ObjStm")
    except Exception:
        return False


def test_grayscale_pdfium_render_channel_consistent(tmp_path):
    task, _ = make_task(tmp_path, [{"orig": solid((10, 10, 10)), "proc": _content_image()}])
    out, _ = export(task, tmp_path, color_mode="grayscale")
    doc = pdfium.PdfDocument(str(out))
    for i in range(len(doc)):
        rgb = np.asarray(doc[i].render(scale=0.5).to_pil().convert("RGB")).astype(int)
        d = max(np.abs(rgb[:, :, 0] - rgb[:, :, 1]).max(),
                np.abs(rgb[:, :, 1] - rgb[:, :, 2]).max())
        assert d <= 1  # 灰度渲染三通道一致
    doc.close()