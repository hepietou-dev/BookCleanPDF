"""M4.7.5 灰度验证测试：DeviceGray 通过 / RGB 失败定位 / 混合页 / 1-bit 位深 / ICCBased 分类 / Form 递归 / 导出集成。"""

import io

import pikepdf
import pytest
from PIL import Image

from bookclean.pdf_verifier import (PDFVerifier, _classify_colorspace,
                                    _iter_image_xobjects)
from test_pdf_exporter import make_task
from test_grayscale_export import mixed_specs


def _png(im):
    b = io.BytesIO(); im.save(b, format="PNG"); return b.getvalue()


def _gray_png():
    return _png(Image.new("L", (40, 60), 120))


def _rgb_png():
    return _png(Image.new("RGB", (40, 60), (200, 30, 30)))


def _mono_png():
    return _png(Image.new("1", (40, 60), 1))


def _write(tmp_path, name, data_list):
    import img2pdf
    p = tmp_path / name
    p.write_bytes(img2pdf.convert(*data_list))
    return p


def test_gray_pdf_passes(tmp_path):
    p = _write(tmp_path, "g.pdf", [_gray_png(), _gray_png()])
    r = PDFVerifier().verify_grayscale(p)
    assert r["status"] == "passed"
    assert r["device_gray_image_count"] == 2
    assert r["unexpected_rgb_image_count"] == 0
    assert r["render_channel_check"] == "passed"
    assert r["max_render_channel_diff"] <= 1
    assert r["failed_pages"] == []


def test_rgb_pdf_fails(tmp_path):
    p = _write(tmp_path, "c.pdf", [_rgb_png()])
    r = PDFVerifier().verify_grayscale(p)
    assert r["status"] == "failed"
    assert r["unexpected_rgb_image_count"] == 1
    assert any(f["page"] == 1 for f in r["failed_pages"])


def test_mixed_identifies_rgb_page(tmp_path):
    p = _write(tmp_path, "m.pdf", [_gray_png(), _rgb_png()])
    r = PDFVerifier().verify_grayscale(p)
    assert r["status"] == "failed"
    assert r["device_gray_image_count"] == 1
    assert r["unexpected_rgb_image_count"] == 1
    assert any(f["page"] == 2 for f in r["failed_pages"])


def test_mono_1bit_fails_bitdepth(tmp_path):
    p = _write(tmp_path, "mono.pdf", [_mono_png()])
    r = PDFVerifier().verify_grayscale(p)
    assert r["status"] == "failed"
    assert any("位深" in f["reason"] for f in r["failed_pages"])


# ---------- 色彩空间分类 ----------

def test_classify_devicegray():
    label, n, gray = _classify_colorspace(pikepdf.Name("/DeviceGray"))
    assert gray and n == 1


def test_classify_devicergb():
    label, n, gray = _classify_colorspace(pikepdf.Name("/DeviceRGB"))
    assert not gray and n == 3


def test_classify_iccbased_gray_and_rgb():
    s1 = pikepdf.Dictionary(N=1)
    s3 = pikepdf.Dictionary(N=3)
    l1, n1, g1 = _classify_colorspace(pikepdf.Array([pikepdf.Name("/ICCBased"), s1]))
    l3, n3, g3 = _classify_colorspace(pikepdf.Array([pikepdf.Name("/ICCBased"), s3]))
    assert g1 and n1 == 1 and "ICCBased" in l1
    assert not g3 and n3 == 3


def test_classify_calgray_and_indexed():
    assert _classify_colorspace(pikepdf.Name("/CalGray"))[2] is True
    idx = pikepdf.Array([pikepdf.Name("/Indexed"), pikepdf.Name("/DeviceRGB"), 255, b""])
    assert _classify_colorspace(idx)[2] is False


# ---------- Form XObject 递归 ----------

def test_iter_recurses_form_xobject(tmp_path):
    with pikepdf.new() as pdf:
        img = pdf.make_stream(b"\x80" * 16, pikepdf.Dictionary(
            Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Image,
            Width=4, Height=4, ColorSpace=pikepdf.Name.DeviceGray, BitsPerComponent=8))
        form = pdf.make_stream(b"q 4 0 0 4 0 0 cm /Im0 Do Q", pikepdf.Dictionary(
            Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Form,
            Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=img))))
        page = pikepdf.Page(pikepdf.Dictionary(
            Type=pikepdf.Name.Page, MediaBox=[0, 0, 4, 4],
            Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary(Fm0=form))))
        pdf.pages.append(page)
        p = tmp_path / "form.pdf"
        pdf.save(p)
    with pikepdf.open(p) as pdf:
        found = list(_iter_image_xobjects(pdf))
    assert len(found) == 1
    pno, name, obj = found[0]
    assert pno == 1 and "Fm0" in name and "Im0" in name


# ---------- 导出集成 ----------

def test_export_grayscale_validation_recorded(tmp_path):
    from bookclean.config import PDFExportConfig
    from bookclean.pdf_exporter import PDFExporter
    task, _ = make_task(tmp_path, mixed_specs())
    out = tmp_path / "g.pdf"
    info = PDFExporter(PDFExportConfig(color_mode="grayscale")).export(task, out)
    gv = info["grayscale_validation"]
    assert gv["enabled"] and gv["status"] == "passed"
    assert gv["device_gray_image_count"] == 8
    assert gv["unexpected_rgb_image_count"] == 0


def test_export_preserve_validation_disabled(tmp_path):
    from bookclean.config import PDFExportConfig
    from bookclean.pdf_exporter import PDFExporter
    task, _ = make_task(tmp_path, mixed_specs())
    out = tmp_path / "p.pdf"
    info = PDFExporter(PDFExportConfig(color_mode="preserve")).export(task, out)
    assert info["grayscale_validation"]["enabled"] is False