"""A. PDF 读取与渲染测试。"""

import numpy as np
import pikepdf
import pytest
from PIL import Image

from bookclean.errors import PDFCorruptError, PDFError, PDFPasswordError
from bookclean.pdf_reader import PDFReader


def test_single_page_count(pdf_factory):
    path = pdf_factory([Image.new("RGB", (100, 200))])
    assert PDFReader().get_page_count(path) == 1


def test_multi_page_count(pdf_factory):
    path = pdf_factory([Image.new("RGB", (100, 200))] * 3)
    assert PDFReader().get_page_count(path) == 3


def test_render_300dpi_size_and_format(pdf_factory):
    path = pdf_factory([Image.new("RGB", (100, 200), (255, 0, 0))])
    page = PDFReader().render_page(path, 0, dpi=300)
    assert page.image.shape == (200, 100, 3)
    assert page.image.dtype == np.uint8
    assert page.image.flags["C_CONTIGUOUS"]
    assert page.color_mode == "RGB"
    assert page.source_kind == "pdf"
    assert page.source_dpi == 300.0
    assert page.metadata["render_dpi"] == 300
    assert tuple(page.image[100, 50]) == (255, 0, 0)


def test_render_custom_dpi(pdf_factory):
    path = pdf_factory([Image.new("RGB", (100, 200))], dpi=300)
    page = PDFReader().render_page(path, 0, dpi=150)
    assert page.image.shape == (100, 50, 3)


def test_different_page_sizes(pdf_factory):
    path = pdf_factory([Image.new("RGB", (100, 200)), Image.new("RGB", (200, 100))])
    reader = PDFReader()
    p0 = reader.render_page(path, 0)
    p1 = reader.render_page(path, 1)
    assert p0.image.shape == (200, 100, 3)
    assert p1.image.shape == (100, 200, 3)
    info = reader.get_page_info(path, 0)
    assert info["width_pt"] == pytest.approx(24.0)
    assert info["height_pt"] == pytest.approx(48.0)


def test_page_rotation_mark(pdf_factory, tmp_path):
    path = pdf_factory([Image.new("RGB", (100, 200), (0, 0, 255))])
    rotated = tmp_path / "rotated.pdf"
    with pikepdf.open(path) as pdf:
        pdf.pages[0]["/Rotate"] = 90
        pdf.save(rotated)
    reader = PDFReader()
    info = reader.get_page_info(rotated, 0)
    assert info["rotation"] == 90
    assert info["width_pt"] == pytest.approx(48.0)  # get_size 返回应用旋转后的视觉尺寸
    assert info["height_pt"] == pytest.approx(24.0)
    page = reader.render_page(rotated, 0)
    assert page.image.shape == (100, 200, 3)  # 渲染自动应用旋转
    assert page.metadata["rotation"] == 90


def test_corrupt_pdf(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 broken content")
    with pytest.raises(PDFCorruptError):
        PDFReader().get_page_count(bad)


def test_encrypted_pdf(pdf_factory, tmp_path):
    path = pdf_factory([Image.new("RGB", (100, 200))])
    enc = tmp_path / "enc.pdf"
    with pikepdf.open(path) as pdf:
        pdf.save(enc, encryption=pikepdf.Encryption(owner="ownerpw", user="userpw", R=4))
    with pytest.raises(PDFPasswordError):
        PDFReader().get_page_count(enc)
    with pytest.raises(PDFPasswordError):
        PDFReader(password="wrong").get_page_count(enc)
    reader = PDFReader(password="userpw")
    assert reader.get_page_count(enc) == 1
    assert reader.render_page(enc, 0).image.shape == (200, 100, 3)


def test_zero_page_pdf(tmp_path):
    empty = tmp_path / "empty.pdf"
    with pikepdf.new() as pdf:
        pdf.save(empty)
    with pytest.raises(PDFError):
        PDFReader().get_page_count(empty)


def test_missing_pdf(tmp_path):
    with pytest.raises(PDFError):
        PDFReader().get_page_count(tmp_path / "nope.pdf")


def test_page_index_out_of_range(pdf_factory):
    path = pdf_factory([Image.new("RGB", (100, 200))])
    with pytest.raises(PDFError):
        PDFReader().render_page(path, 5)


def test_iter_pages_order_and_metadata(pdf_factory):
    images = [Image.new("RGB", (100, 200), (i, i, i)) for i in (10, 20, 30)]
    path = pdf_factory(images)
    pages = list(PDFReader().iter_pages(path))
    assert [p.page_index for p in pages] == [0, 1, 2]
    assert [int(p.image[100, 50][0]) for p in pages] == [10, 20, 30]
    assert all(p.metadata["source_page_index"] == p.page_index for p in pages)