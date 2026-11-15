"""C. 输入收集与自然排序测试。"""

import pytest
from PIL import Image

from bookclean.errors import InputError, UnsupportedFormatError
from bookclean.input_adapter import InputAdapter
from bookclean.utils.natural_sort import natural_sorted


def test_natural_sort_basic():
    names = ["page10.jpg", "page2.jpg", "page1.jpg", "page20.jpg", "page11.jpg"]
    assert natural_sorted(names) == ["page1.jpg", "page2.jpg", "page10.jpg", "page11.jpg", "page20.jpg"]


def test_natural_sort_mixed():
    assert natural_sorted(["img2.png", "img10.png", "IMG1.png"]) == ["IMG1.png", "img2.png", "img10.png"]


def test_folder_natural_sort_and_filter(img_factory, tmp_path):
    for name in ("page10.jpg", "page2.jpg", "page1.jpg", "page20.jpg"):
        img_factory(name)
    (tmp_path / "notes.txt").write_text("not an image")
    (tmp_path / "data.xyz").write_bytes(b"data")
    inputs = InputAdapter().collect([tmp_path])
    assert [f.path.name for f in inputs] == ["page1.jpg", "page2.jpg", "page10.jpg", "page20.jpg"]
    assert all(f.kind == "image" for f in inputs)


def test_case_insensitive_extensions(img_factory, pdf_factory):
    img_factory("Photo.JPG")
    img_factory("scan.Png")
    pdf_factory([Image.new("RGB", (40, 40))], name="book.PdF")
    folder = img_factory("x.png").parent
    inputs = InputAdapter().collect([folder])
    assert {f.kind for f in inputs} == {"image", "pdf"}
    assert len(inputs) == 4  # 含 x.png


def test_explicit_file_order_preserved(pdf_factory):
    b = pdf_factory([Image.new("RGB", (40, 40))], name="b.pdf")
    a = pdf_factory([Image.new("RGB", (40, 40))], name="a.pdf")
    inputs = InputAdapter().collect([b, a])
    assert [f.path.name for f in inputs] == ["b.pdf", "a.pdf"]


def test_unsupported_explicit_file(tmp_path):
    bad = tmp_path / "file.xyz"
    bad.write_bytes(b"data")
    with pytest.raises(UnsupportedFormatError):
        InputAdapter().collect([bad])


def test_missing_path(tmp_path):
    with pytest.raises(InputError):
        InputAdapter().collect([tmp_path / "nope"])


def test_global_page_index_across_files(pdf_factory, img_factory, tmp_path):
    pdf = pdf_factory([Image.new("RGB", (40, 40))] * 2)
    tiff = tmp_path / "multi.tif"
    frames = [Image.new("RGB", (10, 10))] * 3
    frames[0].save(tiff, save_all=True, append_images=frames[1:])
    jpg = img_factory("single.jpg")

    adapter = InputAdapter()
    inputs = adapter.collect([pdf, tiff, jpg])
    pages = list(adapter.iter_pages(inputs))
    assert [p.page_index for p in pages] == list(range(6))
    assert [p.metadata["source_page_index"] for p in pages] == [0, 1, 0, 1, 2, 0]
    assert [p.source_kind for p in pages] == ["pdf", "pdf", "image", "image", "image", "image"]


def test_file_level_error_does_not_crash_batch(img_factory, tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 broken")
    good = img_factory("good.jpg")
    adapter = InputAdapter()
    inputs = adapter.collect([bad, good])
    assert inputs[0].error is not None
    assert inputs[1].error is None

    errors = []
    pages = list(adapter.iter_pages(inputs, on_error=lambda idx, exc: errors.append((idx, exc))))
    assert len(pages) == 1
    assert pages[0].page_index == 0
    assert len(errors) == 1 and errors[0][0] is None