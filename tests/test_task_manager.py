"""D. 标准化输出、manifest、缩略图、原件保护测试。"""

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from bookclean.task_manager import ImportTaskManager


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mixed_import_output(pdf_factory, img_factory, tmp_path):
    pdf = pdf_factory([Image.new("RGB", (100, 200), (255, 0, 0))] * 2, name="book.pdf")
    jpg = img_factory("photo.jpg", size=(60, 40), color=(0, 255, 0))
    tiff = tmp_path / "multi.tif"
    frames = [Image.new("RGB", (30, 50), (0, 0, 255))] * 3
    frames[0].save(tiff, save_all=True, append_images=frames[1:])

    sources = [pdf, jpg, tiff]
    before = {p: sha256(p) for p in sources}

    temp_root = tmp_path / "temp"
    result = ImportTaskManager(temp_root=temp_root).run(sources)
    manifest = result["manifest"]
    task_dir = result["task_dir"]

    assert manifest["status"] == "completed"
    assert manifest["total_pages"] == 6
    assert manifest["completed_pages"] == 6
    assert manifest["failed_pages"] == []
    assert [p["page_index"] for p in manifest["pages"]] == list(range(6))
    assert [p["source_page_index"] for p in manifest["pages"]] == [0, 1, 0, 0, 1, 2]
    assert [p["display_index"] for p in manifest["pages"]] == list(range(1, 7))

    for entry in manifest["pages"]:
        image_path = task_dir / entry["image_path"]
        thumb_path = task_dir / entry["thumbnail_path"]
        assert image_path.is_file() and thumb_path.is_file()
        arr = np.asarray(Image.open(image_path))
        assert arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[2] == 3
        thumb = Image.open(thumb_path)
        assert max(thumb.size) <= 300
        assert thumb.format == "JPEG"

    pdf_page = Image.open(task_dir / manifest["pages"][0]["image_path"])
    assert pdf_page.info["dpi"][0] == pytest.approx(300, abs=0.01)

    assert (task_dir / "original_pages").is_dir()
    assert (task_dir / "thumbnails").is_dir()
    assert (task_dir / "logs").is_dir()
    assert (task_dir / manifest["contact_sheet"]).is_file()
    assert not list(task_dir.rglob("*.tmp"))

    json.loads(result["manifest_path"].read_text(encoding="utf-8"))
    assert {p: sha256(p) for p in sources} == before  # 原件未被修改


def test_page_order_matches_input(pdf_factory, tmp_path):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    pdf = pdf_factory([Image.new("RGB", (40, 20), c) for c in colors])
    result = ImportTaskManager(temp_root=tmp_path / "temp").run([pdf])
    for entry, color in zip(result["manifest"]["pages"], colors):
        arr = np.asarray(Image.open(result["task_dir"] / entry["image_path"]))
        assert tuple(arr[10, 20]) == color


def test_failed_page_continues(img_factory, tmp_path):
    good1 = img_factory("a_good.png", color=(255, 0, 0))
    bad = tmp_path / "b_bad.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\nbroken")
    good2 = img_factory("c_good.png", color=(0, 255, 0))

    result = ImportTaskManager(temp_root=tmp_path / "temp").run([tmp_path])
    manifest = result["manifest"]
    assert manifest["status"] == "completed_with_errors"
    assert manifest["total_pages"] == 2  # 损坏文件在 collect 阶段被记录，无页数
    assert manifest["completed_pages"] == 2
    assert len(manifest["file_errors"]) == 1
    assert "b_bad.png" in manifest["file_errors"][0]
    assert [p["display_index"] for p in manifest["pages"]] == [1, 2]


def test_empty_folder(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    result = ImportTaskManager(temp_root=tmp_path / "temp").run([folder])
    assert result["manifest"]["status"] == "failed"
    assert result["manifest"]["total_pages"] == 0


def test_task_dirs_unique(img_factory, tmp_path):
    img = img_factory("a.png")
    manager = ImportTaskManager(temp_root=tmp_path / "temp")
    r1 = manager.run([img])
    r2 = manager.run([img])
    assert r1["task_dir"] != r2["task_dir"]