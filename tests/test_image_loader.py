"""B. 图片读取、色彩模式、EXIF、多帧 TIFF 测试。"""

import numpy as np
import pytest
from PIL import Image

from bookclean.errors import ImageLoadError
from bookclean.image_loader import ImageLoader


def assert_standard(page, width, height):
    assert page.image.shape == (height, width, 3)
    assert page.image.dtype == np.uint8
    assert page.image.flags["C_CONTIGUOUS"]
    assert page.color_mode == "RGB"
    assert page.source_kind == "image"


@pytest.mark.parametrize("name,fmt", [("a.jpg", "JPEG"), ("a.png", "PNG"), ("a.bmp", "BMP"), ("a.tif", "TIFF")])
def test_common_formats(img_factory, name, fmt):
    path = img_factory(name, size=(40, 20), color=(10, 200, 30))
    page = ImageLoader().load_image(path)
    assert_standard(page, 40, 20)
    assert page.metadata["format"] == fmt
    assert tuple(page.image[10, 20]) == (10, 200, 30)


def test_rgba_composited_on_white(tmp_path):
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (4, 4), (255, 0, 0, 128)).save(path)
    page = ImageLoader().load_image(path)
    assert_standard(page, 4, 4)
    r, g, b = page.image[2, 2]
    assert abs(int(r) - 255) <= 2 and abs(int(g) - 128) <= 2 and abs(int(b) - 128) <= 2


def test_grayscale(img_factory):
    path = img_factory("gray.png", mode="L", color=128)
    page = ImageLoader().load_image(path)
    assert_standard(page, 40, 20)
    assert tuple(page.image[10, 20]) == (128, 128, 128)


def test_palette(img_factory):
    path = img_factory("pal.png", mode="P", color=1)
    page = ImageLoader().load_image(path)
    assert_standard(page, 40, 20)


def test_cmyk_jpeg(tmp_path):
    path = tmp_path / "cmyk.jpg"
    Image.new("CMYK", (40, 20), (255, 0, 0, 0)).save(path)
    page = ImageLoader().load_image(path)
    assert_standard(page, 40, 20)
    r, g, b = (int(v) for v in page.image[10, 20])
    assert r < 60 and g > 195 and b > 195  # 青 -> RGB 近似 (0,255,255)


def test_exif_orientation_rotates_pixels(tmp_path):
    path = tmp_path / "exif.jpg"
    im = Image.new("RGB", (40, 20))
    for x in range(40):
        for y in range(20):
            im.putpixel((x, y), (255, 0, 0) if x < 20 else (0, 0, 255))
    exif = Image.Exif()
    exif[274] = 6
    im.save(path, exif=exif)

    loader = ImageLoader()
    assert loader.get_image_info(path)["exif_orientation"] == 6
    page = loader.load_image(path)
    assert_standard(page, 20, 40)  # 像素真实旋转，不只是改元数据
    assert page.image[5, 10] == pytest.approx((255, 0, 0), abs=3)  # JPEG 有损容差
    assert page.image[35, 10] == pytest.approx((0, 0, 255), abs=3)
    assert page.metadata["raw_width"] == 40
    assert page.metadata["exif_orientation"] == 6


def test_multi_page_tiff(tmp_path):
    path = tmp_path / "multi.tif"
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    frames = [Image.new("RGB", (40, 20), c) for c in colors]
    frames[0].save(path, save_all=True, append_images=frames[1:])

    loader = ImageLoader()
    assert loader.get_image_info(path)["n_frames"] == 3
    pages = list(loader.iter_image_frames(path))
    assert len(pages) == 3
    assert [p.page_index for p in pages] == [0, 1, 2]
    for page, color in zip(pages, colors):
        assert tuple(page.image[10, 20]) == color
        assert page.metadata["source_page_index"] == page.page_index


def test_corrupt_image(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\nbroken")
    with pytest.raises(ImageLoadError):
        ImageLoader().load_image(bad)


def test_missing_image(tmp_path):
    with pytest.raises(ImageLoadError):
        ImageLoader().load_image(tmp_path / "nope.jpg")


def test_decompression_bomb_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    path = tmp_path / "huge.png"
    Image.new("RGB", (30, 30)).save(path)  # 900 像素 > 2 * 100，触发安全上限
    with pytest.raises(ImageLoadError, match="安全"):
        ImageLoader().load_image(path)


def test_dpi_metadata_preserved(img_factory):
    path = img_factory("dpi.png", dpi=(300, 300))
    page = ImageLoader().load_image(path)
    assert page.source_dpi == pytest.approx(300.0, abs=0.01)  # PNG pHYs 整数米制换算容差