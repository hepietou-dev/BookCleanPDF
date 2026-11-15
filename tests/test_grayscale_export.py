"""M4.7.3 PDFExporter 灰度集成测试：混合页书 preserve vs grayscale / 页数顺序尺寸 / DeviceGray / 同目录多次导出不改原始。"""

import numpy as np
import pikepdf
import pypdfium2 as pdfium
import pytest
from PIL import Image

from bookclean.config import PDFExportConfig
from bookclean.pdf_exporter import PDFExporter
from test_pdf_exporter import make_task


def mixed_specs():
    grad = np.tile(np.linspace(0, 255, 100).astype(np.uint8), (140, 1))
    return [
        {"orig": Image.new("RGB", (100, 140), (200, 30, 30)), "proc": Image.new("RGB", (100, 140), (200, 30, 30))},   # 彩色封面
        {"orig": Image.new("RGB", (100, 140), (120, 120, 120)), "proc": Image.new("RGB", (100, 140), (120, 120, 120))},  # 黑白正文(RGB等通道)
        {"orig": Image.new("RGB", (140, 100), (30, 120, 200)), "proc": Image.new("RGB", (140, 100), (30, 120, 200))},   # 彩色地图(横)
        {"orig": Image.fromarray(grad), "proc": Image.fromarray(grad)},                                                  # 灰度照片(L)
        {"orig": Image.new("RGB", (100, 140), (255, 0, 0)), "proc": Image.new("RGB", (100, 140), (255, 0, 0))},          # 红色批注
        {"orig": Image.new("RGB", (100, 140), (0, 150, 0)), "proc": Image.new("RGB", (100, 140), (0, 150, 0))},          # 彩色表格
        {"orig": Image.new("L", (100, 140), 90), "proc": Image.new("L", (100, 140), 90)},                                # 已灰度(L)
        {"kind": "pdf", "orig": Image.new("RGB", (80, 120), (10, 20, 30)), "proc": Image.new("RGB", (80, 120), (10, 20, 30)),
         "metadata": {"width_pt": 400.0, "height_pt": 600.0}},                                                          # 不同物理尺寸
    ]


def export(task, tmp_path, name, **cfg):
    out = tmp_path / name
    info = PDFExporter(PDFExportConfig(**cfg)).export(task, out)
    return out, info


def image_colorspaces(path):
    cs = []
    with pikepdf.open(path) as pdf:
        for page in pdf.pages:
            for name, obj in page.get("/Resources", {}).get("/XObject", {}).items():
                if obj.get("/Subtype") == pikepdf.Name("/Image"):
                    cs.append((str(obj.get("/ColorSpace")), int(obj.get("/BitsPerComponent", 0))))
    return cs


def test_preserve_keeps_color_and_gray(tmp_path):
    task, _ = make_task(tmp_path, mixed_specs())
    out, info = export(task, tmp_path, "preserve.pdf", color_mode="preserve")
    modes = [p["color_mode"] for p in info["pages"]]
    assert modes.count("RGB") == 5   # 封面/地图/批注/表格/尺寸页（真彩色）
    assert modes.count("L") == 3     # 黑白正文(RGB等通道)/灰度照片/已灰度
    assert info["color_mode"] == "preserve"
    assert info["grayscale_conversion"] is None


def test_grayscale_all_pages_single_channel(tmp_path):
    task, _ = make_task(tmp_path, mixed_specs())
    out, info = export(task, tmp_path, "gray.pdf", color_mode="grayscale")
    assert all(p["color_mode"] == "L" for p in info["pages"])
    s = info["summary"]
    assert s["grayscale_pages"] == 8 and s["color_pages"] == 0
    gc = info["grayscale_conversion"]
    assert gc["failed_pages"] == 0
    assert gc["converted_pages"] + gc["already_grayscale_pages"] == 8
    assert gc["source_rgb_pages"] + gc["source_gray_pages"] == 8
    assert gc["bit_depth"] == 8


def test_grayscale_pdf_objects_devicegray_8bit(tmp_path):
    task, _ = make_task(tmp_path, mixed_specs())
    out, info = export(task, tmp_path, "gray.pdf", color_mode="grayscale")
    cs = image_colorspaces(out)
    assert len(cs) == 8
    assert all(c == "/DeviceGray" for c, b in cs)
    assert all(b == 8 for c, b in cs)


def test_preserve_pdf_has_rgb_objects(tmp_path):
    task, _ = make_task(tmp_path, mixed_specs())
    out, info = export(task, tmp_path, "preserve.pdf", color_mode="preserve")
    cs = image_colorspaces(out)
    assert any(c == "/DeviceRGB" for c, b in cs)  # preserve 保留彩色对象


def test_count_order_size_match_between_modes(tmp_path):
    task, _ = make_task(tmp_path, mixed_specs())
    _, ip = export(task, tmp_path, "p.pdf", color_mode="preserve")
    _, ig = export(task, tmp_path, "g.pdf", color_mode="grayscale")
    assert len(ip["pages"]) == len(ig["pages"]) == 8
    for a, b in zip(ip["pages"], ig["pages"]):
        assert a["display_index"] == b["display_index"]
        assert a["output_page_width_pt"] == b["output_page_width_pt"]
        assert a["output_page_height_pt"] == b["output_page_height_pt"]
    # 不同物理尺寸页保持 400x600
    assert ig["pages"][7]["output_page_width_pt"] == 400.0
    assert ig["pages"][7]["output_page_height_pt"] == 600.0


def test_grayscale_render_channels_equal(tmp_path):
    task, _ = make_task(tmp_path, mixed_specs())
    out, _ = export(task, tmp_path, "g.pdf", color_mode="grayscale")
    doc = pdfium.PdfDocument(str(out))
    for i in range(len(doc)):
        rgb = np.asarray(doc[i].render(scale=1.0).to_pil().convert("RGB")).astype(int)
        assert np.abs(rgb[:, :, 0] - rgb[:, :, 1]).max() <= 1
        assert np.abs(rgb[:, :, 1] - rgb[:, :, 2]).max() <= 1
    doc.close()


def test_same_task_dir_six_exports_originals_untouched(tmp_path):
    task, _ = make_task(tmp_path, mixed_specs())
    files = sorted(task.rglob("*.png"))
    before = {f: f.read_bytes() for f in files}
    for comp in ("lossless", "balanced", "compact"):
        for cm in ("preserve", "grayscale"):
            out, info = export(task, tmp_path, f"{cm}_{comp}.pdf",
                               compression=comp, color_mode=cm)
            assert out.exists() and info["export_status"] == "completed"
    for f in files:
        assert f.read_bytes() == before[f]  # 原始/处理结果未被修改
    assert len(list(tmp_path.glob("*.pdf"))) == 6


@pytest.mark.parametrize("comp", ["lossless", "balanced", "compact"])
def test_grayscale_three_compressions_all_gray(tmp_path, comp):
    task, _ = make_task(tmp_path, mixed_specs())
    out, info = export(task, tmp_path, f"g_{comp}.pdf", compression=comp, color_mode="grayscale")
    assert all(p["color_mode"] == "L" for p in info["pages"])
    cs = image_colorspaces(out)
    assert all(c == "/DeviceGray" and b == 8 for c, b in cs)