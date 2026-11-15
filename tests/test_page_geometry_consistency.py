"""M4.8.2 页面尺寸/方向一致性测试。

填补 M4.1 缺口：旋转 180/270、跨页（双页展开宽图）、Custom 横向、
单 PDF 混合横竖逐页 MediaBox 与页序、跨页内容不丢失/不拉伸（渲染核验）。
现有 page_layout 实现已充分，本阶段仅补测试，不改源码。
"""

import json

import numpy as np
import pypdfium2 as pdfium
import pytest
from PIL import Image

from bookclean.config import PDFExportConfig
from bookclean.page_layout import MM_TO_PT, PageLayoutResolver
from bookclean.pdf_exporter import PDFExporter

A4_W, A4_H = 210 * MM_TO_PT, 297 * MM_TO_PT


def solid(color, size=(60, 80)):
    return Image.new("RGB", size, color)


def make_task(tmp_path, specs):
    task = tmp_path / "task"
    (task / "original_pages").mkdir(parents=True)
    (task / "processed_pages").mkdir(parents=True)
    manifest = {"task_id": "t_geo", "pages": [], "inputs": []}
    for i, s in enumerate(specs):
        dpi = s.get("dpi", 150.0)
        rec = {
            "page_index": i, "display_index": i + 1,
            "source_kind": s.get("kind", "image"),
            "source_path": str(tmp_path / f"input_{i}.png"),
            "status": "completed", "image_path": None, "processed_image_path": None,
            "processing_status": "completed", "dpi": dpi,
            "metadata": s.get("metadata", {}),
            "final_image_source": "enhanced", "fallback_level": "none",
            "processing_decision": "accept",
        }
        for key, sub in (("orig", "original_pages"), ("proc", "processed_pages")):
            if s.get(key) is not None:
                rel = f"{sub}/page_{i + 1:06d}.png"
                s[key].save(task / rel, dpi=(dpi, dpi))
                rec["image_path" if key == "orig" else "processed_image_path"] = rel
        manifest["pages"].append(rec)
        manifest["inputs"].append({"path": rec["source_path"], "kind": rec["source_kind"]})
    (task / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return task


def export(task, tmp_path, name="out.pdf", **cfg):
    out = tmp_path / name
    info = PDFExporter(PDFExportConfig(**cfg)).export(task, out)
    return out, info


def resolver(**kwargs):
    return PageLayoutResolver(PDFExportConfig(**kwargs))


def image_page(path="x.png", **extra):
    return {"page_index": 0, "source_kind": "image", "source_path": path,
            "dpi": None, "metadata": {}, **extra}


def pdf_page(w_pt, h_pt, rotation=0, **extra):
    return {"page_index": 0, "source_kind": "pdf", "source_path": "book.pdf", "dpi": 150.0,
            "metadata": {"width_pt": w_pt, "height_pt": h_pt, "rotation": rotation}, **extra}


def _sizes(path):
    doc = pdfium.PdfDocument(str(path))
    sizes = [doc[i].get_size() for i in range(len(doc))]
    doc.close()
    return sizes


# ---------- 旋转 180/270（get_size 已应用 /Rotate，不得二次交换） ----------

def test_pdf_rotation_180_no_swap():
    lay = resolver().resolve(pdf_page(480.0, 742.0, rotation=180), 1000, 1546)
    assert (lay.page_width_pt, lay.page_height_pt) == (480.0, 742.0)


def test_pdf_rotation_270_no_double_swap():
    lay = resolver().resolve(pdf_page(742.0, 487.0, rotation=270), 1546, 1015)
    assert (lay.page_width_pt, lay.page_height_pt) == (742.0, 487.0)


@pytest.mark.parametrize("rot,w,h", [(0, 480, 742), (90, 742, 480), (180, 480, 742), (270, 742, 480)])
def test_pdf_all_rotations_keep_visual_size(rot, w, h):
    lay = resolver().resolve(pdf_page(float(w), float(h), rotation=rot), w * 2, h * 2)
    assert lay.page_width_pt == pytest.approx(w, abs=0.01)
    assert lay.page_height_pt == pytest.approx(h, abs=0.01)


# ---------- 跨页（双页展开宽图） ----------

def test_spread_preserve_landscape_and_aspect(img_factory):
    path = img_factory("spread.png", size=(2000, 1000), dpi=(150, 150))
    lay = resolver().resolve(image_page(path, dpi=150.0), 2000, 1000)
    assert lay.page_width_pt > lay.page_height_pt  # 横向物理尺寸
    assert lay.image_display_width_pt / lay.image_display_height_pt == pytest.approx(2.0, rel=1e-6)


def test_spread_a4_auto_landscape_no_stretch():
    lay = resolver(page_size_mode="A4").resolve(image_page(), 2000, 1000)
    assert lay.page_width_pt == pytest.approx(A4_H, abs=0.01)  # 横版 A4
    assert lay.page_height_pt == pytest.approx(A4_W, abs=0.01)
    assert lay.image_display_width_pt / lay.image_display_height_pt == pytest.approx(2.0, rel=1e-6)
    assert lay.image_display_width_pt <= lay.page_width_pt + 0.01
    assert lay.image_display_height_pt <= lay.page_height_pt + 0.01


def test_custom_landscape_portrait_image_no_stretch():
    lay = resolver(page_size_mode="custom", custom_width_mm=297.0, custom_height_mm=210.0).resolve(
        image_page(), 600, 800)
    assert lay.page_width_pt > lay.page_height_pt
    assert lay.image_display_width_pt / lay.image_display_height_pt == pytest.approx(600 / 800, rel=1e-6)


def test_mixed_orientation_batch_order():
    r = resolver(page_size_mode="A4")
    specs = [(600, 800), (800, 600), (500, 500), (1200, 500)]
    layouts = [r.resolve(image_page(page_index=i), w, h) for i, (w, h) in enumerate(specs)]
    assert [l.page_index for l in layouts] == [0, 1, 2, 3]  # 页序保持
    assert layouts[0].page_height_pt > layouts[0].page_width_pt   # 竖
    assert layouts[1].page_width_pt > layouts[1].page_height_pt   # 横
    assert layouts[3].page_width_pt > layouts[3].page_height_pt   # 跨页宽图->横


# ---------- 端到端：单 PDF 混合横竖 + 页序 + MediaBox ----------

def test_export_mixed_orientation_mediabox_and_order(tmp_path):
    specs = [
        {"orig": solid((200, 30, 30), (60, 80)), "proc": solid((200, 30, 30), (60, 80))},   # 竖
        {"orig": solid((30, 200, 30), (80, 60)), "proc": solid((30, 200, 30), (80, 60))},   # 横
        {"orig": solid((30, 30, 200), (60, 80)), "proc": solid((30, 30, 200), (60, 80))},   # 竖
    ]
    task = make_task(tmp_path, specs)
    out, _ = export(task, tmp_path)
    sizes = _sizes(out)
    assert len(sizes) == 3
    assert sizes[0][1] > sizes[0][0]  # 竖
    assert sizes[1][0] > sizes[1][1]  # 横
    assert sizes[2][1] > sizes[2][0]  # 竖
    # 页序：渲染主色与输入顺序一致
    doc = pdfium.PdfDocument(str(out))
    means = [np.asarray(doc[i].render(scale=0.5).to_pil().convert("RGB")).reshape(-1, 3).mean(axis=0)
             for i in range(3)]
    doc.close()
    assert means[0][0] > means[0][1] and means[0][0] > means[0][2]  # 红
    assert means[1][1] > means[1][0] and means[1][1] > means[1][2]  # 绿
    assert means[2][2] > means[2][0] and means[2][2] > means[2][1]  # 蓝


def test_export_a4_landscape_mediabox(tmp_path):
    specs = [{"orig": solid((120, 120, 120), (200, 100)), "proc": solid((120, 120, 120), (200, 100))}]
    task = make_task(tmp_path, specs)
    out, _ = export(task, tmp_path, page_size_mode="A4")
    w, h = _sizes(out)[0]
    assert w == pytest.approx(A4_H, abs=0.5)
    assert h == pytest.approx(A4_W, abs=0.5)


def test_export_spread_content_not_lost_no_stretch(tmp_path):
    # 跨页宽图：左半红 右半蓝，各为正方形；导出后两色均在且红块不拉伸
    spread = Image.new("RGB", (200, 100), (255, 255, 255))
    spread.paste(solid((220, 40, 40), (100, 100)), (0, 0))
    spread.paste(solid((40, 40, 220), (100, 100)), (100, 0))
    task = make_task(tmp_path, [{"orig": spread, "proc": spread}])
    out, _ = export(task, tmp_path, page_size_mode="A4")
    doc = pdfium.PdfDocument(str(out))
    arr = np.asarray(doc[0].render(scale=1.0).to_pil().convert("RGB"))
    doc.close()
    red = (arr[:, :, 0] > 150) & (arr[:, :, 1] < 100) & (arr[:, :, 2] < 100)
    blue = (arr[:, :, 2] > 150) & (arr[:, :, 0] < 100) & (arr[:, :, 1] < 100)
    assert red.sum() > 0 and blue.sum() > 0  # 左右内容均保留，未丢失/未拆分
    ys, xs = np.where(red)
    assert abs((xs.max() - xs.min()) - (ys.max() - ys.min())) <= 2  # 红块正方形=不拉伸
    assert xs.mean() < arr.shape[1] / 2  # 红在左
    ys2, xs2 = np.where(blue)
    assert xs2.mean() > arr.shape[1] / 2      # 蓝在右


def test_export_custom_landscape_square_no_stretch(tmp_path):
    specs = [{"orig": solid((200, 30, 30), (80, 80)), "proc": solid((200, 30, 30), (80, 80))}]
    task = make_task(tmp_path, specs)
    out, _ = export(task, tmp_path, page_size_mode="custom", custom_width_mm=297.0, custom_height_mm=210.0)
    w, h = _sizes(out)[0]
    assert w > h
    doc = pdfium.PdfDocument(str(out))
    arr = np.asarray(doc[0].render(scale=1.0).to_pil().convert("RGB"))
    doc.close()
    assert tuple(arr[2, 2]) == (255, 255, 255)  # 留白纯白
    red = (arr[:, :, 0] > 150) & (arr[:, :, 1] < 100) & (arr[:, :, 2] < 100)
    ys, xs = np.where(red)
    assert abs((xs.max() - xs.min()) - (ys.max() - ys.min())) <= 2  # 方形不拉伸


def test_export_pdf_rotation90_preserved(tmp_path):
    specs = [{"kind": "pdf", "orig": solid((90, 90, 90), (140, 100)), "proc": solid((90, 90, 90), (140, 100)),
              "metadata": {"width_pt": 742.0, "height_pt": 487.0, "rotation": 90}}]
    task = make_task(tmp_path, specs)
    out, _ = export(task, tmp_path)
    w, h = _sizes(out)[0]
    assert w == pytest.approx(742.0, abs=0.5)
    assert h == pytest.approx(487.0, abs=0.5)