"""M4.3 PDF 合成测试：顺序/页数/尺寸/回退/缺失中止/混合来源/白底/覆盖与输入守卫。"""

import json

import numpy as np
import pypdfium2 as pdfium
import pytest
from PIL import Image

from bookclean.config import PDFExportConfig
from bookclean.pdf_exporter import ExportError, FinalImageResolver, PDFExporter


def solid(color, size=(60, 80)):
    return Image.new("RGB", size, color)


def make_task(tmp_path, specs):
    task = tmp_path / "task"
    (task / "original_pages").mkdir(parents=True)
    (task / "processed_pages").mkdir(parents=True)
    manifest = {"task_id": "t_test", "pages": [], "inputs": []}
    for i, s in enumerate(specs):
        dpi = s.get("dpi", 150.0)
        rec = {
            "page_index": i, "display_index": i + 1,
            "source_kind": s.get("kind", "image"),
            "source_path": str(tmp_path / f"input_{i}.png"),
            "status": "completed", "image_path": None, "processed_image_path": None,
            "processing_status": s.get("pstatus", "completed"),
            "page_type": s.get("page_type"), "dpi": dpi,
            "metadata": s.get("metadata", {}),
            "final_image_source": s.get("fis", "enhanced"),
            "fallback_level": s.get("fb", "none"),
            "processing_decision": s.get("dec", "accept"),
        }
        if s.get("orig") is not None:
            rel = f"original_pages/page_{i + 1:06d}.png"
            s["orig"].save(task / rel, dpi=(dpi, dpi))
            rec["image_path"] = rel
        if s.get("proc") is not None:
            rel = f"processed_pages/page_{i + 1:06d}.png"
            s["proc"].save(task / rel, dpi=(dpi, dpi))
            rec["processed_image_path"] = rel
        manifest["pages"].append(rec)
        manifest["inputs"].append({"path": rec["source_path"], "kind": rec["source_kind"]})
    (task / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return task, manifest


def export(task, tmp_path, name="out.pdf", **cfg):
    out = tmp_path / name
    info = PDFExporter(PDFExportConfig(**cfg)).export(task, out)
    return out, info


def page_colors(path):
    doc = pdfium.PdfDocument(str(path))
    cols = []
    for i in range(len(doc)):
        arr = np.asarray(doc[i].render(scale=0.5).to_pil().convert("RGB"))
        cols.append(tuple(int(v) for v in arr.reshape(-1, 3).mean(axis=0).round()))
    doc.close()
    return cols


# ---------- FinalImageResolver ----------

def test_resolver_completed_uses_processed(tmp_path):
    task, m = make_task(tmp_path, [{"orig": solid((10, 10, 10)), "proc": solid((200, 200, 200))}])
    fi = FinalImageResolver(task).resolve(m, m["pages"][0])
    assert fi.path.name == "page_000001.png" and "processed_pages" in str(fi.path)
    assert fi.source == "enhanced" and not fi.fallback and fi.status == "completed"


def test_resolver_fallback_geometry_flag(tmp_path):
    task, m = make_task(tmp_path, [{"orig": solid((0, 0, 0)), "proc": solid((255, 255, 255)),
                                   "fis": "geometry", "fb": "geometry", "dec": "fallback_geometry"}])
    fi = FinalImageResolver(task).resolve(m, m["pages"][0])
    assert fi.fallback and fi.source == "geometry"
    assert "processed_pages" in str(fi.path)  # 回退结果已是保存的最终图


def test_resolver_failed_falls_back_original(tmp_path):
    task, m = make_task(tmp_path, [{"orig": solid((120, 120, 120)), "proc": None, "pstatus": "failed"}])
    fi = FinalImageResolver(task).resolve(m, m["pages"][0])
    assert fi.fallback and fi.source == "original_fallback"
    assert "original_pages" in str(fi.path)


def test_resolver_missing_when_both_absent(tmp_path):
    task, m = make_task(tmp_path, [{"orig": None, "proc": None, "pstatus": "failed"}])
    fi = FinalImageResolver(task).resolve(m, m["pages"][0])
    assert fi.path is None and fi.source == "missing"


def test_resolver_completed_but_processed_deleted(tmp_path):
    task, m = make_task(tmp_path, [{"orig": solid((1, 2, 3)), "proc": solid((9, 9, 9))}])
    (task / m["pages"][0]["processed_image_path"]).unlink()
    fi = FinalImageResolver(task).resolve(m, m["pages"][0])
    assert fi.fallback and "original_pages" in str(fi.path)


# ---------- 合成 ----------

def test_single_page(tmp_path):
    task, _ = make_task(tmp_path, [{"orig": solid((10, 10, 10)), "proc": solid((200, 30, 30))}])
    out, info = export(task, tmp_path)
    doc = pdfium.PdfDocument(str(out))
    assert len(doc) == 1
    doc.close()
    assert info["summary"]["total_pages"] == 1


def test_multi_page_order_by_index(tmp_path):
    specs = [{"orig": solid((i * 60 % 256, 40, 40)), "proc": solid((i * 60 % 256, 40, 40))}
             for i in range(5)]
    task, _ = make_task(tmp_path, specs)
    out, info = export(task, tmp_path)
    cols = page_colors(out)
    assert len(cols) == 5
    for i, c in enumerate(cols):
        assert abs(c[0] - i * 60 % 256) < 6  # 顺序与页色对应，无重复/遗漏


def test_mixed_pdf_and_image_sizes(tmp_path):
    specs = [
        {"kind": "pdf", "orig": solid((255, 0, 0), (100, 140)), "proc": solid((255, 0, 0), (100, 140)),
         "metadata": {"width_pt": 595.0, "height_pt": 842.0}},
        {"kind": "image", "orig": solid((0, 0, 255), (80, 60)), "proc": solid((0, 0, 255), (80, 60)),
         "dpi": 150.0},
    ]
    task, _ = make_task(tmp_path, specs)
    out, info = export(task, tmp_path)
    doc = pdfium.PdfDocument(str(out))
    w0, h0 = doc[0].get_size()
    w1, h1 = doc[1].get_size()
    doc.close()
    assert (round(w0), round(h0)) == (595, 842)      # PDF 保留原始物理尺寸
    assert w1 > h1                                    # 横图保持横向
    assert info["summary"]["source_pdf_pages"] == 1
    assert info["summary"]["source_image_pages"] == 1


def test_page_size_matches_plan_within_half_pt(tmp_path):
    specs = [{"kind": "pdf", "orig": solid((9, 9, 9), (100, 140)), "proc": solid((9, 9, 9), (100, 140)),
              "metadata": {"width_pt": 400.5, "height_pt": 600.25}}]
    task, _ = make_task(tmp_path, specs)
    out, info = export(task, tmp_path)
    doc = pdfium.PdfDocument(str(out))
    w, h = doc[0].get_size()
    doc.close()
    assert abs(w - 400.5) <= 0.5 and abs(h - 600.25) <= 0.5


def test_contain_white_margins_no_stretch(tmp_path):
    # 方形图放入 A4 竖版 -> 上下留白纯白，不拉伸
    task, _ = make_task(tmp_path, [{"orig": solid((200, 30, 30), (80, 80)),
                                   "proc": solid((200, 30, 30), (80, 80))}])
    out, info = export(task, tmp_path, page_size_mode="A4")
    doc = pdfium.PdfDocument(str(out))
    arr = np.asarray(doc[0].render(scale=1.0).to_pil().convert("RGB"))
    doc.close()
    assert tuple(arr[2, 2]) == (255, 255, 255)          # 角落留白纯白
    assert tuple(arr[-3, -3]) == (255, 255, 255)
    # 内容区为红色且宽高相等（不拉伸）；红色掩膜排除白色留白
    red = (arr[:, :, 0] > 150) & (arr[:, :, 1] < 100) & (arr[:, :, 2] < 100)
    ys, xs = np.where(red)
    assert abs((xs.max() - xs.min()) - (ys.max() - ys.min())) <= 2  # 内容宽高相等=不拉伸


def test_mixed_gray_and_color(tmp_path):
    g = Image.new("RGB", (60, 80), (120, 120, 120))
    c = solid((200, 60, 60))
    task, _ = make_task(tmp_path, [{"orig": g, "proc": g}, {"orig": c, "proc": c}])
    out, info = export(task, tmp_path)
    assert info["summary"]["grayscale_pages"] == 1
    assert info["summary"]["color_pages"] == 1


def test_fallback_pages_not_dropped(tmp_path):
    specs = [
        {"orig": solid((10, 10, 10)), "proc": solid((240, 240, 240))},                      # accept
        {"orig": solid((20, 20, 20)), "proc": solid((250, 250, 250)),
         "fis": "geometry", "fb": "geometry", "dec": "fallback_geometry"},                  # fallback_geometry
        {"orig": solid((30, 30, 30)), "proc": None, "pstatus": "failed"},                   # failed->original
    ]
    task, _ = make_task(tmp_path, specs)
    out, info = export(task, tmp_path)
    doc = pdfium.PdfDocument(str(out))
    assert len(doc) == 3
    doc.close()
    assert info["summary"]["fallback_original_pages"] == 1
    assert info["summary"]["fallback_geometry_pages"] == 1


def test_missing_page_aborts_with_numbers(tmp_path):
    specs = [{"orig": solid((1, 1, 1)), "proc": solid((2, 2, 2))},
             {"orig": None, "proc": None, "pstatus": "failed"}]
    task, _ = make_task(tmp_path, specs)
    with pytest.raises(ExportError) as e:
        export(task, tmp_path)
    assert "2" in str(e.value)


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(ExportError):
        PDFExporter(PDFExportConfig()).export(tmp_path, tmp_path / "x.pdf")


# ---------- 守卫 ----------

def test_no_overwrite_by_default(tmp_path):
    task, _ = make_task(tmp_path, [{"orig": solid((1, 1, 1)), "proc": solid((2, 2, 2))}])
    out, _ = export(task, tmp_path)
    with pytest.raises(ExportError):
        export(task, tmp_path)
    out2, _ = export(task, tmp_path, overwrite=True)
    assert out2.exists()


def test_output_equals_input_rejected(tmp_path):
    task, m = make_task(tmp_path, [{"orig": solid((1, 1, 1)), "proc": solid((2, 2, 2))}])
    inp = tmp_path / "input_0.png"
    solid((1, 1, 1)).save(inp)
    with pytest.raises(ExportError):
        PDFExporter(PDFExportConfig()).export(task, inp)


def test_lossless_render_matches_processed(tmp_path):
    proc = solid((180, 90, 40), (60, 80))
    task, _ = make_task(tmp_path, [{"orig": solid((0, 0, 0)), "proc": proc}])
    out, info = export(task, tmp_path, compression="lossless")
    doc = pdfium.PdfDocument(str(out))
    arr = np.asarray(doc[0].render(scale=150 / 72).to_pil().convert("RGB"))
    doc.close()
    src = np.asarray(proc)
    # pypdfium 渲染可能因取整多出一行/列，比较源尺寸裁剪区
    assert np.array_equal(arr[:src.shape[0], :src.shape[1]], src)
    assert info["pages"][0]["output_format"] == "png"