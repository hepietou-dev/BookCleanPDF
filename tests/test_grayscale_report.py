"""M4.7.6 导出报告扩展测试：json 新字段 + 统计 / txt 可读段 / preserve 语义不变 / 原字段保留。"""

import json

from bookclean.config import PDFExportConfig
from bookclean.pdf_exporter import PDFExporter
from test_pdf_exporter import make_task
from test_grayscale_export import mixed_specs


def _export(tmp_path, name, **cfg):
    task, _ = make_task(tmp_path, mixed_specs())
    out = tmp_path / name
    PDFExporter(PDFExportConfig(**cfg)).export(task, out)
    rep = json.loads((out.parent / "export_report.json").read_text(encoding="utf-8"))
    txt = (out.parent / "export_report.txt").read_text(encoding="utf-8")
    return rep, txt


def test_grayscale_json_fields(tmp_path):
    rep, _ = _export(tmp_path, "g.pdf", color_mode="grayscale")
    assert rep["color_mode"] == "grayscale"
    gc = rep["grayscale_conversion"]
    for k in ("algorithm", "bit_depth", "source_rgb_pages", "source_gray_pages",
              "already_grayscale_pages", "converted_pages", "failed_pages"):
        assert k in gc
    assert gc["bit_depth"] == 8 and gc["failed_pages"] == 0
    gv = rep["grayscale_validation"]
    for k in ("enabled", "status", "device_gray_image_count", "unexpected_rgb_image_count",
              "render_channel_check", "failed_pages"):
        assert k in gv
    assert gv["enabled"] and gv["status"] == "passed"
    assert gv["device_gray_image_count"] == 8 and gv["unexpected_rgb_image_count"] == 0


def test_grayscale_stats(tmp_path):
    rep, _ = _export(tmp_path, "g.pdf", color_mode="grayscale")
    assert rep["final_grayscale_pages"] == 8
    assert rep["gray_png_pages"] + rep["gray_jpeg_pages"] == 8
    assert rep["grayscale_verified_pages"] == 8


def test_grayscale_lossless_all_png(tmp_path):
    rep, _ = _export(tmp_path, "gl.pdf", color_mode="grayscale", compression="lossless")
    assert rep["gray_png_pages"] == 8 and rep["gray_jpeg_pages"] == 0


def test_grayscale_txt_readable(tmp_path):
    _, txt = _export(tmp_path, "g.pdf", color_mode="grayscale")
    for token in ("输出色彩模式: 8 位灰度", "原始彩色页面:", "原始灰度页面:",
                  "实际转换页面:", "最终灰度页面:", "灰度验证: 通过"):
        assert token in txt


def test_preserve_report_semantics(tmp_path):
    rep, txt = _export(tmp_path, "p.pdf", color_mode="preserve")
    assert rep["color_mode"] == "preserve"
    assert rep["grayscale_conversion"] is None
    assert rep["grayscale_validation"]["enabled"] is False
    assert rep["color_pages"] > 0  # 不要求全灰
    assert "preserve" in txt


def test_original_fields_retained(tmp_path):
    rep, _ = _export(tmp_path, "g.pdf", color_mode="grayscale")
    for k in ("task_id", "export_status", "output_file", "page_count", "output_size_bytes",
              "compression_mode", "image_dpi_mode", "page_size_mode", "fit_mode",
              "jpeg_pages", "png_pages", "grayscale_pages", "color_pages",
              "validation_status", "validation", "warnings", "pages", "note"):
        assert k in rep


def test_per_page_conversion_fields(tmp_path):
    rep, _ = _export(tmp_path, "g.pdf", color_mode="grayscale")
    for p in rep["pages"]:
        assert "source_color_mode" in p and "already_grayscale" in p and "grayscale_converted" in p