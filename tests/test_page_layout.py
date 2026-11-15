"""M4.1 页面布局测试：尺寸来源 / 旋转 / Contain / Cover / DPI 可信度 / 配置校验。"""

import math

import pytest

from bookclean.config import PDFExportConfig
from bookclean.page_layout import MM_TO_PT, PageLayoutResolver

A4_W, A4_H = 210 * MM_TO_PT, 297 * MM_TO_PT


def pdf_page(page_index=0, w_pt=487.0, h_pt=742.0, rotation=0, dpi=150.0, **extra):
    entry = {
        "page_index": page_index,
        "source_kind": "pdf",
        "source_path": "book.pdf",
        "dpi": dpi,
        "metadata": {"width_pt": w_pt, "height_pt": h_pt, "rotation": rotation,
                     "render_dpi": int(dpi) if dpi else None},
    }
    entry.update(extra)
    return entry


def image_page(path, page_index=0, dpi=None, **extra):
    entry = {
        "page_index": page_index,
        "source_kind": "image",
        "source_path": str(path),
        "dpi": dpi,
        "metadata": {"format": "PNG"},
    }
    entry.update(extra)
    return entry


def resolver(**kwargs):
    return PageLayoutResolver(PDFExportConfig(**kwargs))


# ---------- 标准纸张 ----------

def test_a4_portrait():
    lay = resolver(page_size_mode="A4").resolve(image_page("x.png"), 600, 800)
    assert lay.page_width_pt == pytest.approx(A4_W, abs=0.01)
    assert lay.page_height_pt == pytest.approx(A4_H, abs=0.01)
    assert lay.source_page_size == "standard_A4"
    assert not lay.estimated_size


def test_a4_landscape():
    lay = resolver(page_size_mode="A4").resolve(image_page("x.png"), 800, 600)
    assert lay.page_width_pt == pytest.approx(A4_H, abs=0.01)
    assert lay.page_height_pt == pytest.approx(A4_W, abs=0.01)


def test_a5():
    lay = resolver(page_size_mode="A5").resolve(image_page("x.png"), 600, 800)
    assert lay.page_width_pt == pytest.approx(148 * MM_TO_PT, abs=0.01)
    assert lay.page_height_pt == pytest.approx(210 * MM_TO_PT, abs=0.01)


def test_b5_iso():
    lay = resolver(page_size_mode="B5").resolve(image_page("x.png"), 600, 800)
    assert lay.page_width_pt == pytest.approx(176 * MM_TO_PT, abs=0.01)
    assert lay.page_height_pt == pytest.approx(250 * MM_TO_PT, abs=0.01)


def test_letter():
    lay = resolver(page_size_mode="Letter").resolve(image_page("x.png"), 600, 800)
    assert lay.page_width_pt == pytest.approx(612.0, abs=0.01)
    assert lay.page_height_pt == pytest.approx(792.0, abs=0.01)


def test_square_image_defaults_portrait():
    lay = resolver(page_size_mode="A4").resolve(image_page("x.png"), 500, 500)
    assert lay.page_height_pt > lay.page_width_pt


def test_custom_size():
    lay = resolver(page_size_mode="custom", custom_width_mm=100.0, custom_height_mm=150.0).resolve(
        image_page("x.png"), 600, 800)
    assert lay.page_width_pt == pytest.approx(100 * MM_TO_PT, abs=0.01)
    assert lay.page_height_pt == pytest.approx(150 * MM_TO_PT, abs=0.01)
    assert lay.source_page_size == "custom"
    assert not lay.estimated_size


# ---------- PDF 来源尺寸保留 ----------

def test_pdf_custom_size_preserved():
    lay = resolver().resolve(pdf_page(w_pt=487.0, h_pt=742.0), 1015, 1546)
    assert lay.page_width_pt == pytest.approx(487.0, abs=0.01)
    assert lay.page_height_pt == pytest.approx(742.0, abs=0.01)
    assert lay.source_page_size == "pdf_original"
    assert not lay.estimated_size


def test_pdf_rotation_no_double_swap():
    # M1 get_size() 已应用 /Rotate：rotation=90 时 metadata 已是视觉横向尺寸
    lay = resolver().resolve(pdf_page(w_pt=742.0, h_pt=487.0, rotation=90), 1546, 1015)
    assert lay.page_width_pt == pytest.approx(742.0, abs=0.01)
    assert lay.page_height_pt == pytest.approx(487.0, abs=0.01)


def test_pdf_rotation_0():
    lay = resolver().resolve(pdf_page(w_pt=480.0, h_pt=742.0, rotation=0), 1000, 1546)
    assert (lay.page_width_pt, lay.page_height_pt) == (480.0, 742.0)


def test_mixed_sizes_batch():
    r = resolver()
    pages = [
        (pdf_page(0, 480.0, 742.0), 1000, 1546),
        (pdf_page(1, 487.0, 742.0), 1015, 1546),
        (pdf_page(2, 595.28, 841.89), 2480, 3508),
    ]
    layouts = [r.resolve(p, w, h) for p, w, h in pages]
    assert [(round(l.page_width_pt, 1), round(l.page_height_pt, 1)) for l in layouts] == [
        (480.0, 742.0), (487.0, 742.0), (595.3, 841.9)]
    assert [l.page_index for l in layouts] == [0, 1, 2]


def test_pdf_missing_metadata_uses_render_dpi():
    page = pdf_page(dpi=300.0)
    page["metadata"] = {"rotation": 0, "render_dpi": 300}
    lay = resolver().resolve(page, 2480, 3508)
    assert lay.page_width_pt == pytest.approx(2480 / 300 * 72, abs=0.01)
    assert lay.estimated_size
    assert lay.warnings


def test_pdf_missing_metadata_and_dpi_auto():
    page = pdf_page(dpi=None)
    page["metadata"] = {}
    lay = resolver().resolve(page, 1000, 1500)
    assert lay.source_page_size == "auto_estimated"
    assert lay.estimated_size


def test_auto_mode_overrides_pdf_size():
    lay = resolver(page_size_mode="auto").resolve(pdf_page(), 1000, 1500)
    assert lay.source_page_size == "auto_estimated"
    assert lay.estimated_size
    assert lay.page_height_pt == pytest.approx(250 * MM_TO_PT, abs=0.01)


# ---------- 图片来源 DPI ----------

def test_scanner_trusted_dpi(img_factory):
    path = img_factory("scan.png", size=(2480, 3508), dpi=(300, 300))
    lay = resolver().resolve(image_page(path, dpi=300.0), 2480, 3508)
    assert lay.source_page_size == "trusted_dpi"
    assert not lay.estimated_size
    assert lay.page_width_pt == pytest.approx(2480 / 300 * 72, abs=0.05)
    assert lay.page_height_pt == pytest.approx(3508 / 300 * 72, abs=0.05)


def test_real_book_dpi_recovery(img_factory):
    # 真实书 150 DPI 渲染页：pHYs 取整为 150.0124，反推尺寸误差应 <0.5pt
    path = img_factory("real.png", size=(1015, 1546), dpi=(150.0124, 150.0124))
    lay = resolver().resolve(image_page(path, dpi=150.0124), 1015, 1546)
    assert lay.source_page_size == "trusted_dpi"
    assert abs(lay.page_width_pt - 487.0) < 0.5
    assert abs(lay.page_height_pt - 742.0) < 0.5


def test_phone_photo_auto(img_factory):
    path = img_factory("phone.jpg", size=(3000, 2100), dpi=(72, 72))
    lay = resolver().resolve(image_page(path, dpi=72.0), 3000, 2100)
    assert lay.source_page_size == "auto_estimated"
    assert lay.estimated_size
    assert lay.page_width_pt == pytest.approx(250 * MM_TO_PT, abs=0.01)
    assert lay.page_height_pt == pytest.approx(175 * MM_TO_PT, abs=0.01)
    assert any("估算" in w for w in lay.warnings)


def test_missing_dpi_auto(img_factory):
    path = img_factory("nodpi.png", size=(2000, 3000))
    lay = resolver().resolve(image_page(path, dpi=300.0), 2000, 3000)
    # 文件无 DPI：即使任务记录 dpi=300（M1 导入默认值）也不可信
    assert lay.source_page_size == "auto_estimated"
    assert lay.estimated_size
    assert lay.page_height_pt == pytest.approx(250 * MM_TO_PT, abs=0.01)
    assert lay.page_width_pt == pytest.approx(250 * MM_TO_PT * 2 / 3, abs=0.01)


def test_untrusted_96_dpi(img_factory):
    path = img_factory("p96.png", size=(2000, 3000), dpi=(96, 96))
    lay = resolver().resolve(image_page(path, dpi=96.0), 2000, 3000)
    assert lay.source_page_size == "auto_estimated"


def test_anisotropic_dpi_untrusted(img_factory):
    path = img_factory("aniso.png", size=(2000, 3000), dpi=(300, 600))
    lay = resolver().resolve(image_page(path, dpi=300.0), 2000, 3000)
    assert lay.source_page_size == "auto_estimated"


def test_dpi_below_range_untrusted(img_factory):
    path = img_factory("low.png", size=(2000, 3000), dpi=(100, 100))
    lay = resolver().resolve(image_page(path, dpi=100.0), 2000, 3000)
    assert lay.source_page_size == "auto_estimated"


def test_auto_long_edge_configurable(img_factory):
    path = img_factory("nodpi.png", size=(3000, 2100))
    lay = resolver(auto_long_edge_mm=297.0).resolve(image_page(path), 3000, 2100)
    assert lay.page_width_pt == pytest.approx(297 * MM_TO_PT, abs=0.01)


def test_source_inaccessible_task_dpi_unverified():
    lay = resolver().resolve(image_page("missing/photo.png", dpi=300.0), 2480, 3508)
    assert lay.source_page_size == "task_recorded_dpi"
    assert lay.estimated_size
    assert any("未核实" in w for w in lay.warnings)


def test_source_inaccessible_untrusted_dpi_auto():
    lay = resolver().resolve(image_page("missing/photo.png", dpi=72.0), 3000, 2100)
    assert lay.source_page_size == "auto_estimated"
    assert lay.estimated_size


# ---------- 放置与比例 ----------

def test_contain_no_stretch():
    # 细长图 400x800 放入 A4：按高度缩放，宽度留白，比例不变
    lay = resolver(page_size_mode="A4").resolve(image_page("x.png"), 400, 800)
    assert lay.image_display_height_pt == pytest.approx(A4_H, abs=0.01)
    assert lay.image_display_width_pt == pytest.approx(A4_H / 2, abs=0.01)
    assert lay.image_display_width_pt < lay.page_width_pt
    src_aspect = 400 / 800
    out_aspect = lay.image_display_width_pt / lay.image_display_height_pt
    assert out_aspect == pytest.approx(src_aspect, rel=1e-6)


def test_contain_wide_image():
    lay = resolver(page_size_mode="A4").resolve(image_page("x.png"), 800, 400)
    # 横图自动横版 A4：按宽度缩放
    assert lay.image_display_width_pt == pytest.approx(A4_H, abs=0.01)
    assert lay.image_display_height_pt == pytest.approx(A4_H / 2, abs=0.01)


def test_contain_pdf_cropped_image_keeps_page():
    # M2 裁边后比例略变：页面保持原始 487x742，图像等比 contain，不拉伸
    lay = resolver().resolve(pdf_page(w_pt=487.0, h_pt=742.0), 1000, 1500)
    assert (lay.page_width_pt, lay.page_height_pt) == (487.0, 742.0)
    scale = min(487.0 / 1000, 742.0 / 1500)
    assert lay.image_display_width_pt == pytest.approx(1000 * scale, abs=0.01)
    assert lay.image_display_height_pt == pytest.approx(1500 * scale, abs=0.01)
    assert lay.image_display_width_pt <= 487.0 + 0.01
    assert lay.image_display_height_pt <= 742.0 + 0.01


def test_cover_covers_and_warns():
    lay = resolver(page_size_mode="A4", fit_mode="cover").resolve(image_page("x.png"), 400, 800)
    assert lay.image_display_width_pt == pytest.approx(A4_W, abs=0.01)
    assert lay.image_display_height_pt == pytest.approx(A4_W * 2, abs=0.01)
    assert lay.image_display_height_pt > lay.page_height_pt  # 允许裁切
    assert any("裁切" in w for w in lay.warnings)


def test_effective_dpi_from_display_size():
    # 2480x3508 @300DPI 源 -> A4 页面 contain 后有效 DPI 按显示尺寸计算
    lay = resolver(page_size_mode="A4").resolve(image_page("x.png"), 2480, 3508)
    expected_x = 2480 / lay.image_display_width_pt * 72
    expected_y = 3508 / lay.image_display_height_pt * 72
    assert lay.effective_dpi_x == pytest.approx(expected_x, rel=1e-6)
    assert lay.effective_dpi_y == pytest.approx(expected_y, rel=1e-6)
    assert lay.effective_dpi_x == pytest.approx(lay.effective_dpi_y, rel=1e-6)  # 等比 -> 各向同性


def test_effective_dpi_pdf_original():
    lay = resolver().resolve(pdf_page(w_pt=487.0, h_pt=742.0), 1015, 1546)
    # 图像比例与页面一致 -> 满页显示，有效 DPI ≈ 150
    assert lay.effective_dpi_x == pytest.approx(150.0, abs=0.5)
    assert lay.effective_dpi_y == pytest.approx(150.0, abs=0.5)


# ---------- 校验与异常 ----------

def test_invalid_dims_raise():
    with pytest.raises(ValueError):
        resolver().resolve(image_page("x.png"), 0, 100)
    with pytest.raises(ValueError):
        resolver().resolve(image_page("x.png"), 100, -1)


def test_stretch_rejected():
    with pytest.raises(ValueError, match="fit_mode"):
        PDFExportConfig(fit_mode="stretch").validate()


@pytest.mark.parametrize("kwargs", [
    {"compression": "zip"},
    {"page_size_mode": "A3"},
    {"fit_mode": "fill"},
    {"target_dpi": 150},
    {"target_dpi": 600},
    {"auto_long_edge_mm": 10},
    {"auto_long_edge_mm": 5000},
    {"page_size_mode": "custom"},
    {"page_size_mode": "custom", "custom_width_mm": 100},
    {"page_size_mode": "custom", "custom_width_mm": 5, "custom_height_mm": 100},
    {"overwrite": "yes"},
])
def test_config_validation_rejects(kwargs):
    with pytest.raises(ValueError):
        PDFExportConfig(**kwargs).validate()


def test_config_defaults():
    cfg = PDFExportConfig()
    cfg.validate()
    assert cfg.compression == "balanced"
    assert cfg.target_dpi is None
    assert cfg.page_size_mode == "preserve"
    assert cfg.fit_mode == "contain"
    assert cfg.auto_long_edge_mm == 250.0
    assert cfg.verify is True
    assert cfg.overwrite is False
    assert cfg.linearize is False


def test_target_dpi_valid_values():
    for dpi in (200, 300, 400):
        PDFExportConfig(target_dpi=dpi).validate()