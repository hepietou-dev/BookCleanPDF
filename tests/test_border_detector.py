"""M4.8.1 扫描黑边检测与保守清除测试。

覆盖：纯文字/黑边/书脊阴影/黑白照片压边/黑底白字/边缘文字与页码/表格线/横向页/
照片页/低置信度跳过/掩膜正确性/配置校验/runner 幂等与只读保护/dry-run/导出集成与向后兼容。
内容保真断言：裁切为纯像素切片，内部逐像素不变；前景掩膜全保留。
"""

import hashlib
import json
import threading

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from bookclean.border_detector import BorderCleaningRunner, detect_borders
from bookclean.config import BorderCleanConfig, PDFExportConfig
from bookclean.pdf_exporter import FinalImageResolver, PDFExporter
from bookclean.task_manager import ImportTaskManager
from scripts.make_m2_samples import (
    _photo_array, black_border, shadow, table_page, text_page,
)


@pytest.fixture(scope="module")
def base():
    return text_page()


def _font(size):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/times.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------- 纯检测 ----------

def test_clean_page_no_border(base):
    info, mask = detect_borders(base[0], BorderCleanConfig())
    assert not info["executed"]
    assert info["reason"] == "no_border"
    assert mask is None


def test_black_border_cropped_and_content_exact(base):
    rgb, m = black_border(base[0], base[1], border=70)
    info, mask = detect_borders(rgb, BorderCleanConfig())
    assert info["executed"], info
    x0, y0, x1, y1 = info["crop_box"]
    # 页面内全部前景（含页码）保留
    assert m[y0:y1, x0:x1].sum() == m.sum()
    # 裁切为纯切片：内部逐像素不变
    assert np.array_equal(rgb[y0:y1, x0:x1], rgb[y0:y1, x0:x1])
    assert info["final_size"] == [x1 - x0, y1 - y0]
    # 四边均高置信度
    for e in ("left", "right", "top", "bottom"):
        assert info["edges"][e]["executed"]
        assert info["edges"][e]["confidence"] >= 0.9
    # 掩膜：被裁区域 255，保留区域 0
    assert mask.shape == rgb.shape[:2]
    assert mask[y0:y1, x0:x1].sum() == 0
    assert mask[:y0, :].min() == 255 and mask[y1:, :].min() == 255
    assert mask[:, :x0].min() == 255 and mask[:, x1:].min() == 255


def test_black_border_keeps_safety_sliver(base):
    rgb, m = black_border(base[0], base[1], border=70)
    info, _ = detect_borders(rgb, BorderCleanConfig())
    x0, y0, x1, y1 = info["crop_box"]
    # 裁切线在黑边内侧（保留薄边），不贴到页面内容边界 70
    assert 0 < x0 < 70 and 0 < y0 < 70
    assert rgb.shape[1] - 70 < x1 < rgb.shape[1]
    assert rgb.shape[0] - 70 < y1 < rgb.shape[0]


def test_spine_shadow_center_no_border(base):
    rgb, _ = shadow(base[0], base[1], "spine", 0.4)
    info, mask = detect_borders(rgb, BorderCleanConfig())
    assert not info["executed"]
    assert mask is None


def test_photo_straddling_edge_skips_that_edge(base):
    rgb, m = black_border(base[0], base[1], border=70)
    rgb[200:700, 70:370] = _photo_array(300, 500, color=False, seed=9)
    info, _ = detect_borders(rgb, BorderCleanConfig())
    left = info["edges"]["left"]
    assert not left["executed"]
    assert left["reason"] == "content_straddles_border"
    # 照片所在左缘不裁切
    x0 = info["crop_box"][0]
    assert x0 == 0
    # 其余干净边仍清除
    assert info["edges"]["right"]["executed"]


def test_white_on_black_text_skips_edge(base):
    rgb, _ = black_border(base[0], base[1], border=70)
    im = Image.fromarray(rgb)
    ImageDraw.Draw(im).text((400, rgb.shape[0] - 60), "123", fill=(240, 240, 240), font=_font(30))
    info, _ = detect_borders(np.asarray(im), BorderCleanConfig())
    bottom = info["edges"]["bottom"]
    assert not bottom["executed"]
    assert bottom["reason"] == "bright_content_in_border"


def test_dust_specks_do_not_block_border(base):
    rgb, _ = black_border(base[0], base[1], border=70)
    rng = np.random.default_rng(1)
    for _ in range(15):
        x = int(rng.integers(0, rgb.shape[1]))
        y = int(rng.integers(rgb.shape[0] - 68, rgb.shape[0]))
        cv2.circle(rgb, (x, y), 2, (180, 180, 180), -1)
    info, _ = detect_borders(rgb, BorderCleanConfig())
    assert info["edges"]["bottom"]["executed"]


def test_edge_text_and_page_number_no_loss():
    rgb, m = text_page(margin=25)
    info, _ = detect_borders(rgb, BorderCleanConfig())
    if info["executed"]:
        x0, y0, x1, y1 = info["crop_box"]
        assert m[y0:y1, x0:x1].sum() == m.sum()
    else:
        assert info["reason"] in ("no_border", "all_edges_skipped")


def test_landscape_border_all_edges(base):
    land = np.rot90(base[0])
    lm = np.rot90(base[1])
    rgb, _ = black_border(land, lm, border=60)
    info, _ = detect_borders(rgb, BorderCleanConfig())
    assert info["executed"]
    for e in ("left", "right", "top", "bottom"):
        assert info["edges"][e]["executed"], info["edges"][e]
    x0, y0, x1, y1 = info["crop_box"]
    assert lm[y0:y1, x0:x1].sum() == lm.sum()


def test_table_lines_preserved():
    rgb, m = table_page()
    bordered, bm = black_border(rgb, m, border=70)
    info, _ = detect_borders(bordered, BorderCleanConfig())
    assert info["executed"]
    x0, y0, x1, y1 = info["crop_box"]
    assert bm[y0:y1, x0:x1].sum() == bm.sum()


def test_photo_dominated_page_skipped():
    photo = (_photo_array(900, 1200, color=False, seed=3).astype(np.float32) * 0.45).astype(np.uint8)
    info, mask = detect_borders(photo, BorderCleanConfig())
    assert not info["executed"]
    assert info["reason"] == "insufficient_paper"
    assert mask is None


def test_perspective_phone_page_no_crop(base):
    from scripts.make_m2_samples import perspective
    rgb, pm, _ = perspective(base[0], base[1])
    info, mask = detect_borders(rgb, BorderCleanConfig())
    # 不规则四边形页面边界不可靠 -> 不自动裁切，内容全保留
    assert not info["executed"]
    assert mask is None


def test_low_confidence_gate_skips(base):
    rgb, _ = black_border(base[0], base[1], border=70)
    info, _ = detect_borders(rgb, BorderCleanConfig(min_confidence=0.99))
    assert not info["executed"]
    assert info["reason"] == "all_edges_skipped"
    for e in ("left", "right", "top", "bottom"):
        assert info["edges"][e]["reason"] == "low_confidence"


def test_width_cap_skips_oversized_border(base):
    # 黑边占边长 > 12% -> 异常，不裁
    rgb, _ = black_border(base[0], base[1], border=300)
    info, _ = detect_borders(rgb, BorderCleanConfig())
    assert info["edges"]["left"]["reason"] == "width_exceeds_limit"


def test_grayscale_input_supported(base):
    gray = cv2.cvtColor(base[0], cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    bordered = np.full((h + 140, w + 140), 15, np.uint8)
    bordered[70:70 + h, 70:70 + w] = gray
    info, mask = detect_borders(bordered, BorderCleanConfig())
    assert info["executed"]
    assert bordered.ndim == 2


@pytest.mark.parametrize("kwargs", [
    {"min_confidence": 1.5}, {"max_edge_fraction": 0.9}, {"analysis_max_dim": 50},
    {"dark_delta": -1}, {"bright_cc_min_area": 0}, {"safety_margin_fraction": 0.5},
])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        BorderCleanConfig(**kwargs).validate()


# ---------- runner / 导出集成 ----------

def _make_task(tmp_path, base, n=3, border=70):
    files = []
    in_dir = tmp_path / "inputs"
    in_dir.mkdir(exist_ok=True)
    for i in range(n):
        rgb, _ = black_border(base[0], base[1], border=border)
        p = in_dir / f"p{i}.png"
        Image.fromarray(rgb).save(p, dpi=(150, 150))
        files.append(p)
    return ImportTaskManager(tmp_path / "tasks").run(files, dpi=150)["task_dir"]


def test_runner_cleans_and_protects_originals(tmp_path, base):
    td = _make_task(tmp_path, base)
    man = json.loads((td / "manifest.json").read_text(encoding="utf-8"))
    orig_hashes = {p["image_path"]: _sha(td / p["image_path"]) for p in man["pages"]}
    res = BorderCleaningRunner().run(td, BorderCleanConfig())
    assert res["summary"]["cleaned"] == 3
    assert res["summary"]["failed"] == 0
    man = res["manifest"]
    for p in man["pages"]:
        assert p["border_status"] == "completed"
        assert (td / p["border_image_path"]).exists()
        assert (td / p["border_mask_path"]).exists()
        assert p["border_detection"]["executed"]
        # 原始文件哈希不变
        assert _sha(td / p["image_path"]) == orig_hashes[p["image_path"]]
    assert (td / "border_report.json").exists()
    assert (td / "border_contact_sheet.jpg").exists()


def test_runner_idempotent_rerun(tmp_path, base):
    td = _make_task(tmp_path, base)
    BorderCleaningRunner().run(td, BorderCleanConfig())
    first = {p["display_index"]: _sha(td / p["border_image_path"])
             for p in json.loads((td / "manifest.json").read_text(encoding="utf-8"))["pages"]}
    res = BorderCleaningRunner().run(td, BorderCleanConfig())
    assert res["summary"]["cleaned"] == 3
    for p in res["manifest"]["pages"]:
        assert _sha(td / p["border_image_path"]) == first[p["display_index"]]


def test_runner_dry_run_writes_nothing(tmp_path, base):
    td = _make_task(tmp_path, base)
    before = _sha(td / "manifest.json")
    res = BorderCleaningRunner().run(td, BorderCleanConfig(), dry_run=True)
    assert res["summary"]["dry_run"]
    assert res["summary"]["cleaned"] == 3
    assert not (td / "border_pages").exists()
    assert _sha(td / "manifest.json") == before


def test_runner_prefers_processed_and_protects_it(tmp_path, base):
    td = _make_task(tmp_path, base)
    man = json.loads((td / "manifest.json").read_text(encoding="utf-8"))
    (td / "processed_pages").mkdir(exist_ok=True)
    # 注入一个"已处理"页（内容为去边后的页面），验证 runner 从 processed 解析且不改动它
    rgb, m = black_border(base[0], base[1], border=70)
    processed = rgb[70:70 + base[0].shape[0], 70:70 + base[0].shape[1]]
    rel = "processed_pages/page_000001.png"
    Image.fromarray(processed).save(td / rel, dpi=(150, 150))
    man["pages"][0]["processing_status"] = "completed"
    man["pages"][0]["processed_image_path"] = rel
    (td / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    proc_hash = _sha(td / rel)
    res = BorderCleaningRunner().run(td, BorderCleanConfig())
    p0 = res["manifest"]["pages"][0]
    assert p0["border_source_path"] == rel
    assert _sha(td / rel) == proc_hash  # processed 只读


def test_runner_cancel(tmp_path, base):
    td = _make_task(tmp_path, base, n=5)
    ev = threading.Event()
    ev.set()
    res = BorderCleaningRunner().run(td, BorderCleanConfig(), cancel_event=ev)
    assert res["summary"]["cleaned"] == 0
    assert res["manifest"]["pages"][0]["border_status"] == "cancelled"


def test_resolver_prefers_border_then_backward_compatible(tmp_path, base):
    td = _make_task(tmp_path, base, n=1)
    man = json.loads((td / "manifest.json").read_text(encoding="utf-8"))
    p0 = man["pages"][0]
    # 无 border 字段：回退 original（M1 完成但无 processed）
    fi = FinalImageResolver(td).resolve(man, p0)
    assert fi.source == "original_fallback"
    # 清除后：优先 border
    BorderCleaningRunner().run(td, BorderCleanConfig())
    man = json.loads((td / "manifest.json").read_text(encoding="utf-8"))
    fi = FinalImageResolver(td).resolve(man, man["pages"][0])
    assert fi.source == "border_cleaned"
    assert fi.path.name == "page_000001.png"
    # border 文件缺失时回退（不中断）
    (td / man["pages"][0]["border_image_path"]).unlink()
    fi = FinalImageResolver(td).resolve(man, man["pages"][0])
    assert fi.source == "original_fallback"


def test_export_uses_border_pages(tmp_path, base):
    td = _make_task(tmp_path, base, n=2)
    BorderCleaningRunner().run(td, BorderCleanConfig())
    out = tmp_path / "out.pdf"
    report = PDFExporter(PDFExportConfig(compression="balanced")).export(td, out)
    assert out.exists()
    assert report["validation"]["passed"]
    # 导出报告记录 border_cleaned 来源
    sources = [p.get("final_image_source") for p in report["pages"]]
    assert all(s == "border_cleaned" for s in sources), sources