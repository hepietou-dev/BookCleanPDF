"""M3.4 安全回退与 Manifest 正式化测试：多级回退按失败阶段、有限重试、独立中间结果、Manifest/报告/联系表。

回退阶梯逻辑用脚本化 QualityChecker 决策确定性验证；Manifest/quality_report/质量联系表用真实 runner 验证。
"""

import json

import numpy as np
import pytest
from PIL import Image

from bookclean.config import TaskConfig
from bookclean.models import PageData
from bookclean.presets import get_preset
from bookclean.processing_pipeline import ProcessingPipeline, ProcessingRunner
from bookclean.task_manager import ImportTaskManager
from scripts.make_m2_samples import text_page, yellow, black_border


def _pd(rgb, idx=0):
    return PageData(idx, rgb, "synthetic", "image", rgb.shape[1], rgb.shape[0], 150.0, "RGB")


def _report(decision, failed=(), reasons=None):
    checks = [{"name": n, "passed": n not in failed, "value": 0.0, "detail": ""}
              for n in ("size", "blank", "foreground", "background", "sharpness", "color", "text_core", "illustration")]
    return {"decision": decision, "checks": checks,
            "reasons": reasons or [f"{n}: simulated" for n in failed]}


@pytest.fixture(scope="module")
def base():
    return text_page()


# ---- 失败阶段 -> 回退起点 映射 ----

def test_fallback_start_geometry_crop_to_original():
    r = _report("fallback_original", failed=("size",))
    assert ProcessingPipeline._fallback_start(r, _DummyBg(False), True) == 2


def test_fallback_start_size_fail_without_geometry_not_original():
    r = _report("fallback_original", failed=("size",))
    assert ProcessingPipeline._fallback_start(r, _DummyBg(False), False) == 1


def test_fallback_start_color_damage_to_geometry():
    r = _report("fallback_original", failed=("color",))
    assert ProcessingPipeline._fallback_start(r, _DummyBg(False), True) == 1


def test_fallback_start_cleaner_suggest_to_geometry():
    r = _report("fallback_original")
    assert ProcessingPipeline._fallback_start(r, _DummyBg(True), True) == 1


def test_fallback_start_enhance_only_to_background():
    r = _report("retry_lower_strength", failed=("sharpness",))
    assert ProcessingPipeline._fallback_start(r, _DummyBg(False), True) == 0


def test_fallback_start_text_damage_to_geometry():
    r = _report("fallback_original", failed=("text_core",))
    assert ProcessingPipeline._fallback_start(r, _DummyBg(False), True) == 1


class _DummyBg:
    def __init__(self, suggest):
        self.suggest_fallback = suggest


# ---- 脚本化回退阶梯 ----

def _scripted(pipe, reports):
    seq = list(reports)
    calls = {"n": 0}

    def stub(original, candidate, analysis, config, geometry_applied=False, deskew_applied=False, regions=None):
        i = min(calls["n"], len(seq) - 1)
        calls["n"] += 1
        return seq[i]

    pipe.quality.evaluate = stub
    return calls


def test_retry_accepted_uses_retry_image(base):
    pipe = ProcessingPipeline()
    _scripted(pipe, [_report("retry_lower_strength"), _report("accept")])
    res = pipe.process_page(_pd(base[0]), get_preset("auto"))
    assert res.quality["decision"] == "accept"
    assert res.quality["retry_count"] == 1
    assert res.quality["final_image_source"] == "retry_enhanced"
    assert res.quality["fallback_level"] == "none"
    assert not res.quality["fallback_events"]


def test_retry_actually_lowers_strength(base):
    pipe = ProcessingPipeline()
    cfg = get_preset("old_book")
    _scripted(pipe, [_report("retry_lower_strength"), _report("accept")])
    res = pipe.process_page(_pd(base[0]), cfg)
    retry = next(s for s in res.steps if s["stage"] == "retry")
    assert retry["params"]["background_strength"] == pytest.approx(cfg.background_strength * 0.5)
    assert retry["params"]["yellow_removal_strength"] == pytest.approx(cfg.yellow_removal_strength * 0.5)


def test_retry_max_once(base):
    pipe = ProcessingPipeline()
    _scripted(pipe, [_report("retry_lower_strength")] * 6)
    res = pipe.process_page(_pd(base[0]), get_preset("auto"))
    assert res.quality["retry_count"] == 1
    assert sum(1 for s in res.steps if s["stage"] == "retry") == 1
    assert res.quality["fallback_level"] != "none"


def test_background_color_damage_falls_to_geometry(base):
    pipe = ProcessingPipeline()
    _scripted(pipe, [_report("fallback_original", failed=("color",)), _report("accept")])
    res = pipe.process_page(_pd(base[0]), get_preset("auto"))
    assert res.quality["fallback_level"] == "geometry"
    assert res.quality["final_image_source"] == "geometry"
    assert res.quality["decision"] == "fallback_geometry"
    assert res.quality["fallback_events"]


def test_enhance_only_falls_to_background(base):
    pipe = ProcessingPipeline()
    _scripted(pipe, [_report("retry_lower_strength", failed=("sharpness",)),
                     _report("retry_lower_strength", failed=("sharpness",)),
                     _report("accept")])
    res = pipe.process_page(_pd(base[0]), get_preset("auto"))
    assert res.quality["fallback_level"] == "background"
    assert res.quality["final_image_source"] == "background"
    assert res.quality["decision"] == "fallback_background"


def test_geometry_crop_damage_falls_to_original(base):
    rgb, m = black_border(*text_page())
    pipe = ProcessingPipeline()
    _scripted(pipe, [_report("fallback_original", failed=("size",))])
    res = pipe.process_page(_pd(rgb), get_preset("auto"))
    assert res.quality["fallback_level"] == "original"
    assert res.quality["final_image_source"] == "original"
    assert res.quality["decision"] == "fallback_original"
    assert np.array_equal(res.image, rgb)


def test_all_stages_unsafe_falls_to_original(base):
    pipe = ProcessingPipeline()
    _scripted(pipe, [_report("fallback_original", failed=("color",)),
                     _report("fallback_original"), _report("fallback_original")])
    res = pipe.process_page(_pd(base[0]), get_preset("auto"))
    assert res.quality["fallback_level"] == "original"
    assert res.quality["final_image_source"] == "original"


def test_fallback_does_not_overwrite_intermediates(base):
    pipe = ProcessingPipeline()
    _scripted(pipe, [_report("fallback_original", failed=("color",)), _report("accept")])
    res = pipe.process_page(_pd(base[0]), get_preset("auto"))
    # 回退到几何结果：最终图应等于几何阶段输出，而非被背景/增强覆盖
    geo_steps = [s for s in res.steps if s["stage"] == "geometry" and s["executed"]]
    assert res.quality["final_image_source"] == "geometry"
    assert res.image.shape[2] == 3 and res.image.dtype == np.uint8


# ---- Manifest / quality_report / 质量联系表 集成 ----

@pytest.fixture()
def task_dir(tmp_path, base):
    in_dir = tmp_path / "inputs"
    in_dir.mkdir()
    y, _ = yellow(base[0], base[1], paper=(232, 210, 150), noise=2.0)
    files = []
    for name, arr in (("clean", base[0]), ("yellow", y)):
        p = in_dir / f"{name}.png"
        Image.fromarray(arr).save(p, dpi=(150, 150))
        files.append(p)
    mgr = ImportTaskManager(tmp_path / "tasks")
    return mgr.run(files, dpi=150)["task_dir"]


def test_manifest_m34_fields_and_backward_compat(task_dir):
    runner = ProcessingRunner()
    runner.run(task_dir, get_preset("auto"))
    manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
    for e in manifest["pages"]:
        assert {"page_index", "display_index", "status", "image_path"} <= set(e)  # M1
        assert {"processing_status", "processed_image_path", "processing_decision"} <= set(e)  # M2
        for k in ("page_type", "region_mask_confidence", "paper_area_ratio", "text_area_ratio",
                  "color_area_ratio", "background_quality_before", "background_quality_after",
                  "text_preservation_score", "color_preservation_score", "illustration_preservation_score",
                  "quality_risk_score", "decision", "retry_count", "fallback_level",
                  "fallback_reason", "final_image_source"):
            assert k in e, k
        assert e["final_image_source"] in ("enhanced", "retry_enhanced", "background", "geometry", "original")


def test_quality_report_json(task_dir):
    runner = ProcessingRunner()
    res = runner.run(task_dir, get_preset("auto"))
    qr = json.loads((task_dir / "quality_report.json").read_text(encoding="utf-8"))
    assert qr["total_pages"] == 2
    assert qr["accept"] + qr["fallback"] <= 2
    assert "page_type_counts" in qr and "avg_ms_by_page_type" in qr
    assert "main_fallback_reasons" in qr and "quality_abnormal_pages" in qr and "manual_review_pages" in qr
    assert res["quality_report"]["total_pages"] == 2
    manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["quality_report"] == "quality_report.json"


def test_quality_contact_sheet_highlights_fallback(task_dir, monkeypatch):
    runner = ProcessingRunner()
    real = runner.pipeline.quality.evaluate

    def stub(original, candidate, analysis, config, geometry_applied=False, deskew_applied=False, regions=None):
        if original[0, 0, 2] < 200:  # 黄纸页：强制走回退，验证联系表突出显示
            return _report("fallback_original", failed=("color",))
        return real(original, candidate, analysis, config, geometry_applied, deskew_applied, regions=regions)

    monkeypatch.setattr(runner.pipeline.quality, "evaluate", stub)
    runner.run(task_dir, get_preset("auto"))
    manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("quality_contact_sheet") == "quality_contact_sheet.jpg"
    assert (task_dir / "quality_contact_sheet.jpg").exists()
    qr = json.loads((task_dir / "quality_report.json").read_text(encoding="utf-8"))
    assert qr["fallback"] >= 1
    assert qr["manual_review_pages"]
    flagged = [e for e in manifest["pages"] if e.get("fallback_level", "none") != "none"]
    assert flagged and flagged[0]["final_image_source"] == "original"


def test_saved_final_image_matches_decision(task_dir, monkeypatch):
    # 最终落盘图片必须与决策记录一致：fallback_original=原图像素；accept=增强后（与原图不同）
    runner = ProcessingRunner()
    real = runner.pipeline.quality.evaluate

    def stub(original, candidate, analysis, config, geometry_applied=False, deskew_applied=False, regions=None):
        if original[0, 0, 2] < 200:
            return _report("fallback_original", failed=("color",))
        return real(original, candidate, analysis, config, geometry_applied, deskew_applied, regions=regions)

    monkeypatch.setattr(runner.pipeline.quality, "evaluate", stub)
    runner.run(task_dir, get_preset("auto"))
    manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
    for e in manifest["pages"]:
        orig = np.asarray(Image.open(task_dir / e["image_path"]).convert("RGB"))
        saved = np.asarray(Image.open(task_dir / e["processed_image_path"]).convert("RGB"))
        if e["final_image_source"] == "original":
            assert e["fallback_level"] == "original"
            assert np.array_equal(saved, orig), "fallback_original 落盘图必须等于原图"
        else:
            assert e["decision"] == "accept"
            assert not np.array_equal(saved, orig), "accept 落盘图应为处理后结果"