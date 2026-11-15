"""M2.4 流水线与批量 runner 测试：单页处理、Manifest 兼容扩展、失败隔离、取消、预设。"""

import hashlib
import json
import threading

import cv2
import numpy as np
import pytest
from PIL import Image

from bookclean.config import TaskConfig
from bookclean.models import PageData
from bookclean.presets import get_preset, list_presets
from bookclean.processing_pipeline import ProcessingPipeline, ProcessingRunner
from bookclean.task_manager import ImportTaskManager
from scripts.make_m2_samples import (
    black_border, color_content, rotate, text_page, yellow,
)


def _gray(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)


def _pd(rgb, idx=0):
    return PageData(idx, rgb, "synthetic", "image", rgb.shape[1], rgb.shape[0], 150.0, "RGB")


@pytest.fixture(scope="module")
def samples():
    base, m = text_page()
    y, _ = yellow(base, m, paper=(232, 210, 150), noise=2.0)
    sk, _ = rotate(base, m, 5.0)
    bb, _ = black_border(base, m)
    col, _, _ = color_content(base, m)
    return {"clean": base, "yellow": y, "skew": sk, "black_border": bb, "color": col}


@pytest.fixture()
def task_dir(tmp_path, samples):
    in_dir = tmp_path / "inputs"
    in_dir.mkdir()
    order = ["clean", "yellow", "skew", "black_border", "color"]
    files = []
    for name in order:
        p = in_dir / f"{name}.png"
        Image.fromarray(samples[name]).save(p, dpi=(150, 150))
        files.append(p)
    mgr = ImportTaskManager(tmp_path / "tasks")
    res = mgr.run(files, dpi=150)
    return res["task_dir"]


def test_pipeline_single_page_yellow(samples):
    pipe = ProcessingPipeline()
    cfg = get_preset("old_book")
    cfg.enable_crop = False  # 关闭裁边以保持尺寸对齐，隔离背景保真度
    res = pipe.process_page(_pd(samples["yellow"]), cfg)
    assert res.executed
    assert res.image.dtype == np.uint8 and res.image.shape[2] == 3
    assert res.image.flags["C_CONTIGUOUS"]
    assert res.quality["decision"] == "accept", res.quality["reasons"]
    core = _gray(samples["yellow"]) < 128
    keep = (_gray(res.image) < 160)[core].sum() / core.sum()
    assert keep >= 0.97
    assert res.steps and res.quality["checks"]


def test_pipeline_clean_accept(samples):
    pipe = ProcessingPipeline()
    res = pipe.process_page(_pd(samples["clean"]), get_preset("auto"))
    assert res.quality["decision"] == "accept"


def test_pipeline_deskew_applied(samples):
    pipe = ProcessingPipeline()
    res = pipe.process_page(_pd(samples["skew"]), get_preset("auto"))
    ops = [s["op"] for s in res.steps if s["executed"]]
    assert "deskew" in ops
    assert res.params["deskew_applied"]


def test_pipeline_color_preserved(samples):
    pipe = ProcessingPipeline()
    rgb = samples["color"]
    res = pipe.process_page(_pd(rgb), get_preset("auto"))
    chroma = (res.image.astype(int).max(2) - res.image.astype(int).min(2))
    assert chroma.mean() > 5  # 仍含彩色


def test_runner_processed_and_manifest(task_dir):
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (task_dir / "original_pages").glob("*.png")}
    runner = ProcessingRunner()
    res = runner.run(task_dir, get_preset("auto"))
    assert res["summary"]["processed"] == 5
    assert res["summary"]["failed"] == 0
    processed = list((task_dir / "processed_pages").glob("*.png"))
    assert len(processed) == 5
    # 原始页面只读，未被修改
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (task_dir / "original_pages").glob("*.png")}
    assert before == after
    manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["pages"]:
        # M1 字段保留
        assert {"page_index", "display_index", "status", "image_path"} <= set(entry)
        # M2 扩展字段
        assert entry["processing_status"] == "completed"
        assert entry["processed_image_path"].startswith("processed_pages/")
        assert "applied_operations" in entry and "quality_warnings" in entry
        assert "fallback_events" in entry and "processing_decision" in entry
    assert (task_dir / "processing_report.json").exists()
    assert manifest.get("processing_contact_sheet")


def test_runner_failure_isolation(task_dir, monkeypatch):
    runner = ProcessingRunner()
    orig = runner.pipeline.process_page

    def flaky(page, config):
        if page.page_index == 2:
            raise RuntimeError("模拟单页算法异常")
        return orig(page, config)

    monkeypatch.setattr(runner.pipeline, "process_page", flaky)
    res = runner.run(task_dir, get_preset("auto"))
    assert res["summary"]["processed"] == 4
    assert res["summary"]["failed"] == 1
    manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pages"][2]["processing_status"] == "failed"
    assert manifest["pages"][3]["processing_status"] == "completed"
    assert manifest["processing_status"] == "completed_with_errors"


def test_runner_cancel(task_dir):
    runner = ProcessingRunner()
    cancel = threading.Event()
    count = {"n": 0}

    def progress(entry):
        count["n"] += 1
        if count["n"] >= 2:
            cancel.set()

    res = runner.run(task_dir, get_preset("auto"), progress_cb=progress, cancel_event=cancel)
    assert res["summary"]["processed"] < 5
    manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
    assert any(p.get("processing_status") == "cancelled" for p in manifest["pages"])
    # 已完成页保留
    assert res["summary"]["processed"] >= 1


def test_presets_valid_and_distinct():
    cfgs = {name: get_preset(name) for name in list_presets()}
    assert set(cfgs) == {"auto", "old_book", "phone_photo", "faithful"}
    for cfg in cfgs.values():
        cfg.validate()
    assert cfgs["old_book"].yellow_removal_strength > cfgs["faithful"].yellow_removal_strength
    assert cfgs["faithful"].sharpen_strength == 0.0
    assert cfgs["phone_photo"].enable_perspective


def test_pipeline_input_not_modified(samples):
    pipe = ProcessingPipeline()
    rgb = samples["yellow"]
    snap = rgb.copy()
    pipe.process_page(_pd(rgb), get_preset("auto"))
    assert np.array_equal(rgb, snap)