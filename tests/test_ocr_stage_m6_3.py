# -*- coding: utf-8 -*-
"""M6.3 验收测试：OCR 任务阶段集成（OcrStageRunner）。

覆盖：缓存键逐维失效（图像/引擎/模型/参数/schema 变即失效；布局/操作参数变不失效）、
逐页缓存（仅 completed 落盘、续跑复用、failed 重试）、Manifest 只增 ocr_* 字段、
失败隔离（failed/全失败/插件缺失不阻塞导出）、OCR 关内容字节级一致、取消页边界协作、
长文档周期性重启、依赖方向隔离（导出内核不导入 OCR；OCR 关不导入任何 OCR 代码）、
OcrConfig 校验、controller 映射与预检告警。

全部离线：合成图像 + FakeProvider（确定性，不启动子进程、不依赖真实插件）。
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pikepdf
import pypdfium2 as pdfium
import pytest
from PIL import Image

from bookclean.config import OCR_ENGINES, OcrConfig, PDFExportConfig
from bookclean.ocr.fake import FakeProvider
from bookclean.ocr.provider import (
    OCR_SCHEMA_VERSION,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_SKIPPED,
    EngineVersion,
    InitResult,
    OcrError,
    OcrParams,
)
from bookclean.ocr.text_source import OcrTextSource
from bookclean.ocr_stage import OcrStageRunner, compute_cache_key
from bookclean.pdf_exporter import PDFExporter


# ---------------------------------------------------------------- 工具

def solid(color, size=(200, 280)):
    return Image.new("RGB", size, color)


def make_task(tmp_path, specs):
    """与 test_pdf_exporter.make_task 同构的任务目录（原始/处理页 + manifest）。"""
    task = tmp_path / "task"
    (task / "original_pages").mkdir(parents=True)
    (task / "processed_pages").mkdir(parents=True)
    manifest = {"task_id": "t_m63", "pages": [], "inputs": []}
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


def three_page_task(tmp_path):
    return make_task(tmp_path, [
        {"orig": solid((10, 10, 10)), "proc": solid((220, 220, 220))},
        {"orig": solid((20, 20, 20)), "proc": solid((200, 200, 200))},
        {"orig": solid((30, 30, 30)), "proc": solid((180, 180, 180))},
    ])


def run_stage(task, provider=None, cancel_event=None, resume=False, progress_cb=None, **cfg):
    config = OcrConfig(enabled=True, **cfg)
    return OcrStageRunner().run(task, config, progress_cb=progress_cb,
                                cancel_event=cancel_event, resume=resume, provider=provider)


def read_manifest(task):
    return json.loads((task / "manifest.json").read_text(encoding="utf-8"))


def export(task, tmp_path, name, text_source=None, **cfg):
    out = tmp_path / name
    cfg.setdefault("overwrite", True)
    info = PDFExporter(PDFExportConfig(**cfg)).export(task, out, text_source=text_source)
    return out, info


def extract_pages(path):
    doc = pdfium.PdfDocument(str(path))
    texts = [doc[i].get_textpage().get_text_bounded() for i in range(len(doc))]
    doc.close()
    return texts


def render_arrays(path, dpi=100):
    doc = pdfium.PdfDocument(str(path))
    arrs = [np.asarray(doc[i].render(scale=dpi / 72.0).to_pil().convert("RGB"), dtype=np.int16)
            for i in range(len(doc))]
    doc.close()
    return arrs


def pdf_content_bytes(path):
    """PDF 实质内容字节：逐页内容流 + 图像 XObject + 资源键（不含元数据/ID）。"""
    sig = []
    with pikepdf.open(path) as pdf:
        for page in pdf.pages:
            c = page.obj.get("/Contents")
            if isinstance(c, pikepdf.Array):
                data = b"".join(s.read_bytes() for s in c)
            elif c is not None:
                data = c.read_bytes()
            else:
                data = b""
            res = page.obj.get("/Resources", {})
            res_keys = sorted(str(k) for k in res.keys())
            imgs = []
            for name, xo in res.get("/XObject", {}).items():
                if xo.get("/Subtype") == pikepdf.Name("/Image"):
                    imgs.append((str(name), xo.read_raw_bytes()))
            sig.append((data, res_keys, sorted(imgs)))
    return sig


class TrackingProvider(FakeProvider):
    """记录 initialize/shutdown 调用次数，验证长文档周期性重启。"""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.init_calls = 0
        self.shutdown_calls = 0

    def initialize(self, params=None):
        self.init_calls += 1
        return super().initialize(params)

    def shutdown(self):
        self.shutdown_calls += 1
        super().shutdown()


def _isolation_subprocess(imports, forbidden_prefixes):
    src = str(Path(__file__).resolve().parent.parent / "src")
    code = (
        "import sys; " + imports + "; "
        "bad=[m for m in sys.modules if any(m==p or m.startswith(p+'.') for p in "
        + repr(forbidden_prefixes) + ")]; "
        "print(','.join(sorted(bad))); sys.exit(1 if bad else 0)"
    )
    env = {**os.environ, "PYTHONPATH": src}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)


# ---------------------------------------------------------------- 缓存键逐维失效

_BASE_VER = EngineVersion("ocr-rapid", "1.4.4", "ppocrv4", "a" * 64, 1)
_BASE_PARAMS = OcrParams(threads=4, lang="auto", text_score=0.5, page_timeout_s=120.0)
_BASE_SHA = "f" * 64


def _key(sha=_BASE_SHA, ver=_BASE_VER, params=_BASE_PARAMS):
    return compute_cache_key(sha, ver, params)


def test_cache_key_deterministic():
    assert _key() == _key()
    assert len(_key()) == 64  # sha256 hex


def test_cache_key_image_change_invalidates():
    assert _key(sha="b" * 64) != _key()


@pytest.mark.parametrize("field,value", [
    ("engine_id", "ocr-paddle-bridge"),
    ("engine_version", "1.5.0"),
    ("model_id", "ppocrv5"),
    ("model_sha256", "c" * 64),
    ("params_schema_version", 2),
])
def test_cache_key_engine_fingerprint_change_invalidates(field, value):
    ver = EngineVersion(**{**_BASE_VER.to_dict(), field: value})
    assert _key(ver=ver) != _key()


@pytest.mark.parametrize("field,value", [
    ("threads", 8),
    ("lang", "ch"),
    ("text_score", 0.7),
])
def test_cache_key_recognition_params_change_invalidates(field, value):
    params = OcrParams(**{**_BASE_PARAMS.to_dict(), field: value})
    assert _key(params=params) != _key()


def test_cache_key_page_timeout_does_not_invalidate():
    # page_timeout_s 是操作参数，不影响识别结果 ⇒ 不入键
    params = OcrParams(**{**_BASE_PARAMS.to_dict(), "page_timeout_s": 30.0})
    assert _key(params=params) == _key()


def test_cache_key_schema_version_in_payload(monkeypatch):
    import bookclean.ocr_stage as stage
    base = _key()
    monkeypatch.setattr(stage, "OCR_SCHEMA_VERSION", OCR_SCHEMA_VERSION + 1)
    assert stage.compute_cache_key(_BASE_SHA, _BASE_VER, _BASE_PARAMS) != base


# ---------------------------------------------------------------- 阶段：成功/失败/空/跳过

def test_stage_all_completed(tmp_path):
    task, _ = three_page_task(tmp_path)
    r = run_stage(task, provider=FakeProvider())
    s = r["summary"]
    assert s["completed"] == 3 and s["failed"] == 0 and s["skipped"] == 0 and s["reused"] == 0
    assert s["total_pages"] == 3 and s["coverage"] == 1.0
    assert s["engine_id"] == "fake"
    assert r["available"] is True
    assert r["text_source"] is not None and len(r["text_source"]) == 3
    assert read_manifest(task)["ocr_status"] == "completed"


def test_stage_failed_page_isolated(tmp_path):
    task, _ = three_page_task(tmp_path)
    r = run_stage(task, provider=FakeProvider(fail_pages={1}))
    s = r["summary"]
    assert s["completed"] == 2 and s["failed"] == 1
    assert len(r["text_source"]) == 2  # failed 页不入文字层
    m = read_manifest(task)
    assert m["ocr_status"] == "completed_with_errors"
    assert [p["ocr_status"] for p in m["pages"]] == ["completed", "failed", "completed"]
    assert m["pages"][1]["ocr_error"].startswith("ENGINE_ERROR:")
    assert m["pages"][0]["ocr_error"] is None
    # failed 页不缓存；completed 页缓存
    assert (task / "ocr" / "page_000001.json").is_file()
    assert not (task / "ocr" / "page_000002.json").exists()
    assert (task / "ocr" / "page_000003.json").is_file()


def test_stage_empty_page_completed_but_no_text(tmp_path):
    task, _ = three_page_task(tmp_path)
    r = run_stage(task, provider=FakeProvider(empty_pages={0}))
    s = r["summary"]
    assert s["completed"] == 3  # 空页仍 completed
    assert len(r["text_source"]) == 2  # 无行 ⇒ 不注入
    assert s["coverage"] == round(2 / 3, 4)  # coverage 按"有文字"页计
    assert read_manifest(task)["pages"][0]["ocr_text_chars"] == 0


def test_stage_missing_image_skipped(tmp_path):
    task, _ = make_task(tmp_path, [
        {"orig": solid((10, 10, 10)), "proc": solid((220, 220, 220))},
        {},  # 无原始/处理图 ⇒ FinalImageResolver path None ⇒ skipped
    ])
    r = run_stage(task, provider=FakeProvider())
    s = r["summary"]
    assert s["completed"] == 1 and s["skipped"] == 1 and s["failed"] == 0
    m = read_manifest(task)
    assert m["pages"][1]["ocr_status"] == STATUS_SKIPPED
    assert m["pages"][1]["ocr_source"] == "none"
    assert m["pages"][1]["ocr_error"].startswith("MISSING_IMAGE:")
    assert len(r["text_source"]) == 1


def test_stage_all_failed_text_source_none(tmp_path):
    task, _ = three_page_task(tmp_path)
    r = run_stage(task, provider=FakeProvider(fail_pages={0, 1, 2}))
    assert r["summary"]["failed"] == 3 and r["summary"]["completed"] == 0
    assert r["text_source"] is None  # 全失败 ⇒ 钩子 None ⇒ 导出与 M5 一致
    assert r["available"] is True
    assert read_manifest(task)["ocr_status"] == "completed_with_errors"
    assert not (task / "ocr" / "page_000001.json").exists()


def test_stage_per_page_source_and_progress_order(tmp_path):
    task, _ = three_page_task(tmp_path)
    seen = []
    r = run_stage(task, provider=FakeProvider(), progress_cb=lambda e: seen.append(e["display_index"]))
    assert seen == [1, 2, 3]  # 页序进度
    m = read_manifest(task)
    assert all(p["ocr_source"] == "fresh" for p in m["pages"])
    assert all(p["ocr_status"] == STATUS_COMPLETED for p in m["pages"])
    assert all(isinstance(p["ocr_ms"], (int, float)) for p in m["pages"])


# ---------------------------------------------------------------- 缓存复用 / 续跑

def test_stage_cache_reuse_second_run(tmp_path):
    task, _ = three_page_task(tmp_path)
    run_stage(task, provider=FakeProvider())
    r2 = run_stage(task, provider=FakeProvider(), resume=True)
    s2 = r2["summary"]
    assert s2["reused"] == 3 and s2["completed"] == 3 and s2["failed"] == 0
    assert all(p["ocr_source"] == "cached" for p in read_manifest(task)["pages"])
    assert len(r2["text_source"]) == 3


def test_stage_resume_retries_failed(tmp_path):
    task, _ = three_page_task(tmp_path)
    run_stage(task, provider=FakeProvider(fail_pages={1}))
    r2 = run_stage(task, provider=FakeProvider(), resume=True)  # 不再失败
    s2 = r2["summary"]
    assert s2["reused"] == 2 and s2["completed"] == 3 and s2["failed"] == 0
    m = read_manifest(task)
    assert m["ocr_status"] == "completed"
    assert m["pages"][1]["ocr_source"] == "fresh"  # failed 页重跑
    assert (task / "ocr" / "page_000002.json").is_file()  # 重试成功后落盘


def test_stage_cache_key_stable_across_layout_change(tmp_path):
    # 布局不入缓存键：导出布局变化后重跑 OCR 仍全部命中缓存（只重注入，不重识别）
    task, _ = three_page_task(tmp_path)
    run_stage(task, provider=FakeProvider())
    keys_before = [p["ocr_cache_key"] for p in read_manifest(task)["pages"]]
    export(task, tmp_path, "layout_a.pdf", text_source=OcrTextSource({}), target_dpi=200)
    export(task, tmp_path, "layout_b.pdf", text_source=OcrTextSource({}), target_dpi=400)
    r2 = run_stage(task, provider=FakeProvider(), resume=True)
    keys_after = [p["ocr_cache_key"] for p in read_manifest(task)["pages"]]
    assert keys_before == keys_after
    assert r2["summary"]["reused"] == 3


def test_stage_corrupt_cache_falls_back_to_fresh(tmp_path):
    task, _ = three_page_task(tmp_path)
    run_stage(task, provider=FakeProvider())
    (task / "ocr" / "page_000001.json").write_text("{ not json", encoding="utf-8")
    r2 = run_stage(task, provider=FakeProvider(), resume=True)
    assert r2["summary"]["reused"] == 2 and r2["summary"]["completed"] == 3
    assert read_manifest(task)["pages"][0]["ocr_source"] == "fresh"


# ---------------------------------------------------------------- Manifest 只增 / 产物

def test_manifest_ocr_fields_additive_only(tmp_path):
    task, _ = three_page_task(tmp_path)
    before = read_manifest(task)
    before_pages = copy.deepcopy(before["pages"])
    before_top = {k: v for k, v in before.items() if k != "pages"}
    run_stage(task, provider=FakeProvider(fail_pages={1}))
    after = read_manifest(task)
    # 顶层既有字段不变，仅新增 ocr_status / ocr_summary
    for k, v in before_top.items():
        assert after[k] == v
    assert set(after) - set(before) == {"ocr_status", "ocr_summary"}
    # 每页 M1–M5 字段逐一不变，仅新增 ocr_* 字段
    for bp, ap in zip(before_pages, after["pages"]):
        for k, v in bp.items():
            assert ap[k] == v, f"既有字段被改动: {k}"
        added = set(ap) - set(bp)
        assert added == {"ocr_status", "ocr_cache_key", "ocr_text_chars",
                         "ocr_source", "ocr_ms", "ocr_error"}


def test_stage_writes_ocr_manifest_and_report(tmp_path):
    task, _ = three_page_task(tmp_path)
    run_stage(task, provider=FakeProvider(fail_pages={2}))
    om = json.loads((task / "ocr" / "ocr_manifest.json").read_text(encoding="utf-8"))
    assert om["schema_version"] == 1
    assert om["ocr_schema_version"] == OCR_SCHEMA_VERSION
    assert om["engine_version"]["engine_id"] == "fake"
    assert set(om["pages"]) == {"0", "1", "2"}
    assert om["pages"]["2"]["status"] == STATUS_FAILED
    assert om["pages"]["0"]["cache_key"]
    report = json.loads((task / "ocr_report.json").read_text(encoding="utf-8"))
    assert report["completed"] == 2 and report["failed"] == 1


# ---------------------------------------------------------------- 插件缺失降级

def test_plugin_missing_degrades_without_side_effects(tmp_path):
    task, _ = three_page_task(tmp_path)
    empty_root = tmp_path / "no_plugins"
    empty_root.mkdir()
    r = run_stage(task, provider=None, plugins_root=str(empty_root))
    assert r["available"] is False
    assert r["text_source"] is None and r["results"] == {}
    s = r["summary"]
    assert s["skipped"] == 3 and s["completed"] == 0 and "reason" in s
    m = read_manifest(task)
    assert m["ocr_status"] == "unavailable"
    # 降级不产生任何新目录/文件
    assert not (task / "ocr").exists()
    assert not (task / "ocr_report.json").exists()
    # 普通导出照常（M5 路径）
    out, info = export(task, tmp_path, "degraded.pdf")
    assert info["export_status"] == "completed"
    assert all(t == "" for t in extract_pages(out))


def test_degraded_export_byte_identical_to_off(tmp_path):
    task, _ = three_page_task(tmp_path)
    empty_root = tmp_path / "no_plugins"
    empty_root.mkdir()
    run_stage(task, provider=None, plugins_root=str(empty_root))
    off, _ = export(task, tmp_path, "off.pdf")
    degraded, _ = export(task, tmp_path, "degraded.pdf", text_source=None)
    assert pdf_content_bytes(off) == pdf_content_bytes(degraded)


def _write_plugin_manifest(root, engine_id, data):
    pdir = root / engine_id
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.json").write_text(data, encoding="utf-8")


def test_corrupt_plugin_degrades(tmp_path):
    # 损坏清单（非法 JSON）⇒ discover 跳过 ⇒ select None ⇒ 阶段降级，导出照常
    task, _ = three_page_task(tmp_path)
    root = tmp_path / "plugins"
    _write_plugin_manifest(root, "ocr-rapid", "{ not valid json")
    r = run_stage(task, provider=None, plugins_root=str(root))
    assert r["available"] is False and r["text_source"] is None
    assert read_manifest(task)["ocr_status"] == "unavailable"
    assert not (task / "ocr").exists()
    out, info = export(task, tmp_path, "corrupt.pdf")
    assert info["export_status"] == "completed"


def test_version_mismatch_plugin_degrades(tmp_path):
    # 协议版本不匹配 ⇒ discover 跳过 ⇒ 降级（不崩溃、不阻塞导出）
    task, _ = three_page_task(tmp_path)
    root = tmp_path / "plugins"
    _write_plugin_manifest(root, "ocr-rapid", json.dumps({
        "schema": 1, "protocol_version": 999, "engine_id": "ocr-rapid",
        "engine_version": "1.4.4", "python": "runtime/python.exe", "worker": "rapid",
        "models": [{"id": "m", "path": "m.onnx", "sha256": "x"}],
    }))
    r = run_stage(task, provider=None, plugins_root=str(root))
    assert r["available"] is False and r["text_source"] is None
    assert read_manifest(task)["ocr_status"] == "unavailable"


class _InitFailProvider(FakeProvider):
    """模拟模型校验失败：initialize 返回 ok=False（MODEL_MISMATCH）。"""

    def initialize(self, params=None):
        return InitResult(ok=False, error=OcrError("MODEL_MISMATCH", "模型哈希不匹配"))


def test_init_failure_all_failed_export_ok(tmp_path):
    # 引擎初始化失败 ⇒ 全页 failed（隔离）⇒ 钩子 None ⇒ 导出与 M5 一致，不阻塞
    task, _ = three_page_task(tmp_path)
    off, _ = export(task, tmp_path, "off.pdf")
    r = run_stage(task, provider=_InitFailProvider())
    s = r["summary"]
    assert s["failed"] == 3 and s["completed"] == 0
    assert r["text_source"] is None and r["available"] is True
    m = read_manifest(task)
    assert m["ocr_status"] == "completed_with_errors"
    assert m["pages"][0]["ocr_error"].startswith("MODEL_MISMATCH:")
    assert all(p["ocr_status"] == STATUS_FAILED for p in m["pages"])
    out, info = export(task, tmp_path, "initfail.pdf", text_source=r["text_source"])
    assert info["export_status"] == "completed"
    assert pdf_content_bytes(off) == pdf_content_bytes(out)


# ---------------------------------------------------------------- 取消

def test_cancel_at_page_boundary_preserves_completed_cache(tmp_path):
    task, _ = three_page_task(tmp_path)
    ev = threading.Event()
    seen = []

    def cb(entry):
        seen.append(entry["display_index"])
        if len(seen) == 1:
            ev.set()  # 第 1 页完成后取消

    r = run_stage(task, provider=FakeProvider(), cancel_event=ev, progress_cb=cb)
    assert r["summary"]["completed"] == 1
    m = read_manifest(task)
    assert m["ocr_status"] == "cancelled"
    assert m["pages"][0]["ocr_status"] == STATUS_COMPLETED
    assert m["pages"][1]["ocr_status"] == "cancelled"
    assert (task / "ocr" / "page_000001.json").is_file()  # 已完成页缓存保留
    assert not (task / "ocr" / "page_000002.json").exists()
    assert len(r["text_source"]) == 1


def test_cancel_before_start_processes_nothing(tmp_path):
    task, _ = three_page_task(tmp_path)
    ev = threading.Event()
    ev.set()
    r = run_stage(task, provider=FakeProvider(), cancel_event=ev)
    assert r["summary"]["completed"] == 0
    m = read_manifest(task)
    assert m["ocr_status"] == "cancelled"
    assert m["pages"][0]["ocr_status"] == "cancelled"
    assert r["text_source"] is None


def test_cancel_then_resume_completes(tmp_path):
    task, _ = three_page_task(tmp_path)
    ev = threading.Event()
    seen = []

    def cb(entry):
        seen.append(entry["display_index"])
        if len(seen) == 1:
            ev.set()

    run_stage(task, provider=FakeProvider(), cancel_event=ev, progress_cb=cb)
    r2 = run_stage(task, provider=FakeProvider(), resume=True)  # 无取消
    assert r2["summary"]["reused"] == 1 and r2["summary"]["completed"] == 3
    assert read_manifest(task)["ocr_status"] == "completed"


# ---------------------------------------------------------------- 长文档周期性重启

def test_periodic_restart_every_n_pages(tmp_path):
    task, _ = make_task(tmp_path, [{"orig": solid((i, i, i)), "proc": solid((200, 200, 200))}
                                   for i in range(5)])
    prov = TrackingProvider()
    r = run_stage(task, provider=prov, restart_every_n_pages=2)
    assert r["summary"]["completed"] == 5
    assert r["summary"]["restarts"] == 2  # 满 2/4 页后各重启一次
    assert prov.init_calls == 3  # 初次 + 2 次重启
    assert prov.shutdown_calls == 3  # 2 次重启 + 末尾关闭


def test_no_restart_when_disabled(tmp_path):
    task, _ = three_page_task(tmp_path)
    prov = TrackingProvider()
    r = run_stage(task, provider=prov, restart_every_n_pages=0)
    assert r["summary"]["restarts"] == 0
    assert prov.init_calls == 1 and prov.shutdown_calls == 1


def test_restart_does_not_break_cache_reuse(tmp_path):
    task, _ = make_task(tmp_path, [{"orig": solid((i, i, i)), "proc": solid((200, 200, 200))}
                                   for i in range(5)])
    run_stage(task, provider=TrackingProvider(), restart_every_n_pages=2)
    r2 = run_stage(task, provider=TrackingProvider(), resume=True, restart_every_n_pages=2)
    assert r2["summary"]["reused"] == 5 and r2["summary"]["restarts"] == 0  # 全缓存⇒不重启


# ---------------------------------------------------------------- 导出集成

def test_export_failed_page_does_not_block(tmp_path):
    task, _ = three_page_task(tmp_path)
    r = run_stage(task, provider=FakeProvider(fail_pages={1}))
    out, info = export(task, tmp_path, "searchable.pdf", text_source=r["text_source"])
    assert info["validation_status"] == "passed"
    assert info["export_status"] == "completed"
    texts = extract_pages(out)
    assert len(texts[0]) > 0 and len(texts[2]) > 0 and len(texts[1]) == 0
    assert info["ocr"]["pages_with_text"] == 2


def test_export_ocr_on_pixel_identical_to_off(tmp_path):
    task, _ = three_page_task(tmp_path)
    off, _ = export(task, tmp_path, "off.pdf")
    r = run_stage(task, provider=FakeProvider())
    search, info = export(task, tmp_path, "search.pdf", text_source=r["text_source"])
    a, b = render_arrays(off), render_arrays(search)
    assert all(int(np.abs(x - y).max()) == 0 for x, y in zip(a, b))
    assert info["validation"]["passed"] is True


def test_export_ocr_off_content_byte_identical_empty_hook(tmp_path):
    task, _ = three_page_task(tmp_path)
    a, _ = export(task, tmp_path, "a.pdf")
    c, _ = export(task, tmp_path, "c.pdf", text_source=OcrTextSource({}))
    assert pdf_content_bytes(a) == pdf_content_bytes(c)
    for t in extract_pages(a):
        assert t == ""


def test_export_all_failed_identical_to_off(tmp_path):
    task, _ = three_page_task(tmp_path)
    off, _ = export(task, tmp_path, "off.pdf")
    r = run_stage(task, provider=FakeProvider(fail_pages={0, 1, 2}))
    assert r["text_source"] is None
    failed, _ = export(task, tmp_path, "failed.pdf", text_source=r["text_source"])
    assert pdf_content_bytes(off) == pdf_content_bytes(failed)


# ---------------------------------------------------------------- 依赖方向隔离

def test_isolation_exporter_imports_no_ocr():
    r = _isolation_subprocess(
        "import bookclean.pdf_exporter, bookclean.text_layer",
        ["bookclean.ocr", "bookclean.ocr_stage"])
    assert r.returncode == 0, f"导出内核导入了 OCR 模块: {r.stdout} {r.stderr}"


def test_isolation_ocr_off_worker_imports_no_ocr():
    # OCR 关路径：worker/controller/config 均不连带导入任何 OCR 代码（惰性导入守卫）
    r = _isolation_subprocess(
        "import bookclean.task_worker, bookclean.app_controller, bookclean.config",
        ["bookclean.ocr", "bookclean.ocr_stage"])
    assert r.returncode == 0, f"OCR 关路径导入了 OCR 模块: {r.stdout} {r.stderr}"


def test_isolation_ocr_stage_lazy_imports_exporter():
    # ocr_stage 模块加载不得连带导入导出内核（FinalImageResolver 在 run 内惰性导入）
    r = _isolation_subprocess("import bookclean.ocr_stage", ["bookclean.pdf_exporter"])
    assert r.returncode == 0, f"ocr_stage 模块级导入了导出内核: {r.stdout} {r.stderr}"


# ---------------------------------------------------------------- OcrConfig 校验

def test_ocr_config_default_valid_and_disabled():
    c = OcrConfig()
    c.validate()
    assert c.enabled is False and c.engine == "auto"


def test_ocr_config_valid_boundaries():
    OcrConfig(enabled=True, engine="ocr-rapid", threads=1, text_score=0.0,
              page_timeout_s=1.0, restart_every_n_pages=0).validate()
    OcrConfig(enabled=True, threads=64, text_score=1.0,
              page_timeout_s=3600.0, restart_every_n_pages=100000).validate()


@pytest.mark.parametrize("kw", [
    {"enabled": "yes"},
    {"engine": "bogus"},
    {"threads": 0},
    {"threads": 65},
    {"threads": 4.0},
    {"text_score": -0.1},
    {"text_score": 1.1},
    {"page_timeout_s": 0.5},
    {"page_timeout_s": 3601},
    {"restart_every_n_pages": -1},
    {"restart_every_n_pages": 100001},
])
def test_ocr_config_invalid_rejected(kw):
    with pytest.raises(ValueError):
        OcrConfig(**kw).validate()


def test_ocr_engines_enum():
    assert OCR_ENGINES == ("auto", "ocr-rapid", "ocr-paddle-bridge")


# ---------------------------------------------------------------- controller 映射 / 预检

def test_controller_build_ocr_config_disabled_returns_none(tmp_path):
    from bookclean.app_controller import AppController, TaskSettings
    from bookclean.user_paths import UserPaths
    ctl = AppController(paths=UserPaths(root=tmp_path / "data"))
    ctl.set_settings(TaskSettings(ocr_enabled=False))
    assert ctl.build_ocr_config() is None


def test_controller_build_ocr_config_enabled(tmp_path):
    from bookclean.app_controller import AppController, TaskSettings
    from bookclean.user_paths import UserPaths
    ctl = AppController(paths=UserPaths(root=tmp_path / "data"))
    ctl.set_settings(TaskSettings(ocr_enabled=True, ocr_engine="ocr-rapid"))
    cfg = ctl.build_ocr_config()
    assert isinstance(cfg, OcrConfig) and cfg.enabled is True and cfg.engine == "ocr-rapid"


def test_controller_build_ocr_config_invalid_engine(tmp_path):
    from bookclean.app_controller import AppController, TaskSettings
    from bookclean.user_paths import UserPaths
    ctl = AppController(paths=UserPaths(root=tmp_path / "data"))
    ctl.set_settings(TaskSettings(ocr_enabled=True, ocr_engine="bogus"))
    with pytest.raises(ValueError):
        ctl.build_ocr_config()


def test_controller_build_configs_still_three_tuple(tmp_path):
    # 向后兼容：build_configs 签名不变（OCR 走独立 build_ocr_config）
    from bookclean.app_controller import AppController, TaskSettings
    from bookclean.user_paths import UserPaths
    ctl = AppController(paths=UserPaths(root=tmp_path / "data"))
    ctl.set_settings(TaskSettings(ocr_enabled=True))
    assert len(ctl.build_configs()) == 3


def test_controller_validate_warns_when_plugin_missing(tmp_path, img_factory, monkeypatch):
    from bookclean.app_controller import AppController, TaskSettings
    from bookclean.ocr.plugin_manager import PluginManager
    from bookclean.ocr.provider import Availability
    from bookclean.user_paths import UserPaths
    monkeypatch.setattr(PluginManager, "check_available",
                        lambda self, engine_id="auto": Availability(installed=False, reason="未安装"))
    ctl = AppController(paths=UserPaths(root=tmp_path / "data"))
    ctl.inspect_inputs([img_factory("a.png")])
    ctl.set_settings(TaskSettings(ocr_enabled=True, output_dir=str(tmp_path / "out")))
    r = ctl.validate_start()
    assert r.ok
    assert any("未安装 OCR 插件" in w for w in r.warnings)


def test_controller_validate_no_warning_when_plugin_present(tmp_path, img_factory, monkeypatch):
    from bookclean.app_controller import AppController, TaskSettings
    from bookclean.ocr.plugin_manager import PluginManager
    from bookclean.ocr.provider import Availability
    from bookclean.user_paths import UserPaths
    monkeypatch.setattr(PluginManager, "check_available",
                        lambda self, engine_id="auto": Availability(
                            installed=True, engine_id="ocr-rapid", engine_version="1.4.4", models_ok=True))
    ctl = AppController(paths=UserPaths(root=tmp_path / "data"))
    ctl.inspect_inputs([img_factory("a.png")])
    ctl.set_settings(TaskSettings(ocr_enabled=True, output_dir=str(tmp_path / "out")))
    r = ctl.validate_start()
    assert r.ok
    assert not any("未安装 OCR 插件" in w for w in r.warnings)


def test_controller_validate_ocr_off_no_plugin_check(tmp_path, img_factory, monkeypatch):
    # OCR 关 ⇒ 预检不触碰插件发现（即使 check_available 会抛错也不应被调用）
    from bookclean.app_controller import AppController, TaskSettings
    from bookclean.ocr.plugin_manager import PluginManager
    from bookclean.user_paths import UserPaths

    def boom(self, engine_id="auto"):
        raise AssertionError("OCR 关时不应检查插件")

    monkeypatch.setattr(PluginManager, "check_available", boom)
    ctl = AppController(paths=UserPaths(root=tmp_path / "data"))
    ctl.inspect_inputs([img_factory("a.png")])
    ctl.set_settings(TaskSettings(ocr_enabled=False, output_dir=str(tmp_path / "out")))
    r = ctl.validate_start()
    assert r.ok


def test_controller_validate_invalid_ocr_engine_error(tmp_path, img_factory):
    from bookclean.app_controller import AppController, TaskSettings
    from bookclean.user_paths import UserPaths
    ctl = AppController(paths=UserPaths(root=tmp_path / "data"))
    ctl.inspect_inputs([img_factory("a.png")])
    ctl.set_settings(TaskSettings(ocr_enabled=True, ocr_engine="bogus",
                                  output_dir=str(tmp_path / "out")))
    r = ctl.validate_start()
    assert not r.ok and any("OCR 设置不合法" in e for e in r.errors)