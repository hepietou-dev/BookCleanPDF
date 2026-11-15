"""M5.3 结果视图测试：真实 120 页语料 manifest（只读）+  authored 边界夹具。

覆盖：回退不计完全优化、失败页列表（导入/处理/黑边）、回退分级、仅正式 PDF 计完成、
.partial 拒绝、缺失 manifest 优雅降级、输出大小回退到 stat。不以模拟代替真实数据校验。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bookclean.task_result import build_result_view

REAL_TASK = Path(__file__).resolve().parent.parent / "temp" / "m4_6_subset" / "tasks" / "task_20261112_140743_86d276"


def _write_task(tmp_path, pages, proc, border=None, report=None, output_name="out.pdf",
                pdf_bytes=None, manifest_extra=None):
    task_dir = tmp_path / "task_x"
    task_dir.mkdir(exist_ok=True)
    manifest = {
        "task_id": "task_x", "total_pages": len(pages), "pages": pages,
        "inputs": [{"path": str(tmp_path / f"src{i}.png"), "kind": "image", "page_count": 1, "error": None}
                   for i in range(len(pages))],
        "processing_summary": proc, "elapsed_seconds": 1.0,
        "file_errors": [],
    }
    if border is not None:
        manifest["border_summary"] = border
    if manifest_extra:
        manifest.update(manifest_extra)
    (task_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / output_name
    if report is not None:
        report = dict(report)
        report.setdefault("output_file", str(out))
        (tmp_path / "export_report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    if pdf_bytes is not None:
        out.write_bytes(pdf_bytes)
    return task_dir, out


def _page(idx, **kw):
    base = {"page_index": idx - 1, "display_index": idx, "status": "completed",
            "processing_status": "completed", "fallback_level": "none", "fallback_reason": "",
            "final_image_source": "enhanced", "source_path": f"/src{idx}.png",
            "image_path": f"original_pages/page_{idx:06d}.png",
            "processed_image_path": f"processed_pages/page_{idx:06d}.png"}
    base.update(kw)
    return base


# ---- 真实语料（只读，不重跑处理） ----------------------------------------

@pytest.mark.skipif(not (REAL_TASK / "manifest.json").exists(), reason="真实 120 页语料任务目录不存在")
def test_real_corpus_result_consistency():
    r = build_result_view(REAL_TASK)
    assert r["manifest_exists"] is True
    assert r["total_pages"] == 120
    assert r["processed"] == 120
    assert r["fallbacks"] == 38
    assert r["optimized"] == 82  # 120 - 38，回退不计完全优化
    assert r["failed"] == 0
    assert r["fallback_breakdown"] == {"background": 0, "geometry": 32, "original": 6}
    assert len(r["fallback_pages"]) == 38
    assert sum(r["fallback_breakdown"].values()) == r["fallbacks"]
    assert r["border"] == {"cleaned": 81, "skipped": 39, "failed": 0}
    assert r["failed_pages"] == []
    # 每页回退级别与清单一致
    levels = {p["level"] for p in r["fallback_pages"]}
    assert levels <= {"background", "geometry", "original"}


# ---- 完成态（authored 夹具 + 真实 PDF 文件） -----------------------------

def test_completed_optimized_excludes_fallback(tmp_path):
    pages = [_page(1), _page(2, fallback_level="geometry", fallback_reason="text_core",
                             final_image_source="geometry"), _page(3)]
    proc = {"processed": 3, "fallbacks": 1, "failed": 0, "reused": 0, "elapsed_seconds": 10.0}
    border = {"cleaned": 2, "skipped": 1, "failed": 0, "elapsed_seconds": 1.0}
    report = {"export_status": "completed", "output_size_bytes": 1000, "validation_status": "passed",
              "validation": {"issues": []}, "color_mode": "preserve", "compression_mode": "balanced",
              "total_export_time_seconds": 0.5, "warnings": []}
    task_dir, out = _write_task(tmp_path, pages, proc, border, report, pdf_bytes=b"%PDF-1.5\n dummy")
    r = build_result_view(task_dir, out)
    assert r["optimized"] == 2
    assert r["fallbacks"] == 1
    assert r["failed"] == 0
    assert r["fallback_breakdown"]["geometry"] == 1
    assert len(r["fallback_pages"]) == 1
    assert r["fallback_pages"][0]["display_index"] == 2
    assert r["fallback_pages"][0]["reason"] == "text_core"
    assert r["output_exists"] is True
    assert r["output_size_bytes"] == 1000
    assert r["validation_status"] == "passed"
    assert r["elapsed_seconds"] == pytest.approx(12.5)  # 1 import + 10 proc + 1 border + 0.5 export
    assert Path(r["report_path"]).exists()


def test_fallback_page_image_path_prefers_border(tmp_path):
    pages = [_page(1, fallback_level="geometry", border_status="completed",
                   border_image_path="border_pages/page_000001.png")]
    proc = {"processed": 1, "fallbacks": 1, "failed": 0}
    task_dir, out = _write_task(tmp_path, pages, proc)
    r = build_result_view(task_dir, out)
    assert r["fallback_pages"][0]["image_path"].replace("\\", "/").endswith("border_pages/page_000001.png")


# ---- 失败页列表（导入/处理/黑边） ----------------------------------------

def test_failed_pages_all_stages(tmp_path):
    pages = [
        _page(1, status="failed", error="PDFCorruptError: 损坏"),
        _page(2, processing_status="failed", processing_error="ValueError: bad"),
        _page(3, border_status="failed", border_error="RuntimeError: boom"),
    ]
    proc = {"processed": 0, "fallbacks": 0, "failed": 3}
    task_dir, out = _write_task(tmp_path, pages, proc)
    r = build_result_view(task_dir, out)
    assert r["failed"] == 3
    stages = {p["stage"] for p in r["failed_pages"]}
    assert stages == {"import", "processing", "border"}
    reasons = {p["display_index"]: p["reason"] for p in r["failed_pages"]}
    assert "PDFCorruptError" in reasons[1]
    assert "ValueError: bad" in reasons[2]
    assert "RuntimeError: boom" in reasons[3]
    assert r["output_exists"] is False


# ---- 取消/部分完成：无正式输出 -------------------------------------------

def test_cancelled_no_final_output(tmp_path):
    pages = [_page(1), _page(2)]
    proc = {"processed": 2, "fallbacks": 0, "failed": 0}
    task_dir, out = _write_task(tmp_path, pages, proc)  # 无 export_report、无 PDF
    r = build_result_view(task_dir, out)
    assert r["output_exists"] is False
    assert r["export_status"] == ""
    assert r["validation_status"] == ""
    assert r["optimized"] == 2


def test_partial_pdf_not_counted(tmp_path):
    pages = [_page(1)]
    proc = {"processed": 1, "fallbacks": 0, "failed": 0}
    task_dir, _ = _write_task(tmp_path, pages, proc, output_name="out.partial.pdf",
                              pdf_bytes=b"%PDF-1.5\n partial")
    partial = tmp_path / "out.partial.pdf"
    r = build_result_view(task_dir, partial)
    assert r["output_exists"] is False  # .partial 一律不算完成


def test_output_size_falls_back_to_stat(tmp_path):
    pages = [_page(1)]
    proc = {"processed": 1, "fallbacks": 0, "failed": 0}
    report = {"export_status": "completed", "validation_status": "passed", "validation": {"issues": []}}
    task_dir, out = _write_task(tmp_path, pages, proc, report=report, pdf_bytes=b"%PDF-1.5\n1234567890")
    r = build_result_view(task_dir, out)
    assert r["output_exists"] is True
    assert r["output_size_bytes"] == out.stat().st_size  # 报告无 size → 用真实文件 stat


# ---- 缺失 manifest 优雅降级 ----------------------------------------------

def test_missing_manifest_graceful(tmp_path):
    task_dir = tmp_path / "empty_task"
    task_dir.mkdir()
    r = build_result_view(task_dir, tmp_path / "none.pdf")
    assert r["manifest_exists"] is False
    assert r["total_pages"] == 0
    assert r["optimized"] == 0
    assert r["failed_pages"] == []
    assert r["output_exists"] is False


def test_validation_issues_surfaced(tmp_path):
    pages = [_page(1)]
    proc = {"processed": 1, "fallbacks": 0, "failed": 0}
    report = {"export_status": "failed", "validation_status": "failed",
              "validation": {"issues": ["page 1 size mismatch", "page 1 blank"]}}
    task_dir, out = _write_task(tmp_path, pages, proc, report=report)
    r = build_result_view(task_dir, out)
    assert r["validation_status"] == "failed"
    assert len(r["validation_issues"]) == 2
    assert r["output_exists"] is False