"""M5.4 结果视图状态/可恢复派生 + 逐页总数（resume 正确）测试。

关键回归：续跑时 processing_summary.processed 只计新处理页（复用页计入 reused），
结果视图总数必须以逐页 processing_status/fallback_level 为准，否则续跑后会少计。
state/resumable 由现有 Manifest/报告字段派生（架构 §4.1/§4.2），.partial 一律不认终。
authored 夹具，不重跑真实处理。
"""
from __future__ import annotations

import json
from pathlib import Path

from bookclean.task_result import build_result_view


def _page(idx, **kw):
    base = {"page_index": idx - 1, "display_index": idx, "status": "completed",
            "processing_status": "completed", "fallback_level": "none", "fallback_reason": "",
            "final_image_source": "enhanced", "source_path": f"/src{idx}.png",
            "image_path": f"original_pages/page_{idx:06d}.png",
            "processed_image_path": f"processed_pages/page_{idx:06d}.png"}
    base.update(kw)
    return base


def _write(tmp_path, pages, proc=None, report=None, pdf_bytes=None, output_name="out.pdf",
           manifest_extra=None):
    task_dir = tmp_path / "task_x"
    task_dir.mkdir(exist_ok=True)
    manifest = {"task_id": "task_x", "total_pages": len(pages), "pages": pages,
                "inputs": [], "file_errors": [], "elapsed_seconds": 1.0}
    if proc is not None:
        manifest["processing_summary"] = proc
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


# ---- 逐页总数：续跑 summary 少计时仍以逐页为准 ----------------------------

def test_per_page_totals_ignore_resume_undercount(tmp_path):
    pages = [_page(1), _page(2, fallback_level="geometry", fallback_reason="text_core"), _page(3)]
    # 续跑：仅 1 页新处理，2 页复用 → summary.processed=1（少计），reused=2
    proc = {"processed": 1, "fallbacks": 0, "failed": 0, "reused": 2, "elapsed_seconds": 5.0}
    task_dir, out = _write(tmp_path, pages, proc)
    r = build_result_view(task_dir, out)
    assert r["processed"] == 3      # 逐页 processing_status==completed，而非 summary 的 1
    assert r["fallbacks"] == 1      # 逐页 fallback_level
    assert r["optimized"] == 2      # 3 - 1，回退不计完全优化
    assert r["reused"] == 2         # 复用数仍取自 summary
    assert r["failed"] == 0


def test_per_page_failed_counts_all_stages(tmp_path):
    pages = [_page(1, status="failed", error="ImportError"),
             _page(2, processing_status="failed", processing_error="ValueError"),
             _page(3, border_status="failed", border_error="RuntimeError")]
    proc = {"processed": 0, "fallbacks": 0, "failed": 0}  # summary 故意少计
    task_dir, out = _write(tmp_path, pages, proc)
    r = build_result_view(task_dir, out)
    assert r["failed"] == 3  # 以逐页失败列表为准
    assert {p["stage"] for p in r["failed_pages"]} == {"import", "processing", "border"}


# ---- state 派生 ----------------------------------------------------------

def test_state_completed(tmp_path):
    pages = [_page(1)]
    report = {"export_status": "completed", "validation_status": "passed", "validation": {"issues": []}}
    task_dir, out = _write(tmp_path, pages, {"processed": 1}, report, pdf_bytes=b"%PDF-1.5\nx")
    r = build_result_view(task_dir, out)
    assert r["state"] == "completed"
    assert r["resumable"] is False
    assert r["output_exists"] is True


def test_state_cancelled(tmp_path):
    pages = [_page(1), _page(2, processing_status="cancelled")]
    task_dir, out = _write(tmp_path, pages, {"processed": 1})
    r = build_result_view(task_dir, out)
    assert r["state"] == "cancelled"
    assert r["resumable"] is True


def test_state_incomplete_no_output(tmp_path):
    pages = [_page(1), _page(2)]
    task_dir, out = _write(tmp_path, pages, {"processed": 2})  # 处理完成但无导出/PDF
    r = build_result_view(task_dir, out)
    assert r["state"] == "incomplete"
    assert r["resumable"] is True
    assert r["output_exists"] is False


def test_state_failed_manifest(tmp_path):
    pages = [_page(1, status="failed", error="PDFCorruptError", image_path=None,
                   processed_image_path=None, processing_status="skipped")]
    task_dir, out = _write(tmp_path, pages, {"processed": 0, "failed": 1},
                           manifest_extra={"status": "failed"})
    r = build_result_view(task_dir, out)
    assert r["state"] == "failed"
    # 无已导入页（image_path 缺失）→ 不可恢复（绝不重跑 M1）
    assert r["imported"] == 0
    assert r["resumable"] is False


def test_state_failed_validation(tmp_path):
    pages = [_page(1)]
    report = {"export_status": "failed", "validation_status": "failed",
              "validation": {"issues": ["page 1 blank"]}}
    task_dir, out = _write(tmp_path, pages, {"processed": 1}, report)
    r = build_result_view(task_dir, out)
    assert r["state"] == "failed"
    assert r["resumable"] is True  # 有已导入页，可续跑重新导出


def test_state_missing(tmp_path):
    task_dir = tmp_path / "empty"
    task_dir.mkdir()
    r = build_result_view(task_dir, tmp_path / "none.pdf")
    assert r["state"] == "missing"
    assert r["resumable"] is False
    assert r["manifest_exists"] is False


def test_partial_pdf_never_completed(tmp_path):
    pages = [_page(1)]
    report = {"export_status": "completed", "validation_status": "passed"}
    task_dir, _ = _write(tmp_path, pages, {"processed": 1}, report,
                         pdf_bytes=b"%PDF-1.5\npartial", output_name="out.partial.pdf")
    r = build_result_view(task_dir, tmp_path / "out.partial.pdf")
    assert r["output_exists"] is False
    assert r["state"] != "completed"  # .partial 一律不认终
    assert r["resumable"] is True


def test_resumable_requires_imported_pages(tmp_path):
    # 处理状态 cancelled 但无任何已导入页 → 不可恢复
    pages = [_page(1, status="failed", image_path=None, processing_status="cancelled")]
    task_dir, out = _write(tmp_path, pages, {"processed": 0})
    r = build_result_view(task_dir, out)
    assert r["state"] == "cancelled"
    assert r["imported"] == 0
    assert r["resumable"] is False