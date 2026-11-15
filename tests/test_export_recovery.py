"""M4.4 原子输出与中断/冲突保护测试。

覆盖：成功后无 .partial 残留；验证失败不覆盖既有正确 PDF；写入中断不破坏既有 PDF；
失败后重试成功且处理结果只读不被改动；导出报告(json/txt)落盘；Windows 文件占用时不破坏既有 PDF。
"""

import json
import os

import pikepdf
import pytest

from bookclean import pdf_exporter
from bookclean.pdf_exporter import ExportError
from test_pdf_exporter import export, make_task, solid


def _task(tmp_path):
    return make_task(tmp_path, [
        {"orig": solid((10, 10, 10)), "proc": solid((220, 220, 220))},
        {"orig": solid((20, 20, 20)), "proc": solid((60, 60, 60))},
    ])


def _fail_verify(self, pdf_path, info, contact_dir=None, progress_cb=None):
    return {"passed": False, "issues": ["模拟验证失败"],
            "page_count": info["summary"]["total_pages"], "per_page": [],
            "contact_sheet": None, "previews": []}


def test_success_leaves_no_partial(tmp_path):
    task, _ = _task(tmp_path)
    out, info = export(task, tmp_path)  # verify 默认开
    assert out.exists()
    assert not (tmp_path / "out.partial.pdf").exists()
    assert info["export_status"] == "completed"
    assert info["validation_status"] == "passed"


def test_verify_failure_keeps_existing_pdf(tmp_path, monkeypatch):
    task, _ = _task(tmp_path)
    out, _ = export(task, tmp_path, verify=False)
    original = out.read_bytes()
    monkeypatch.setattr(pdf_exporter.PDFVerifier, "verify", _fail_verify)
    with pytest.raises(ExportError):
        export(task, tmp_path, overwrite=True)  # verify 默认开 -> 失败
    assert out.read_bytes() == original
    assert not (tmp_path / "out.partial.pdf").exists()


def test_save_interruption_keeps_existing_pdf(tmp_path, monkeypatch):
    task, _ = _task(tmp_path)
    out, _ = export(task, tmp_path, verify=False)
    original = out.read_bytes()

    def boom(self, *a, **k):
        raise OSError("模拟写入中断")

    monkeypatch.setattr(pikepdf.Pdf, "save", boom)
    with pytest.raises(OSError):
        export(task, tmp_path, overwrite=True, verify=False)
    assert out.read_bytes() == original
    assert not (tmp_path / "out.partial.pdf").exists()


def test_retry_after_failure_succeeds_and_processed_untouched(tmp_path, monkeypatch):
    task, m = _task(tmp_path)
    proc_rel = m["pages"][0]["processed_image_path"]
    proc_bytes = (task / proc_rel).read_bytes()
    export(task, tmp_path, verify=False)
    monkeypatch.setattr(pdf_exporter.PDFVerifier, "verify", _fail_verify)
    with pytest.raises(ExportError):
        export(task, tmp_path, overwrite=True)
    monkeypatch.undo()
    out2, info2 = export(task, tmp_path, overwrite=True)
    assert out2.exists() and info2["validation_status"] == "passed"
    assert (task / proc_rel).read_bytes() == proc_bytes  # 处理结果只读，未被改动
    assert not (tmp_path / "out.partial.pdf").exists()


def test_reports_written(tmp_path):
    task, _ = _task(tmp_path)
    out, _ = export(task, tmp_path)
    rj = tmp_path / "export_report.json"
    rt = tmp_path / "export_report.txt"
    assert rj.exists() and rt.exists()
    rep = json.loads(rj.read_text(encoding="utf-8"))
    assert rep["page_count"] == 2
    assert rep["output_size_bytes"] == out.stat().st_size
    assert rep["validation_status"] == "passed"
    assert "图像型" in rep["note"]
    assert "导出报告" in rt.read_text(encoding="utf-8")


@pytest.mark.skipif(os.name != "nt", reason="Windows 文件占用语义")
def test_output_occupied_keeps_existing(tmp_path):
    task, _ = _task(tmp_path)
    out, _ = export(task, tmp_path, verify=False)
    original = out.read_bytes()
    with open(out, "rb"):
        with pytest.raises(PermissionError):
            export(task, tmp_path, overwrite=True, verify=False)
        assert not (tmp_path / "out.partial.pdf").exists()
    assert out.read_bytes() == original