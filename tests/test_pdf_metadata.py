"""M4.3 元数据测试：字段/用户优先/Creator/Producer/日期/XMP 与 DocumentInfo 一致/读回验证。"""

import pikepdf
import pytest

from bookclean import __version__
from bookclean.config import PDFExportConfig
from bookclean.pdf_exporter import PDFExporter
from bookclean import pdf_metadata as pm
from test_pdf_exporter import make_task, solid


def _export(tmp_path, **cfg):
    task, _ = make_task(tmp_path, [{"orig": solid((1, 1, 1)), "proc": solid((200, 200, 200))}])
    out = tmp_path / "my_book.pdf"
    info = PDFExporter(PDFExportConfig(**cfg)).export(task, out)
    return out, info


def test_default_title_from_filename(tmp_path):
    out, info = _export(tmp_path)
    got = pm.read_metadata(out)
    assert got["docinfo"]["title"] == "my_book"
    assert got["xmp"]["title"] == "my_book"


def test_user_fields_priority(tmp_path):
    out, info = _export(tmp_path, title="Forgotten Armies", author="C. A. Bayly",
                        subject="Historical scan", keywords="war,history")
    got = pm.read_metadata(out)
    di = got["docinfo"]
    assert di["title"] == "Forgotten Armies"
    assert di["author"] == "C. A. Bayly"
    assert di["subject"] == "Historical scan"
    assert di["keywords"] == "war,history"
    assert got["xmp"]["author"] == "C. A. Bayly"


def test_creator_producer_version(tmp_path):
    out, _ = _export(tmp_path)
    di = pm.read_metadata(out)["docinfo"]
    assert di["creator"] == "BookClean PDF"
    assert di["producer"] == f"BookClean PDF {__version__}"


def test_dates_present(tmp_path):
    out, _ = _export(tmp_path)
    got = pm.read_metadata(out)
    assert got["docinfo"]["creation_date"].startswith("D:")
    assert got["docinfo"]["mod_date"].startswith("D:")
    assert got["xmp"]["creation_date"] and got["xmp"]["mod_date"]


def test_no_author_when_unspecified(tmp_path):
    out, _ = _export(tmp_path)
    got = pm.read_metadata(out)
    assert got["docinfo"]["author"] is None
    assert got["xmp"]["author"] is None


def test_xmp_docinfo_consistency(tmp_path):
    out, info = _export(tmp_path, title="T", author="A", keywords="k")
    issues = pm.verify_metadata(out, info["metadata"])
    assert issues == []


def test_verify_detects_mismatch(tmp_path):
    out, info = _export(tmp_path, title="Real")
    bad = dict(info["metadata"], title="Wrong")
    issues = pm.verify_metadata(out, bad)
    assert any("title" in i for i in issues)


def test_metadata_survives_reopen(tmp_path):
    out, info = _export(tmp_path, title="Persist")
    with pikepdf.open(out) as pdf:
        assert str(pdf.docinfo.Title) == "Persist"