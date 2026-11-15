"""M4.5 CLI 端到端测试：参数映射/默认值/进度显示/退出码/覆盖与输入守卫/元数据/验证失败。

直接调用 scripts/export_pdf.py 的 main(argv)（importlib 加载），不启子进程。
"""

import importlib.util
import json
from pathlib import Path

import pikepdf
import pypdfium2 as pdfium
import pytest

from test_pdf_exporter import make_task, solid

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_pdf.py"
_spec = importlib.util.spec_from_file_location("export_pdf_cli", _SCRIPT)
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def _task(tmp_path):
    return make_task(tmp_path, [
        {"orig": solid((10, 10, 10)), "proc": solid((220, 220, 220))},
        {"orig": solid((20, 20, 20)), "proc": solid((60, 60, 60))},
    ])


def run(tmp_path, task, out_name="out.pdf", extra=()):
    out = tmp_path / out_name
    code = cli.main(["--task-dir", str(task), "--output", str(out), *extra])
    return code, out


def _report(tmp_path):
    return json.loads((tmp_path / "export_report.json").read_text(encoding="utf-8"))


def test_cli_success_default_shows_progress_and_reports(tmp_path, capsys):
    task, _ = _task(tmp_path)
    code, out = run(tmp_path, task)
    assert code == 0 and out.exists()
    txt = capsys.readouterr().out
    assert "输入页数: 2" in txt
    assert "[1/2]" in txt and "[2/2]" in txt      # 逐页解析/编码/合成进度
    assert "验证 [2/2]" in txt                    # 验证进度
    assert "导出成功" in txt and "大小:" in txt    # 验证后才报成功
    assert (tmp_path / "export_report.json").exists()
    assert (tmp_path / "export_report.txt").exists()
    assert not (tmp_path / "out.partial.pdf").exists()


@pytest.mark.parametrize("mode", ["lossless", "balanced", "compact"])
def test_cli_compression_modes(tmp_path, mode):
    task, _ = _task(tmp_path)
    code, out = run(tmp_path, task, extra=["--compression", mode])
    assert code == 0 and out.exists()
    assert _report(tmp_path)["compression_mode"] == mode


def test_cli_dpi_downsample_only(tmp_path):
    task, _ = make_task(tmp_path, [{"orig": solid((0, 0, 0), (300, 400)),
                                   "proc": solid((120, 120, 120), (300, 400)), "dpi": 300.0}])
    code, out = run(tmp_path, task, extra=["--dpi", "200"])
    assert code == 0
    assert _report(tmp_path)["downsampled_pages"] == 1


def test_cli_page_size_a4(tmp_path):
    task, _ = _task(tmp_path)
    code, out = run(tmp_path, task, extra=["--page-size", "A4"])
    assert code == 0
    doc = pdfium.PdfDocument(str(out))
    w, h = doc[0].get_size()
    doc.close()
    assert abs(w - 595.28) < 1 and abs(h - 841.89) < 1


def test_cli_custom_page_size(tmp_path):
    task, _ = _task(tmp_path)
    code, out = run(tmp_path, task, extra=["--page-size", "custom", "--width-mm", "100", "--height-mm", "150"])
    assert code == 0
    doc = pdfium.PdfDocument(str(out))
    w, h = doc[0].get_size()
    doc.close()
    assert abs(w - 283.46) < 1 and abs(h - 425.20) < 1


def test_cli_custom_missing_dims_rejected(tmp_path):
    task, _ = _task(tmp_path)
    code, _ = run(tmp_path, task, extra=["--page-size", "custom"])
    assert code == 2


def test_cli_overwrite_guard(tmp_path):
    task, _ = _task(tmp_path)
    code, _ = run(tmp_path, task)
    assert code == 0
    code2, _ = run(tmp_path, task)
    assert code2 == 1                       # 默认不覆盖
    code3, _ = run(tmp_path, task, extra=["--overwrite"])
    assert code3 == 0


def test_cli_output_equals_input_rejected(tmp_path):
    task, m = _task(tmp_path)
    inp = Path(m["pages"][0]["source_path"])
    solid((1, 1, 1)).save(inp)
    code, _ = run(tmp_path, task, out_name=inp.name)
    assert code == 1


def test_cli_missing_task_dir(tmp_path):
    code = cli.main(["--task-dir", str(tmp_path / "nope"), "--output", str(tmp_path / "o.pdf")])
    assert code == 2


def test_cli_invalid_compression_exit2(tmp_path):
    task, _ = _task(tmp_path)
    with pytest.raises(SystemExit) as e:
        run(tmp_path, task, extra=["--compression", "bogus"])
    assert e.value.code == 2


def test_cli_no_verify(tmp_path):
    task, _ = _task(tmp_path)
    code, out = run(tmp_path, task, extra=["--no-verify"])
    assert code == 0
    assert _report(tmp_path)["validation_status"] == "skipped"


def test_cli_verify_failure_exit1_keeps_existing(tmp_path, monkeypatch):
    from bookclean import pdf_exporter
    task, _ = _task(tmp_path)
    code, out = run(tmp_path, task, extra=["--no-verify"])
    assert code == 0
    original = out.read_bytes()

    def fail(self, pdf_path, info, contact_dir=None, progress_cb=None):
        return {"passed": False, "issues": ["模拟验证失败"], "page_count": 2,
                "per_page": [], "contact_sheet": None, "previews": []}

    monkeypatch.setattr(pdf_exporter.PDFVerifier, "verify", fail)
    code2, _ = run(tmp_path, task, extra=["--overwrite"])
    assert code2 == 1
    assert out.read_bytes() == original


def test_cli_metadata_title_author(tmp_path):
    task, _ = _task(tmp_path)
    code, out = run(tmp_path, task, extra=["--title", "我的书", "--author", "张三"])
    assert code == 0
    with pikepdf.open(out) as pk:
        assert str(pk.docinfo.get("/Title")) == "我的书"
        assert str(pk.docinfo.get("/Author")) == "张三"


def test_cli_cover_warns_crop(tmp_path):
    task, _ = _task(tmp_path)
    code, out = run(tmp_path, task, extra=["--fit", "cover", "--page-size", "A4"])
    assert code == 0
    assert any("裁切" in w for w in _report(tmp_path)["warnings"])