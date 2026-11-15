"""M4.7.4 CLI --color-mode 测试：默认 preserve / grayscale / 非法值拒绝 / help 说明 / 与压缩独立。"""

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from test_pdf_exporter import make_task

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_pdf.py"
_spec = importlib.util.spec_from_file_location("export_pdf_cli", _SCRIPT)
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def _task(tmp_path):
    return make_task(tmp_path, [
        {"orig": Image.new("RGB", (60, 80), (200, 30, 30)), "proc": Image.new("RGB", (60, 80), (200, 30, 30))},
        {"orig": Image.new("L", (60, 80), 120), "proc": Image.new("L", (60, 80), 120)},
    ])[0]


def _run(tmp_path, task, name, extra):
    out = tmp_path / name
    rc = cli.main(["--task-dir", str(task), "--output", str(out)] + extra)
    rep = json.loads((out.parent / "export_report.json").read_text(encoding="utf-8"))
    return rc, rep


def test_cli_default_preserve(tmp_path):
    task = _task(tmp_path)
    rc, rep = _run(tmp_path, task, "a.pdf", [])
    assert rc == 0
    assert any(p["color_mode"] == "RGB" for p in rep["pages"])  # 默认保留彩色


def test_cli_grayscale(tmp_path):
    task = _task(tmp_path)
    rc, rep = _run(tmp_path, task, "b.pdf", ["--color-mode", "grayscale"])
    assert rc == 0
    assert all(p["color_mode"] == "L" for p in rep["pages"])


def test_cli_grayscale_lossless_independent(tmp_path):
    task = _task(tmp_path)
    rc, rep = _run(tmp_path, task, "c.pdf", ["--color-mode", "grayscale", "--compression", "lossless"])
    assert rc == 0
    assert rep["compression_mode"] == "lossless"
    assert all(p["color_mode"] == "L" for p in rep["pages"])


def test_cli_invalid_color_mode_rejected(tmp_path):
    task = _task(tmp_path)
    with pytest.raises(SystemExit) as e:
        cli.main(["--task-dir", str(task), "--output", str(tmp_path / "d.pdf"),
                  "--color-mode", "sepia"])
    assert e.value.code == 2


def test_cli_help_explains_grayscale(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    text = capsys.readouterr().out
    assert "grayscale" in text and "preserve" in text
    assert "8 位灰度" in text
    assert "不会改变原始任务文件" in text