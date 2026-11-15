"""M0 冒烟测试：配置校验、数据结构、日志、诊断、入口。"""

import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bookclean import __version__
from bookclean.config import TaskConfig
from bookclean.logging_setup import setup_logging
from bookclean.models import PageData, PageFeatures, ProcessResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_config_defaults_valid():
    TaskConfig().validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"background_strength": 1.5},
        {"denoise_strength": -0.1},
        {"render_dpi": 30},
        {"output_dpi": 1200},
        {"output_mode": "binary"},
        {"preset": "unknown"},
        {"compression_preset": "tiny"},
        {"enable_dewarp": True},
        {"min_paper_confidence": 1.2},
        {"quality_text_core_min": 1.5},
        {"quality_bg_std_ratio_max": 0.5},
        {"quality_photo_blank_std_min": -1.0},
        {"quality_risk_accept_max": 150.0},
    ],
)
def test_config_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        TaskConfig(**kwargs).validate()


def test_models_instantiate():
    image = np.zeros((8, 6, 3), dtype=np.uint8)
    page = PageData(
        page_index=0,
        image=image,
        source_path="x.png",
        source_kind="image",
        original_width=6,
        original_height=8,
        source_dpi=None,
        color_mode="RGB",
    )
    assert page.image.shape == (8, 6, 3)
    features = PageFeatures()
    assert features.skew_confidence == 0.0
    assert features.warnings == []
    result = ProcessResult(image=image, executed=False)
    assert not result.suggest_fallback


def test_logging_writes_file(tmp_path):
    logger = setup_logging(tmp_path)
    logger.info("m0 smoke log entry")
    for handler in logger.handlers:
        handler.flush()
    content = (tmp_path / "bookclean.log").read_text(encoding="utf-8")
    assert "m0 smoke log entry" in content
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_diagnostics_all_pass():
    import app

    result = app.run_diagnostics()
    failed = [(name, detail) for name, ok, detail in result["checks"] if not ok]
    assert result["ok"], f"诊断失败项: {failed}"


def test_cli_version():
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "app.py"), "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0
    assert __version__ in proc.stdout


def test_cli_diagnose_exit_code():
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "app.py"), "--diagnose"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr