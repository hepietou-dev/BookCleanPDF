"""M5.3 错误映射单元测试：内核异常类型→简明用户提示+建议措施，不甩 traceback。"""
from __future__ import annotations

import pytest

from bookclean.error_messages import ErrorInfo, describe_error


KNOWN = [
    "UnsupportedFormatError", "PDFPasswordError", "PDFCorruptError", "PDFError",
    "ImageLoadError", "InputError", "BookCleanError", "ExportError",
]


@pytest.mark.parametrize("exc_type", KNOWN)
def test_known_types_have_title_message_action(exc_type):
    info = describe_error(exc_type, f"{exc_type}: 示例原因")
    assert isinstance(info, ErrorInfo)
    assert info.title and info.message and info.action
    assert "示例原因" in info.message  # 原始原因作为详情保留


def test_export_validation_failure_maps_to_verification():
    info = describe_error("ExportError", "ExportError: PDF 验证失败: size mismatch")
    assert info.title == "PDF 验证未通过"
    assert "日志" in info.action
    assert "size mismatch" in info.message


def test_export_grayscale_validation_failure():
    info = describe_error("ExportError", "ExportError: 灰度验证失败: page 3 not gray")
    assert info.title == "PDF 验证未通过"


def test_export_output_exists_suggests_overwrite():
    info = describe_error("ExportError", "ExportError: 输出已存在: x.pdf（默认不覆盖）")
    assert info.title == "输出文件已存在"
    assert "覆盖" in info.action


def test_export_output_equals_input():
    info = describe_error("ExportError", "ExportError: 输出路径等于原始输入文件: x.png")
    assert info.title == "输出与输入相同"
    assert "更改" in info.action


def test_export_missing_pages():
    info = describe_error("ExportError", "ExportError: 缺失页码（无可用图像），中止导出: [2, 3]")
    assert info.title == "缺少可用页面"


def test_export_missing_manifest():
    info = describe_error("ExportError", "ExportError: 任务目录缺少 manifest.json: /x")
    assert info.title == "任务数据缺失"


def test_unknown_type_generic_keeps_detail():
    info = describe_error("WhateverError", "boom")
    assert info.title == "处理失败"
    assert "boom" in info.message
    assert info.action


def test_empty_raw_still_valid():
    info = describe_error("ImageLoadError", "")
    assert info.title == "图片无法读取"
    assert info.action


def test_no_traceback_injected():
    info = describe_error("ExportError", "ExportError: 输出已存在: x.pdf")
    assert "Traceback" not in info.message
    assert 'File "' not in info.message