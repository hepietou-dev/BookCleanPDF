"""内核异常 → 用户友好提示 + 建议措施。

说明书 §9.2：错误提示应说明文件名/页码/原因/可采取措施，不只显示 Python traceback。
worker 已把完整 traceback 写入日志文件；UI 只展示本模块产出的简明信息。

纯映射，不导入 Tkinter，也不导入内核（按异常类型名 str 匹配），便于单元测试。
未知类型给通用提示并保留原始原因，绝不静默吞掉错误。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorInfo:
    title: str
    message: str
    action: str


# 异常类型名 → (标题, 友好原因, 建议措施)。类型名来自 errors.py 与 pdf_exporter.ExportError。
_MAP: dict[str, tuple[str, str, str]] = {
    "UnsupportedFormatError": ("不支持的文件格式", "该文件格式无法导入。", "改用 PDF / JPG / PNG / TIFF / BMP 文件。"),
    "PDFPasswordError": ("PDF 受密码保护", "该 PDF 需要密码才能打开。", "先解密 PDF 再导入。"),
    "PDFCorruptError": ("PDF 损坏", "该 PDF 损坏或无法解析。", "更换文件或重新导出该 PDF。"),
    "PDFError": ("PDF 读取失败", "该 PDF 打开、结构或渲染出错。", "更换文件或检查 PDF 是否完整。"),
    "ImageLoadError": ("图片无法读取", "图片损坏、无法解码或尺寸异常。", "更换或修复该图片。"),
    "InputError": ("输入不可用", "输入路径不存在或不可读。", "检查路径与访问权限。"),
    "BookCleanError": ("处理错误", "处理过程中发生错误。", "查看错误日志了解详情，或更换输入后重试。"),
}

_GENERIC = ("处理失败", "处理过程中发生错误。", "查看错误日志了解详情，或更换输入后重试。")


def _export_detail(detail: str) -> tuple[str, str, str]:
    """ExportError 按原因细分（缺失页/验证未通过/输出已存在/输出=输入/任务数据缺失）。"""
    if "验证失败" in detail:
        return ("PDF 验证未通过", "生成的 PDF 未通过验证，未输出正式文件。",
                "查看错误日志了解具体验证问题后重试。")
    if "输出已存在" in detail:
        return ("输出文件已存在", "目标输出文件已存在，默认不覆盖以保护既有成果。",
                "更换输出文件名/目录，或勾选“允许覆盖”。")
    if "等于原始输入" in detail or "输出路径等于" in detail:
        return ("输出与输入相同", "输出路径不能与原始输入文件相同（禁止覆盖输入）。",
                "更改输出目录或文件名。")
    if "缺失页码" in detail:
        return ("缺少可用页面", "部分页码无可用图像，导出中止。",
                "检查处理失败页，补全后重试。")
    if "manifest" in detail:
        return ("任务数据缺失", "任务目录缺少 manifest.json。", "重新导入并处理。")
    return ("导出中止", "PDF 导出未能完成。", "更换输出名/目录或勾选覆盖，并确认页面完整。")


def describe_error(exc_type: str, raw: str = "") -> ErrorInfo:
    """把异常类型名 + 原始信息映射为简明用户提示。原始信息作为“详情”附在原因后。"""
    detail = (raw or "").strip()
    if exc_type == "ExportError":
        title, message, action = _export_detail(detail)
    else:
        title, message, action = _MAP.get(exc_type, _GENERIC)
    if detail:
        message = f"{message}\n详情：{detail}"
    return ErrorInfo(title=title, message=message, action=action)