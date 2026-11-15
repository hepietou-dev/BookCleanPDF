"""结果视图：只读 manifest.json + export_report.json，派生 GUI 结果页所需事实。

M5.3 职责（说明书 §9.1 处理报告 / §6.2 结果页）：
- 汇总输入/输出/页数/优化/回退/失败/耗时/输出大小/PDF 验证结果。
- 失败页列表（页码 + 阶段 + 原因）；回退页列表（页码 + 级别 + 原因 + 来源）。
- **回退页不计为完全成功优化页**：optimized = processed - fallbacks。
- **只认 os.replace 后的正式 PDF**：.partial 一律视为未完成。

本模块只读取既有任务目录文件，绝不修改 Manifest 或任务目录结构；缺失/部分数据优雅降级。
不导入 Tkinter；由 app_controller.load_result 调用，保持 ui → controller → kernel 单向。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("bookclean")

_FALLBACK_LEVELS = ("background", "geometry", "original")


def _read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as exc:
        logger.warning("读取结果数据失败 %s: %s", path, exc)
        return {}


def _is_final_pdf(path: Path) -> bool:
    """仅 os.replace 后的正式 .pdf 计完成；.partial.pdf 不算。"""
    return path.suffix.casefold() == ".pdf" and not path.name.casefold().endswith(".partial.pdf") and path.is_file()


def _page_image(task_dir: Path, p: dict) -> str:
    """该页最终采用图像的绝对路径（供“一键定位”外部打开，不在 UI 内加载像素）。"""
    rel = ""
    if p.get("border_status") == "completed" and p.get("border_image_path"):
        rel = p["border_image_path"]
    elif p.get("processed_image_path"):
        rel = p["processed_image_path"]
    elif p.get("image_path"):
        rel = p["image_path"]
    return str(task_dir / rel) if rel else ""


def _derive_state(manifest: dict, pages: list[dict], output_exists: bool,
                  export_status: str, validation_status: str,
                  processed: int, failed: int) -> str:
    """由现有 Manifest/报告字段派生 GUI 任务状态（架构 §4.1/§4.2，不新增 Manifest 字段）。

    completed 仅认 os.replace 后正式 PDF + 导出 completed + 验证 passed；
    .partial/缺失输出一律不算完成。cancelled 看逐页协作取消标记。
    """
    if not manifest:
        return "missing"
    if output_exists and export_status == "completed" and validation_status == "passed":
        return "completed"
    if any(p.get("processing_status") == "cancelled" or p.get("border_status") == "cancelled"
           for p in pages):
        return "cancelled"
    if export_status == "failed" or validation_status == "failed":
        return "failed"
    if manifest.get("status") == "failed":
        return "failed"
    if processed == 0 and failed > 0:
        return "failed"
    return "incomplete"


def build_result_view(task_dir: str | Path, output_path: str | Path | None = None) -> dict:
    """从任务目录与导出报告派生结果视图。task_dir 缺失时返回最小降级视图。"""
    task_dir = Path(task_dir)
    manifest = _read_json(task_dir / "manifest.json")
    pages = manifest.get("pages", []) if isinstance(manifest.get("pages"), list) else []

    out = Path(output_path) if output_path else None
    report_path = (out.parent / "export_report.json") if out else (task_dir / "export_report.json")
    report = _read_json(report_path)
    if out is None and report.get("output_file"):
        out = Path(report["output_file"])

    proc = manifest.get("processing_summary", {}) or {}

    breakdown = {level: 0 for level in _FALLBACK_LEVELS}
    failed_pages: list[dict] = []
    fallback_pages: list[dict] = []
    for p in pages:
        idx = p.get("display_index", p.get("page_index", 0))
        img = _page_image(task_dir, p)
        src = p.get("source_path", "")
        if p.get("status") == "failed":
            failed_pages.append({"display_index": idx, "stage": "import",
                                 "reason": p.get("error") or "导入失败",
                                 "image_path": img, "source_path": src})
        if p.get("processing_status") == "failed":
            failed_pages.append({"display_index": idx, "stage": "processing",
                                 "reason": p.get("processing_error") or "处理失败",
                                 "image_path": img, "source_path": src})
        if p.get("border_status") == "failed":
            failed_pages.append({"display_index": idx, "stage": "border",
                                 "reason": p.get("border_error") or "黑边清除失败",
                                 "image_path": img, "source_path": src})
        if p.get("ocr_status") == "failed":
            failed_pages.append({"display_index": idx, "stage": "ocr",
                                 "reason": p.get("ocr_error") or "OCR 识别失败",
                                 "image_path": img, "source_path": src})
        level = p.get("fallback_level", "none")
        if level in breakdown:
            breakdown[level] += 1
            fallback_pages.append({"display_index": idx, "level": level,
                                   "reason": p.get("fallback_reason", ""),
                                   "source": p.get("final_image_source", ""),
                                   "image_path": img, "source_path": src})

    # 逐页派生总数（resume 正确）：续跑时 summary.processed 只计新处理页、复用页计入
    # reused，故总数必须以逐页 processing_status/fallback_level 为准，而非 summary。
    processed = sum(1 for p in pages if p.get("processing_status") == "completed")
    fallbacks = sum(breakdown.values())
    failed = len(failed_pages)
    optimized = max(0, processed - fallbacks)
    imported = sum(1 for p in pages if p.get("status") == "completed" and p.get("image_path"))

    border = manifest.get("border_summary", {}) or {}
    ocr_summary = manifest.get("ocr_summary", {}) or {}
    ocr_status = manifest.get("ocr_status", "")
    ocr = {
        "enabled": bool(ocr_summary) or bool(ocr_status),
        "status": ocr_status,
        "engine_id": ocr_summary.get("engine_id", ""),
        "engine_version": ocr_summary.get("engine_version", ""),
        "completed": int(ocr_summary.get("completed", 0)),
        "failed": int(ocr_summary.get("failed", 0)),
        "skipped": int(ocr_summary.get("skipped", 0)),
        "reused": int(ocr_summary.get("reused", 0)),
        "total_text_chars": int(ocr_summary.get("total_text_chars", 0)),
        "coverage": float(ocr_summary.get("coverage", 0.0)),
        "elapsed_seconds": float(ocr_summary.get("elapsed_seconds", 0.0)),
        "reason": ocr_summary.get("reason", ""),
    }
    validation = report.get("validation", {}) or {}
    output_exists = bool(out and _is_final_pdf(out))
    output_size = int(report.get("output_size_bytes", 0)) if report else 0
    if output_exists and not output_size:
        try:
            output_size = out.stat().st_size
        except OSError:
            output_size = 0

    export_status = report.get("export_status", "")
    validation_status = report.get("validation_status", "")
    state = _derive_state(manifest, pages, output_exists, export_status, validation_status,
                          processed, failed)
    resumable = bool(manifest) and state in ("cancelled", "incomplete", "failed") and imported > 0

    elapsed = (float(manifest.get("elapsed_seconds", 0) or 0)
               + float(proc.get("elapsed_seconds", 0) or 0)
               + float(border.get("elapsed_seconds", 0) or 0)
               + float(ocr_summary.get("elapsed_seconds", 0) or 0)
               + float(report.get("total_export_time_seconds", 0) or 0))

    inputs = [{"name": Path(i.get("path", "")).name, "kind": i.get("kind", ""),
               "page_count": i.get("page_count", 0), "error": i.get("error")}
              for i in manifest.get("inputs", []) if isinstance(i, dict)]

    return {
        "task_dir": str(task_dir),
        "task_id": manifest.get("task_id", task_dir.name),
        "manifest_exists": bool(manifest),
        "state": state,
        "resumable": resumable,
        "total_pages": int(manifest.get("total_pages", len(pages))),
        "imported": imported,
        "inputs": inputs,
        "optimized": optimized,
        "processed": processed,
        "fallbacks": fallbacks,
        "failed": failed,
        "reused": int(proc.get("reused", 0)),
        "fallback_breakdown": breakdown,
        "failed_pages": failed_pages,
        "fallback_pages": fallback_pages,
        "border": {"cleaned": int(border.get("cleaned", 0)),
                   "skipped": int(border.get("skipped", 0)),
                   "failed": int(border.get("failed", 0))},
        "ocr": ocr,
        "output_path": str(out) if out else "",
        "output_exists": output_exists,
        "output_size_bytes": output_size,
        "export_status": export_status,
        "validation_status": validation_status,
        "validation_issues": list(validation.get("issues", []) or []),
        "color_mode": report.get("color_mode", ""),
        "compression_mode": report.get("compression_mode", ""),
        "elapsed_seconds": round(elapsed, 2),
        "report_path": str(report_path),
        "report_txt_path": str(report_path.with_name("export_report.txt")),
        "warnings": list(report.get("warnings", []) or []),
    }