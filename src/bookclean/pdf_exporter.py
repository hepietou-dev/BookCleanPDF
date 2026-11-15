"""M4.3 PDF 合成：FinalImageResolver + PDFExporter。

统一调用 FinalImageResolver / PageLayoutResolver / ImageEncoder / img2pdf / pikepdf。
页面顺序与最终图像映射一律来自 Manifest（全局 page_index），不按文件名扫描目录。

合成方案 B（M4.0 实测内存更优）：逐页编码 -> img2pdf 单页 PDF -> pikepdf 逐页追加合并。
任意时刻内存只持有一页解码像素与一页 PDF 字节；不调用 pikepdf optimize_images（会重编码图像）。
页面白底、Contain 留白纯白、不拉伸；允许混合尺寸/方向/灰度彩色。

输出（M4.4）：先写同目录 .partial.pdf -> PDFVerifier 重读逐页验证 -> 通过才 os.replace 为正式输出；
任何异常清理临时文件、不覆盖既有正确 PDF。验证全部通过才落盘导出报告(json/txt)与联系表/预览。
"""

from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import img2pdf
import pikepdf
from PIL import Image

from .config import PDFExportConfig
from .image_encoder import ImageEncoder
from .page_layout import PageLayout, PageLayoutResolver
from . import pdf_metadata
from .pdf_verifier import PDFVerifier
from .text_layer import PageText, TextLayerWriter
from .utils.atomic_file import atomic_write_json

logger = logging.getLogger("bookclean")


class ExportError(Exception):
    """导出中止（缺失页 / 覆盖冲突 / 输出等于输入等）。"""


@dataclass
class FinalImage:
    path: Path | None
    source: str
    status: str
    fallback: bool
    warnings: list[str] = field(default_factory=list)


class FinalImageResolver:
    """只选择实际存在且为最终结果的图像；失败/跳过回退 original；两者皆缺则标记 missing。"""

    def __init__(self, task_dir: str | Path):
        self.task_dir = Path(task_dir)

    def resolve(self, manifest: dict, page: dict) -> FinalImage:
        status = page.get("processing_status") or page.get("status") or "unknown"
        warnings: list[str] = []
        proc_rel = page.get("processed_image_path")
        orig_rel = page.get("image_path")
        proc = self.task_dir / proc_rel if proc_rel else None
        orig = self.task_dir / orig_rel if orig_rel else None

        # M4.8.1：黑边清除结果优先（仅在 Manifest 含 border_* 字段且文件存在时生效）
        border_rel = page.get("border_image_path")
        border = self.task_dir / border_rel if border_rel else None
        if page.get("border_status") == "completed" and border is not None and border.exists():
            if self._valid_image(border):
                fallback = (page.get("fallback_level") not in (None, "none")) or \
                    str(page.get("processing_decision", "")).startswith("fallback")
                warnings.extend(page.get("quality_warnings") or [])
                return FinalImage(border, "border_cleaned", status, fallback, warnings)
            warnings.append("黑边清除结果文件损坏，回退处理结果/原始页面")

        if status == "completed" and proc is not None and proc.exists():
            if self._valid_image(proc):
                source = page.get("final_image_source") or "processed"
                fallback = (page.get("fallback_level") not in (None, "none")) or \
                    str(page.get("processing_decision", "")).startswith("fallback")
                warnings.extend(page.get("quality_warnings") or [])
                return FinalImage(proc, source, status, fallback, warnings)
            warnings.append("处理结果文件损坏，回退原始页面")

        if orig is not None and orig.exists():
            if status == "completed":
                warnings.append("处理结果文件缺失，回退原始页面")
            else:
                warnings.append(f"M2/M3 状态 {status}，回退原始页面")
            return FinalImage(orig, "original_fallback", status, True, warnings)

        warnings.append(f"第 {page.get('display_index', page.get('page_index'))} 页无可用图像（处理结果与原始页面均缺失）")
        return FinalImage(None, "missing", status, False, warnings)

    @staticmethod
    def _valid_image(path: Path) -> bool:
        """完整性守卫：文件可被完整解码（PNG CRC/结构校验），损坏则不可用。非仅存在性。"""
        try:
            with Image.open(path) as im:
                im.verify()
            return True
        except Exception:
            return False


class PDFExporter:
    def __init__(self, config: PDFExportConfig):
        config.validate()
        self.config = config

    # ---------- 合成 ----------

    def build(self, task_dir: str | Path, output_name: str = "output.pdf",
              progress_cb=None, text_source=None) -> tuple[pikepdf.Pdf, dict]:
        """合成 PDF。text_source 为 M6.2 可选文字层钩子：

        text_source(page_index, image_path, layout) -> PageText | None。
        默认 None ⇒ 不注入任何文字层，输出与 M5 字节级一致（不导入任何 OCR 代码）。
        """
        task_dir = Path(task_dir)
        manifest = _load_manifest(task_dir)
        pages = sorted(manifest["pages"], key=lambda p: p["page_index"])
        resolver = FinalImageResolver(task_dir)
        layout_resolver = PageLayoutResolver(self.config)
        encoder = ImageEncoder(self.config)

        resolved = [(p, resolver.resolve(manifest, p)) for p in pages]
        missing = [p for p, fi in resolved if fi.path is None]
        if missing:
            nums = ", ".join(str(p.get("display_index", p["page_index"] + 1)) for p in missing)
            raise ExportError(f"缺失页码（无可用图像），中止导出: {nums}")

        started = time.monotonic()
        dest = pikepdf.new()
        text_writer = TextLayerWriter(dest) if text_source is not None else None
        per_page: list[dict] = []
        warnings: list[str] = []
        for page, fi in resolved:
            with Image.open(fi.path) as im:
                w, h = im.size
            layout = layout_resolver.resolve(page, w, h)
            enc = encoder.encode(fi.path, layout, page.get("page_type"))
            pw, ph = layout.page_width_pt, layout.page_height_pt
            iw, ih = layout.image_display_width_pt, layout.image_display_height_pt
            lf = lambda wp, hp, dpi: (pw, ph, iw, ih)
            single = img2pdf.convert(enc.data, layout_fun=lf)
            with pikepdf.open(io.BytesIO(single)) as src:
                dest.pages.append(src.pages[0])
            ocr_chars = 0
            if text_writer is not None:
                page_text = text_source(page["page_index"], fi.path, layout)
                if page_text is not None:
                    ocr_chars = text_writer.inject_page(dest.pages[-1], page_text, layout)
            warnings.extend(fi.warnings)
            warnings.extend(enc.warnings)
            warnings.extend(layout.warnings)
            per_page.append({
                "page_index": page["page_index"],
                "display_index": page.get("display_index", page["page_index"] + 1),
                "final_image_source": fi.source,
                "source_image_path": str(fi.path),
                "page_status": fi.status,
                "fallback": fi.fallback,
                "output_page_width_pt": round(pw, 3),
                "output_page_height_pt": round(ph, 3),
                "image_width_px": enc.width_px,
                "image_height_px": enc.height_px,
                "effective_dpi_x": round(enc.effective_dpi_x, 2),
                "effective_dpi_y": round(enc.effective_dpi_y, 2),
                "source_page_size": layout.source_page_size,
                "estimated_size": layout.estimated_size,
                "output_format": enc.format,
                "compression_method": enc.compression_method,
                "jpeg_quality": enc.jpeg_quality,
                "color_mode": enc.color_mode,
                "source_color_mode": enc.source_color_mode,
                "already_grayscale": enc.already_grayscale,
                "grayscale_converted": enc.grayscale_converted,
                "downsampled": enc.downsampled,
                "original_image_size": enc.original_size_bytes,
                "encoded_image_size": enc.encoded_size_bytes,
                "page_export_status": "completed",
                "page_warnings": fi.warnings + enc.warnings + layout.warnings,
            })
            if text_writer is not None:
                per_page[-1]["ocr_text_chars"] = ocr_chars
            if progress_cb is not None:
                progress_cb(per_page[-1])

        meta = pdf_metadata.apply_metadata(dest, self.config, output_name)
        elapsed = time.monotonic() - started
        info = {
            "task_id": manifest.get("task_id"),
            "color_mode": self.config.color_mode,
            "grayscale_conversion": _grayscale_conversion(per_page, self.config.color_mode),
            "pages": per_page,
            "metadata": meta,
            "warnings": warnings,
            "summary": _summary(manifest, per_page, elapsed),
        }
        return dest, info

    # ---------- 输出 ----------

    def export(self, task_dir: str | Path, output_path: str | Path,
               progress_cb=None, verify_progress_cb=None, text_source=None) -> dict:
        task_dir = Path(task_dir)
        output_path = Path(output_path)
        manifest = _load_manifest(task_dir)
        self._guard_output(manifest, output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        partial = output_path.with_name(output_path.stem + ".partial.pdf")
        pdf, info = self.build(task_dir, output_name=output_path.name, progress_cb=progress_cb,
                               text_source=text_source)
        try:
            try:
                pdf.save(partial, compress_streams=True,
                         object_stream_mode=pikepdf.ObjectStreamMode.generate,
                         linearize=self.config.linearize)
            finally:
                pdf.close()
            if self.config.verify:
                if text_source is not None:
                    text_pages = {i for i, p in enumerate(info["pages"])
                                  if p.get("ocr_text_chars", 0) > 0}
                    v = PDFVerifier().verify(partial, info, contact_dir=output_path.parent,
                                             progress_cb=verify_progress_cb, text_pages=text_pages)
                else:
                    # OCR 关：以 M5 原签名调用（不传 text_pages），保证调用路径与 M5 完全一致
                    v = PDFVerifier().verify(partial, info, contact_dir=output_path.parent,
                                             progress_cb=verify_progress_cb)
                info["validation"] = {
                    "passed": v["passed"], "issues": v["issues"],
                    "page_count": v["page_count"], "per_page": v["per_page"],
                    "contact_sheet": v["contact_sheet"], "previews": v["previews"],
                    "ocr_text_check": v.get("ocr_text_check"),
                }
                if not v["passed"]:
                    raise ExportError("PDF 验证失败: " + "; ".join(v["issues"]))
            if self.config.color_mode == "grayscale":
                gv = PDFVerifier().verify_grayscale(partial, progress_cb=verify_progress_cb)
                info["grayscale_validation"] = gv
                if gv["status"] != "passed":
                    detail = "; ".join(f"第{f['page']}页 {f['object']}: {f['reason']}"
                                       for f in gv["failed_pages"])
                    raise ExportError("灰度验证失败: " + detail)
            else:
                info["grayscale_validation"] = {"enabled": False, "status": "skipped"}
            os.replace(partial, output_path)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        info["output_file"] = str(output_path)
        info["output_size_bytes"] = output_path.stat().st_size
        info["export_status"] = "completed"
        info["validation_status"] = "passed" if self.config.verify else "skipped"
        if text_source is not None:
            with_text = sum(1 for p in info["pages"] if p.get("ocr_text_chars", 0) > 0)
            total = len(info["pages"])
            info["ocr"] = {
                "enabled": True,
                "text_layer": "W2 pikepdf CIDFontType0（不嵌字体，render_mode 3 不可见）",
                "pages_with_text": with_text,
                "pages_without_text": total - with_text,
                "total_text_chars": sum(p.get("ocr_text_chars", 0) for p in info["pages"]),
                "coverage": round(with_text / total, 4) if total else 0.0,
            }
        _write_reports(output_path, info, self.config)
        return info

    def _guard_output(self, manifest: dict, output_path: Path) -> None:
        out = output_path.resolve()
        if out.exists() and not self.config.overwrite:
            raise ExportError(f"输出已存在: {out}（默认不覆盖，需显式 overwrite）")
        inputs = set()
        for item in manifest.get("inputs", []):
            p = item.get("path") if isinstance(item, dict) else item
            if p:
                inputs.add(Path(p).resolve())
        for page in manifest.get("pages", []):
            if page.get("source_path"):
                inputs.add(Path(page["source_path"]).resolve())
        if out in inputs:
            raise ExportError(f"输出路径等于原始输入文件: {out}（禁止覆盖输入）")


def _load_manifest(task_dir: Path) -> dict:
    import json
    path = task_dir / "manifest.json"
    if not path.exists():
        raise ExportError(f"任务目录缺少 manifest.json: {task_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def _grayscale_conversion(per_page: list[dict], color_mode: str) -> dict | None:
    """grayscale 模式的转换统计；preserve 返回 None（不要求全灰）。"""
    if color_mode != "grayscale":
        return None
    color_src = ("RGB", "RGBA", "P", "PA", "LA", "CMYK")
    return {
        "algorithm": "OpenCV COLOR_RGB2GRAY",
        "bit_depth": 8,
        "source_rgb_pages": sum(1 for r in per_page if r.get("source_color_mode") in color_src),
        "source_gray_pages": sum(1 for r in per_page if r.get("source_color_mode") in ("L", "1")),
        "already_grayscale_pages": sum(1 for r in per_page if r.get("already_grayscale")),
        "converted_pages": sum(1 for r in per_page if r.get("grayscale_converted")),
        "failed_pages": sum(1 for r in per_page if r.get("color_mode") != "L"),
    }


def _summary(manifest: dict, per_page: list[dict], elapsed: float) -> dict:
    src_pdf = sum(1 for p in manifest["pages"] if p.get("source_kind") == "pdf")
    src_img = sum(1 for p in manifest["pages"] if p.get("source_kind") != "pdf")
    return {
        "total_pages": len(per_page),
        "source_pdf_pages": src_pdf,
        "source_image_pages": src_img,
        "fallback_original_pages": sum(1 for r in per_page if r["final_image_source"] == "original_fallback"),
        "fallback_geometry_pages": sum(1 for r in per_page if r["final_image_source"] == "geometry"),
        "jpeg_pages": sum(1 for r in per_page if r["output_format"] == "jpeg"),
        "png_pages": sum(1 for r in per_page if r["output_format"] == "png"),
        "grayscale_pages": sum(1 for r in per_page if r["color_mode"] in ("L", "1")),
        "color_pages": sum(1 for r in per_page if r["color_mode"] == "RGB"),
        "downsampled_pages": sum(1 for r in per_page if r["downsampled"]),
        "total_export_time_seconds": round(elapsed, 3),
    }


def _write_reports(output_path: Path, info: dict, config: PDFExportConfig) -> None:
    s = info["summary"]
    gc = info.get("grayscale_conversion")
    gv = info.get("grayscale_validation") or {}
    pages = info["pages"]
    gray_png = sum(1 for r in pages if r["color_mode"] == "L" and r["output_format"] == "png")
    gray_jpeg = sum(1 for r in pages if r["color_mode"] == "L" and r["output_format"] == "jpeg")
    final_gray = sum(1 for r in pages if r["color_mode"] == "L")
    failed_nums = {f["page"] for f in gv.get("failed_pages", []) if f.get("page")}
    verified_pages = (s["total_pages"] - len(failed_nums)) if gv.get("enabled") else 0
    rep = {
        "task_id": info.get("task_id"),
        "export_status": info["export_status"],
        "output_file": info["output_file"],
        "page_count": s["total_pages"],
        "output_size_bytes": info["output_size_bytes"],
        "compression_mode": config.compression,
        "image_dpi_mode": "preserve_source" if config.target_dpi is None else str(config.target_dpi),
        "page_size_mode": config.page_size_mode,
        "fit_mode": config.fit_mode,
        "source_pdf_pages": s["source_pdf_pages"],
        "source_image_pages": s["source_image_pages"],
        "fallback_original_pages": s["fallback_original_pages"],
        "fallback_geometry_pages": s["fallback_geometry_pages"],
        "jpeg_pages": s["jpeg_pages"],
        "png_pages": s["png_pages"],
        "grayscale_pages": s["grayscale_pages"],
        "color_pages": s["color_pages"],
        "downsampled_pages": s["downsampled_pages"],
        "total_export_time_seconds": s["total_export_time_seconds"],
        "validation_status": info["validation_status"],
        "validation": info.get("validation"),
        "color_mode": info.get("color_mode", config.color_mode),
        "grayscale_conversion": gc,
        "grayscale_validation": gv or None,
        "gray_png_pages": gray_png,
        "gray_jpeg_pages": gray_jpeg,
        "final_grayscale_pages": final_gray,
        "grayscale_verified_pages": verified_pages,
        "warnings": info["warnings"],
        "pages": info["pages"],
        "note": ("图像型 PDF + 不可见 OCR 文字层（可搜索/复制）；无超链接/书签/表单。"
                 if info.get("ocr", {}).get("enabled")
                 else "图像型 PDF：无 OCR 文字层/超链接/书签/表单。"),
    }
    if info.get("ocr", {}).get("enabled"):
        rep["ocr"] = info["ocr"]
    atomic_write_json(output_path.parent / "export_report.json", rep)
    v = info.get("validation") or {}
    lines = [
        "BookClean PDF 导出报告",
        f"任务: {info.get('task_id')}",
        f"输出: {info['output_file']}",
        f"输入页数: {s['total_pages']}（PDF {s['source_pdf_pages']} / 图片 {s['source_image_pages']}）",
        f"成功导出页数: {s['total_pages']}",
        f"最终 PDF 大小: {info['output_size_bytes']} 字节",
        f"压缩模式: {config.compression}",
        f"输出色彩模式: {'8 位灰度' if config.color_mode == 'grayscale' else 'preserve（保留原始彩色/灰度）'}",
        f"灰度页: {s['grayscale_pages']}  彩色页: {s['color_pages']}",
        f"JPEG 页: {s['jpeg_pages']}  PNG 页: {s['png_pages']}",
        f"回退原始页: {s['fallback_original_pages']}  回退几何页: {s['fallback_geometry_pages']}",
        f"验证: {info['validation_status']}" + (f"（问题 {len(v.get('issues', []))}）" if v else ""),
        f"耗时: {s['total_export_time_seconds']}s",
        ("说明: 图像型 PDF + 不可见 OCR 文字层（可搜索/复制）。"
         if info.get("ocr", {}).get("enabled") else "说明: 图像型 PDF，无 OCR 文字层。"),
    ]
    ocr = info.get("ocr")
    if ocr and ocr.get("enabled"):
        lines += [
            f"OCR 文字层: {ocr['text_layer']}",
            f"有文字页: {ocr['pages_with_text']}  无文字页: {ocr['pages_without_text']}  "
            f"覆盖率: {ocr['coverage'] * 100:.1f}%",
            f"文字层总字符: {ocr['total_text_chars']}",
        ]
    if gc:
        lines += [
            f"原始彩色页面: {gc['source_rgb_pages']}",
            f"原始灰度页面: {gc['source_gray_pages']}",
            f"实际转换页面: {gc['converted_pages']}",
            f"已灰度跳过: {gc['already_grayscale_pages']}",
            f"最终灰度页面: {final_gray}",
            f"灰度 PNG: {gray_png}  灰度 JPEG: {gray_jpeg}",
            f"灰度验证: {'通过' if gv.get('status') == 'passed' else gv.get('status', '未启用')}"
            + (f"（DeviceGray {gv.get('device_gray_image_count', 0)} / 意外RGB {gv.get('unexpected_rgb_image_count', 0)}）"
               if gv.get("enabled") else ""),
        ]
    if info["warnings"]:
        lines.append("警告:")
        lines.extend(f"  - {w}" for w in info["warnings"])
    (output_path.parent / "export_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")