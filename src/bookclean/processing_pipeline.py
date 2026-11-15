"""统一处理流水线（M2.4）：单页处理 + 任务目录批量 runner。

单页流程：分析 -> 几何校正 -> 重分析 -> 背景清理 -> 文字增强 -> 质量检查 -> 自动回退。
批量 runner 读取 M1 任务目录的 original_pages，输出 processed_pages，兼容扩展 Manifest
（只增不改 M1 字段，原子写入），逐页隔离失败，保护已完成成果。
输入/输出均为 RGB / uint8 / C-contiguous。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .background_cleaner import BackgroundCleaner
from .config import TaskConfig
from .geometry_corrector import GeometryCorrector
from .models import PageData, ProcessResult
from .page_analyzer import PageAnalyzer
from .quality_checker import QualityChecker
from .recovery import can_reuse_processed, sha256_file
from .text_enhancer import TextEnhancer
from .utils.atomic_file import atomic_save_png, atomic_write_json

logger = logging.getLogger("bookclean")


def _step(stage: str, res: ProcessResult) -> dict:
    return {
        "stage": stage,
        "op": res.params.get("operation", stage),
        "executed": res.executed,
        "params": res.params,
        "elapsed_ms": round(res.elapsed_ms, 1),
        "suggest_fallback": res.suggest_fallback,
    }


class ProcessingPipeline:
    def __init__(self):
        self.analyzer = PageAnalyzer()
        self.geometry = GeometryCorrector()
        self.background = BackgroundCleaner()
        self.enhancer = TextEnhancer()
        self.quality = QualityChecker()

    def process_page(self, page: PageData, config: TaskConfig) -> ProcessResult:
        start = time.perf_counter()
        original = page.image
        image = original
        steps: list[dict] = []
        warnings: list[str] = []
        fallback_events: list[dict] = []

        analysis = self.analyzer.analyze(page)
        steps.append({"stage": "analyze", "op": "analyze", "executed": True,
                      "params": {"skew_angle": round(analysis.skew_angle, 3),
                                 "skew_confidence": round(analysis.skew_confidence, 3),
                                 "recommended_steps": analysis.recommended_steps,
                                 "warnings": analysis.warnings},
                      "elapsed_ms": 0.0, "suggest_fallback": False})

        # 几何校正
        deskew_applied = False
        r = self.geometry.deskew(image, analysis, config)
        steps.append(_step("geometry", r)); image = r.image; deskew_applied = r.executed
        r = self.geometry.crop(image, analysis, config)
        steps.append(_step("geometry", r)); image = r.image
        r = self.geometry.correct_perspective(image, analysis, config)
        steps.append(_step("geometry", r)); image = r.image
        if r.executed:
            a_geo = self.analyzer.analyze(PageData(page.page_index, image, page.source_path,
                                                   page.source_kind, image.shape[1], image.shape[0],
                                                   page.source_dpi, page.color_mode))
            r2 = self.geometry.deskew(image, a_geo, config)
            steps.append(_step("geometry", r2)); image = r2.image; deskew_applied = deskew_applied or r2.executed
        geometry_applied = any(s["executed"] for s in steps if s["stage"] == "geometry")
        geometry_image = image

        # 背景清理 + 文字增强：各阶段保留独立中间结果，回退不覆盖前序
        def enhance_pass(img: np.ndarray, cfg: TaskConfig):
            a_bg = self.analyzer.photometrics(img)
            bg_res = self.background.process(img, a_bg, cfg)
            a_en = self.analyzer.photometrics(bg_res.image)
            en_res = self.enhancer.process(bg_res.image, a_en, cfg)
            return bg_res, en_res

        def record_bg(bg_res, stage):
            for s in bg_res.steps:
                steps.append({"stage": stage, **s, "elapsed_ms": round(bg_res.elapsed_ms, 1),
                              "suggest_fallback": False})

        bg_res, en_res = enhance_pass(geometry_image, config)
        record_bg(bg_res, "background")
        steps.append(_step("enhance", en_res))
        bg_image, en_image = bg_res.image, en_res.image
        regions = bg_res.regions
        cleaner_chroma = next((s.get("chroma_retention") for s in bg_res.steps
                               if s["op"] == "color_protection_check"), None)
        warnings.extend(bg_res.warnings); warnings.extend(en_res.warnings)

        def evaluate(img, cfg, regs):
            return self.quality.evaluate(original, img, analysis, cfg,
                                         geometry_applied, deskew_applied, regions=regs)

        report = evaluate(en_image, config, regions)
        retry_count = 0
        final_image, final_source = en_image, "enhanced"
        fallback_level, fallback_reason = "none", ""

        # 有限重试：最多一次，且必须实际降低强度
        if report["decision"] == "retry_lower_strength":
            retry_count = 1
            half = replace(config, background_strength=config.background_strength * 0.5,
                           yellow_removal_strength=config.yellow_removal_strength * 0.5,
                           denoise_strength=config.denoise_strength * 0.5,
                           sharpen_strength=config.sharpen_strength * 0.5)
            bg2, en2 = enhance_pass(geometry_image, half)
            record_bg(bg2, "retry_background")
            steps.append({"stage": "retry", "op": "half_strength", "executed": True,
                          "params": {"background_strength": half.background_strength,
                                     "yellow_removal_strength": half.yellow_removal_strength,
                                     "denoise_strength": half.denoise_strength,
                                     "sharpen_strength": half.sharpen_strength},
                          "elapsed_ms": round(en2.elapsed_ms, 1), "suggest_fallback": False})
            warnings.extend(bg2.warnings); warnings.extend(en2.warnings)
            report = evaluate(en2.image, half, bg2.regions)
            if report["decision"] == "accept":
                final_image, final_source = en2.image, "retry_enhanced"

        if report["decision"] != "accept":
            # 依据失败阶段选择回退起点，逐级向下直到目标通过质量检查；original 为最终保底
            start_idx = self._fallback_start(report, bg_res, geometry_applied)
            ladder = [("background", bg_image), ("geometry", geometry_image), ("original", original)]
            fallback_reason = "; ".join(report["reasons"]) or report["decision"]
            for level, target in ladder[start_idx:]:
                if level == "original":
                    final_image, final_source, fallback_level = original, "original", "original"
                    break
                if evaluate(target, config, regions)["decision"] == "accept":
                    final_image, final_source, fallback_level = target, level, level
                    break
            else:
                final_image, final_source, fallback_level = original, "original", "original"
            fallback_events.append({"stage": "fallback", "reason": fallback_reason,
                                    "target": final_source, "fallback_level": fallback_level,
                                    "checker_decision": report["decision"]})
            warnings.append(f"fallback_to_{fallback_level}")

        decision = "accept" if fallback_level == "none" else f"fallback_{fallback_level}"
        dim_chroma = report.get("dimensions", {}).get("color_preservation", {}).get("chroma_retention")
        color_score = dim_chroma if dim_chroma is not None else cleaner_chroma
        image = np.ascontiguousarray(final_image)
        return ProcessResult(
            image=image,
            executed=True,
            params={"operation": "pipeline", "preset": config.preset, "output_mode": config.output_mode,
                    "geometry_applied": geometry_applied, "deskew_applied": deskew_applied},
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
            warnings=warnings,
            suggest_fallback=bool(fallback_events),
            steps=steps,
            quality={**report, "decision": decision, "checker_decision": report["decision"],
                     "fallback_level": fallback_level, "fallback_reason": fallback_reason,
                     "retry_count": retry_count, "final_image_source": final_source,
                     "color_preservation_score": color_score,
                     "fallback_events": fallback_events},
        )

    @staticmethod
    def _fallback_start(report: dict, bg_res: ProcessResult, geometry_applied: bool) -> int:
        """按失败阶段决定回退起点：0=保留背景去增强, 1=几何结果, 2=原始页面。"""
        checks = {c["name"]: c for c in report["checks"]}

        def failed(name):
            c = checks.get(name)
            return c is not None and not c["passed"]

        if geometry_applied and failed("size"):
            return 2  # 几何裁切损伤内容 -> 原始
        if (bg_res.suggest_fallback or failed("color") or failed("illustration")
                or failed("text_core") or failed("foreground") or failed("blank")):
            return 1  # 背景/增强损伤内容 -> 几何结果
        if failed("sharpness"):
            return 0  # 仅增强阶段清晰度问题 -> 保留背景清理
        return 1


class ProcessingRunner:
    """读取 M1 任务目录，逐页处理并兼容扩展 Manifest。"""

    def __init__(self, pipeline: ProcessingPipeline | None = None):
        self.pipeline = pipeline or ProcessingPipeline()

    def run(self, task_dir: str | Path, config: TaskConfig,
            progress_cb: Callable[[dict], None] | None = None,
            cancel_event=None, resume: bool = False) -> dict:
        task_dir = Path(task_dir)
        manifest_path = task_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        (task_dir / "processed_pages").mkdir(exist_ok=True)
        started = time.monotonic()
        timings: list[float] = []
        decisions: dict[str, int] = {}
        processed = failed = fallbacks = reused = 0

        for entry in manifest["pages"]:
            if cancel_event is not None and cancel_event.is_set():
                entry["processing_status"] = "cancelled"
                break
            if entry.get("status") != "completed" or not entry.get("image_path"):
                entry["processing_status"] = "skipped"
                continue
            if resume:
                ok, _reasons = can_reuse_processed(task_dir, entry, config)
                if ok:
                    reused += 1
                    if progress_cb is not None:
                        progress_cb(entry)
                    continue
            entry["processing_status"] = "processing"
            try:
                page = self._load_page(task_dir, entry)
                result = self.pipeline.process_page(page, config)
                rel, meta = self._save_processed(task_dir, entry, result, config)
                self._update_entry(entry, result, rel, config, meta)
                processed += 1
                decisions[result.quality["decision"]] = decisions.get(result.quality["decision"], 0) + 1
                if result.quality["fallback_events"]:
                    fallbacks += 1
                timings.append(result.elapsed_ms)
                page.image = None
                del page, result
            except Exception as exc:
                entry["processing_status"] = "failed"
                entry["processing_error"] = f"{type(exc).__name__}: {exc}"
                failed += 1
                logger.error("页面处理失败 第 %d 页: %s", entry["display_index"], exc)
            atomic_write_json(manifest_path, manifest)
            if progress_cb is not None:
                progress_cb(entry)

        summary = {
            "total_pages": manifest["total_pages"],
            "processed": processed, "failed": failed, "fallbacks": fallbacks,
            "reused": reused,
            "decisions": decisions,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "avg_page_ms": round(sum(timings) / len(timings), 1) if timings else 0.0,
            "preset": config.preset, "output_mode": config.output_mode,
        }
        manifest["processing_status"] = "completed" if failed == 0 else "completed_with_errors"
        manifest["processing_summary"] = summary
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(task_dir / "processing_report.json", summary)
        quality_report = self._quality_report(manifest["pages"], config)
        atomic_write_json(task_dir / "quality_report.json", quality_report)
        manifest["quality_report"] = "quality_report.json"
        if processed:
            sheet = self._comparison_sheet(task_dir, manifest["pages"])
            manifest["processing_contact_sheet"] = sheet
            qsheet = self._quality_sheet(task_dir, manifest["pages"])
            manifest["quality_contact_sheet"] = qsheet
        atomic_write_json(manifest_path, manifest)
        return {"task_dir": task_dir, "manifest": manifest, "summary": summary,
                "quality_report": quality_report}

    @staticmethod
    def _load_page(task_dir: Path, entry: dict) -> PageData:
        im = Image.open(task_dir / entry["image_path"]).convert("RGB")
        arr = np.ascontiguousarray(np.asarray(im, dtype=np.uint8))
        dpi = entry.get("dpi")
        return PageData(entry["page_index"], arr, entry["source_path"], entry["source_kind"],
                        arr.shape[1], arr.shape[0], dpi, "RGB", entry.get("metadata", {}))

    @staticmethod
    def _save_processed(task_dir: Path, entry: dict, result: ProcessResult, config: TaskConfig) -> tuple[str, dict]:
        name = f"page_{entry['display_index']:06d}.png"
        rel = f"processed_pages/{name}"
        dpi = entry.get("dpi") or config.output_dpi
        path = task_dir / rel
        atomic_save_png(Image.fromarray(result.image), path, dpi=dpi)
        with Image.open(path) as im:
            meta = {"processed_sha256": sha256_file(path),
                    "processed_width_px": im.width, "processed_height_px": im.height,
                    "processed_color_mode": im.mode, "processed_format": (im.format or "PNG").upper()}
        return rel, meta

    @staticmethod
    def _update_entry(entry: dict, result: ProcessResult, rel: str, config: TaskConfig, meta: dict) -> None:
        applied, skipped = [], []
        for s in result.steps:
            if s["stage"] in ("analyze", "retry"):
                continue
            label = s.get("op", s["stage"])
            if s["executed"]:
                applied.append(label)
            else:
                skipped.append({"op": label, "reason": s.get("params", {}).get("reason", s.get("reason", ""))})
        q = result.quality
        dim = q.get("dimensions", {})
        pb = dim.get("paper_background", {})
        ar = q.get("region_area_ratios", {})
        tp = dim.get("text_preservation", {})
        text_score = tp.get("text_core_retention")
        if text_score is None:
            text_score = tp.get("foreground_retention")
        entry.update({
            "processing_status": "completed",
            "processed_image_path": rel,
            **meta,
            "applied_operations": applied,
            "skipped_operations": skipped,
            "processing_parameters": {"preset": config.preset, "output_mode": config.output_mode,
                                      "background_strength": config.background_strength,
                                      "yellow_removal_strength": config.yellow_removal_strength,
                                      "denoise_strength": config.denoise_strength,
                                      "sharpen_strength": config.sharpen_strength},
            "quality_warnings": result.warnings + q.get("reasons", []),
            "fallback_events": q.get("fallback_events", []),
            "processing_decision": q["decision"],
            "processing_ms": round(result.elapsed_ms, 1),
            # M3.4 分区质量与回退追踪字段（只增不改 M1/M2 字段）
            "page_type": q.get("page_type"),
            "region_mask_confidence": q.get("region_mask_confidence"),
            "paper_area_ratio": ar.get("paper"),
            "text_area_ratio": ar.get("text"),
            "color_area_ratio": ar.get("color"),
            "background_quality_before": pb.get("before"),
            "background_quality_after": pb.get("after"),
            "text_preservation_score": text_score,
            "color_preservation_score": q.get("color_preservation_score"),
            "illustration_preservation_score": dim.get("illustration_preservation", {}).get("gray_std_retention"),
            "quality_risk_score": q.get("risk_score"),
            "decision": q["decision"],
            "retry_count": q.get("retry_count", 0),
            "fallback_level": q.get("fallback_level", "none"),
            "fallback_reason": q.get("fallback_reason", ""),
            "final_image_source": q.get("final_image_source"),
        })

    @staticmethod
    def _comparison_sheet(task_dir: Path, pages: list[dict]) -> str:
        done = [p for p in pages if p.get("processing_status") == "completed" and p.get("processed_image_path")]
        cols, cell_w, cell_h, label_h, pad = 4, 200, 280, 22, 8
        rows = (len(done) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * (2 * cell_w + pad) + pad, rows * (cell_h + label_h + pad) + pad), (245, 245, 245))
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype("msyh.ttc", 13)
            label = "第 {} 页  原图 | 修复"
        except OSError:
            font = ImageFont.load_default(size=13)
            label = "Page {}  orig | fixed"
        for i, entry in enumerate(done):
            x = pad + (i % cols) * (2 * cell_w + pad)
            y = pad + (i // cols) * (cell_h + label_h + pad)
            for j, key in enumerate(("image_path", "processed_image_path")):
                thumb = Image.open(task_dir / entry[key])
                thumb.thumbnail((cell_w, cell_h), Image.LANCZOS)
                sheet.paste(thumb, (x + j * cell_w + (cell_w - thumb.width) // 2, y))
                thumb.close()
            draw.text((x, y + cell_h + 3), label.format(entry["display_index"]), fill=(60, 60, 60), font=font)
        rel = "processing_contact_sheet.jpg"
        sheet.save(task_dir / rel, format="JPEG", quality=85)
        return rel

    @staticmethod
    def _quality_report(pages: list[dict], config: TaskConfig) -> dict:
        done = [p for p in pages if p.get("processing_status") == "completed"]
        type_counts: dict[str, int] = {}
        type_ms: dict[str, list[float]] = {}
        reasons: dict[str, int] = {}
        accept = retry = fallback = 0
        abnormal, review = [], []
        for p in done:
            pt = p.get("page_type") or "unknown"
            type_counts[pt] = type_counts.get(pt, 0) + 1
            type_ms.setdefault(pt, []).append(p.get("processing_ms", 0.0))
            dec = p.get("decision", "accept")
            if dec == "accept":
                accept += 1
            if p.get("retry_count", 0) > 0:
                retry += 1
            lvl = p.get("fallback_level", "none")
            if lvl != "none":
                fallback += 1
                r = p.get("fallback_reason") or dec
                reasons[r] = reasons.get(r, 0) + 1
            risky = (p.get("quality_risk_score") or 0) > config.quality_risk_accept_max
            if lvl != "none" or p.get("retry_count", 0) > 0 or risky:
                abnormal.append(p["display_index"])
            if lvl != "none" or dec != "accept":
                review.append(p["display_index"])
        for p in pages:
            if p.get("processing_status") == "failed":
                review.append(p["display_index"])
        top_reasons = sorted(reasons.items(), key=lambda kv: -kv[1])[:5]
        return {
            "total_pages": len(pages),
            "processed": len(done),
            "accept": accept, "retry": retry, "fallback": fallback,
            "page_type_counts": type_counts,
            "avg_ms_by_page_type": {k: round(sum(v) / len(v), 1) for k, v in type_ms.items()},
            "main_fallback_reasons": [{"reason": r, "count": c} for r, c in top_reasons],
            "quality_abnormal_pages": sorted(abnormal),
            "manual_review_pages": sorted(set(review)),
        }

    @staticmethod
    def _quality_sheet(task_dir: Path, pages: list[dict]) -> str:
        flagged = [p for p in pages
                   if p.get("processing_status") == "completed" and p.get("processed_image_path")
                   and (p.get("fallback_level", "none") != "none" or p.get("retry_count", 0) > 0)]
        cols, cell_w, cell_h, label_h, pad = 3, 220, 300, 40, 8
        rows = max(1, (len(flagged) + cols - 1) // cols)
        sheet = Image.new("RGB", (cols * (2 * cell_w + pad) + pad, rows * (cell_h + label_h + pad) + pad), (250, 244, 240))
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype("msyh.ttc", 13)
            small = ImageFont.truetype("msyh.ttc", 11)
        except OSError:
            font = ImageFont.load_default(size=13)
            small = ImageFont.load_default(size=11)
        if not flagged:
            draw.text((pad, pad), "无重试/回退页面 (no retry/fallback pages)", fill=(60, 60, 60), font=font)
        for i, entry in enumerate(flagged):
            x = pad + (i % cols) * (2 * cell_w + pad)
            y = pad + (i // cols) * (cell_h + label_h + pad)
            for j, key in enumerate(("image_path", "processed_image_path")):
                thumb = Image.open(task_dir / entry[key])
                thumb.thumbnail((cell_w, cell_h), Image.LANCZOS)
                sheet.paste(thumb, (x + j * cell_w + (cell_w - thumb.width) // 2, y))
                thumb.close()
            draw.rectangle((x - 2, y - 2, x + 2 * cell_w + 2, y + cell_h + label_h + 2), outline=(200, 60, 40), width=2)
            draw.text((x, y + cell_h + 3),
                      f"第 {entry['display_index']} 页  {entry.get('decision', '')}", fill=(160, 40, 30), font=font)
            draw.text((x, y + cell_h + 20),
                      f"fallback={entry.get('fallback_level', 'none')} retry={entry.get('retry_count', 0)} "
                      f"risk={entry.get('quality_risk_score') or 0}", fill=(90, 60, 50), font=small)
        rel = "quality_contact_sheet.jpg"
        sheet.save(task_dir / rel, format="JPEG", quality=88)
        return rel