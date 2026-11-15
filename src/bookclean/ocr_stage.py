"""M6.3 OCR 任务阶段：border 之后、export 之前，对每页最终图像逐页 OCR。

设计（docs/m6_0_ocr_architecture.md §9）：
- 缓存键 = 图像 sha256 + 引擎指纹 + 识别参数 + 结果 schema；**布局不入键**（布局变只重注入）。
- 逐页结果落盘 ocr/page_NNNNNN.json（仅 completed）+ ocr/ocr_manifest.json（原子写）；
  Manifest **只增** ocr_* 字段，不改 M1–M5 任何既有字段。
- **failed 页不阻塞导出**：导出仅注入 completed 页；全页失败/插件缺失 ⇒ 钩子 None ⇒ 与 M5 一致。
- 长文档每 N 页重启 worker 释放碎片；取消页边界协作，已完成页缓存保留；续跑复用缓存、重试 failed。
- 流式逐页：每页完成即落盘 ⇒ 崩溃/取消后从断点续跑，最坏重跑 1 页。

依赖方向：本模块导入 bookclean.ocr（插件子进程，基础包零新运行时依赖）与 FinalImageResolver；
**导出内核 pdf_exporter 不导入本模块**（依赖倒置）。worker 仅在 OCR 启用时惰性导入本模块，
故 OCR 关⇒不导入任何 OCR 代码。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Callable

from .config import OcrConfig
from .ocr.plugin_manager import PluginManager
from .ocr.provider import (
    OCR_SCHEMA_VERSION,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_SKIPPED,
    EngineVersion,
    OcrError,
    OcrParams,
    PageOcrResult,
)
from .ocr.paddle_provider import PaddleBridgeProvider
from .ocr.rapid_provider import RapidOcrProvider
from .ocr.text_source import OcrTextSource
from .recovery import sha256_file
from .utils.atomic_file import atomic_write_json

logger = logging.getLogger("bookclean")

OCR_MANIFEST_SCHEMA = 1


def _params_fingerprint(params: OcrParams) -> str:
    """识别参数指纹（threads/lang/text_score）。page_timeout_s 为操作参数，不影响识别结果，不入键。"""
    return json.dumps(
        {"threads": params.threads, "lang": params.lang, "text_score": params.text_score},
        sort_keys=True,
    )


def compute_cache_key(image_sha: str, version: EngineVersion, params: OcrParams) -> str:
    """OCR 缓存键（§9.3）：图像/引擎/模型/参数/schema 任一变化即失效；布局变化不失效。"""
    payload = "|".join([
        image_sha, version.fingerprint(), _params_fingerprint(params), str(OCR_SCHEMA_VERSION),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _text_chars(result: PageOcrResult) -> int:
    return sum(len(line.text) for line in result.lines)


def _make_provider(plugin, params: OcrParams):
    """按插件 worker 种类分派 Provider（rapid / paddle-bridge）。"""
    if plugin.worker_kind == "paddle-bridge":
        return PaddleBridgeProvider(plugin, params)
    return RapidOcrProvider(plugin, params)


class OcrStageRunner:
    """读取任务目录，对每页最终图像（FinalImageResolver，border 优先）逐页 OCR。

    原始页面/processed_pages/border_pages 只读；输出 ocr/ 缓存 + Manifest ocr_* 扩展字段。
    provider 可注入（测试用 FakeProvider）；默认经 PluginManager 选插件，按 worker 种类构造
    RapidOcrProvider / PaddleBridgeProvider。
    """

    def run(self, task_dir: str | Path, config: OcrConfig,
            progress_cb: Callable[[dict], None] | None = None,
            cancel_event=None, resume: bool = False, provider=None) -> dict:
        from .pdf_exporter import FinalImageResolver  # 惰性导入，避免与导出内核循环

        task_dir = Path(task_dir)
        config.validate()
        manifest_path = task_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pages = sorted(manifest["pages"], key=lambda p: p["page_index"])
        resolver = FinalImageResolver(task_dir)
        params = OcrParams(threads=config.threads, lang=config.lang,
                           text_score=config.text_score, page_timeout_s=config.page_timeout_s)
        started = time.monotonic()

        # ---- 选择引擎（测试注入 provider 时跳过插件发现） ----
        if provider is None:
            plugin = PluginManager(config.plugins_root or None).select(config.engine)
            if plugin is None:
                return self._degrade(manifest, manifest_path, pages, config, started,
                                     status="unavailable",
                                     reason="未安装 OCR 插件。OCR 为可选功能，已跳过文字层，普通导出不受影响。")
            provider = _make_provider(plugin, params)
        version = provider.get_version()

        # ---- 逐页解析最终图 + 计算缓存键 + 缓存命中判定（不加载模型） ----
        ocr_dir = task_dir / "ocr"
        records: list[dict] = []
        for entry in pages:
            fi = resolver.resolve(manifest, entry)
            rec = {"entry": entry, "fi": fi, "key": "", "image_sha": "", "cached": None}
            if fi.path is not None:
                try:
                    rec["image_sha"] = sha256_file(fi.path)
                    rec["key"] = compute_cache_key(rec["image_sha"], version, params)
                    rec["cached"] = self._load_cache(ocr_dir, entry, rec["key"])
                except OSError as exc:
                    rec["error"] = f"{type(exc).__name__}: {exc}"
            records.append(rec)

        need_fresh = any(r["cached"] is None and r["fi"].path is not None for r in records)
        if need_fresh:
            ocr_dir.mkdir(exist_ok=True)

        # ---- 流式逐页：缓存复用 / 识别 / 落盘 / 进度（页序） ----
        results: dict[int, PageOcrResult] = {}
        completed = failed = skipped = reused = 0
        total_chars = 0
        timings: list[float] = []
        restarts = 0
        fresh_count = 0
        initialized = False
        init_failed = False
        cancelled = False

        try:
            for rec in records:
                entry = rec["entry"]
                pidx = entry["page_index"]
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    entry["ocr_status"] = "cancelled"
                    break
                t0 = time.perf_counter()
                if rec["fi"].path is None:
                    result = PageOcrResult(page_no=pidx, status=STATUS_SKIPPED,
                                           engine_version=version,
                                           error=OcrError("MISSING_IMAGE", "无可用最终图像"))
                    source = "none"
                    skipped += 1
                elif rec["cached"] is not None:
                    result = rec["cached"]
                    source = "cached"
                    reused += 1
                elif init_failed:
                    result = PageOcrResult(page_no=pidx, status=STATUS_FAILED,
                                           image_sha256=rec["image_sha"], engine_version=version,
                                           error=OcrError("INIT_FAILED", "OCR 引擎初始化失败，已跳过该页"))
                    source = "none"
                else:
                    # 长文档周期性重启 worker 释放内存碎片（在识别满 N 页后、下一页前重启）
                    if (config.restart_every_n_pages and fresh_count
                            and fresh_count % config.restart_every_n_pages == 0):
                        provider.shutdown()
                        initialized = False
                        restarts += 1
                    if not initialized:
                        init = provider.initialize(params)
                        if not init.ok:
                            init_failed = True
                            err = init.error or OcrError("INIT_FAILED", "OCR 引擎初始化失败")
                            result = PageOcrResult(page_no=pidx, status=STATUS_FAILED,
                                                   image_sha256=rec["image_sha"],
                                                   engine_version=version, error=err)
                            source = "none"
                        else:
                            initialized = True
                    if initialized:
                        result = provider.recognize_page(rec["fi"].path, pidx)
                        fresh_count += 1
                        source = "fresh"
                        if result.status == STATUS_COMPLETED:
                            self._write_cache(ocr_dir, entry, rec["key"], result)

                elapsed_ms = (time.perf_counter() - t0) * 1000
                timings.append(elapsed_ms)
                chars = _text_chars(result) if result.status == STATUS_COMPLETED else 0
                if result.status == STATUS_COMPLETED:
                    completed += 1
                    total_chars += chars
                    results[pidx] = result
                elif result.status == STATUS_FAILED:
                    failed += 1
                self._update_entry(entry, result, rec["key"], chars, source, elapsed_ms)
                atomic_write_json(manifest_path, manifest)
                if progress_cb is not None:
                    progress_cb(entry)
        finally:
            try:
                provider.shutdown()
            except Exception as exc:  # shutdown 绝不抛出到阶段层
                logger.warning("OCR provider 关闭异常（已忽略）: %s", exc)

        with_text = sum(1 for r in results.values() if _text_chars(r) > 0)
        total = len(pages)
        summary = {
            "total_pages": total,
            "completed": completed, "failed": failed, "skipped": skipped, "reused": reused,
            "total_text_chars": total_chars,
            "coverage": round(with_text / total, 4) if total else 0.0,
            "engine_id": version.engine_id, "engine_version": version.engine_version,
            "restarts": restarts,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "avg_page_ms": round(sum(timings) / len(timings), 1) if timings else 0.0,
        }
        if cancelled:
            manifest["ocr_status"] = "cancelled"
        else:
            manifest["ocr_status"] = "completed" if failed == 0 else "completed_with_errors"
        manifest["ocr_summary"] = summary
        atomic_write_json(manifest_path, manifest)
        if ocr_dir.is_dir():
            atomic_write_json(ocr_dir / "ocr_manifest.json", {
                "schema_version": OCR_MANIFEST_SCHEMA,
                "engine_version": version.to_dict(),
                "params": params.to_dict(),
                "ocr_schema_version": OCR_SCHEMA_VERSION,
                "summary": summary,
                "pages": {
                    str(r["entry"]["page_index"]): {
                        "display_index": r["entry"].get("display_index"),
                        "cache_key": r["key"],
                        "status": r["entry"].get("ocr_status"),
                        "text_chars": r["entry"].get("ocr_text_chars", 0),
                    } for r in records
                },
            })
        atomic_write_json(task_dir / "ocr_report.json", summary)

        # 全页失败/无 completed ⇒ 钩子 None ⇒ 导出与 M5 完全一致
        text_source = OcrTextSource.from_ocr_results(results) if results else None
        if text_source is not None and len(text_source) == 0:
            text_source = None
        return {"task_dir": task_dir, "manifest": manifest, "summary": summary,
                "results": results, "text_source": text_source, "available": True}

    # ---- 降级（插件缺失/不可用）：不产生任何新目录/文件，导出照常 ----

    def _degrade(self, manifest: dict, manifest_path: Path, pages: list[dict],
                 config: OcrConfig, started: float, status: str, reason: str) -> dict:
        logger.warning("OCR 阶段降级（%s）: %s", status, reason)
        summary = {
            "total_pages": len(pages), "completed": 0, "failed": 0, "skipped": len(pages),
            "reused": 0, "total_text_chars": 0, "coverage": 0.0,
            "engine_id": "", "engine_version": "", "restarts": 0,
            "elapsed_seconds": round(time.monotonic() - started, 2), "avg_page_ms": 0.0,
            "reason": reason,
        }
        manifest["ocr_status"] = status
        manifest["ocr_summary"] = summary
        atomic_write_json(manifest_path, manifest)
        return {"task_dir": manifest_path.parent, "manifest": manifest, "summary": summary,
                "results": {}, "text_source": None, "available": False}

    # ---- 缓存读写 ----

    @staticmethod
    def _cache_path(ocr_dir: Path, entry: dict) -> Path:
        return ocr_dir / f"page_{entry.get('display_index', entry['page_index'] + 1):06d}.json"

    def _load_cache(self, ocr_dir: Path, entry: dict, key: str) -> PageOcrResult | None:
        path = self._cache_path(ocr_dir, entry)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if data.get("cache_key") != key:
            return None
        result = PageOcrResult.from_dict(data.get("result", {}))
        if result.status != STATUS_COMPLETED:
            return None
        return result

    def _write_cache(self, ocr_dir: Path, entry: dict, key: str, result: PageOcrResult) -> None:
        atomic_write_json(self._cache_path(ocr_dir, entry),
                          {"cache_key": key, "result": result.to_dict()})

    @staticmethod
    def _update_entry(entry: dict, result: PageOcrResult, key: str, chars: int,
                      source: str, elapsed_ms: float) -> None:
        entry["ocr_status"] = result.status
        entry["ocr_cache_key"] = key
        entry["ocr_text_chars"] = chars
        entry["ocr_source"] = source
        entry["ocr_ms"] = round(elapsed_ms, 1)
        entry["ocr_error"] = (f"{result.error.code}: {result.error.message}"
                              if result.error else None)