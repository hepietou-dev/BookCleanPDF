"""M4.8.6 综合长任务实测：300 页合成（整流水线）+ 120 页真实书（复用，不重跑 M1–M3）。

合成：M1 导入 -> M2/M3 修复 -> M4.8.1 黑边清除 -> M4 导出(balanced)+验证 -> 重开逐页校验；
      另跑一次 resume=True 验证 M4.8.4 断点复用（reused 计数）。
真实：复用 temp/m4_6_subset 已完成任务目录（processed_pages/border_pages 只读），
      先做 M4.8.4 恢复校验（can_reuse_processed），再 M4 导出 preserve+grayscale(balanced)+验证。
记录：总处理时间/峰值内存(Windows 工作集)/文件体积/页面成功率/失败页/回退页/PDF 验证。
不联网、不重跑真实书 M1–M3、不改动原始与既有产物。输出 temp/m4_8_review/m4_8_6_long_metrics.json。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import numpy as np
import pypdfium2 as pdfium
from PIL import Image

from test_m2_long_task import _mixed_pages, working_set_mb

from bookclean.config import PDFExportConfig
from bookclean.border_detector import BorderCleaningRunner
from bookclean.pdf_exporter import PDFExporter
from bookclean.presets import get_preset
from bookclean.processing_pipeline import ProcessingRunner
from bookclean.recovery import can_reuse_processed
from bookclean.task_manager import ImportTaskManager

REVIEW = ROOT / "temp" / "m4_8_review"
REAL_TASK = ROOT / "temp" / "m4_6_subset" / "tasks" / "task_20261112_140743_86d276"
SYN_PAGES = 300


def _reopen_check(pdf_path: Path, info: dict, full=True) -> dict:
    doc = pdfium.PdfDocument(str(pdf_path))
    n = len(doc)
    size_ok = True
    rendered = 0
    errors = 0
    for i in range(n):
        w, h = doc[i].get_size()
        exp = info["pages"][i]
        if abs(w - exp["output_page_width_pt"]) > 0.5 or abs(h - exp["output_page_height_pt"]) > 0.5:
            size_ok = False
        if full or i in (0, n // 2, n - 1):
            try:
                doc[i].render(scale=60 / 72).to_pil().convert("RGB")
                rendered += 1
            except Exception:
                errors += 1
    doc.close()
    return {"reopen_pages": n, "page_count_match": n == info["summary"]["total_pages"],
            "per_page_size_ok": size_ok, "rendered_pages": rendered, "render_errors": errors}


def _export(task: Path, out: Path, mode: str, color: str, samples: list) -> dict:
    esamples = [working_set_mb()]
    t0 = time.monotonic()

    def cb(rec):
        if rec["page_index"] % 10 == 0:
            esamples.append(working_set_mb())

    cfg = PDFExportConfig(compression=mode, color_mode=color, verify=True)
    info = PDFExporter(cfg).export(task, out, progress_cb=cb)
    esamples.append(working_set_mb())
    elapsed = time.monotonic() - t0
    s = info["summary"]
    rec = {
        "file": str(out.relative_to(ROOT)), "size_bytes": info["output_size_bytes"],
        "size_mb": round(info["output_size_bytes"] / 1e6, 2),
        "pages": s["total_pages"], "export_seconds": round(elapsed, 2),
        "peak_mb": round(max(esamples), 1), "growth_mb": round(esamples[-1] - esamples[0], 1),
        "validation": info["validation_status"],
        "jpeg_pages": s["jpeg_pages"], "png_pages": s["png_pages"],
        "grayscale_pages": s["grayscale_pages"], "color_pages": s["color_pages"],
        "fallback_original_pages": s["fallback_original_pages"],
        "fallback_geometry_pages": s["fallback_geometry_pages"],
        "downsampled_pages": s["downsampled_pages"],
    }
    rec.update(_reopen_check(out, info, full=True))
    samples.append(working_set_mb())
    return rec


def run_synthetic() -> dict:
    out = REVIEW / "long300"
    out.mkdir(parents=True, exist_ok=True)
    m = {"kind": f"synthetic_{SYN_PAGES}"}
    in_dir = out / "inputs"
    in_dir.mkdir(exist_ok=True)
    inputs = []
    for i, rgb in enumerate(_mixed_pages(SYN_PAGES)):
        p = in_dir / f"page_{i + 1:03d}.png"
        Image.fromarray(rgb).save(p, dpi=(150, 150))
        inputs.append(str(p))

    t0 = time.monotonic()
    m1 = ImportTaskManager(out / "tasks").run(inputs, dpi=150)
    task = m1["task_dir"]
    m["m1_seconds"] = round(time.monotonic() - t0, 2)
    m["m1_pages"] = m1["manifest"]["total_pages"]
    print(f"[合成] M1: {m['m1_pages']} 页 {m['m1_seconds']}s")

    samples = [working_set_mb()]
    t0 = time.monotonic()

    def pcb(e):
        if e["page_index"] % 10 == 0:
            samples.append(working_set_mb())

    res = ProcessingRunner().run(task, get_preset("auto"), progress_cb=pcb)
    samples.append(working_set_mb())
    s = res["summary"]
    m["m2m3"] = {"seconds": round(time.monotonic() - t0, 2), "processed": s["processed"],
                 "failed": s["failed"], "fallbacks": s["fallbacks"],
                 "peak_mb": round(max(samples), 1), "growth_mb": round(samples[-1] - samples[0], 1),
                 "avg_page_ms": s["avg_page_ms"]}
    print(f"[合成] M2/M3: processed={s['processed']} failed={s['failed']} fallbacks={s['fallbacks']} "
          f"{m['m2m3']['seconds']}s peak={m['m2m3']['peak_mb']}MB")

    t0 = time.monotonic()
    bres = BorderCleaningRunner().run(task)
    bs = bres["summary"]
    m["border"] = {"seconds": round(time.monotonic() - t0, 2), "cleaned": bs["cleaned"],
                   "skipped": bs["skipped"], "failed": bs["failed"]}
    print(f"[合成] 黑边: cleaned={bs['cleaned']} skipped={bs['skipped']} failed={bs['failed']} {m['border']['seconds']}s")

    m["exports"] = {"balanced_preserve": _export(task, out / "long_balanced.pdf", "balanced", "preserve", samples)}
    e = m["exports"]["balanced_preserve"]
    print(f"[合成] M4[balanced]: {e['pages']} 页 {e['size_mb']}MB {e['export_seconds']}s "
          f"peak={e['peak_mb']}MB validation={e['validation']} reopen={e['reopen_pages']} size_ok={e['per_page_size_ok']}")

    t0 = time.monotonic()
    rres = ProcessingRunner().run(task, get_preset("auto"), resume=True)
    m["resume"] = {"seconds": round(time.monotonic() - t0, 2), "reused": rres["summary"]["reused"],
                   "processed": rres["summary"]["processed"]}
    print(f"[合成] resume: reused={m['resume']['reused']} processed={m['resume']['processed']} {m['resume']['seconds']}s")

    m["overall_peak_mb"] = round(max(samples), 1)
    m["success_rate"] = round(s["processed"] / SYN_PAGES, 4)
    return m


def run_real() -> dict:
    m = {"kind": "real_120_reuse", "task_dir": str(REAL_TASK.relative_to(ROOT))}
    manifest = json.loads((REAL_TASK / "manifest.json").read_text(encoding="utf-8"))
    pages = sorted(manifest["pages"], key=lambda p: p["page_index"])
    m["pages"] = len(pages)

    cfg = get_preset("auto")
    t0 = time.monotonic()
    reusable = sum(1 for p in pages if can_reuse_processed(REAL_TASK, p, cfg)[0])
    m["recovery"] = {"reusable": reusable, "seconds": round(time.monotonic() - t0, 2)}
    print(f"[真实] 恢复校验: reusable={reusable}/{len(pages)} {m['recovery']['seconds']}s（不重跑 M1–M3）")

    from collections import Counter
    m["manifest_fallback"] = dict(Counter(p.get("fallback_level") for p in pages))
    m["manifest_border"] = dict(Counter(p.get("border_status") for p in pages))
    m["page_types"] = dict(Counter(p.get("page_type") for p in pages))

    samples = [working_set_mb()]
    m["exports"] = {
        "balanced_preserve": _export(REAL_TASK, REVIEW / "real120_preserve.pdf", "balanced", "preserve", samples),
        "balanced_grayscale": _export(REAL_TASK, REVIEW / "real120_grayscale.pdf", "balanced", "grayscale", samples),
    }
    for k, e in m["exports"].items():
        print(f"[真实] M4[{k}]: {e['pages']} 页 {e['size_mb']}MB {e['export_seconds']}s "
              f"peak={e['peak_mb']}MB validation={e['validation']} reopen={e['reopen_pages']} size_ok={e['per_page_size_ok']}")
    m["overall_peak_mb"] = round(max(samples), 1)
    return m


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    REVIEW.mkdir(parents=True, exist_ok=True)
    print("=== M4.8.6 300 页合成长任务（整流水线）===")
    syn = run_synthetic()
    print("\n=== M4.8.6 120 页真实书长任务（复用，不重跑 M1–M3）===")
    real = run_real()
    metrics = {"synthetic_300": syn, "real_120": real}
    out = REVIEW / "m4_8_6_long_metrics.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nOK -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())