"""M3.5 综合回归：50 页混合长任务（与 M2 同输入同预设）性能/重试/回退/内存 + M2 对比。

复用 tests/test_m2_long_task 的 _mixed_pages 与 working_set_mb，保证与 M2 基线 apples-to-apples。
产出 temp/m3_long_task/{long_task_report.json, quality_report.json, manifest_excerpt.json, 联系表}。
运行: python scripts/make_m3_long_task.py [输出目录]。本地合成，不联网。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE.parent / "tests"))
sys.path.insert(0, str(_HERE.parent))

from test_m2_long_task import PAGE_COUNT, _mixed_pages, working_set_mb  # noqa: E402

from bookclean.presets import get_preset  # noqa: E402
from bookclean.processing_pipeline import ProcessingRunner  # noqa: E402
from bookclean.task_manager import ImportTaskManager  # noqa: E402

# M2 基线（temp/m2_long_memory.log，50 页 auto 预设）
M2_BASELINE = {"growth_mb": 32.0, "peak_mb": 103.0, "processed": 50, "failed": 0,
               "elapsed_seconds": 74.72, "avg_page_ms": 1406.3,
               "decisions": {"accept": 45, "retry_lower_strength": 5}}


def main(out_dir="temp/m3_long_task"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    in_dir = out / "inputs"
    in_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for i, rgb in enumerate(_mixed_pages(PAGE_COUNT)):
        p = in_dir / f"page_{i + 1:03d}.png"
        Image.fromarray(rgb).save(p, dpi=(150, 150))
        files.append(p)
    task = ImportTaskManager(out / "tasks").run(files, dpi=150)["task_dir"]

    samples = [working_set_mb()]

    def progress(entry):
        if entry["page_index"] % 10 == 0:
            samples.append(working_set_mb())

    res = ProcessingRunner().run(task, get_preset("auto"), progress_cb=progress)
    samples.append(working_set_mb())
    s = res["summary"]
    manifest = res["manifest"]
    pages = manifest["pages"]
    retry = sum(p.get("retry_count", 0) for p in pages)
    fallback = sum(1 for p in pages if p.get("fallback_level", "none") != "none")
    growth = samples[-1] - samples[0]
    peak = max(samples)

    report = {
        "page_count": PAGE_COUNT,
        "processed": s["processed"], "failed": s["failed"],
        "elapsed_seconds": s["elapsed_seconds"], "avg_page_ms": s["avg_page_ms"],
        "peak_memory_mb": round(peak, 1), "net_memory_growth_mb": round(growth, 1),
        "memory_samples_mb": [round(x, 1) for x in samples],
        "retry_count": retry, "fallback_count": fallback,
        "decisions": s["decisions"],
        "m2_baseline": M2_BASELINE,
    }
    (out / "long_task_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "quality_report.json").write_text(
        json.dumps(res["quality_report"], ensure_ascii=False, indent=2), encoding="utf-8")
    keys = ("display_index", "page_type", "decision", "retry_count", "fallback_level",
            "final_image_source", "quality_risk_score", "processing_ms")
    (out / "manifest_excerpt.json").write_text(
        json.dumps([{k: p.get(k) for k in keys} for p in pages], ensure_ascii=False, indent=2), encoding="utf-8")
    for f in ("quality_contact_sheet.jpg", "processing_contact_sheet.jpg"):
        if (task / f).exists():
            (out / f).write_bytes((task / f).read_bytes())

    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert s["processed"] == PAGE_COUNT and s["failed"] == 0, s
    assert growth < 100, f"内存持续增长 {growth:.1f} MB"
    assert peak - samples[0] < 250, f"内存峰值异常 {peak:.1f} MB"
    print(f"\n证据目录 -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "temp/m3_long_task"))