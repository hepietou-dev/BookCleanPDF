"""M3.4 安全回退与 Manifest 正式化证据：真实 runner 任务 + 强制回退演示联系表。

真实任务：对 10 类区域样本跑 ProcessingRunner，产出 Manifest(M3.4 字段)/quality_report.json/联系表。
演示任务：人为强制一页走回退，验证质量联系表突出显示重试/回退页（机制演示，非真实损伤）。
运行: python scripts/make_m3_fallback_evidence.py [输出目录]，默认 temp/m3_fallback_evidence/。本地合成，不联网。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_m2_samples import build_region_samples  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bookclean.presets import get_preset  # noqa: E402
from bookclean.processing_pipeline import ProcessingRunner  # noqa: E402
from bookclean.task_manager import ImportTaskManager  # noqa: E402


def _build_task(root: Path, names_samples, tag):
    in_dir = root / f"{tag}_inputs"
    in_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for name, rgb in names_samples:
        p = in_dir / f"{name}.png"
        Image.fromarray(rgb).save(p, dpi=(150, 150))
        files.append(p)
    mgr = ImportTaskManager(root / f"{tag}_tasks")
    return mgr.run(files, dpi=150)["task_dir"]


def _excerpt(manifest):
    keys = ("display_index", "page_type", "region_mask_confidence", "paper_area_ratio",
            "text_preservation_score", "color_preservation_score", "illustration_preservation_score",
            "quality_risk_score", "decision", "retry_count", "fallback_level", "final_image_source")
    return [{k: e.get(k) for k in keys} for e in manifest["pages"]]


def main(out_dir="temp/m3_fallback_evidence"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = build_region_samples()
    cfg = get_preset("old_book")

    # 真实任务
    task = _build_task(out, [(n, s["image"]) for n, s in samples.items()], "real")
    res = ProcessingRunner().run(task, cfg)
    qr = res["quality_report"]
    (out / "real_quality_report.json").write_text(json.dumps(qr, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "real_manifest_excerpt.json").write_text(json.dumps(_excerpt(res["manifest"]), ensure_ascii=False, indent=2), encoding="utf-8")
    for f in ("quality_contact_sheet.jpg", "processing_contact_sheet.jpg"):
        if (task / f).exists():
            (out / f"real_{f}").write_bytes((task / f).read_bytes())
    print("== 真实任务 ==")
    print(json.dumps(qr, ensure_ascii=False, indent=2))

    # 演示任务：强制彩色页走回退，验证联系表突出显示
    demo_pairs = [("pure_bw_text", samples["pure_bw_text"]["image"]),
                  ("yellow_color_map", samples["yellow_color_map"]["image"])]
    dtask = _build_task(out, demo_pairs, "demo")
    runner = ProcessingRunner()
    real = runner.pipeline.quality.evaluate

    def stub(original, candidate, analysis, config, geometry_applied=False, deskew_applied=False, regions=None):
        if original[0, 0, 2] < 200:  # 黄纸彩色页 -> 强制回退（机制演示）
            return {"decision": "fallback_original",
                    "checks": [{"name": "color", "passed": False, "value": 0.0, "detail": "demo forced"}],
                    "reasons": ["color: demo forced fallback"]}
        return real(original, candidate, analysis, config, geometry_applied, deskew_applied, regions=regions)

    runner.pipeline.quality.evaluate = stub
    dres = runner.run(dtask, cfg)
    (out / "demo_quality_report.json").write_text(json.dumps(dres["quality_report"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "demo_manifest_excerpt.json").write_text(json.dumps(_excerpt(dres["manifest"]), ensure_ascii=False, indent=2), encoding="utf-8")
    if (dtask / "quality_contact_sheet.jpg").exists():
        (out / "demo_quality_contact_sheet.jpg").write_bytes((dtask / "quality_contact_sheet.jpg").read_bytes())
    print("\n== 演示任务（强制回退） ==")
    print(json.dumps(dres["quality_report"], ensure_ascii=False, indent=2))
    print(f"\n证据目录 -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "temp/m3_fallback_evidence"))