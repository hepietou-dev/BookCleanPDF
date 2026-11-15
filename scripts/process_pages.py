"""处理 M1 任务目录的命令行工具（M2.4）。

读取 task_dir/original_pages，按预设逐页修复，输出 processed_pages/，
兼容扩展 manifest.json，生成前后对比联系表与（可选）全分辨率对比图。

用法:
  python scripts/process_pages.py <task_dir> [--preset auto] [--output-mode clean_gray]
                                  [--compare N] [--background-strength 0.7]
原始页面只读，不覆盖 M1 产物。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from PIL import Image

from bookclean.config import TaskConfig
from bookclean.presets import PRESET_LABELS, get_preset, list_presets
from bookclean.processing_pipeline import ProcessingRunner


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="BookClean PDF M2 页面修复")
    parser.add_argument("task_dir", help="M1 任务目录（含 manifest.json 与 original_pages）")
    parser.add_argument("--preset", default="auto", choices=list_presets(),
                        help="预设: " + ", ".join(f"{k}={v}" for k, v in PRESET_LABELS.items()))
    parser.add_argument("--output-mode", default=None, choices=["clean_gray", "adaptive_bw", "mixed"])
    parser.add_argument("--background-strength", type=float, default=None)
    parser.add_argument("--yellow-removal-strength", type=float, default=None)
    parser.add_argument("--compare", type=int, default=0, metavar="N",
                        help="为前 N 个完成页生成全分辨率前后对比图")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑：复用已通过完整性/SHA256/尺寸/色彩/配置校验的 processed_pages，不重跑 M1–M3")
    args = parser.parse_args(argv)

    task_dir = Path(args.task_dir)
    if not (task_dir / "manifest.json").exists():
        print(f"错误: {task_dir} 不是有效任务目录（缺少 manifest.json）")
        return 2

    config = get_preset(args.preset)
    if args.output_mode:
        config.output_mode = args.output_mode
    if args.background_strength is not None:
        config.background_strength = args.background_strength
    if args.yellow_removal_strength is not None:
        config.yellow_removal_strength = args.yellow_removal_strength
    config.validate()

    def progress(entry):
        st = entry.get("processing_status", "?")
        print(f"  第 {entry['display_index']} 页: {st} "
              f"({entry.get('processing_decision', '')}) {entry.get('processing_ms', '')}ms")

    print(f"预设: {PRESET_LABELS[config.preset]}  输出模式: {config.output_mode}")
    runner = ProcessingRunner()
    result = runner.run(task_dir, config, progress_cb=progress, resume=args.resume)
    s = result["summary"]
    print("\n=== 处理摘要 ===")
    print(f"总页数 {s['total_pages']}  成功 {s['processed']}  失败 {s['failed']}  回退 {s['fallbacks']}  复用 {s.get('reused', 0)}")
    print(f"决策分布 {s['decisions']}")
    print(f"耗时 {s['elapsed_seconds']}s  平均 {s['avg_page_ms']}ms/页")
    print(f"对比联系表: {result['manifest'].get('processing_contact_sheet')}")

    if args.compare > 0:
        comp_dir = task_dir / "comparisons"
        comp_dir.mkdir(exist_ok=True)
        done = [p for p in result["manifest"]["pages"]
                if p.get("processing_status") == "completed"][:args.compare]
        for entry in done:
            orig = Image.open(task_dir / entry["image_path"]).convert("RGB")
            proc = Image.open(task_dir / entry["processed_image_path"]).convert("RGB")
            h = min(orig.height, proc.height)
            ow = int(orig.width * h / orig.height)
            pw = int(proc.width * h / proc.height)
            orig = orig.resize((ow, h))
            proc = proc.resize((pw, h))
            canvas = Image.new("RGB", (ow + pw + 20, h), (255, 255, 255))
            canvas.paste(orig, (0, 0))
            canvas.paste(proc, (ow + 20, 0))
            out = comp_dir / f"page_{entry['display_index']:06d}_compare.png"
            canvas.save(out)
            print(f"对比图: {out}")
    return 0 if s["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())