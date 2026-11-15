"""M4.8.1 扫描黑边清除命令行工具。

对 M1 任务目录的每页最终图像（processed 优先，回退 original）做保守黑边清除，
输出 border_pages/ + border_masks/，兼容扩展 manifest.json，生成前后对比联系表。
原始页面与 processed_pages 只读，不重跑 M1–M3。

用法:
  python scripts/clean_borders.py <task_dir> [--min-confidence 0.6] [--max-edge-fraction 0.12]
                                  [--dry-run] [--compare N]
清除后可直接用 scripts/export_pdf.py 导出（FinalImageResolver 自动优先 border_pages）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image

from bookclean.border_detector import BorderCleaningRunner
from bookclean.config import BorderCleanConfig


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="BookClean PDF M4.8.1 扫描黑边清除")
    parser.add_argument("task_dir", help="M1 任务目录（含 manifest.json）")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--max-edge-fraction", type=float, default=0.12)
    parser.add_argument("--dry-run", action="store_true", help="只检测不写文件，打印每页判定")
    parser.add_argument("--compare", type=int, default=0, metavar="N",
                        help="为前 N 个清除页生成全分辨率前后对比图")
    args = parser.parse_args(argv)

    task_dir = Path(args.task_dir)
    if not (task_dir / "manifest.json").exists():
        print(f"错误: {task_dir} 不是有效任务目录（缺少 manifest.json）")
        return 2

    config = BorderCleanConfig(min_confidence=args.min_confidence,
                               max_edge_fraction=args.max_edge_fraction)
    config.validate()

    def progress(entry):
        st = entry.get("border_status", "?")
        det = entry.get("border_detection", {})
        if args.dry_run:
            edges = {e: d["reason"] for e, d in det.get("edges", {}).items() if d.get("width_analysis")}
            print(f"  第 {entry['display_index']} 页: {det.get('reason', '')} {edges}")
        else:
            print(f"  第 {entry['display_index']} 页: {st} ({det.get('reason', '')}) "
                  f"{entry.get('border_ms', '')}ms")

    runner = BorderCleaningRunner()
    result = runner.run(task_dir, config, progress_cb=progress, dry_run=args.dry_run)
    s = result["summary"]
    print("\n=== 黑边清除摘要 ===")
    print(f"总页数 {s['total_pages']}  清除 {s['cleaned']}  跳过 {s['skipped']}  失败 {s['failed']}"
          + ("  [dry-run]" if s["dry_run"] else ""))
    print(f"判定分布 {s['reasons']}")
    print(f"耗时 {s['elapsed_seconds']}s  平均 {s['avg_page_ms']}ms/页")

    if not args.dry_run:
        print(f"对比联系表: {result['manifest'].get('border_contact_sheet')}")
        if args.compare > 0:
            comp_dir = task_dir / "border_comparisons"
            comp_dir.mkdir(exist_ok=True)
            done = [p for p in result["manifest"]["pages"]
                    if p.get("border_status") == "completed"][:args.compare]
            for entry in done:
                with Image.open(task_dir / entry["border_source_path"]) as orig, \
                        Image.open(task_dir / entry["border_image_path"]) as proc:
                    orig, proc = orig.convert("RGB"), proc.convert("RGB")
                    h = min(orig.height, proc.height)
                    ow, pw = int(orig.width * h / orig.height), int(proc.width * h / proc.height)
                    orig, proc = orig.resize((ow, h)), proc.resize((pw, h))
                    canvas = Image.new("RGB", (ow + pw + 20, h), (255, 255, 255))
                    canvas.paste(orig, (0, 0))
                    canvas.paste(proc, (ow + 20, 0))
                    out = comp_dir / f"page_{entry['display_index']:06d}_compare.png"
                    canvas.save(out)
                    print(f"对比图: {out}")
    return 0 if s["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())