"""M1 输入检查工具：导入 PDF/图片/文件夹，生成标准化页面、缩略图、联系表和 manifest。

用法:
    python scripts/inspect_input.py "输入文件或文件夹" [更多输入...] [--dpi 300] [--temp-root temp]
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bookclean.errors import BookCleanError  # noqa: E402
from bookclean.logging_setup import setup_logging  # noqa: E402
from bookclean.task_manager import ImportTaskManager  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="BookClean PDF 输入检查工具（M1）")
    parser.add_argument("inputs", nargs="+", help="PDF / 图片 / 文件夹")
    parser.add_argument("--dpi", type=int, default=300, help="PDF 渲染 DPI（默认 300）")
    parser.add_argument("--temp-root", default=str(PROJECT_ROOT / "temp"), help="任务目录根位置")
    args = parser.parse_args(argv)

    setup_logging(PROJECT_ROOT / "logs")
    manager = ImportTaskManager(temp_root=args.temp_root)
    try:
        result = manager.run(args.inputs, dpi=args.dpi)
    except BookCleanError as exc:
        print(f"输入/任务错误: {exc}")
        return 1

    manifest = result["manifest"]
    print("输入文件:")
    for f in manifest["inputs"]:
        status = f"错误: {f['error']}" if f["error"] else f"{f['page_count']} 页"
        print(f"  [{f['kind']}] {f['path']} — {status}")
    print(f"\n任务目录: {result['task_dir']}")
    print(f"Manifest: {result['manifest_path']}")
    print(f"渲染 DPI: {args.dpi}")
    print(f"总页数: {manifest['total_pages']}  完成: {manifest['completed_pages']}  失败: {manifest['failed_pages']}")
    print(f"任务状态: {manifest['status']}  耗时: {manifest['elapsed_seconds']}s")
    if manifest["contact_sheet"]:
        print(f"联系表: {result['task_dir'] / manifest['contact_sheet']}")
    for entry in manifest["pages"][:10]:
        size = f"{entry.get('image_width')}x{entry.get('image_height')}" if entry.get("image_width") else "-"
        print(f"  第 {entry['display_index']} 页 [{entry['status']}] {size} <- {Path(entry['source_path']).name}#{entry['source_page_index']}")
    if manifest["total_pages"] > 10:
        print(f"  ... 其余 {manifest['total_pages'] - 10} 页见 manifest.json")
    return 0 if manifest["status"] == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())