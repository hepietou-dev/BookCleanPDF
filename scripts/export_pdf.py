"""BookClean PDF M4.5 命令行端到端导出：已有 M1/M2/M3 任务目录 -> 最终 PDF。

不重新执行 M1–M3 图像修复；只读任务目录与原始输入。默认 balanced/preserve/contain/
验证开/不覆盖/不线性化。原子输出：验证全部通过后才替换为正式 PDF 并显示成功。

用法:
  python scripts/export_pdf.py --task-dir temp/task_xxx --output output/book.pdf
                               [--compression balanced] [--dpi preserve]
                               [--page-size preserve] [--fit contain]
                               [--title 书名] [--author 作者] [--overwrite]
退出码: 0 成功 / 1 导出或验证失败 / 2 参数或任务目录无效 / 130 用户取消。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bookclean.config import (EXPORT_COLOR_MODES, EXPORT_COMPRESSION_MODES,
                              EXPORT_FIT_MODES, EXPORT_PAGE_SIZE_MODES, OCR_ENGINES,
                              OcrConfig, PDFExportConfig)
from bookclean.pdf_exporter import ExportError, PDFExporter


def _build_config(args) -> PDFExportConfig:
    return PDFExportConfig(
        compression=args.compression,
        color_mode=args.color_mode,
        target_dpi=None if args.dpi == "preserve" else int(args.dpi),
        page_size_mode=args.page_size,
        fit_mode=args.fit,
        auto_long_edge_mm=args.auto_long_edge_mm,
        custom_width_mm=args.width_mm,
        custom_height_mm=args.height_mm,
        title=args.title,
        author=args.author,
        subject=args.subject,
        keywords=args.keywords,
        overwrite=args.overwrite,
        linearize=args.linearize,
        verify=args.verify,
    )


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="BookClean PDF M4 端到端导出（任务目录 -> PDF）")
    parser.add_argument("--task-dir", required=True, help="M1/M2/M3 任务目录（含 manifest.json）")
    parser.add_argument("--output", required=True, help="输出 PDF 路径")
    parser.add_argument("--compression", default="balanced", choices=EXPORT_COMPRESSION_MODES)
    parser.add_argument("--color-mode", default="preserve", choices=EXPORT_COLOR_MODES,
                        help="preserve=保留原始彩色和灰度内容（默认，与 M4 行为一致）；"
                             "grayscale=整本 PDF 转换为 8 位灰度，所有彩色内容均会转换为灰度。"
                             "灰度输出无法恢复原始彩色信息，但不会改变原始任务文件。")
    parser.add_argument("--dpi", default="preserve", choices=["preserve", "200", "300", "400"],
                        help="preserve=保留源分辨率；其余仅下采样，不上采样")
    parser.add_argument("--page-size", default="preserve", choices=EXPORT_PAGE_SIZE_MODES)
    parser.add_argument("--width-mm", type=float, default=None, help="custom 模式页宽(mm)")
    parser.add_argument("--height-mm", type=float, default=None, help="custom 模式页高(mm)")
    parser.add_argument("--auto-long-edge-mm", type=float, default=250.0,
                        help="auto/照片估算长边参考尺寸(mm)，默认 250")
    parser.add_argument("--fit", default="contain", choices=EXPORT_FIT_MODES)
    parser.add_argument("--title", default=None)
    parser.add_argument("--author", default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--keywords", default=None)
    parser.add_argument("--overwrite", action="store_true", help="显式允许覆盖已有输出")
    parser.add_argument("--linearize", action=argparse.BooleanOptionalAction, default=False,
                        help="线性化（默认关）；--no-linearize 显式关闭")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True,
                        help="导出后重读验证（默认开）；--no-verify 关闭")
    parser.add_argument("--ocr", action="store_true",
                        help="生成可搜索 PDF（M6.3 可选本地 OCR 文字层）；需已安装 OCR 插件，"
                             "缺插件时自动跳过文字层、普通导出不受影响")
    parser.add_argument("--ocr-engine", default="auto", choices=OCR_ENGINES)
    parser.add_argument("--ocr-threads", type=int, default=4)
    parser.add_argument("--ocr-lang", default="auto")
    parser.add_argument("--ocr-text-score", type=float, default=0.5)
    parser.add_argument("--ocr-page-timeout", type=float, default=120.0)
    parser.add_argument("--ocr-restart-every", type=int, default=200,
                        help="长文档每 N 页重启 OCR worker 释放内存（0=不重启）")
    parser.add_argument("--plugins-root", default="", help="OCR 插件根目录（空=默认用户插件目录）")
    args = parser.parse_args(argv)

    task_dir = Path(args.task_dir)
    if not (task_dir / "manifest.json").exists():
        print(f"错误: {task_dir} 不是有效任务目录（缺少 manifest.json）")
        return 2
    try:
        config = _build_config(args)
        config.validate()
    except ValueError as exc:
        print(f"参数错误: {exc}")
        return 2

    manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
    pages = manifest.get("pages", [])
    total = len(pages)
    src_pdf = sum(1 for p in pages if p.get("source_kind") == "pdf")
    print(f"任务: {manifest.get('task_id')}  输入页数: {total}（PDF {src_pdf} / 图片 {total - src_pdf}）")
    print(f"配置: compression={config.compression} color_mode={config.color_mode} dpi={args.dpi} "
          f"page_size={config.page_size_mode} fit={config.fit_mode} verify={config.verify} "
          f"overwrite={config.overwrite} linearize={config.linearize}")
    if config.color_mode == "grayscale":
        print("提示: 灰度输出无法恢复原始彩色信息，但不会改变原始任务文件。")

    def page_cb(rec):
        q = f" q{rec['jpeg_quality']}" if rec.get("jpeg_quality") else ""
        print(f"  [{rec['display_index']}/{total}] 解析→编码→合成 "
              f"{rec['output_format']}{q} {rec['output_page_width_pt']:.0f}x{rec['output_page_height_pt']:.0f}pt")

    def verify_cb(i, n):
        step = max(1, n // 20)
        if i % step == 0 or i == n:
            print(f"  验证 [{i}/{n}]")

    started = time.monotonic()
    text_source = None
    if args.ocr:
        from bookclean.ocr_stage import OcrStageRunner  # 惰性导入：--ocr 才触碰 OCR 代码
        ocr_config = OcrConfig(enabled=True, engine=args.ocr_engine, threads=args.ocr_threads,
                               lang=args.ocr_lang, text_score=args.ocr_text_score,
                               page_timeout_s=args.ocr_page_timeout,
                               restart_every_n_pages=args.ocr_restart_every,
                               plugins_root=args.plugins_root)
        try:
            ocr_config.validate()
        except ValueError as exc:
            print(f"OCR 参数错误: {exc}")
            return 2

        def ocr_cb(entry):
            print(f"  OCR [{entry['display_index']}/{total}] {entry.get('ocr_status', '')} "
                  f"({entry.get('ocr_text_chars', 0)} 字, {entry.get('ocr_source', '')})")

        print("OCR 文字识别（可选本地引擎，失败页不阻塞导出）...")
        ocr_result = OcrStageRunner().run(task_dir, ocr_config, progress_cb=ocr_cb)
        os_ = ocr_result["summary"]
        text_source = ocr_result["text_source"]
        print(f"OCR 摘要: completed {os_['completed']} / failed {os_['failed']} / "
              f"skipped {os_['skipped']} / reused {os_['reused']}，字符 {os_['total_text_chars']}，"
              f"覆盖率 {os_['coverage']}，引擎 {os_['engine_id'] or '无'}")
        if not ocr_result["available"]:
            print(f"提示: {os_.get('reason', 'OCR 不可用')}（普通导出照常）")
    try:
        info = PDFExporter(config).export(task_dir, args.output,
                                          progress_cb=page_cb, verify_progress_cb=verify_cb,
                                          text_source=text_source)
    except ExportError as exc:
        print(f"导出失败: {exc}")
        return 1
    except KeyboardInterrupt:
        print("已取消导出（临时文件已清理，既有输出与原始输入未受影响）")
        return 130
    except OSError as exc:
        print(f"导出失败(IO): {exc}")
        return 1
    elapsed = time.monotonic() - started

    s = info["summary"]
    v = info.get("validation") or {}
    print("\n=== 导出成功（已通过重读验证） ===" if config.verify else "\n=== 导出完成（验证已关闭） ===")
    print(f"输出: {info['output_file']}")
    print(f"色彩模式: {config.color_mode}")
    print(f"大小: {info['output_size_bytes']} 字节（{info['output_size_bytes'] / 1e6:.2f} MB）")
    print(f"页数: {s['total_pages']}  灰度 {s['grayscale_pages']} / 彩色 {s['color_pages']}  "
          f"JPEG {s['jpeg_pages']} / PNG {s['png_pages']}")
    print(f"回退: 原始 {s['fallback_original_pages']} / 几何 {s['fallback_geometry_pages']}  "
          f"下采样 {s['downsampled_pages']}")
    print(f"验证: {info['validation_status']}" + (f"（问题 {len(v.get('issues', []))}）" if v else ""))
    if info.get("ocr"):
        oc = info["ocr"]
        print(f"OCR 文字层: {oc['pages_with_text']}/{s['total_pages']} 页有文字，"
              f"共 {oc['total_text_chars']} 字，覆盖率 {oc['coverage']}（W2 不嵌字体，像素不变）")
    print(f"耗时: {elapsed:.2f}s（合成 {s['total_export_time_seconds']}s）")
    print(f"报告: {Path(info['output_file']).parent / 'export_report.json'} / export_report.txt")
    if v.get("contact_sheet"):
        print(f"联系表: {v['contact_sheet']}")
    if info["warnings"]:
        print(f"警告 {len(info['warnings'])} 条（详见报告）")
    return 0


if __name__ == "__main__":
    sys.exit(main())