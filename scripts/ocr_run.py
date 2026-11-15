"""BookClean PDF M6.1 命令行 OCR：图像 -> 逐页 OCR JSON（不写 PDF、不改任务目录）。

OCR 为可选功能：未安装插件时明确提示并以退出码 2 结束，绝不触碰 M1–M5 数据。
逐页原子落盘（.tmp -> os.replace），单页失败不中断整批；Ctrl+C 页边界取消。

用法:
  python scripts/ocr_run.py --images temp/m6_review/pages --out temp/m6_1_run
                            [--engine auto] [--plugins-root temp/plugins]
                            [--fake] [--threads 4] [--timeout 120]
                            [--verify-models]
退出码: 0 全部完成 / 1 存在失败页或初始化失败 / 2 无插件或参数无效 / 130 用户取消。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bookclean.ocr.plugin_manager import PluginManager
from bookclean.ocr.provider import OcrParams, STATUS_COMPLETED

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _collect_images(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(q for q in p.iterdir()
                                if q.is_file() and q.suffix.lower() in IMAGE_EXTS))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f"输入路径不存在: {p}")
    if not files:
        raise FileNotFoundError("未找到任何图像文件")
    return files


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="BookClean PDF M6.1 命令行 OCR（图像 -> 逐页 JSON）")
    parser.add_argument("--images", nargs="+", required=True, help="图像目录或文件（可多个）")
    parser.add_argument("--out", required=True, help="输出目录（逐页 JSON + summary.json）")
    parser.add_argument("--engine", default="auto", help="引擎 id 或 auto（默认）")
    parser.add_argument("--plugins-root", default=None, help="插件根目录（默认用户数据目录/plugins）")
    parser.add_argument("--fake", action="store_true", help="使用确定性假引擎（测试/演示，不启动子进程）")
    parser.add_argument("--threads", type=int, default=4, help="ONNX 线程数（默认 4）")
    parser.add_argument("--timeout", type=float, default=120.0, help="单页超时秒数（默认 120）")
    parser.add_argument("--verify-models", action="store_true",
                        help="仅校验插件模型哈希后退出（0 通过 / 1 不匹配 / 2 无插件）")
    args = parser.parse_args(argv)

    try:
        images = _collect_images(args.images)
    except (FileNotFoundError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if args.fake:
        from bookclean.ocr.fake import FakeProvider
        provider = FakeProvider()
        engine_label = "fake 0.0.0"
    else:
        from bookclean.ocr.paddle_provider import PaddleBridgeProvider
        from bookclean.ocr.rapid_provider import RapidOcrProvider
        pm = PluginManager(args.plugins_root)
        if args.verify_models:
            plugin = pm.select(args.engine)
            if plugin is None:
                print("错误: " + pm.check_available(args.engine).reason, file=sys.stderr)
                return 2
            problems = PluginManager.verify_models(plugin)
            for m in plugin.models:
                print(f"模型 {m['id']}: {'不通过' if any(m['id'] in p for p in problems) else '通过'}")
            if problems:
                for p in problems:
                    print("错误: " + p, file=sys.stderr)
                return 1
            print(f"模型校验全部通过（{plugin.engine_id} {plugin.engine_version}）")
            return 0
        plugin = pm.select(args.engine)
        if plugin is None:
            print("错误: " + pm.check_available(args.engine).reason, file=sys.stderr)
            return 2
        provider_cls = (PaddleBridgeProvider if plugin.worker_kind == "paddle-bridge"
                        else RapidOcrProvider)
        provider = provider_cls(plugin, OcrParams(threads=args.threads,
                                                  page_timeout_s=args.timeout))
        engine_label = f"{plugin.engine_id} {plugin.engine_version}"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = OcrParams(threads=args.threads, page_timeout_s=args.timeout)
    init = provider.initialize(params)
    if not init.ok:
        print(f"错误: OCR 初始化失败 [{init.error.code}] {init.error.message}", file=sys.stderr)
        return 1
    version = provider.get_version()
    print(f"引擎就绪: {engine_label}（{init.device}，初始化 {init.init_seconds:.2f}s）")

    pages = list(enumerate(images, start=1))
    results: list[dict] = []
    cancelled = False
    t0 = time.perf_counter()

    def on_result(res):
        payload = res.to_dict()
        payload["image_name"] = images[res.page_no - 1].name
        _atomic_write_json(out_dir / f"page_{res.page_no:04d}.json", payload)
        results.append(payload)
        if res.status == STATUS_COMPLETED:
            print(f"  页 {res.page_no}/{len(pages)} {images[res.page_no - 1].name}: "
                  f"{len(res.lines)} 行，{res.elapsed_s:.2f}s")
        else:
            print(f"  页 {res.page_no}/{len(pages)} {images[res.page_no - 1].name}: "
                  f"失败 [{res.error.code}] {res.error.message}")

    try:
        summary = provider.recognize_batch(pages, on_result)
    except KeyboardInterrupt:
        cancelled = True
        provider.cancel()
        summary = None
    finally:
        provider.shutdown()

    completed = sum(1 for r in results if r["status"] == STATUS_COMPLETED)
    failed = len(results) - completed
    elapsed = round(time.perf_counter() - t0, 3)
    summary_payload = {
        "engine": version.to_dict(),
        "params": params.to_dict(),
        "issued": len(results) if summary is None else summary.issued,
        "completed": completed,
        "failed": failed,
        "total_pages": len(pages),
        "cancelled": cancelled,
        "elapsed_s": elapsed,
        "pages": [{"page_no": r["page_no"], "image_name": r["image_name"],
                   "status": r["status"], "lines": len(r["lines"]),
                   "elapsed_s": r["elapsed_s"],
                   "error": r["error"]} for r in results],
    }
    _atomic_write_json(out_dir / "summary.json", summary_payload)

    if cancelled:
        print(f"已取消：完成 {completed}/{len(pages)} 页，结果保留在 {out_dir}")
        return 130
    print(f"完成: {completed} 成功 / {failed} 失败 / 共 {len(pages)} 页，耗时 {elapsed:.1f}s")
    print(f"结果目录: {out_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())