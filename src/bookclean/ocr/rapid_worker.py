# -*- coding: utf-8 -*-
"""RapidOCR worker：运行于插件独立 runtime（非 BookClean 主环境）。

协议：JSON-lines over stdin/stdout（M6.0 §6.3）。
纪律：只读图像、只写 stdout/stderr，绝不落盘（落盘全在主进程）；
单页异常 → failed 响应后继续，绝不因一页崩溃整批。
本脚本不 import bookclean（插件 runtime 无 BookClean 依赖）。
"""

from __future__ import annotations

import json
import sys
import time


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _init(params: dict) -> dict:
    from importlib.metadata import version as pkg_version

    from rapidocr_onnxruntime import RapidOCR

    kwargs = {
        "intra_op_num_threads": max(1, int(params.get("threads", 4))),
        "inter_op_num_threads": 1,
    }
    for key in ("det_model_path", "rec_model_path", "cls_model_path"):
        if params.get(key):
            kwargs[key] = str(params[key])
    if params.get("text_score") is not None:
        kwargs["text_score"] = float(params["text_score"])

    t0 = time.perf_counter()
    engine = RapidOCR(**kwargs)
    elapsed = time.perf_counter() - t0
    version = pkg_version("rapidocr_onnxruntime")
    _log(f"RapidOCR {version} 初始化完成（{elapsed:.2f}s，threads={kwargs['intra_op_num_threads']}）")
    return engine, {
        "type": "ready",
        "engine": "ocr-rapid",
        "engine_version": version,
        "device": "cpu",
        "init_seconds": round(elapsed, 3),
    }


def _recognize(engine, page_no, image: str) -> dict:
    from PIL import Image

    t0 = time.perf_counter()
    with Image.open(image) as im:
        width, height = im.size
    result, _elapse = engine(image)
    lines = []
    if result:
        for i, item in enumerate(result):
            box, text, score = item[0], item[1], float(item[2])
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            lines.append({
                "line_id": i,
                "text": str(text),
                "score": round(score, 4),
                "bbox_px": [min(xs), min(ys), max(xs), max(ys)],
                "polygon_px": [[float(p[0]), float(p[1])] for p in box],
            })
    return {
        "type": "page",
        "page_no": page_no,
        "status": "completed",
        "lines": lines,
        "reading_order": [l["line_id"] for l in lines],
        "image_width_px": width,
        "image_height_px": height,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }


def main() -> int:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    engine = None
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            _emit({"type": "error", "code": "BAD_REQUEST", "message": "请求不是合法 JSON"})
            continue
        cmd = req.get("cmd")
        if cmd == "init":
            try:
                engine, ready = _init(req.get("params") or {})
                _emit(ready)
            except Exception as exc:  # noqa: BLE001 — 初始化失败必须回报而非静默退出
                engine = None
                _emit({"type": "error", "code": "INIT_FAILED",
                       "message": f"{type(exc).__name__}: {exc}"})
        elif cmd == "recognize":
            page_no = req.get("page_no")
            if engine is None:
                _emit({"type": "page", "page_no": page_no, "status": "failed",
                       "error": {"code": "NOT_INITIALIZED", "message": "引擎未初始化"}})
                continue
            try:
                _emit(_recognize(engine, page_no, req.get("image")))
            except Exception as exc:  # noqa: BLE001 — 单页失败隔离：回报后继续下一页
                _log(f"第 {page_no} 页识别失败: {type(exc).__name__}: {exc}")
                _emit({"type": "page", "page_no": page_no, "status": "failed",
                       "error": {"code": "ENGINE_ERROR",
                                 "message": f"{type(exc).__name__}: {exc}"}})
        elif cmd == "cancel":
            _emit({"type": "cancelled"})
        elif cmd == "shutdown":
            _emit({"type": "bye"})
            break
        else:
            _emit({"type": "error", "code": "UNKNOWN_CMD", "message": str(cmd)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())