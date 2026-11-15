# -*- coding: utf-8 -*-
"""PP-OCRv5 桥接 worker：只读复用用户既有 DeepSeekOCR_v2 安装（M6.0 §3/§4.2，M6.5）。

运行于用户既有项目的 Python 运行时（非 BookClean 主环境、非 BookClean 分发）。
协议：与 rapid_worker 相同的 JSON-lines over stdin/stdout。
纪律：
  - 只读对方项目（sys.path 只读 import src.ocr_engine），绝不修改其任何文件；
  - 只写 stdout/stderr，绝不落盘（对方运行时被 AV 禁写文件）；
  - 只调用 init_paddle / run_ocr，不 import/不调用任何 DeepSeek 云端校对路径，
    并主动移除 DEEPSEEK_API_KEY（物理隔离，离线红线）；
  - 模型缺失即 INIT_FAILED，绝不触发 PaddleOCR 联网下载；
  - init_paddle/predict 的 print 日志重定向到 stderr，保护 stdout JSON 协议；
  - 单页异常 → failed 响应后继续，绝不因一页崩溃整批。
本脚本不 import bookclean。
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


# ----------------------------- 行重建 ------------------------------------
# PP-OCRv5 server 英文词级过切分（实测 385 行/页）：同一视觉行的词各为独立框。
# W2 每行一个文本对象，不合并则跨词搜索失败。此处按几何把词框聚成视觉行：
#   y 重叠 ≥ overlap_ratio×min(高) 且 高度比 ≤ height_ratio 且 水平间隙 ≤ gap_ratio×min(高)。
# 中文行框本为整行（相邻行 y 不重叠）不会误并；表格宽间隙单元格不并。
# 竖排（多数框高>宽）保持引擎原顺序不合并（extract_ocr_lines 已做竖排排序）。


def _join_text(a: str, b: str) -> str:
    if a and b and a[-1].isascii() and b[0].isascii():
        return a + " " + b
    return a + b


def _extraction_order(lines: list[dict], overlap_ratio: float = 0.5) -> list[dict]:
    """把合并行重排为 PDFium 提取的不动点序（「提取文本 == OCR 文本」的前提）。

    PDFium 提取：同一视觉行的文本对象按 x 升序输出，行带之间保持内容序（不做全局
    y 排序，基线抖动的相邻行亦按内容序）。故此处按内容序聚行带——y 重叠 >
    overlap_ratio×min(高) 即并入并扩展并集——行带内按 x0 升序，行带间保持内容序。
    M6.5 取证：该序在 10 页真实书（目录/正文/中文表单）全文提取==OCR 验证通过。
    """
    groups: list[dict] = []
    for l in lines:
        x0, y0, x1, y1 = (float(v) for v in l["bbox_px"])
        if groups:
            g = groups[-1]
            overlap = min(y1, g["y1"]) - max(y0, g["y0"])
            hh = min(y1 - y0, g["y1"] - g["y0"])
            if overlap > overlap_ratio * hh:
                g["members"].append(l)
                g["y0"] = min(g["y0"], y0)
                g["y1"] = max(g["y1"], y1)
                continue
        groups.append({"y0": y0, "y1": y1, "members": [l]})
    out: list[dict] = []
    for g in groups:
        out.extend(sorted(g["members"], key=lambda m: m["bbox_px"][0]))
    return out


def merge_word_lines(lines: list[dict], gap_ratio: float = 0.6,
                     overlap_ratio: float = 0.5, height_ratio: float = 1.6) -> list[dict]:
    """输入 [{text, score, bbox_px}]；输出合并后的行（line_id 从 0 重编号）。

    两遍：先按 y 重叠聚成行带（容忍基线抖动，与 x 顺序无关），再在行带内按
    水平间隙切段（词间隙小→并；列间隙大→分）。
    行 bbox=成员并集；行 score=成员最小值（保守）；行文本=成员按 x 拼接
    （ASCII 边界补空格，其余直接拼接；不改写任何词的内容）。
    """
    boxes = []
    tall = 0
    for l in lines:
        text = str(l.get("text") or "")
        x0, y0, x1, y1 = (float(v) for v in l["bbox_px"])
        if not text or not (x1 > x0 and y1 > y0):
            continue
        if y1 - y0 > (x1 - x0) * 1.5:
            tall += 1
        boxes.append({"text": text, "score": float(l.get("score") or 0.0),
                      "x0": x0, "y0": y0, "x1": x1, "y1": y1})
    if not boxes:
        return []
    if tall * 2 > len(boxes):  # 竖排为主：保持引擎原顺序
        return [{"line_id": i, "text": b["text"], "score": b["score"],
                 "bbox_px": [b["x0"], b["y0"], b["x1"], b["y1"]]}
                for i, b in enumerate(boxes)]

    # 第一遍：y 重叠聚行带
    boxes.sort(key=lambda b: (b["y0"], b["x0"]))
    groups: list[dict] = []
    for b in boxes:
        h = b["y1"] - b["y0"]
        placed = False
        for g in groups:
            gh = g["y1"] - g["y0"]
            overlap = min(b["y1"], g["y1"]) - max(b["y0"], g["y0"])
            if overlap <= 0 or overlap / min(h, gh) < overlap_ratio:
                continue
            if max(h, gh) / min(h, gh) > height_ratio:
                continue
            g["y0"] = min(g["y0"], b["y0"])
            g["y1"] = max(g["y1"], b["y1"])
            g["members"].append(b)
            placed = True
            break
        if not placed:
            groups.append({"y0": b["y0"], "y1": b["y1"], "members": [b]})

    # 第二遍：行带内按 x 间隙切段并拼接
    for g in groups:
        members = sorted(g["members"], key=lambda b: b["x0"])
        segments: list[list[dict]] = [[members[0]]]
        for b in members[1:]:
            seg = segments[-1]
            right = max(m["x1"] for m in seg)
            hh = min(b["y1"] - b["y0"], seg[-1]["y1"] - seg[-1]["y0"])
            if b["x0"] - right > gap_ratio * hh:
                segments.append([b])
            else:
                seg.append(b)
        g["lines"] = []
        for seg in segments:
            text = seg[0]["text"]
            for m in seg[1:]:
                text = _join_text(text, m["text"])
            g["lines"].append({
                "text": text,
                "score": min(m["score"] for m in seg),
                "bbox_px": [min(m["x0"] for m in seg), min(m["y0"] for m in seg),
                            max(m["x1"] for m in seg), max(m["y1"] for m in seg)],
            })
    # 行带自上而下展开后，再重排为 PDFium 提取不动点序（行内 x 升序、行带间内容序），
    # 使「提取文本 == OCR 文本」成立；全局 y 中心排序会在混高行/基线抖动页打乱顺序。
    out = []
    for g in sorted(groups, key=lambda g: g["y0"]):
        out.extend(g["lines"])
    out = _extraction_order(out)
    for i, l in enumerate(out):
        l["line_id"] = i
    return out


# ----------------------------- 引擎初始化 ---------------------------------


def _init(params: dict):
    source_dir = Path(str(params.get("source_dir") or ""))
    if not source_dir.is_dir():
        raise RuntimeError(f"桥接源码目录不存在: {source_dir}（请运行 ocr_bridge_setup.py 重新配置）")
    models_dir = Path(str(params.get("models_dir") or (source_dir / "models")))
    for name in ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"):
        model_file = models_dir / name / "inference.pdiparams"
        if not model_file.is_file():
            raise RuntimeError(f"模型缺失（桥接模式不下载模型）: {model_file}")

    # 离线红线：禁模型源检查/联网下载；移除云端校对凭据；CPU 线程数
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["OMP_NUM_THREADS"] = str(max(1, int(params.get("threads", 4))))
    os.environ.pop("DEEPSEEK_API_KEY", None)
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("FLAGS_minloglevel", "2")

    sys.path.insert(0, str(source_dir))
    sys.dont_write_bytecode = True  # 只读复用：绝不在对方项目内生成 __pycache__
    t0 = time.perf_counter()
    # 对方 init_paddle 用 print 输出日志 ⇒ 重定向 stdout 保护 JSON 协议
    with contextlib.redirect_stdout(sys.stderr):
        from src import ocr_engine as oe  # 只读 import 用户既有项目
        ocr = oe.init_paddle("server", "cpu", models_dir_override=models_dir)
    elapsed = time.perf_counter() - t0

    from importlib.metadata import PackageNotFoundError, version as pkg_version
    try:
        version = pkg_version("paddleocr")
    except PackageNotFoundError:
        version = "unknown"  # 假引擎测试环境无 paddleocr 元数据；真实运行时必有
    _log(f"PP-OCRv5 server 桥接初始化完成（paddleocr {version}，{elapsed:.2f}s，"
         f"source={source_dir}，models={models_dir}）")
    return (oe, ocr), {
        "type": "ready",
        "engine": "ocr-paddle-bridge",
        "engine_version": version,
        "device": "cpu",
        "init_seconds": round(elapsed, 3),
    }


def _recognize(state, page_no, image: str, text_score: float) -> dict:
    import numpy as np
    from PIL import Image

    oe, ocr = state
    t0 = time.perf_counter()
    with Image.open(image) as im:
        width, height = im.size
        img = np.asarray(im.convert("RGB"))
    with contextlib.redirect_stdout(sys.stderr):
        engine_lines = oe.run_ocr(ocr, img)
    raw = [{"text": l.text, "score": float(l.score), "bbox_px": list(l.bbox_px)}
           for l in engine_lines]
    lines = merge_word_lines(raw)
    if text_score > 0:
        lines = [l for l in lines if l["score"] >= text_score]
    for l in lines:
        l["polygon_px"] = None  # OCRLine 不提供多边形，不虚构
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
    state = None
    text_score = 0.5
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
                params = req.get("params") or {}
                text_score = float(params.get("text_score", 0.5))
                state, ready = _init(params)
                _emit(ready)
            except Exception as exc:  # noqa: BLE001 — 初始化失败必须回报而非静默退出
                state = None
                _emit({"type": "error", "code": "INIT_FAILED",
                       "message": f"{type(exc).__name__}: {exc}"})
        elif cmd == "recognize":
            page_no = req.get("page_no")
            if state is None:
                _emit({"type": "page", "page_no": page_no, "status": "failed",
                       "error": {"code": "NOT_INITIALIZED", "message": "引擎未初始化"}})
                continue
            try:
                _emit(_recognize(state, page_no, req.get("image"), text_score))
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