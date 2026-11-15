# -*- coding: utf-8 -*-
"""测试用假 OCR worker：与 rapid_worker 相同的 JSON-lines 协议，仅标准库。

init params 控制行为：
    mode        ok(默认) / init_error / init_crash / bad_json / always_crash / always_hang
    fail_pages  这些页返回 status=failed（进程不退出，验证页级隔离）
    crash_page  该页首次请求时进程崩溃；once_file 存在则正常（验证重启后重试成功）
    hang_page   该页首次请求时挂起 60s；once_file 存在则正常（验证超时重启）
    once_file   崩溃/挂起"只发生一次"的状态文件路径（测试夹具允许落盘）
    slow_s      每页识别前休眠秒数
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _page_ok(page_no: int) -> dict:
    return {
        "type": "page", "page_no": page_no, "status": "completed",
        "lines": [
            {"line_id": 0, "text": f"fake worker page {page_no}", "score": 0.99,
             "bbox_px": [10.0, 10.0, 200.0, 40.0],
             "polygon_px": [[10.0, 10.0], [200.0, 10.0], [200.0, 40.0], [10.0, 40.0]]},
        ],
        "reading_order": [0],
        "image_width_px": 400, "image_height_px": 600,
        "elapsed_s": 0.01,
    }


def main() -> int:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    params: dict = {}
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
            params = req.get("params") or {}
            mode = params.get("mode", "ok")
            if mode == "init_crash":
                os._exit(3)
            if mode == "init_error":
                _emit({"type": "error", "code": "INIT_FAILED", "message": "fake 初始化失败"})
                continue
            _emit({"type": "ready", "engine": "fake-worker",
                   "engine_version": "0.0.0-test", "device": "cpu", "init_seconds": 0.0})
        elif cmd == "recognize":
            page_no = int(req.get("page_no", 0))
            mode = params.get("mode", "ok")
            if mode == "bad_json":
                sys.stdout.write("this is not json\n{broken\n")
                sys.stdout.flush()
            if mode == "always_crash" or page_no == params.get("crash_page"):
                once = params.get("once_file")
                if mode == "always_crash" or not (once and Path(once).exists()):
                    if once:
                        Path(once).write_text("crashed", encoding="utf-8")
                    os._exit(1)
            if mode == "always_hang" or page_no == params.get("hang_page"):
                once = params.get("once_file")
                if mode == "always_hang" or not (once and Path(once).exists()):
                    if once:
                        Path(once).write_text("hung", encoding="utf-8")
                    time.sleep(60)
            if params.get("slow_s"):
                time.sleep(float(params["slow_s"]))
            if page_no in (params.get("fail_pages") or []):
                _emit({"type": "page", "page_no": page_no, "status": "failed",
                       "error": {"code": "ENGINE_ERROR", "message": f"fake 失败页 {page_no}"}})
                continue
            _emit(_page_ok(page_no))
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