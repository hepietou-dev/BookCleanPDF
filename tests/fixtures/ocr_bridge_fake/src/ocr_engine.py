# -*- coding: utf-8 -*-
"""假 ocr_engine：与真实 DeepSeekOCR_v2 引擎函数同名同签名，供桥接 worker 协议测试。

故意行为：
  - init_paddle 向 stdout print 垃圾（验证 worker 的 stdout 重定向保护）；
  - 断言 DEEPSEEK_API_KEY 已移除、模型源检查已禁用（离线红线）；
  - run_ocr 返回词级框（验证行重建）；纯红像素页抛错（页级隔离）、纯蓝像素页崩溃（重启）；
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_STATE = {}


@dataclass
class OCRLine:
    line_id: int
    text: str
    score: float
    bbox_px: tuple


def init_paddle(model_kind, device, models_dir_override=None):
    print("FAKE-STDOUT-POLLUTION 模型目录 print 日志")
    assert os.environ.get("DEEPSEEK_API_KEY") is None, "云端凭据未移除"
    assert os.environ.get("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK") == "True", "未禁用模型源检查"
    assert device == "cpu"
    _STATE["marker_dir"] = str(Path(models_dir_override).parent)
    return {"kind": model_kind, "device": device, "models_dir": str(models_dir_override)}


def run_ocr(ocr, img):
    import numpy as np
    assert isinstance(img, np.ndarray)
    # 像素标记控制行为（worker 只传 ndarray，不传文件名）：
    #   纯红 = 单页抛错；纯蓝 = 首次进程崩溃（marker 存在则正常，验证重启重试）
    px = img[0, 0]
    if tuple(int(v) for v in px) == (255, 0, 0):
        raise RuntimeError("fake 引擎单页失败")
    if tuple(int(v) for v in px) == (0, 0, 255):
        marker = Path(_STATE["marker_dir"]) / "crash.marker"
        if not marker.exists():
            marker.write_text("crashed", encoding="utf-8")
            os._exit(3)
    # 同一基线的三个英文词框 + 下一行中文框（行重建应把词并成一行）
    return [
        OCRLine(1, "hello", 0.95, (10.0, 10.0, 60.0, 30.0)),
        OCRLine(2, "world", 0.90, (66.0, 11.0, 120.0, 31.0)),
        OCRLine(3, "foo", 0.99, (126.0, 10.0, 160.0, 30.0)),
        OCRLine(4, "北京大学", 0.97, (10.0, 60.0, 130.0, 90.0)),
    ]