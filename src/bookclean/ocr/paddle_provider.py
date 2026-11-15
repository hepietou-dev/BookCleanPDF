"""PP-OCRv5 桥接 Provider：只读复用用户既有 DeepSeekOCR_v2 安装（M6.0 §4.2，M6.5）。

与 RapidOcrProvider 共用 WorkerClient/协议/批处理/取消逻辑；差异仅在：
  - worker 种类 paddle-bridge（paddle_bridge_worker.py）；
  - init 注入 source_dir/models_dir（来自插件清单 extra 字段）与 device=cpu。
引擎运行时与模型全部在用户既有安装内（0 下载）；BookClean 只分发 <1MB 桥接脚本。
"""

from __future__ import annotations

from pathlib import Path

from .plugin_manager import PluginError
from .rapid_provider import RapidOcrProvider

BRIDGE_WORKER_SCRIPT = Path(__file__).resolve().parent / "paddle_bridge_worker.py"


class PaddleBridgeProvider(RapidOcrProvider):
    engine_id = "ocr-paddle-bridge"
    expected_worker_kind = "paddle-bridge"
    worker_script = BRIDGE_WORKER_SCRIPT

    def _extra_init_params(self) -> dict:
        extra = self._plugin.extra
        source_dir = str(extra.get("source_dir") or "")
        if not source_dir:
            raise PluginError("桥接插件清单缺少 extra.source_dir（请运行 ocr_bridge_setup.py 重新配置）")
        params = {"source_dir": source_dir, "device": "cpu"}
        if extra.get("models_dir"):
            params["models_dir"] = str(extra["models_dir"])
        return params