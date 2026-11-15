"""OCR 插件发现与可用性检查（M6.0 §5/§6）。

插件目录布局（独立安装/更新/卸载 = 整目录增删）：
    <plugins_root>/<engine_id>/<version>/plugin.json
    <plugins_root>/<engine_id>/<version>/runtime/...   （独立 Python 运行时+模型）

plugin.json 字段（schema=1）：
    engine_id / engine_version / protocol_version / params_schema_version
    python   相对插件目录的解释器路径（如 runtime/Scripts/python.exe）
    worker   worker 种类（"rapid" → BookClean 自带 rapid_worker.py）
    models   [{id, path(相对插件目录), sha256}]

约束：discover/check_available 只读、无副作用、不加载模型、不计算哈希
（哈希校验在 initialize 时经 verify_models 执行）。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..user_paths import UserPaths
from .provider import Availability

logger = logging.getLogger("bookclean")

PLUGIN_SCHEMA = 1
PROTOCOL_VERSION = 1
# auto 选择优先级（M6.0 §4 推荐序）
ENGINE_PRIORITY = ("ocr-rapid", "ocr-paddle-bridge")


class PluginError(Exception):
    """插件缺失/损坏/不兼容。"""


@dataclass(frozen=True)
class PluginInfo:
    engine_id: str
    engine_version: str
    protocol_version: int
    params_schema_version: int
    plugin_dir: Path
    python_exe: Path
    worker_kind: str
    models: tuple[dict, ...]  # {id, path(绝对), sha256}
    extra: dict = field(default_factory=dict)  # 引擎专属配置（如桥接的 source_dir/models_dir）

    def model_fingerprint(self) -> tuple[str, str]:
        """(model_id, model_sha256)：多模型拼接 id、哈希逐文件列表的 sha256。"""
        model_id = "+".join(m["id"] for m in self.models)
        joined = "|".join(m["sha256"] for m in self.models)
        digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
        return model_id, digest


def default_plugins_root() -> Path:
    return UserPaths().data_root / "plugins"


class _Reverse:
    """包装比较使排序结果为"新优先"（sort 升序 + 反转比较 = 降序）。"""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        return other.value < self.value

    def __eq__(self, other):
        return self.value == other.value


def _version_key(v: str):
    try:
        from packaging.version import Version
        return (0, _Reverse(Version(v)))
    except Exception:
        return (1, _Reverse(v))


class PluginManager:
    def __init__(self, plugins_root: str | Path | None = None):
        self.plugins_root = Path(plugins_root) if plugins_root is not None else default_plugins_root()

    def discover(self) -> list[PluginInfo]:
        """扫描并校验插件清单。损坏/不兼容的条目跳过并记录，绝不抛出中断扫描。"""
        found: list[PluginInfo] = []
        if not self.plugins_root.is_dir():
            return found
        candidates = sorted(self.plugins_root.glob("*/plugin.json")) + \
            sorted(self.plugins_root.glob("*/*/plugin.json"))
        for mf in candidates:
            try:
                info = self._load_manifest(mf)
            except (PluginError, OSError, ValueError) as exc:
                logger.warning("跳过无效 OCR 插件清单 %s: %s", mf, exc)
                continue
            found.append(info)
        found.sort(key=lambda p: (ENGINE_PRIORITY.index(p.engine_id)
                                  if p.engine_id in ENGINE_PRIORITY else len(ENGINE_PRIORITY),
                                  _version_key(p.engine_version)))
        return found

    def _load_manifest(self, mf: Path) -> PluginInfo:
        data = json.loads(mf.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise PluginError("清单不是 JSON 对象")
        if int(data.get("schema", 0)) != PLUGIN_SCHEMA:
            raise PluginError(f"不支持的插件 schema: {data.get('schema')!r}")
        if int(data.get("protocol_version", 0)) != PROTOCOL_VERSION:
            raise PluginError(f"协议版本不兼容: {data.get('protocol_version')!r}")
        for key in ("engine_id", "engine_version", "python", "worker"):
            if not data.get(key):
                raise PluginError(f"缺少字段 {key}")
        plugin_dir = mf.parent
        python_exe = (plugin_dir / data["python"]).resolve()
        if not python_exe.is_file():
            raise PluginError(f"插件解释器不存在: {python_exe}")
        models = []
        for m in data.get("models", []):
            mp = (plugin_dir / m["path"]).resolve()
            if not mp.is_file():
                raise PluginError(f"模型文件不存在: {mp}")
            if not m.get("id") or not m.get("sha256"):
                raise PluginError("模型条目缺少 id/sha256")
            models.append({"id": str(m["id"]), "path": str(mp), "sha256": str(m["sha256"])})
        if not models:
            raise PluginError("插件未声明任何模型")
        return PluginInfo(
            engine_id=str(data["engine_id"]),
            engine_version=str(data["engine_version"]),
            protocol_version=int(data["protocol_version"]),
            params_schema_version=int(data.get("params_schema_version", 1)),
            plugin_dir=plugin_dir,
            python_exe=python_exe,
            worker_kind=str(data["worker"]),
            models=tuple(models),
            extra=dict(data.get("extra") or {}),
        )

    def select(self, engine_id: str = "auto") -> PluginInfo | None:
        plugins = self.discover()
        if engine_id != "auto":
            plugins = [p for p in plugins if p.engine_id == engine_id]
        return plugins[0] if plugins else None

    def check_available(self, engine_id: str = "auto") -> Availability:
        try:
            plugin = self.select(engine_id)
        except OSError as exc:
            return Availability(installed=False, reason=f"插件目录不可读: {exc}")
        if plugin is None:
            return Availability(
                installed=False,
                reason="未安装 OCR 插件。请安装 ocr-rapid 插件后重试（OCR 为可选功能，不影响普通导出）。",
            )
        return Availability(
            installed=True,
            engine_id=plugin.engine_id,
            engine_version=plugin.engine_version,
            models_ok=True,  # discover 已校验文件存在；内容哈希在 initialize 校验
            reason="",
            plugin_dir=str(plugin.plugin_dir),
        )

    @staticmethod
    def verify_models(plugin: PluginInfo) -> list[str]:
        """逐文件 sha256 校验，返回不匹配/缺失描述列表（空=全部通过）。"""
        problems: list[str] = []
        for m in plugin.models:
            p = Path(m["path"])
            if not p.is_file():
                problems.append(f"模型缺失: {m['id']} ({p})")
                continue
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != m["sha256"]:
                problems.append(f"模型哈希不匹配: {m['id']}（文件可能损坏或被替换）")
        return problems