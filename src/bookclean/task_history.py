"""GUI 历史索引与配置持久化（M5.4）。

历史索引（user_paths.history_path）只存 {task_dir, output_path, settings, label,
created_at, last_state}，**不复制页面数据、不改 Manifest**（架构 §4.2/§8.3）。
页面级事实仍以任务目录 manifest 为准；索引仅作“最近任务”入口与恢复线索。
崩溃恢复：worker 在导入建立任务目录后即登记索引，故中途崩溃仍有索引可续跑。

纯标准库 + 内核原子写工具，不导入 Tkinter/ui。读写失败优雅降级（不抛到 UI）。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .utils.atomic_file import atomic_write_json

logger = logging.getLogger("bookclean")

MAX_ENTRIES = 50


def _read_json(path: Path, default):
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, type(default)) else default
    except (OSError, ValueError):
        return default


def load_history(path: str | Path) -> list[dict]:
    """读历史索引（最新在前）。缺失/损坏返回空列表，不抛异常。"""
    entries = _read_json(Path(path), [])
    return [e for e in entries if isinstance(e, dict) and e.get("task_dir")]


def save_history(path: str | Path, entries: list[dict]) -> None:
    atomic_write_json(Path(path), entries[:MAX_ENTRIES])


def record_task(path: str | Path, *, task_dir: str | Path, output_path: str = "",
                settings: dict | None = None, label: str = "",
                last_state: str = "running") -> dict:
    """登记/更新一条历史（按 task_dir 去重，保留首次 created_at）。返回该条目。"""
    task_dir = str(task_dir)
    entries = load_history(path)
    entry = next((e for e in entries if e.get("task_dir") == task_dir), None)
    if entry is None:
        entry = {"task_dir": task_dir, "created_at": datetime.now(timezone.utc).isoformat()}
    else:
        entries.remove(entry)  # 既有任务更新后置顶（最近活动在前）
    entries.insert(0, entry)
    entry.update({
        "output_path": str(output_path or entry.get("output_path", "")),
        "settings": dict(settings or entry.get("settings", {})),
        "label": label or entry.get("label", "") or Path(task_dir).name,
        "last_state": last_state,
    })
    save_history(path, entries)
    return entry


def update_task_state(path: str | Path, task_dir: str | Path, last_state: str,
                      output_path: str | None = None) -> None:
    """终止事件后更新索引状态（不新增条目；索引缺失时静默跳过）。"""
    task_dir = str(task_dir)
    entries = load_history(path)
    for e in entries:
        if e.get("task_dir") == task_dir:
            e["last_state"] = last_state
            if output_path is not None:
                e["output_path"] = str(output_path)
            save_history(path, entries)
            return


def find_entry(path: str | Path, task_dir: str | Path) -> dict | None:
    task_dir = str(task_dir)
    return next((e for e in load_history(path) if e.get("task_dir") == task_dir), None)


def load_config(path: str | Path) -> dict:
    """读 GUI 配置（设置快照）。缺失/损坏返回空 dict。"""
    return _read_json(Path(path), {})


def save_config(path: str | Path, cfg: dict) -> None:
    atomic_write_json(Path(path), dict(cfg))