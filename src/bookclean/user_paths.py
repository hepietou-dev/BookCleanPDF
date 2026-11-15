"""用户数据与资源路径解析：开发环境 vs PyInstaller 打包（frozen）。

打包后任务/日志/配置写入 %LOCALAPPDATA%/BookCleanPDF（用户可写、卸载不删），
输出 PDF 默认写入用户 Documents/BookCleanPDF；绝不依赖开发环境绝对路径。
开发环境沿用 CLI 约定：任务根 = PROJECT_ROOT/temp，日志 = PROJECT_ROOT/logs。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR_NAME = "BookCleanPDF"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def project_root() -> Path:
    """开发环境为仓库根；打包后为 PyInstaller 资源根（_MEIPASS）或可执行文件目录。"""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent.parent.parent


def _documents_dir() -> Path:
    home = Path.home()
    docs = home / "Documents"
    return docs if docs.is_dir() else home


def user_data_root() -> Path:
    if is_frozen():
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_DIR_NAME
    return project_root() / "temp"


class UserPaths:
    """集中解析 GUI 所需路径。可传入 root 覆盖（便于测试与自定义数据位置）。"""

    def __init__(self, root: str | Path | None = None):
        self.frozen = is_frozen()
        self._root = Path(root) if root is not None else user_data_root()

    @property
    def data_root(self) -> Path:
        return self._root

    @property
    def tasks_root(self) -> Path:
        # 开发：temp/ 直接容纳 task_*（与 CLI 一致）；打包：独立 tasks/ 子目录
        return self._root if not self.frozen else self._root / "tasks"

    @property
    def logs_dir(self) -> Path:
        return project_root() / "logs" if not self.frozen else self._root / "logs"

    @property
    def config_path(self) -> Path:
        name = "gui_config.json" if not self.frozen else "config.json"
        return self._root / name

    @property
    def history_path(self) -> Path:
        name = "gui_tasks_index.json" if not self.frozen else "tasks_index.json"
        return self._root / name

    @property
    def default_output_dir(self) -> Path:
        return _documents_dir() / APP_DIR_NAME

    def resource_path(self, *parts: str) -> Path:
        return project_root().joinpath("assets", *parts)

    def ensure_dirs(self) -> None:
        for d in (self.tasks_root, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)