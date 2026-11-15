"""M5.1 用户路径解析测试：开发环境 vs 打包(frozen)，不依赖开发绝对路径。"""

import sys
from pathlib import Path

import pytest

from bookclean import user_paths
from bookclean.user_paths import APP_DIR_NAME, UserPaths


@pytest.fixture
def dev_mode(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)


@pytest.fixture
def frozen_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localapp"))
    return tmp_path


def test_dev_paths(dev_mode):
    p = UserPaths()
    root = user_paths.project_root()
    assert p.frozen is False
    assert p.tasks_root == root / "temp"
    assert p.logs_dir == root / "logs"
    assert p.config_path.name == "gui_config.json"
    assert p.history_path.name == "gui_tasks_index.json"


def test_frozen_paths(frozen_mode):
    p = UserPaths()
    base = frozen_mode / "localapp" / APP_DIR_NAME
    assert p.frozen is True
    assert p.data_root == base
    assert p.tasks_root == base / "tasks"
    assert p.logs_dir == base / "logs"
    assert p.config_path.name == "config.json"
    assert p.history_path.name == "tasks_index.json"


def test_frozen_project_root_uses_meipass(frozen_mode):
    assert user_paths.project_root() == frozen_mode / "bundle"


def test_default_output_dir_under_home(dev_mode):
    p = UserPaths()
    out = p.default_output_dir
    assert out.name == APP_DIR_NAME
    assert out.parent == Path.home() or out.parent == Path.home() / "Documents"


def test_explicit_root_override(dev_mode, tmp_path):
    p = UserPaths(root=tmp_path / "custom")
    assert p.data_root == tmp_path / "custom"
    assert p.tasks_root == tmp_path / "custom"  # dev: root 直接容纳 task_*


def test_ensure_dirs_creates(dev_mode, tmp_path):
    p = UserPaths(root=tmp_path / "data")
    p.ensure_dirs()
    assert p.tasks_root.is_dir()
    assert p.logs_dir.is_dir()


def test_resource_path_joins_assets(dev_mode):
    p = UserPaths()
    assert p.resource_path("icons", "app.ico") == user_paths.project_root() / "assets" / "icons" / "app.ico"


def test_frozen_without_localappdata_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    p = UserPaths()
    assert p.data_root == Path.home() / "AppData" / "Local" / APP_DIR_NAME