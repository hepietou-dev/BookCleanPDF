# -*- coding: utf-8 -*-
"""M6.5 验收测试：PP-OCRv5 桥接（行重建/清单 extra/Provider 守卫/worker 协议）+ 插件分发。

全部离线：worker 协议测试用假 src.ocr_engine（tests/fixtures/ocr_bridge_fake，主 venv 可运行，
无 paddle）；分发测试用 tmp_path 合成插件；真实桥接端到端在无既有安装环境自动跳过。
不修改 D:\\LLM-OCR 任何文件（真实用例只读）。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from bookclean.ocr.paddle_bridge_worker import merge_word_lines
from bookclean.ocr.paddle_provider import PaddleBridgeProvider
from bookclean.ocr.plugin_manager import PluginError, PluginInfo, PluginManager
from bookclean.ocr.provider import OcrParams
from bookclean.ocr.rapid_provider import RapidOcrProvider
from bookclean.ocr.worker_client import WorkerClient

ROOT = Path(__file__).resolve().parent.parent
BRIDGE_WORKER = ROOT / "src" / "bookclean" / "ocr" / "paddle_bridge_worker.py"
FAKE_SRC = ROOT / "tests" / "fixtures" / "ocr_bridge_fake" / "src"
REAL_PLUGINS_ROOT = ROOT / "temp" / "plugins"
REAL_BRIDGE_SOURCE = Path("D:/LLM-OCR/DeepSeekOCR_v2")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _line(text, score, x0, y0, x1, y1):
    return {"text": text, "score": score, "bbox_px": [x0, y0, x1, y1]}


# ---------------------------------------------------------------- 行重建


class TestMergeWordLines:
    def test_english_words_merged_with_spaces(self):
        lines = [_line("none", 0.9, 0, 10, 50, 30),
                 _line("of", 0.8, 56, 11, 80, 31),
                 _line("their", 0.95, 86, 10, 140, 30)]
        out = merge_word_lines(lines)
        assert len(out) == 1
        assert out[0]["text"] == "none of their"
        assert out[0]["bbox_px"] == [0, 10, 140, 31]
        assert out[0]["score"] == pytest.approx(0.8)  # 成员最小值（保守）
        assert out[0]["line_id"] == 0

    def test_separate_lines_not_merged(self):
        lines = [_line("first line", 0.9, 0, 10, 140, 30),
                 _line("second line", 0.9, 0, 60, 150, 80)]
        out = merge_word_lines(lines)
        assert len(out) == 2
        assert [l["text"] for l in out] == ["first line", "second line"]

    def test_two_columns_not_merged(self):
        # 同一 y 带但水平间隙远超 0.6×高 ⇒ 两列不并
        lines = [_line("left", 0.9, 0, 10, 60, 30),
                 _line("right", 0.9, 400, 10, 460, 30)]
        out = merge_word_lines(lines)
        assert len(out) == 2

    def test_cjk_joined_without_space(self):
        lines = [_line("北京", 0.9, 0, 10, 60, 40),
                 _line("大学", 0.9, 64, 10, 124, 40)]
        out = merge_word_lines(lines)
        assert len(out) == 1
        assert out[0]["text"] == "北京大学"

    def test_punctuation_boundary_gets_space(self):
        lines = [_line("shy,", 0.9, 0, 10, 50, 30),
                 _line("though", 0.9, 56, 10, 120, 30)]
        out = merge_word_lines(lines)
        assert out[0]["text"] == "shy, though"

    def test_mixed_cjk_latin_boundary_no_space(self):
        lines = [_line("北京大学", 0.9, 0, 10, 120, 40),
                 _line("2026", 0.9, 124, 10, 180, 40)]
        out = merge_word_lines(lines)
        assert out[0]["text"] == "北京大学2026"

    def test_vertical_layout_preserved(self):
        # 高>宽×1.5 的竖排框为主 ⇒ 保持引擎原顺序，不合并
        lines = [_line("竖", 0.9, 100, 0, 130, 60),
                 _line("排", 0.9, 100, 66, 130, 126),
                 _line("文", 0.9, 100, 132, 130, 192)]
        out = merge_word_lines(lines)
        assert len(out) == 3
        assert [l["text"] for l in out] == ["竖", "排", "文"]

    def test_empty_and_invalid_filtered(self):
        assert merge_word_lines([]) == []
        lines = [_line("", 0.9, 0, 0, 10, 10),          # 空文本
                 _line("bad", 0.9, 10, 10, 10, 20),      # 零宽
                 _line("ok", 0.9, 0, 0, 40, 20)]
        out = merge_word_lines(lines)
        assert len(out) == 1 and out[0]["text"] == "ok"

    def test_reading_order_top_to_bottom(self):
        lines = [_line("bottom", 0.9, 0, 100, 100, 120),
                 _line("top", 0.9, 0, 0, 100, 20)]
        out = merge_word_lines(lines)
        assert [l["text"] for l in out] == ["top", "bottom"]
        assert [l["line_id"] for l in out] == [0, 1]


# ---------------------------------------------------------------- 插件清单 extra


def _bridge_manifest(tmp_path, *, extra=True, abs_paths=True):
    pd = tmp_path / "ocr-paddle-bridge" / "3.2.0"
    pd.mkdir(parents=True)
    src = tmp_path / "bridge_src"
    (src / "src").mkdir(parents=True)
    models = tmp_path / "bridge_models"
    for name in ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"):
        d = models / name
        d.mkdir(parents=True)
        (d / "inference.pdiparams").write_bytes(b"fake-weights-" + name.encode())
    python = tmp_path / "fake_python.exe"
    python.write_bytes(b"not-a-real-interpreter")
    manifest = {
        "schema": 1, "protocol_version": 1,
        "engine_id": "ocr-paddle-bridge", "engine_version": "3.2.0",
        "params_schema_version": 1,
        "python": str(python) if abs_paths else "runtime/python.exe",
        "worker": "paddle-bridge",
        "models": [
            {"id": "ppocrv5-server-det",
             "path": str(models / "PP-OCRv5_server_det" / "inference.pdiparams"),
             "sha256": _sha256(b"fake-weights-PP-OCRv5_server_det")},
            {"id": "ppocrv5-server-rec",
             "path": str(models / "PP-OCRv5_server_rec" / "inference.pdiparams"),
             "sha256": _sha256(b"fake-weights-PP-OCRv5_server_rec")},
        ],
    }
    if extra:
        manifest["extra"] = {"source_dir": str(src), "models_dir": str(models)}
    (pd / "plugin.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return pd


class TestBridgePluginManifest:
    def test_extra_field_loaded(self, tmp_path):
        pd = _bridge_manifest(tmp_path)
        info = PluginManager(tmp_path)._load_manifest(pd / "plugin.json")
        assert info.worker_kind == "paddle-bridge"
        assert info.extra["source_dir"].endswith("bridge_src")
        assert info.extra["models_dir"].endswith("bridge_models")

    def test_extra_default_empty(self, tmp_path):
        pd = _bridge_manifest(tmp_path, extra=False)
        info = PluginManager(tmp_path)._load_manifest(pd / "plugin.json")
        assert info.extra == {}

    def test_absolute_paths_resolve(self, tmp_path):
        pd = _bridge_manifest(tmp_path)
        info = PluginManager(tmp_path)._load_manifest(pd / "plugin.json")
        assert info.python_exe.is_file()
        assert all(Path(m["path"]).is_absolute() and Path(m["path"]).is_file()
                   for m in info.models)

    def test_discover_priority_rapid_before_bridge(self, tmp_path):
        _bridge_manifest(tmp_path)
        rapid = tmp_path / "ocr-rapid" / "1.4.4"
        rapid.mkdir(parents=True)
        py = rapid / "python.exe"
        py.write_bytes(b"x")
        mp = rapid / "m.onnx"
        mp.write_bytes(b"m")
        (rapid / "plugin.json").write_text(json.dumps({
            "schema": 1, "protocol_version": 1, "engine_id": "ocr-rapid",
            "engine_version": "1.4.4", "python": "python.exe", "worker": "rapid",
            "models": [{"id": "det", "path": "m.onnx", "sha256": _sha256(b"m")}],
        }), encoding="utf-8")
        pm = PluginManager(tmp_path)
        assert pm.select("auto").engine_id == "ocr-rapid"
        assert pm.select("ocr-paddle-bridge").engine_id == "ocr-paddle-bridge"


# ---------------------------------------------------------------- Provider 守卫


class TestPaddleBridgeProviderGuards:
    def test_rejects_wrong_worker_kind(self, tmp_path):
        pd = _bridge_manifest(tmp_path)
        info = PluginManager(tmp_path)._load_manifest(pd / "plugin.json")
        object.__setattr__(info, "worker_kind", "rapid")
        with pytest.raises(PluginError):
            PaddleBridgeProvider(info)

    def test_missing_source_dir_raises(self, tmp_path):
        pd = _bridge_manifest(tmp_path, extra=False)
        info = PluginManager(tmp_path)._load_manifest(pd / "plugin.json")
        provider = PaddleBridgeProvider(info)
        with pytest.raises(PluginError):
            provider.initialize()

    def test_extra_init_params(self, tmp_path):
        pd = _bridge_manifest(tmp_path)
        info = PluginManager(tmp_path)._load_manifest(pd / "plugin.json")
        provider = PaddleBridgeProvider(info)
        extra = provider._extra_init_params()
        assert extra["device"] == "cpu"
        assert extra["source_dir"].endswith("bridge_src")
        assert extra["models_dir"].endswith("bridge_models")

    def test_engine_id_and_script(self, tmp_path):
        pd = _bridge_manifest(tmp_path)
        info = PluginManager(tmp_path)._load_manifest(pd / "plugin.json")
        provider = PaddleBridgeProvider(info)
        assert provider.engine_id == "ocr-paddle-bridge"
        assert provider.worker_script.name == "paddle_bridge_worker.py"
        assert provider.engine_id == info.engine_id

    def test_version_fingerprint_differs_from_rapid(self, tmp_path):
        pd = _bridge_manifest(tmp_path)
        info = PluginManager(tmp_path)._load_manifest(pd / "plugin.json")
        v = PaddleBridgeProvider(info).get_version()
        assert v.engine_id == "ocr-paddle-bridge"
        assert v.model_id == "ppocrv5-server-det+ppocrv5-server-rec"
        assert len(v.model_sha256) == 64


class TestStageDispatch:
    def test_make_provider_dispatch(self, tmp_path):
        from bookclean.ocr_stage import _make_provider
        pd = _bridge_manifest(tmp_path)
        info = PluginManager(tmp_path)._load_manifest(pd / "plugin.json")
        assert isinstance(_make_provider(info, OcrParams()), PaddleBridgeProvider)
        object.__setattr__(info, "worker_kind", "rapid")
        assert isinstance(_make_provider(info, OcrParams()), RapidOcrProvider)


# ---------------------------------------------------------------- worker 协议（假引擎）


@pytest.fixture
def bridge_env(tmp_path):
    """假既有安装：src 包（假引擎）+ server det/rec 模型占位文件。"""
    source = tmp_path / "bridge_src"
    shutil.copytree(FAKE_SRC, source / "src")
    models = tmp_path / "bridge_models"
    for name in ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"):
        d = models / name
        d.mkdir(parents=True)
        (d / "inference.pdiparams").write_bytes(b"fake")
    return {"source_dir": str(source), "models_dir": str(models),
            "threads": 2, "text_score": 0.5}


def _bridge_client(tmp_path, params, **kw) -> WorkerClient:
    kw.setdefault("page_timeout_s", 15.0)
    kw.setdefault("init_timeout_s", 30.0)
    return WorkerClient([sys.executable, "-u", str(BRIDGE_WORKER)], cwd=tmp_path, **kw)


def _image(tmp_path, name, color=(200, 200, 200)):
    from PIL import Image
    p = tmp_path / name
    Image.new("RGB", (200, 120), color).save(p)
    return p


class TestBridgeWorkerProtocol:
    def test_init_ready_and_stdout_pollution_absorbed(self, tmp_path, bridge_env):
        c = _bridge_client(tmp_path, bridge_env)
        ready = c.initialize(bridge_env)
        assert ready["type"] == "ready"
        assert ready["engine"] == "ocr-paddle-bridge"
        assert ready["device"] == "cpu"
        # 假 init_paddle 向 stdout print 垃圾；若重定向失效，协议会收到畸形 JSON 而失败
        r = c.recognize(1, _image(tmp_path, "a.png"))
        assert r["status"] == "completed"
        c.shutdown()

    def test_recognize_merges_word_lines(self, tmp_path, bridge_env):
        c = _bridge_client(tmp_path, bridge_env)
        c.initialize(bridge_env)
        r = c.recognize(1, _image(tmp_path, "a.png"))
        texts = [l["text"] for l in r["lines"]]
        assert texts == ["hello world foo", "北京大学"]
        assert r["lines"][0]["score"] == pytest.approx(0.90)
        assert r["lines"][0]["polygon_px"] is None  # 引擎未提供，不虚构
        assert r["reading_order"] == [0, 1]
        c.shutdown()

    def test_page_failure_isolation(self, tmp_path, bridge_env):
        c = _bridge_client(tmp_path, bridge_env)
        c.initialize(bridge_env)
        ok = c.recognize(1, _image(tmp_path, "ok.png"))
        bad = c.recognize(2, _image(tmp_path, "fail.png", (255, 0, 0)))
        ok2 = c.recognize(3, _image(tmp_path, "ok2.png"))
        assert ok["status"] == "completed"
        assert bad["status"] == "failed"
        assert bad["error"]["code"] == "ENGINE_ERROR"
        assert ok2["status"] == "completed"
        assert c.alive
        c.shutdown()

    def test_crash_restart_once_then_success(self, tmp_path, bridge_env):
        c = _bridge_client(tmp_path, bridge_env)
        c.initialize(bridge_env)
        r = c.recognize(1, _image(tmp_path, "crash.png", (0, 0, 255)))
        assert r["status"] == "completed"  # 崩溃后重启一次并重试成功
        assert c._restarts_left == 0
        c.shutdown()

    def test_missing_models_init_failed(self, tmp_path, bridge_env):
        env = dict(bridge_env)
        env["models_dir"] = str(tmp_path / "no_models")
        c = _bridge_client(tmp_path, env)
        with pytest.raises(Exception) as ei:
            c.initialize(env)
        assert "模型缺失" in str(ei.value)
        c.shutdown()

    def test_missing_source_init_failed(self, tmp_path, bridge_env):
        env = dict(bridge_env)
        env["source_dir"] = str(tmp_path / "no_source")
        c = _bridge_client(tmp_path, env)
        with pytest.raises(Exception) as ei:
            c.initialize(env)
        assert "桥接源码目录不存在" in str(ei.value)
        c.shutdown()

    def test_text_score_filter(self, tmp_path, bridge_env):
        env = dict(bridge_env, text_score=0.95)
        c = _bridge_client(tmp_path, env)
        c.initialize(env)
        r = c.recognize(1, _image(tmp_path, "a.png"))
        # 英文行 score=min(0.95,0.90,0.99)=0.90 < 0.95 被过滤；中文行 0.97 保留
        assert [l["text"] for l in r["lines"]] == ["北京大学"]
        c.shutdown()

    def test_cancel_then_shutdown(self, tmp_path, bridge_env):
        c = _bridge_client(tmp_path, bridge_env)
        c.initialize(bridge_env)
        c.cancel()
        r = c.recognize(1, _image(tmp_path, "a.png"))
        assert r["status"] == "completed"  # cancel 页边界协作，不中断当前页
        c.shutdown()
        assert not c.alive


# ---------------------------------------------------------------- 分发脚本


def _fake_rapid_plugin(root: Path) -> Path:
    pd = root / "ocr-rapid" / "9.9.9"
    (pd / "runtime").mkdir(parents=True)
    (pd / "runtime" / "python.exe").write_bytes(b"fake-interpreter")
    model = pd / "runtime" / "m.onnx"
    model.write_bytes(b"fake-model-bytes")
    (pd / "plugin.json").write_text(json.dumps({
        "schema": 1, "protocol_version": 1, "engine_id": "ocr-rapid",
        "engine_version": "9.9.9", "params_schema_version": 1,
        "python": "runtime/python.exe", "worker": "rapid",
        "models": [{"id": "det", "path": "runtime/m.onnx", "sha256": _sha256(b"fake-model-bytes")}],
    }), encoding="utf-8")
    return pd


class TestDistributionScripts:
    def test_build_and_install_roundtrip(self, tmp_path):
        from scripts.build_ocr_plugin import main as build_main
        from scripts.install_ocr_plugin import main as install_main
        pd = _fake_rapid_plugin(tmp_path / "src_plugins")
        zip_path = tmp_path / "dist" / "test-plugin.zip"
        assert build_main(["--plugin-dir", str(pd), "--out", str(zip_path)]) == 0
        assert zip_path.is_file()
        sidecar = json.loads(zip_path.with_suffix(".zip.json").read_text(encoding="utf-8"))
        assert sidecar["sha256"] == _sha256(zip_path.read_bytes())
        target_root = tmp_path / "target_plugins"
        assert install_main([str(zip_path), "--plugins-root", str(target_root)]) == 0
        pm = PluginManager(target_root)
        assert pm.check_available("ocr-rapid").installed
        assert (target_root / "ocr-rapid" / "9.9.9" / "runtime" / "m.onnx").is_file()

    def test_install_sha_mismatch_rejected(self, tmp_path):
        from scripts.build_ocr_plugin import main as build_main
        from scripts.install_ocr_plugin import main as install_main
        pd = _fake_rapid_plugin(tmp_path / "src_plugins")
        zip_path = tmp_path / "dist" / "test-plugin.zip"
        assert build_main(["--plugin-dir", str(pd), "--out", str(zip_path)]) == 0
        assert install_main([str(zip_path), "--plugins-root", str(tmp_path / "t"),
                             "--sha256", "0" * 64]) == 1

    def test_install_zip_slip_rejected(self, tmp_path):
        from scripts.install_ocr_plugin import main as install_main
        evil = tmp_path / "evil.zip"
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr("plugin.json", json.dumps({
                "schema": 1, "protocol_version": 1, "engine_id": "ocr-rapid",
                "engine_version": "1.0", "python": "p.exe", "worker": "rapid",
                "models": [{"id": "d", "path": "m", "sha256": "x"}]}))
            zf.writestr("../escaped.txt", "pwned")
        rc = install_main([str(evil), "--plugins-root", str(tmp_path / "t")])
        assert rc == 2
        assert not (tmp_path / "escaped.txt").exists()

    def test_install_existing_refused_and_force(self, tmp_path):
        from scripts.build_ocr_plugin import main as build_main
        from scripts.install_ocr_plugin import main as install_main
        pd = _fake_rapid_plugin(tmp_path / "src_plugins")
        zip_path = tmp_path / "dist" / "test-plugin.zip"
        build_main(["--plugin-dir", str(pd), "--out", str(zip_path)])
        target = tmp_path / "t"
        assert install_main([str(zip_path), "--plugins-root", str(target)]) == 0
        assert install_main([str(zip_path), "--plugins-root", str(target)]) == 2
        assert install_main([str(zip_path), "--plugins-root", str(target), "--force"]) == 0
        assert (target / "ocr-rapid" / "9.9.9.bak").is_dir()

    def test_bridge_setup_writes_manifest(self, tmp_path):
        from scripts.ocr_bridge_setup import main as setup_main
        source = tmp_path / "FakeOCR"
        (source / "src").mkdir(parents=True)
        (source / "src" / "ocr_engine.py").write_text(
            "def init_paddle(*a, **k):\n    pass\ndef run_ocr(*a, **k):\n    pass\n",
            encoding="utf-8")
        models = tmp_path / "models"
        for name in ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"):
            d = models / name
            d.mkdir(parents=True)
            (d / "inference.pdiparams").write_bytes(b"w-" + name.encode())
        plugins_root = tmp_path / "plugins"
        rc = setup_main(["--source", str(source), "--python", sys.executable,
                         "--models-dir", str(models), "--plugins-root", str(plugins_root),
                         "--skip-import-check"])
        assert rc == 0
        pm = PluginManager(plugins_root)
        info = pm.select("ocr-paddle-bridge")
        assert info is not None
        assert info.worker_kind == "paddle-bridge"
        assert info.extra["source_dir"] == str(source.resolve())
        assert PluginManager.verify_models(info) == []

    def test_bridge_setup_missing_source_exit2(self, tmp_path):
        from scripts.ocr_bridge_setup import main as setup_main
        rc = setup_main(["--source", str(tmp_path / "nope"),
                         "--plugins-root", str(tmp_path / "p")])
        assert rc == 2


# ---------------------------------------------------------------- GUI（真实 Tk）


tk = pytest.importorskip("tkinter")


@pytest.fixture
def settings_win(tk_root, tmp_path):
    from bookclean.app_controller import AppController
    from bookclean.ui.settings_panel import SettingsPanel
    from bookclean.user_paths import UserPaths
    ctl = AppController(paths=UserPaths(root=tmp_path / "data"))
    top = tk.Toplevel(tk_root)
    panel = SettingsPanel(top, controller=ctl)
    top.update_idletasks()
    yield panel, ctl
    try:
        top.destroy()
    except tk.TclError:
        pass


class TestGuiBridge:
    def test_engine_combo_has_bridge(self, settings_win):
        panel, ctl = settings_win
        values = list(panel.ocr_combo.cget("values"))
        assert any("PP-OCRv5" in v for v in values)
        panel.ocr_enabled_var.set(True)
        panel.ocr_engine_var.set("PP-OCRv5 桥接（高精度）")
        panel._sync()
        assert ctl.settings.ocr_engine == "ocr-paddle-bridge"

    def test_bridge_status_memory_warning(self, settings_win, monkeypatch):
        panel, ctl = settings_win
        monkeypatch.setattr(ctl, "ocr_plugin_status", lambda engine=None: {
            "installed": True, "engine_id": "ocr-paddle-bridge",
            "engine_version": "3.2.0", "models_ok": True, "reason": "", "plugin_dir": ""})
        panel.ocr_enabled_var.set(True)
        panel.ocr_engine_var.set("PP-OCRv5 桥接（高精度）")
        panel._refresh_ocr_status()
        text = panel.ocr_status_var.get()
        assert "已就绪" in text
        assert "5.7GB" in text  # 内存警示（验收门槛）

    def test_bridge_missing_guidance(self, settings_win, monkeypatch):
        panel, ctl = settings_win
        monkeypatch.setattr(ctl, "ocr_plugin_status", lambda engine=None: {
            "installed": False, "engine_id": "", "engine_version": "",
            "models_ok": False, "reason": "未安装", "plugin_dir": ""})
        panel.ocr_enabled_var.set(True)
        panel.ocr_engine_var.set("PP-OCRv5 桥接（高精度）")
        panel._refresh_ocr_status()
        text = panel.ocr_status_var.get()
        assert "ocr_bridge_setup" in text
        assert "0 下载" in text

    def test_install_help_mentions_tools(self):
        from bookclean.app_controller import AppController
        help_text = AppController.ocr_install_help()
        assert "install_ocr_plugin.py" in help_text
        assert "ocr_bridge_setup.py" in help_text
        assert "sha256" in help_text

    def test_gui_construction_does_not_import_ocr(self, tmp_path):
        code = (
            "import sys, tkinter as tk\n"
            "sys.path.insert(0, 'src')\n"
            "from bookclean.app_controller import AppController\n"
            "from bookclean.ui.settings_panel import SettingsPanel\n"
            "from bookclean.user_paths import UserPaths\n"
            "root = tk.Tk(); root.withdraw()\n"
            "ctl = AppController(paths=UserPaths(root=r'" + str(tmp_path / "d") + "'))\n"
            "p = SettingsPanel(root, controller=ctl)\n"
            "assert not any(m == 'bookclean.ocr' or m.startswith('bookclean.ocr.')\n"
            "               for m in sys.modules), 'OCR 关时构造 GUI 泄漏 OCR 导入'\n"
            "root.destroy(); print('ISOLATION_OK')\n"
        )
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
        assert "ISOLATION_OK" in r.stdout


# ---------------------------------------------------------------- 真实桥接（可选）


@pytest.mark.skipif(not (REAL_BRIDGE_SOURCE / "src" / "ocr_engine.py").is_file()
                    or not (REAL_PLUGINS_ROOT / "ocr-paddle-bridge").is_dir(),
                    reason="无既有 DeepSeekOCR_v2 安装或未配置桥接插件")
class TestRealBridge:
    def test_end_to_end_synth_page(self, tmp_path):
        """真实 PP-OCRv5 server：合成中英页 → 行重建 → 可提取文本（只读对方项目）。"""
        from PIL import Image, ImageDraw, ImageFont
        img = tmp_path / "synth.png"
        im = Image.new("RGB", (900, 300), "white")
        d = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 36)
        except OSError:
            font = ImageFont.load_default()
        d.text((40, 40), "北京大学 graduate school", fill="black", font=font)
        d.text((40, 140), "BookClean searchable PDF", fill="black", font=font)
        im.save(img)

        pm = PluginManager(REAL_PLUGINS_ROOT)
        plugin = pm.select("ocr-paddle-bridge")
        provider = PaddleBridgeProvider(plugin, OcrParams(threads=4, page_timeout_s=180.0))
        try:
            init = provider.initialize()
            assert init.ok, init.error
            assert init.device == "cpu"
            result = provider.recognize_page(img, 1)
            assert result.status == "completed", result.error
            text = " ".join(l.text for l in result.lines)
            assert "北京大学" in text
            assert "searchable" in text.lower()
            # 行重建：英文词框应被并成整行（不存在孤立的 "graduate" 词行）
            assert not any(l.text.strip() == "graduate" for l in result.lines)
        finally:
            provider.shutdown()