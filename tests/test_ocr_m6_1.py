# -*- coding: utf-8 -*-
"""M6.1 验收测试：OCR 插件基础设施（协议/崩溃/超时/取消/发现/隔离）。

全部离线：worker 协议测试用标准库假 worker（tests/fixtures/ocr_fake_worker.py），
插件发现测试用 tmp_path 合成清单；真实插件端到端用例在无插件环境自动跳过。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from bookclean.ocr.fake import FakeProvider
from bookclean.ocr.plugin_manager import (
    ENGINE_PRIORITY,
    PluginError,
    PluginInfo,
    PluginManager,
)
from bookclean.ocr.provider import (
    EngineVersion,
    OcrError,
    OcrLine,
    OcrParams,
    OcrProvider,
    PageOcrResult,
    STATUS_COMPLETED,
    STATUS_FAILED,
)
from bookclean.ocr.rapid_provider import RapidOcrProvider
from bookclean.ocr.worker_client import WorkerClient, WorkerError

ROOT = Path(__file__).resolve().parent.parent
FAKE_WORKER = ROOT / "tests" / "fixtures" / "ocr_fake_worker.py"
REAL_PLUGINS_ROOT = ROOT / "temp" / "plugins"


# ---------------------------------------------------------------- 工具


def _client(tmp_path, params_extra=None, **kw) -> WorkerClient:
    kw.setdefault("page_timeout_s", 5.0)
    kw.setdefault("init_timeout_s", 10.0)
    c = WorkerClient([sys.executable, "-u", str(FAKE_WORKER)], cwd=tmp_path, **kw)
    c._test_params = {"mode": "ok"}
    if params_extra:
        c._test_params.update(params_extra)
    return c


def _init(c: WorkerClient) -> dict:
    return c.initialize(c._test_params)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_plugin(root: Path, engine_id="ocr-rapid", version="1.4.4", *,
                 worker="rapid", python="runtime/python.exe", schema=1,
                 protocol=1, models=True, model_bytes=b"fake-model",
                 model_sha=None, extra=None) -> Path:
    """合成插件目录（发现/校验测试用，不启动真实解释器）。"""
    pd = root / engine_id / version
    pd.mkdir(parents=True, exist_ok=True)
    py = pd / python
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_bytes(b"not-a-real-interpreter")
    manifest = {
        "schema": schema, "protocol_version": protocol,
        "engine_id": engine_id, "engine_version": version,
        "params_schema_version": 1, "python": python, "worker": worker,
        "models": [],
    }
    if models:
        mp = pd / "models" / "m.onnx"
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_bytes(model_bytes)
        manifest["models"].append({
            "id": "fake-det", "path": "models/m.onnx",
            "sha256": model_sha or _sha256(model_bytes),
        })
    if extra:
        manifest.update(extra)
    (pd / "plugin.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return pd


@pytest.fixture
def small_image(tmp_path):
    from PIL import Image
    p = tmp_path / "page.png"
    Image.new("RGB", (60, 40), (200, 30, 30)).save(p)
    return p


# ---------------------------------------------------------------- WorkerClient 协议


class TestWorkerProtocol:
    def test_roundtrip_and_shutdown(self, tmp_path):
        c = _client(tmp_path)
        ready = _init(c)
        assert ready["type"] == "ready"
        assert c.alive
        r1 = c.recognize(1, tmp_path / "a.png")
        r2 = c.recognize(2, tmp_path / "b.png")
        assert r1["status"] == STATUS_COMPLETED and r1["page_no"] == 1
        assert r2["page_no"] == 2
        assert r1["lines"][0]["text"] == "fake worker page 1"
        c.shutdown()
        assert not c.alive

    def test_page_failure_isolation(self, tmp_path):
        c = _client(tmp_path, {"fail_pages": [2]})
        _init(c)
        r1 = c.recognize(1, "x.png")
        r2 = c.recognize(2, "x.png")
        r3 = c.recognize(3, "x.png")
        assert r1["status"] == STATUS_COMPLETED
        assert r2["status"] == STATUS_FAILED
        assert r2["error"]["code"] == "ENGINE_ERROR"
        assert r3["status"] == STATUS_COMPLETED  # 失败页之后继续
        assert c.alive
        c.shutdown()

    def test_crash_restart_once_then_success(self, tmp_path):
        once = tmp_path / "crash.marker"
        c = _client(tmp_path, {"crash_page": 2, "once_file": str(once)})
        _init(c)
        assert c.recognize(1, "x.png")["status"] == STATUS_COMPLETED
        r2 = c.recognize(2, "x.png")  # 首次崩溃 → 重启 → 重试成功
        assert r2["status"] == STATUS_COMPLETED and r2["page_no"] == 2
        assert once.exists()
        assert c.alive
        assert c.recognize(3, "x.png")["status"] == STATUS_COMPLETED
        c.shutdown()

    def test_crash_budget_exhausted(self, tmp_path):
        c = _client(tmp_path, {"mode": "always_crash"})
        _init(c)
        with pytest.raises(WorkerError) as ei:
            c.recognize(1, "x.png")
        assert ei.value.code == "WORKER_CRASHED"
        assert "exit=" in str(ei.value)
        with pytest.raises(WorkerError):
            c.recognize(2, "x.png")  # 预算已耗尽，不再重启
        c.shutdown()

    def test_timeout_restart_once_then_success(self, tmp_path):
        once = tmp_path / "hang.marker"
        c = _client(tmp_path, {"hang_page": 2, "once_file": str(once)},
                    page_timeout_s=1.0)
        _init(c)
        r2 = c.recognize(2, "x.png")  # 首次挂起 → 超时 → 重启 → 成功
        assert r2["status"] == STATUS_COMPLETED and r2["page_no"] == 2
        assert c.alive
        c.shutdown()

    def test_timeout_budget_exhausted(self, tmp_path):
        c = _client(tmp_path, {"mode": "always_hang"}, page_timeout_s=0.8)
        _init(c)
        with pytest.raises(WorkerError) as ei:
            c.recognize(1, "x.png")
        assert ei.value.code == "TIMEOUT"
        c.shutdown(grace_s=0.5)
        assert not c.alive  # 挂起进程必须被杀掉

    def test_bad_json_ignored(self, tmp_path):
        c = _client(tmp_path, {"mode": "bad_json"})
        _init(c)
        r = c.recognize(7, "x.png")
        assert r["status"] == STATUS_COMPLETED and r["page_no"] == 7
        c.shutdown()

    def test_init_error_reported(self, tmp_path):
        c = _client(tmp_path, {"mode": "init_error"})
        with pytest.raises(WorkerError) as ei:
            _init(c)
        assert ei.value.code == "INIT_FAILED"
        c.shutdown()

    def test_init_crash_reported(self, tmp_path):
        c = _client(tmp_path, {"mode": "init_crash"})
        with pytest.raises(WorkerError) as ei:
            _init(c)
        assert ei.value.code == "WORKER_CRASHED"
        c.shutdown()

    def test_cancel_then_continue(self, tmp_path):
        c = _client(tmp_path)
        _init(c)
        c.cancel()  # 页边界协作取消：worker 回 cancelled，后续请求仍可用
        r = c.recognize(1, "x.png")
        assert r["status"] == STATUS_COMPLETED
        c.shutdown()


# ---------------------------------------------------------------- PluginManager


class TestPluginManager:
    def test_empty_root(self, tmp_path):
        pm = PluginManager(tmp_path / "nonexistent")
        assert pm.discover() == []
        avail = pm.check_available()
        assert not avail.installed and "插件" in avail.reason

    def test_discover_valid(self, tmp_path):
        _make_plugin(tmp_path)
        plugins = PluginManager(tmp_path).discover()
        assert len(plugins) == 1
        p = plugins[0]
        assert p.engine_id == "ocr-rapid" and p.engine_version == "1.4.4"
        assert p.worker_kind == "rapid"
        assert p.python_exe.is_file()
        assert len(p.models) == 1 and Path(p.models[0]["path"]).is_file()

    def test_discover_skips_invalid(self, tmp_path):
        _make_plugin(tmp_path)  # 唯一有效
        _make_plugin(tmp_path, engine_id="bad-json", version="1.0")
        (tmp_path / "bad-json" / "1.0" / "plugin.json").write_text("{oops", encoding="utf-8")
        _make_plugin(tmp_path, engine_id="bad-schema", version="1.0", schema=99)
        _make_plugin(tmp_path, engine_id="bad-protocol", version="1.0", protocol=99)
        _make_plugin(tmp_path, engine_id="no-python", version="1.0", python="runtime/missing.exe")
        (tmp_path / "no-python" / "1.0" / "runtime" / "missing.exe").unlink()
        _make_plugin(tmp_path, engine_id="no-models", version="1.0", models=False)
        _make_plugin(tmp_path, engine_id="missing-model-file", version="1.0")
        (tmp_path / "missing-model-file" / "1.0" / "models" / "m.onnx").unlink()
        plugins = PluginManager(tmp_path).discover()
        assert [p.engine_id for p in plugins] == ["ocr-rapid"]

    def test_select_priority_and_version(self, tmp_path):
        _make_plugin(tmp_path, engine_id="ocr-paddle-bridge", version="1.0")
        _make_plugin(tmp_path, engine_id="ocr-rapid", version="1.9.0")
        _make_plugin(tmp_path, engine_id="ocr-rapid", version="1.10.0")
        pm = PluginManager(tmp_path)
        assert pm.select("auto").engine_id == "ocr-rapid"
        assert pm.select("auto").engine_version == "1.10.0"  # 语义化版本排序
        assert pm.select("ocr-paddle-bridge").engine_id == "ocr-paddle-bridge"
        assert pm.select("no-such-engine") is None
        assert ENGINE_PRIORITY[0] == "ocr-rapid"

    def test_check_available_installed(self, tmp_path):
        _make_plugin(tmp_path)
        avail = PluginManager(tmp_path).check_available()
        assert avail.installed and avail.models_ok
        assert avail.engine_id == "ocr-rapid" and avail.plugin_dir

    def test_verify_models(self, tmp_path):
        pd = _make_plugin(tmp_path)
        plugin = PluginManager(tmp_path).select()
        assert PluginManager.verify_models(plugin) == []
        (pd / "models" / "m.onnx").write_bytes(b"tampered")
        problems = PluginManager.verify_models(plugin)
        assert len(problems) == 1 and "哈希不匹配" in problems[0]
        (pd / "models" / "m.onnx").unlink()
        problems = PluginManager.verify_models(plugin)
        assert len(problems) == 1 and "缺失" in problems[0]

    def test_model_fingerprint_stable_and_sensitive(self, tmp_path):
        _make_plugin(tmp_path)
        plugin = PluginManager(tmp_path).select()
        mid1, sha1 = plugin.model_fingerprint()
        mid2, sha2 = plugin.model_fingerprint()
        assert (mid1, sha1) == (mid2, sha2)
        assert mid1 == "fake-det" and len(sha1) == 64
        other = PluginInfo(
            engine_id=plugin.engine_id, engine_version=plugin.engine_version,
            protocol_version=1, params_schema_version=1, plugin_dir=plugin.plugin_dir,
            python_exe=plugin.python_exe, worker_kind=plugin.worker_kind,
            models=({"id": "fake-det", "path": "x", "sha256": "f" * 64},),
        )
        assert other.model_fingerprint()[1] != sha1


# ---------------------------------------------------------------- 数据结构


class TestResultStructures:
    def test_page_result_json_roundtrip(self):
        res = PageOcrResult(
            page_no=3, status=STATUS_COMPLETED,
            lines=(OcrLine(0, "中文 line", 0.97, (1.0, 2.0, 30.0, 40.0),
                           ((1.0, 2.0), (30.0, 2.0), (30.0, 40.0), (1.0, 40.0))),),
            reading_order=(0,), image_width_px=100, image_height_px=200,
            image_sha256="ab" * 32, elapsed_s=1.25,
            engine_version=EngineVersion("ocr-rapid", "1.4.4", "m", "cd" * 32),
        )
        back = PageOcrResult.from_dict(json.loads(json.dumps(res.to_dict(), ensure_ascii=False)))
        assert back == res

    def test_failed_result_roundtrip(self):
        res = PageOcrResult(page_no=1, status=STATUS_FAILED,
                            error=OcrError("TIMEOUT", "超时"))
        back = PageOcrResult.from_dict(json.loads(json.dumps(res.to_dict())))
        assert back == res and back.error.code == "TIMEOUT"

    def test_engine_version_fingerprint(self):
        v = EngineVersion("ocr-rapid", "1.4.4", "m", "s")
        assert v.fingerprint() == "ocr-rapid|1.4.4|m|s|1"
        assert EngineVersion("ocr-rapid", "1.4.5", "m", "s").fingerprint() != v.fingerprint()
        assert EngineVersion("ocr-rapid", "1.4.4", "m2", "s").fingerprint() != v.fingerprint()


# ---------------------------------------------------------------- FakeProvider


class TestFakeProvider:
    def test_protocol_conformance(self):
        assert isinstance(FakeProvider(), OcrProvider)

    def test_batch_failure_isolation_and_empty(self, tmp_path, small_image):
        prov = FakeProvider(fail_pages={3}, empty_pages={4})
        prov.initialize()
        results = []
        summary = prov.recognize_batch(
            [(i, small_image) for i in range(1, 6)], results.append)
        assert summary.issued == 5 and summary.completed == 4 and summary.failed == 1
        assert [r.page_no for r in results] == [1, 2, 3, 4, 5]
        assert results[2].status == STATUS_FAILED
        assert results[3].status == STATUS_COMPLETED and results[3].lines == ()
        ok = results[0]
        assert ok.image_sha256 == _sha256(small_image.read_bytes())
        assert (ok.image_width_px, ok.image_height_px) == (60, 40)
        assert ok.lines[0].text == "fake line page 1"

    def test_cancel_stops_at_page_boundary(self, tmp_path, small_image):
        prov = FakeProvider()
        prov.initialize()
        ev = threading.Event()
        results = []

        def on_result(r):
            results.append(r)
            ev.set()  # 第一页完成后取消

        summary = prov.recognize_batch([(i, small_image) for i in range(1, 10)],
                                       on_result, cancel_event=ev)
        assert summary.issued == 1 and len(results) == 1

    def test_not_initialized(self, small_image):
        prov = FakeProvider()
        r = prov.recognize_page(small_image, 1)
        assert r.status == STATUS_FAILED and r.error.code == "NOT_INITIALIZED"


# ---------------------------------------------------------------- RapidOcrProvider（无真实引擎）


class TestRapidProviderGuards:
    def test_rejects_wrong_worker_kind(self, tmp_path):
        _make_plugin(tmp_path, worker="paddle")
        plugin = PluginManager(tmp_path).select()
        with pytest.raises(PluginError):
            RapidOcrProvider(plugin)

    def test_model_mismatch_blocks_init(self, tmp_path):
        _make_plugin(tmp_path, model_sha="0" * 64)
        plugin = PluginManager(tmp_path).select()
        prov = RapidOcrProvider(plugin)
        avail = prov.check_available()
        assert avail.installed and not avail.models_ok
        init = prov.initialize()
        assert not init.ok and init.error.code == "MODEL_MISMATCH"
        assert not prov._client.alive  # 校验失败绝不启动 worker

    def test_not_initialized_page(self, tmp_path, small_image):
        _make_plugin(tmp_path)
        prov = RapidOcrProvider(PluginManager(tmp_path).select())
        r = prov.recognize_page(small_image, 1)
        assert r.status == STATUS_FAILED and r.error.code == "NOT_INITIALIZED"

    def test_unreadable_image_maps_to_failed(self, tmp_path, small_image, monkeypatch):
        _make_plugin(tmp_path)
        prov = RapidOcrProvider(PluginManager(tmp_path).select())
        monkeypatch.setattr(prov, "_initialized", True)
        r = prov.recognize_page(tmp_path / "missing.png", 1)
        assert r.status == STATUS_FAILED and r.error.code == "IMAGE_UNREADABLE"

    def test_worker_error_maps_to_failed(self, tmp_path, small_image, monkeypatch):
        _make_plugin(tmp_path)
        prov = RapidOcrProvider(PluginManager(tmp_path).select())
        monkeypatch.setattr(prov, "_initialized", True)

        def boom(page_no, image_path):
            raise WorkerError("TIMEOUT", "测试超时")

        monkeypatch.setattr(prov._client, "recognize", boom)
        r = prov.recognize_page(small_image, 1)
        assert r.status == STATUS_FAILED and r.error.code == "TIMEOUT"
        assert r.image_sha256 == _sha256(small_image.read_bytes())

    def test_version_fingerprint_from_models(self, tmp_path):
        _make_plugin(tmp_path)
        plugin = PluginManager(tmp_path).select()
        prov = RapidOcrProvider(plugin)
        v = prov.get_version()
        assert v.engine_id == "ocr-rapid" and v.engine_version == "1.4.4"
        assert v.model_id == "fake-det"
        assert v.model_sha256 == plugin.model_fingerprint()[1]


# ---------------------------------------------------------------- 基础包隔离


class TestBaseIsolation:
    def test_base_pipeline_never_imports_ocr(self):
        """M1–M5 主链路导入后 sys.modules 不得出现 bookclean.ocr / OCR 运行时。"""
        code = (
            "import sys; sys.path.insert(0, 'src')\n"
            "import bookclean.task_manager\n"
            "import bookclean.processing_pipeline\n"
            "import bookclean.pdf_exporter\n"
            "import bookclean.app_controller\n"
            "import bookclean.ui.main_window\n"
            "bad = [m for m in sys.modules if m.startswith('bookclean.ocr')\n"
            "       or m.split('.')[0] in ('rapidocr_onnxruntime', 'onnxruntime', 'paddleocr')]\n"
            "assert not bad, bad\n"
            "print('ISOLATION_OK')\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                              capture_output=True, text=True, encoding="utf-8", timeout=120)
        assert proc.returncode == 0, proc.stderr
        assert "ISOLATION_OK" in proc.stdout


# ---------------------------------------------------------------- 真实插件（可跳过）


@pytest.mark.skipif(not (REAL_PLUGINS_ROOT / "ocr-rapid").is_dir(),
                    reason="未安装 ocr-rapid 插件（可选功能）")
class TestRealPlugin:
    def test_end_to_end_synth_page(self, small_image):
        pm = PluginManager(REAL_PLUGINS_ROOT)
        plugin = pm.select("auto")
        assert plugin is not None and plugin.engine_id == "ocr-rapid"
        prov = RapidOcrProvider(plugin, OcrParams(threads=4))
        try:
            init = prov.initialize()
            assert init.ok, init.error
            assert init.device == "cpu"
            res = prov.recognize_page(small_image, 1)
            assert res.status == STATUS_COMPLETED, res.error
            assert res.image_sha256 == _sha256(small_image.read_bytes())
            assert (res.image_width_px, res.image_height_px) == (60, 40)
            assert res.engine_version.engine_version == "1.4.4"
        finally:
            prov.shutdown()