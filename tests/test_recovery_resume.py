"""M4.8.4 长任务稳定性与断点续跑测试。

覆盖：原子图像写入中断不留损坏文件；处理结果多维校验（存在/大小/解码/格式/尺寸/色彩/SHA256/配置）；
续跑复用 validated processed_pages（不重跑 M1–M3）；损坏/篡改/配置不符触发重处理；
磁盘满与 manifest 写入中断的清洁失败；导出遇损坏处理结果优雅回退（不崩溃、不覆盖已验证 PDF）；
原始图与处理结果只读保护。所有指标为文件/像素口径，非 OCR。
"""

import json
import os

import numpy as np
import pytest
from PIL import Image

import bookclean.utils.atomic_file as af
import bookclean.processing_pipeline as pp
from bookclean.config import PDFExportConfig
from bookclean.pdf_exporter import FinalImageResolver, PDFExporter
from bookclean.presets import get_preset
from bookclean.processing_pipeline import ProcessingRunner
from bookclean.recovery import (
    can_reuse_processed, processing_config_matches, sha256_file, validate_processed_image,
)
from bookclean.task_manager import ImportTaskManager
from bookclean.utils.atomic_file import atomic_save_png, atomic_write_json
from scripts.make_m2_samples import text_page


def _make_task(tmp_path, n=3):
    files = []
    for i in range(n):
        rgb, _ = text_page(400, 560)
        p = tmp_path / f"in_{i}.png"
        Image.fromarray(rgb).save(p, dpi=(150, 150))
        files.append(p)
    return ImportTaskManager(tmp_path / "tasks").run(files, dpi=150)["task_dir"]


def _manifest(task_dir):
    return json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))


def _processed(task_dir, n=3):
    td = _make_task(task_dir, n)
    ProcessingRunner().run(td, get_preset("auto"))
    return td


# ---------- 原子图像写入 ----------

def test_atomic_save_png_roundtrip(tmp_path):
    p = tmp_path / "x.png"
    atomic_save_png(Image.new("RGB", (40, 30), (10, 20, 30)), p, dpi=150)
    with Image.open(p) as im:
        assert im.size == (40, 30) and im.format == "PNG"
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_save_png_interruption_keeps_prior(tmp_path, monkeypatch):
    p = tmp_path / "x.png"
    atomic_save_png(Image.new("RGB", (40, 30), (1, 2, 3)), p)
    prior = p.read_bytes()

    def boom(*a, **k):
        raise OSError("模拟 replace 中断")

    monkeypatch.setattr(af.os, "replace", boom)
    with pytest.raises(OSError):
        atomic_save_png(Image.new("RGB", (40, 30), (9, 9, 9)), p)
    assert p.read_bytes() == prior  # 旧文件完整
    assert not list(tmp_path.glob("*.tmp"))  # 临时文件已清理


def test_atomic_save_png_disk_full_no_partial(tmp_path, monkeypatch):
    p = tmp_path / "new.png"

    def boom(*a, **k):
        raise OSError("模拟磁盘满")

    monkeypatch.setattr(af.os, "replace", boom)
    with pytest.raises(OSError):
        atomic_save_png(Image.new("RGB", (40, 30)), p)
    assert not p.exists()  # 不留半个 PNG
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_retries_transient_permission_error(tmp_path, monkeypatch):
    # Windows 杀毒/索引器短暂占用目标 ⇒ os.replace 瞬态 PermissionError；短退避重试后成功
    p = tmp_path / "m.json"
    calls = {"n": 0}
    real_replace = af.os.replace

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(5, "拒绝访问（模拟瞬态占用）")
        return real_replace(src, dst)

    monkeypatch.setattr(af.os, "replace", flaky)
    monkeypatch.setattr(af.time, "sleep", lambda *_: None)
    atomic_write_json(p, {"page": 1})
    assert calls["n"] == 3  # 前两次失败，第三次成功
    assert json.loads(p.read_text(encoding="utf-8")) == {"page": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_persistent_permission_error_raises_and_cleans(tmp_path, monkeypatch):
    p = tmp_path / "m.json"

    def always_denied(src, dst):
        raise PermissionError(5, "拒绝访问（模拟持续占用）")

    monkeypatch.setattr(af.os, "replace", always_denied)
    monkeypatch.setattr(af.time, "sleep", lambda *_: None)
    with pytest.raises(PermissionError):
        atomic_write_json(p, {"page": 1})
    assert not p.exists()
    assert not list(tmp_path.glob("*.tmp"))  # 临时文件已清理


# ---------- 多维校验 ----------

def test_validate_valid_png(tmp_path):
    p = tmp_path / "ok.png"
    Image.new("RGB", (50, 60), (200, 200, 200)).save(p)
    ok, reasons = validate_processed_image(p, {"sha256": sha256_file(p), "width_px": 50,
                                               "height_px": 60, "color_mode": "RGB", "format": "PNG"})
    assert ok and not reasons


def test_validate_missing_and_zero(tmp_path):
    assert not validate_processed_image(tmp_path / "no.png")[0]
    z = tmp_path / "zero.png"
    z.write_bytes(b"")
    ok, reasons = validate_processed_image(z)
    assert not ok and "大小" in reasons[0]


def test_validate_truncated_fails(tmp_path):
    p = tmp_path / "bad.png"
    Image.new("RGB", (50, 60)).save(p)
    p.write_bytes(p.read_bytes()[:20])  # 截断
    ok, reasons = validate_processed_image(p)
    assert not ok and any("解码" in r for r in reasons)


def test_validate_wrong_format_fails(tmp_path):
    p = tmp_path / "img.jpg"
    Image.new("RGB", (50, 60)).save(p, format="JPEG")
    ok, reasons = validate_processed_image(p, {"format": "PNG"})
    assert not ok and any("格式" in r for r in reasons)


def test_validate_sha256_mismatch_fails(tmp_path):
    p = tmp_path / "ok.png"
    Image.new("RGB", (50, 60), (1, 2, 3)).save(p)
    ok, reasons = validate_processed_image(p, {"sha256": "0" * 64})
    assert not ok and any("SHA256" in r for r in reasons)


def test_validate_dimension_mismatch_fails(tmp_path):
    p = tmp_path / "ok.png"
    Image.new("RGB", (50, 60)).save(p)
    ok, reasons = validate_processed_image(p, {"width_px": 99, "height_px": 99})
    assert not ok and any("尺寸" in r for r in reasons)


def test_validate_color_mode_mismatch_fails(tmp_path):
    p = tmp_path / "ok.png"
    Image.new("RGB", (50, 60)).save(p)
    ok, reasons = validate_processed_image(p, {"color_mode": "L"})
    assert not ok and any("色彩" in r for r in reasons)


# ---------- 配置匹配 ----------

def test_config_matches():
    cfg = get_preset("auto")
    entry = {"processing_parameters": {k: getattr(cfg, k) for k in
             ("preset", "output_mode", "background_strength", "yellow_removal_strength",
              "denoise_strength", "sharpen_strength")}}
    assert processing_config_matches(entry, cfg)
    entry["processing_parameters"]["background_strength"] += 0.1
    assert not processing_config_matches(entry, cfg)
    assert not processing_config_matches({}, cfg)  # 缺记录保守不匹配


# ---------- 续跑复用 ----------

def test_resume_reuses_all_and_protects_files(tmp_path):
    td = _processed(tmp_path)
    m = _manifest(td)
    orig_sha = {p["image_path"]: sha256_file(td / p["image_path"]) for p in m["pages"]}
    proc_sha = {p["processed_image_path"]: sha256_file(td / p["processed_image_path"]) for p in m["pages"]}
    res = ProcessingRunner().run(td, get_preset("auto"), resume=True)
    assert res["summary"]["processed"] == 0 and res["summary"]["reused"] == 3
    m2 = _manifest(td)
    assert all(sha256_file(td / p["image_path"]) == orig_sha[p["image_path"]] for p in m2["pages"])
    assert all(sha256_file(td / p["processed_image_path"]) == proc_sha[p["processed_image_path"]] for p in m2["pages"])


def test_default_no_resume_reprocesses(tmp_path):
    td = _processed(tmp_path)
    res = ProcessingRunner().run(td, get_preset("auto"))  # resume 默认 False
    assert res["summary"]["processed"] == 3 and res["summary"]["reused"] == 0


def test_resume_reprocesses_corrupt_page(tmp_path):
    td = _processed(tmp_path)
    m = _manifest(td)
    (td / m["pages"][0]["processed_image_path"]).write_bytes(b"\x89PNG\r\n\x1a\nCORRUPT")
    res = ProcessingRunner().run(td, get_preset("auto"), resume=True)
    assert res["summary"]["processed"] == 1 and res["summary"]["reused"] == 2
    ok, _ = validate_processed_image(td / _manifest(td)["pages"][0]["processed_image_path"])
    assert ok  # 重处理后恢复完整


def test_resume_reprocesses_tampered_page(tmp_path):
    td = _processed(tmp_path)
    m = _manifest(td)
    rel = m["pages"][0]["processed_image_path"]
    w, h = m["pages"][0]["processed_width_px"], m["pages"][0]["processed_height_px"]
    Image.new("RGB", (w, h), (123, 123, 123)).save(td / rel)  # 合法 PNG 但内容被篡改
    res = ProcessingRunner().run(td, get_preset("auto"), resume=True)
    assert res["summary"]["processed"] == 1 and res["summary"]["reused"] == 2  # SHA256 不符触发重处理


def test_resume_reprocesses_on_config_change(tmp_path):
    td = _processed(tmp_path)
    cfg = get_preset("auto")
    cfg.background_strength *= 0.5
    res = ProcessingRunner().run(td, cfg, resume=True)
    assert res["summary"]["processed"] == 3 and res["summary"]["reused"] == 0


def test_resume_after_cancel(tmp_path):
    import threading
    td = _make_task(tmp_path)
    ev = threading.Event()
    cfg = get_preset("auto")
    runner = ProcessingRunner()
    # 处理第一页后取消
    orig_cb = None

    def cb(entry):
        if entry["page_index"] == 0:
            ev.set()

    runner.run(td, cfg, progress_cb=cb, cancel_event=ev)
    m = _manifest(td)
    assert m["pages"][0]["processing_status"] == "completed"
    assert any(p.get("processing_status") == "cancelled" for p in m["pages"][1:])
    res = ProcessingRunner().run(td, cfg, resume=True)
    assert res["summary"]["reused"] == 1  # 已完成页复用
    assert res["summary"]["processed"] == 2  # 取消页续跑


# ---------- 磁盘满 / manifest 中断 ----------

def test_disk_full_marks_failed_no_corrupt(tmp_path, monkeypatch):
    td = _make_task(tmp_path)

    def boom(*a, **k):
        raise OSError("模拟磁盘满")

    monkeypatch.setattr(pp, "atomic_save_png", boom)
    res = ProcessingRunner().run(td, get_preset("auto"))
    assert res["summary"]["processed"] == 0 and res["summary"]["failed"] == 3
    assert not list((td / "processed_pages").glob("*.png"))  # 无损坏产物
    m = _manifest(td)  # manifest 仍为合法 JSON
    assert m["processing_status"] == "completed_with_errors"
    assert all(p["processing_status"] == "failed" for p in m["pages"])


def test_manifest_write_interruption_keeps_prior(tmp_path, monkeypatch):
    p = tmp_path / "manifest.json"
    atomic_write_json(p, {"a": 1})
    prior = p.read_bytes()
    with pytest.raises(TypeError):
        atomic_write_json(p, {"bad": {1, 2, 3}})  # set 不可序列化
    assert p.read_bytes() == prior
    assert not list(tmp_path.glob("*.tmp"))


# ---------- 导出优雅回退 ----------

def test_export_falls_back_on_corrupt_processed(tmp_path):
    td = _processed(tmp_path)
    m = _manifest(td)
    (td / m["pages"][1]["processed_image_path"]).write_bytes(b"\x89PNG\r\n\x1a\nBAD")
    out = tmp_path / "out.pdf"
    info = PDFExporter(PDFExportConfig(compression="balanced")).export(td, out)
    assert info["validation"]["passed"]
    assert info["pages"][1]["final_image_source"] == "original_fallback"
    assert out.exists()


def test_resolver_corrupt_processed_selects_original(tmp_path):
    td = _processed(tmp_path)
    m = _manifest(td)
    (td / m["pages"][0]["processed_image_path"]).write_bytes(b"\x89PNG\r\n\x1a\nBAD")
    fi = FinalImageResolver(td).resolve(m, m["pages"][0])
    assert fi.source == "original_fallback" and fi.fallback
    assert any("损坏" in w for w in fi.warnings)


def test_export_does_not_overwrite_verified_pdf_on_corrupt(tmp_path):
    td = _processed(tmp_path)
    out = tmp_path / "out.pdf"
    PDFExporter(PDFExportConfig(compression="balanced")).export(td, out)
    verified = out.read_bytes()
    m = _manifest(td)
    (td / m["pages"][0]["processed_image_path"]).write_bytes(b"\x89PNG\r\n\x1a\nBAD")
    # 默认不覆盖：即便处理结果损坏，既有已验证 PDF 不被触碰
    from bookclean.pdf_exporter import ExportError
    with pytest.raises(ExportError):
        PDFExporter(PDFExportConfig(compression="balanced")).export(td, out)
    assert out.read_bytes() == verified