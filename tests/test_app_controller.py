"""M5.1 应用服务层测试：输入检查/格式判断/错误捕获/配置映射/预检/详情。

全部使用离线生成的真实 PDF/图片文件（真实 I/O 与页数解析），不 mock 内核。
"""

import json
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

from bookclean.app_controller import AppController, InputInfo, TaskSettings
from bookclean.user_paths import UserPaths


@pytest.fixture
def ctl(tmp_path):
    return AppController(paths=UserPaths(root=tmp_path / "data"))


def _write_pdf(path, pages=1):
    import img2pdf
    import io
    imgs = []
    for i in range(pages):
        im = Image.new("RGB", (40, 30), (i * 10 % 256, 100, 150))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        imgs.append(buf.getvalue())
    path.write_bytes(img2pdf.convert(*imgs))
    return path


# ---- 输入检查与格式判断 -------------------------------------------------

def test_inspect_image(ctl, img_factory):
    p = img_factory("a.png")
    infos = ctl.inspect_inputs([p])
    assert len(infos) == 1
    i = infos[0]
    assert i.kind == "image" and i.page_count == 1 and i.size_bytes > 0 and i.error is None
    assert i.name == "a.png"


def test_inspect_multipage_pdf(ctl, tmp_path):
    p = _write_pdf(tmp_path / "book.pdf", pages=3)
    infos = ctl.inspect_inputs([p])
    assert infos[0].kind == "pdf" and infos[0].page_count == 3 and infos[0].error is None


def test_inspect_unsupported(ctl, tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("hello")
    infos = ctl.inspect_inputs([p])
    assert infos[0].error is not None and infos[0].kind == "file"


def test_inspect_missing(ctl, tmp_path):
    infos = ctl.inspect_inputs([tmp_path / "nope.png"])
    assert infos[0].error is not None


def test_inspect_corrupt_pdf(ctl, tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"not a pdf")
    infos = ctl.inspect_inputs([p])
    assert infos[0].kind == "pdf" and infos[0].error is not None and infos[0].page_count == 0


def test_inspect_corrupt_image(ctl, tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"notpng")
    infos = ctl.inspect_inputs([p])
    assert infos[0].error is not None


def test_inspect_folder_natural_sort(ctl, img_factory, tmp_path):
    folder = tmp_path / "imgs"
    folder.mkdir()
    for name in ("page10.png", "page2.png", "page1.png"):
        Image.new("RGB", (10, 10), (255, 0, 0)).save(folder / name)
    (folder / "ignore.txt").write_text("x")
    infos = ctl.inspect_inputs([folder])
    names = [i.name for i in infos]
    assert names == ["page1.png", "page2.png", "page10.png"]  # 自然排序，跳过 txt


def test_inspect_empty_folder(ctl, tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    infos = ctl.inspect_inputs([folder])
    assert len(infos) == 1 and infos[0].kind == "folder" and infos[0].error is not None


def test_inspect_dedup(ctl, img_factory):
    p = img_factory("a.png")
    ctl.inspect_inputs([p])
    ctl.inspect_inputs([p])
    assert len(ctl.inputs) == 1


def test_inspect_batch_does_not_abort_on_bad(ctl, img_factory, tmp_path):
    good = img_factory("good.png")
    bad = tmp_path / "bad.txt"
    bad.write_text("x")
    infos = ctl.inspect_inputs([bad, good])
    assert len(infos) == 2
    assert any(i.error for i in infos) and any(i.error is None for i in infos)


def test_remove_and_clear(ctl, img_factory):
    a = img_factory("a.png")
    b = img_factory("b.png")
    ctl.inspect_inputs([a, b])
    ctl.remove_input(a)
    assert [i.name for i in ctl.inputs] == ["b.png"]
    ctl.clear_inputs()
    assert ctl.inputs == []


# ---- 设置与配置映射 -----------------------------------------------------

def test_set_settings_rejects_invalid(ctl):
    with pytest.raises(ValueError):
        ctl.set_settings(TaskSettings(preset="unknown"))
    with pytest.raises(ValueError):
        ctl.set_settings(TaskSettings(color_mode="sepia"))
    with pytest.raises(ValueError):
        ctl.set_settings(TaskSettings(compression="tiny"))


def test_build_configs_defaults(ctl):
    cfg, export, border = ctl.build_configs()
    assert cfg.preset == "auto" and cfg.render_dpi == 300
    assert export.compression == "balanced" and export.color_mode == "preserve" and export.verify is True
    assert border is not None


def test_build_configs_maps_settings(ctl):
    ctl.set_settings(TaskSettings(preset="old_book", color_mode="grayscale", compression="compact", border_clean=False))
    cfg, export, border = ctl.build_configs()
    assert cfg.preset == "old_book"
    assert export.color_mode == "grayscale" and export.compression == "compact"
    assert border is None


def test_default_output_path(ctl, img_factory, tmp_path):
    p = img_factory("mybook.png")
    ctl.inspect_inputs([p])
    ctl.set_settings(TaskSettings(compression="compact", color_mode="grayscale", output_dir=str(tmp_path / "out")))
    out = ctl.default_output_path()
    assert out.name == "mybook_compact_grayscale.pdf" and out.parent == tmp_path / "out"


# ---- 预检 ---------------------------------------------------------------

def test_validate_no_inputs(ctl):
    r = ctl.validate_start()
    assert not r.ok and r.errors


def test_validate_ok(ctl, img_factory, tmp_path):
    ctl.inspect_inputs([img_factory("a.png")])
    ctl.set_settings(TaskSettings(output_dir=str(tmp_path / "out")))
    r = ctl.validate_start()
    assert r.ok and not r.errors and r.output_path.endswith(".pdf")


def test_validate_only_invalid_inputs(ctl, tmp_path):
    bad = tmp_path / "x.txt"
    bad.write_text("x")
    ctl.inspect_inputs([bad])
    r = ctl.validate_start()
    assert not r.ok


def test_validate_warns_on_partial_invalid(ctl, img_factory, tmp_path):
    good = img_factory("a.png")
    bad = tmp_path / "x.txt"
    bad.write_text("x")
    ctl.inspect_inputs([good, bad])
    ctl.set_settings(TaskSettings(output_dir=str(tmp_path / "out")))
    r = ctl.validate_start()
    assert r.ok and r.warnings


def test_validate_output_conflict_no_overwrite(ctl, img_factory, tmp_path):
    p = img_factory("a.png")
    ctl.inspect_inputs([p])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    ctl.set_settings(TaskSettings(output_dir=str(out_dir)))
    target = ctl.default_output_path()
    target.write_bytes(b"%PDF-1.4 existing")
    r = ctl.validate_start()
    assert not r.ok and any("已存在" in e for e in r.errors)


def test_validate_output_conflict_with_overwrite(ctl, img_factory, tmp_path):
    p = img_factory("a.png")
    ctl.inspect_inputs([p])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    ctl.set_settings(TaskSettings(output_dir=str(out_dir), overwrite=True))
    ctl.default_output_path().write_bytes(b"%PDF-1.4 existing")
    r = ctl.validate_start()
    assert r.ok


def test_validate_output_equals_input(ctl, tmp_path, monkeypatch):
    # 默认命名总追加 _质量_类型 后缀，天然不会与输入同名；此守卫为防御性（与内核 _guard_output 一致）。
    # 直接构造“输出路径==输入路径”的条件来验证该分支。
    p = _write_pdf(tmp_path / "x.pdf", pages=1)
    ctl.inspect_inputs([p])
    ctl.set_settings(TaskSettings(output_dir=str(tmp_path)))
    monkeypatch.setattr(ctl, "default_output_path", lambda: p)
    r = ctl.validate_start()
    assert not r.ok and any("相同" in e for e in r.errors)


def test_validate_does_not_create_dirs(ctl, img_factory, tmp_path):
    ctl.inspect_inputs([img_factory("a.png")])
    out_dir = tmp_path / "not_created"
    ctl.set_settings(TaskSettings(output_dir=str(out_dir)))
    ctl.validate_start()
    assert not out_dir.exists()  # 预检无副作用


# ---- 详情与打开 ---------------------------------------------------------

def test_describe_json_safe(ctl, img_factory):
    ctl.inspect_inputs([img_factory("a.png")])
    d = ctl.describe()
    json.dumps(d)  # 不抛异常
    assert d["input_count"] == 1 and d["valid_count"] == 1 and d["total_pages"] == 1


def test_open_path_missing_returns_false(ctl, tmp_path):
    assert ctl.open_path(tmp_path / "nope") is False


def test_open_path_success(ctl, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(os, "startfile", lambda p: calls.append(p), raising=False)
    d = tmp_path / "out"
    d.mkdir()
    assert ctl.open_path(d) is True
    assert calls and Path(calls[0]) == d


def test_open_path_guarded_on_error(ctl, tmp_path, monkeypatch):
    def boom(_p):
        raise OSError("shell failed")
    monkeypatch.setattr(os, "startfile", boom, raising=False)
    d = tmp_path / "out"
    d.mkdir()
    assert ctl.open_path(d) is False  # 不抛出到 UI


# ---- 架构约束：服务层不依赖 Tkinter ------------------------------------

def test_controller_is_tk_free():
    import subprocess
    code = (
        "import sys; sys.path.insert(0, 'src'); "
        "import bookclean.app_controller; "
        "assert 'tkinter' not in sys.modules, 'app_controller 不应导入 tkinter'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parent.parent)
    assert proc.returncode == 0, proc.stdout + proc.stderr