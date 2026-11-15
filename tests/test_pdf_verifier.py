"""M4.4 PDFVerifier 测试：重读逐页结构/内容/视觉检查，启发式像素口径（非 OCR）。

通过导出真实 PDF（verify=False）拿到 info 计划，再篡改计划/源图路径来触发各检查分支，
验证 verifier 能发现尺寸/页数/空白/内容/深色/重复/损坏文件等问题；正常导出应全部通过。
"""

from bookclean.pdf_verifier import PDFVerifier
from test_pdf_exporter import export, make_task, solid


def _save(tmp_path, name, color, size=(60, 80)):
    p = tmp_path / name
    solid(color, size).save(p, dpi=(150, 150))
    return str(p)


def _export_info(task, tmp_path, **cfg):
    return export(task, tmp_path, verify=False, **cfg)


def test_good_export_passes(tmp_path):
    task, _ = make_task(tmp_path, [
        {"orig": solid((0, 0, 0)), "proc": solid((220, 220, 220))},
        {"orig": solid((0, 0, 0)), "proc": solid((60, 60, 60))},
        {"orig": solid((0, 0, 0)), "proc": solid((140, 140, 140))},
    ])
    out, info = _export_info(task, tmp_path)
    v = PDFVerifier().verify(out, info, contact_dir=tmp_path / "v")
    assert v["passed"], v["issues"]
    assert v["page_count"] == 3
    assert v["contact_sheet"] and (tmp_path / "v" / "verify_contact_sheet.png").exists()
    assert v["previews"]


def test_white_source_page_not_blank_error(tmp_path):
    task, _ = make_task(tmp_path, [{"orig": solid((255, 255, 255)), "proc": solid((255, 255, 255))}])
    out, info = _export_info(task, tmp_path)
    v = PDFVerifier().verify(out, info, contact_dir=tmp_path / "v")
    assert v["passed"], v["issues"]


def test_size_mismatch_detected(tmp_path):
    task, _ = make_task(tmp_path, [{"orig": solid((0, 0, 0)), "proc": solid((120, 120, 120))}])
    out, info = _export_info(task, tmp_path)
    info["pages"][0]["output_page_width_pt"] += 20
    v = PDFVerifier().verify(out, info, contact_dir=tmp_path / "v")
    assert not v["passed"]
    assert any("尺寸" in i for i in v["issues"])


def test_page_count_mismatch_detected(tmp_path):
    task, _ = make_task(tmp_path, [{"orig": solid((0, 0, 0)), "proc": solid((120, 120, 120))}])
    out, info = _export_info(task, tmp_path)
    info["summary"]["total_pages"] = 5
    v = PDFVerifier().verify(out, info, contact_dir=tmp_path / "v")
    assert any("页数" in i for i in v["issues"])


def test_blank_render_with_content_source_detected(tmp_path):
    task, _ = make_task(tmp_path, [{"orig": solid((255, 255, 255)), "proc": solid((255, 255, 255))}])
    out, info = _export_info(task, tmp_path)
    info["pages"][0]["source_image_path"] = _save(tmp_path, "content.png", (100, 100, 100))
    v = PDFVerifier().verify(out, info, contact_dir=tmp_path / "v")
    assert any("空白" in i for i in v["issues"])


def test_content_mismatch_detected(tmp_path):
    task, _ = make_task(tmp_path, [{"orig": solid((0, 0, 0)), "proc": solid((200, 200, 200))}])
    out, info = _export_info(task, tmp_path)
    info["pages"][0]["source_image_path"] = _save(tmp_path, "dark.png", (40, 40, 40))
    v = PDFVerifier().verify(out, info, contact_dir=tmp_path / "v")
    assert any("内容与源图差异" in i for i in v["issues"])


def test_abnormal_dark_detected(tmp_path):
    task, _ = make_task(tmp_path, [{"orig": solid((0, 0, 0)), "proc": solid((20, 20, 20))}])
    out, info = _export_info(task, tmp_path)
    info["pages"][0]["source_image_path"] = _save(tmp_path, "light.png", (200, 200, 200))
    v = PDFVerifier().verify(out, info, contact_dir=tmp_path / "v")
    assert any("深色" in i for i in v["issues"])


def test_duplicate_render_different_source_detected(tmp_path):
    task, _ = make_task(tmp_path, [
        {"orig": solid((0, 0, 0)), "proc": solid((100, 100, 100))},
        {"orig": solid((0, 0, 0)), "proc": solid((100, 100, 100))},
    ])
    out, info = _export_info(task, tmp_path)
    info["pages"][1]["source_image_path"] = _save(tmp_path, "other.png", (130, 130, 130))
    v = PDFVerifier().verify(out, info, contact_dir=tmp_path / "v")
    assert any("重复" in i for i in v["issues"])


def test_missing_file(tmp_path):
    v = PDFVerifier().verify(tmp_path / "nope.pdf", {"pages": [], "summary": {}}, contact_dir=tmp_path / "v")
    assert not v["passed"] and any("不存在" in i for i in v["issues"])


def test_zero_size_file(tmp_path):
    p = tmp_path / "empty.pdf"
    p.write_bytes(b"")
    v = PDFVerifier().verify(p, {"pages": [], "summary": {}}, contact_dir=tmp_path / "v")
    assert any("大小" in i for i in v["issues"])


def test_corrupt_file(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"this is not a pdf")
    v = PDFVerifier().verify(p, {"pages": [], "summary": {}}, contact_dir=tmp_path / "v")
    assert any("无法打开" in i for i in v["issues"])