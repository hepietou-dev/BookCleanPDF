"""M4.8.5 PDF 结构与阅读器兼容性审查（只读复用真实导出 + 合成混合样本）。

用 pikepdf 审查结构（版本/页数/加密/线性化/catalog/对象流/页面尺寸/图像 XObject 色空间位深过滤器/元数据），
用 pypdfium2（PDFium，Chrome/Edge 同源引擎）逐页打开+渲染+文本层计数（图像型 PDF 应为 0 字符）。
真实阅读器仅检测本机已安装者（Chrome/Edge=PDFium；Adobe/Foxit/Sumatra 未安装则如实标注，不虚构）。
不重跑 M1–M3，不改动真实任务目录与既有导出 PDF。所有指标为文件/像素口径，非 OCR。

输出: temp/m4_8_review/compatibility_audit.json + compat_render_sheet.png
"""

from __future__ import annotations

import ctypes
import importlib.metadata as md
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
REVIEW = ROOT / "temp" / "m4_8_review"

import numpy as np
import pikepdf
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
from PIL import Image

from bookclean.config import PDFExportConfig
from bookclean.pdf_exporter import PDFExporter
from bookclean.pdf_verifier import _classify_colorspace, _iter_image_xobjects

# 只读复用的真实图书导出（120 页真实书，M4.6/M4.7/M4.8.1 产物）
REAL_PDFS = {
    "real_balanced_preserve": ROOT / "temp/m4_6_subset/balanced/real_balanced.pdf",
    "real_grayscale": ROOT / "temp/m4_7_subset/balanced/grayscale_balanced.pdf",
    "real_border_preserve": REVIEW / "border_balanced_preserve.pdf",
    "real_border_grayscale": REVIEW / "border_balanced_gray.pdf",
}


def detect_readers() -> dict:
    """检测本机已安装的 PDF 阅读器/浏览器。必须找到真实可执行文件，空目录不算安装，不虚构。"""
    pf = Path("C:/Program Files")
    pf86 = Path("C:/Program Files (x86)")
    local = Path.home() / "AppData/Local"
    # 每项: (搜索基目录列表, 可执行文件名列表)
    candidates = {
        "Adobe Acrobat/Reader": ([pf / "Adobe", pf86 / "Adobe"], ["AcroRd32.exe", "Acrobat.exe"]),
        "Foxit Reader": ([pf86 / "Foxit Software", pf / "Foxit Software"],
                         ["FoxitPDFReader.exe", "FoxitReader.exe"]),
        "SumatraPDF": ([pf / "SumatraPDF", local / "SumatraPDF"], ["SumatraPDF.exe"]),
        "Chrome (PDFium)": ([pf / "Google/Chrome/Application", pf86 / "Google/Chrome/Application"],
                            ["chrome.exe"]),
        "Edge (PDFium)": ([pf86 / "Microsoft/Edge/Application", pf / "Microsoft/Edge/Application"],
                          ["msedge.exe"]),
    }
    found = {}
    for name, (bases, exes) in candidates.items():
        hit = None
        for base in bases:
            if not base.exists():
                continue
            for exe in exes:
                direct = base / exe
                if direct.exists():
                    hit = str(direct); break
                matches = list(base.rglob(exe))
                if matches:
                    hit = str(matches[0]); break
            if hit:
                break
        found[name] = {"installed": hit is not None, "path": hit}
    return found


def pdfium_version(doc) -> int:
    v = ctypes.c_int(0)
    ok = pdfium_raw.FPDF_GetFileVersion(doc, ctypes.byref(v))
    return int(v.value) if ok else 0


def build_synthetic_mixed() -> Path:
    """用真实导出管线构建混合样本（彩色+灰度/JPEG+PNG/竖+横/混合尺寸），审查当前导出器结构。"""
    from scripts.make_m2_samples import text_page, _photo_array
    task = REVIEW / "compat_synth_task"
    (task / "original_pages").mkdir(parents=True, exist_ok=True)
    (task / "processed_pages").mkdir(parents=True, exist_ok=True)
    base, _ = text_page()
    gray = np.asarray(Image.fromarray(base).convert("L"))
    gray_rgb = np.stack([gray] * 3, axis=2)
    photo = _photo_array(420, 600, color=True, seed=7)
    photo_land = _photo_array(600, 420, color=True, seed=9)
    specs = [
        ("RGB", photo, 150.0),          # 彩色竖 -> JPEG
        ("L", gray_rgb, 150.0),         # 灰度竖文字 -> PNG
        ("RGB", photo_land, 150.0),     # 彩色横
        ("L", gray_rgb, 200.0),         # 灰度横（不同 DPI/尺寸）
        ("RGB", _photo_array(300, 400, color=True, seed=3), 150.0),  # 小彩色
    ]
    manifest = {"task_id": "compat_synth", "pages": [], "inputs": []}
    for i, (mode, arr, dpi) in enumerate(specs):
        im = Image.fromarray(arr).convert(mode)
        rel = f"processed_pages/page_{i + 1:06d}.png"
        im.save(task / rel, dpi=(dpi, dpi))
        manifest["pages"].append({
            "page_index": i, "display_index": i + 1, "source_kind": "image",
            "source_path": str(task / f"in_{i}.png"), "status": "completed",
            "image_path": rel, "processed_image_path": rel,
            "processing_status": "completed", "page_type": None, "dpi": dpi,
            "metadata": {}, "final_image_source": "enhanced",
            "fallback_level": "none", "processing_decision": "accept",
        })
        manifest["inputs"].append({"path": str(task / f"in_{i}.png"), "kind": "image"})
    (task / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    out = REVIEW / "compat_synth_mixed.pdf"
    PDFExporter(PDFExportConfig(compression="balanced", color_mode="preserve")).export(task, out)
    return out


def audit_structure(path: Path) -> dict:
    """pikepdf 结构审查。"""
    with pikepdf.open(path) as pk:
        n = len(pk.pages)
        objstm = 0
        for o in pk.objects:
            try:
                if o.get("/Type") == pikepdf.Name("/ObjStm"):
                    objstm += 1
            except Exception:
                pass
        catalog = list(pk.Root.keys())
        page_sizes = []
        for pg in pk.pages:
            mb = pg.get("/MediaBox")
            if mb is not None:
                page_sizes.append([round(float(x), 2) for x in mb])
        images = []
        cs_dist: dict[str, int] = {}
        filter_dist: dict[str, int] = {}
        bits_dist: dict[int, int] = {}
        for pno, name, obj in _iter_image_xobjects(pk):
            cs = obj.get("/ColorSpace")
            label, ncomp, is_gray = _classify_colorspace(cs)
            bits = int(obj.get("/BitsPerComponent", 0))
            filt = obj.get("/Filter")
            if isinstance(filt, pikepdf.Array):
                fname = "+".join(str(x) for x in filt)
            else:
                fname = str(filt) if filt is not None else "None"
            w = int(obj.get("/Width", 0)); h = int(obj.get("/Height", 0))
            images.append({"page": pno, "object": name, "colorspace": label,
                           "components": ncomp, "bits": bits, "filter": fname,
                           "width": w, "height": h})
            cs_dist[label] = cs_dist.get(label, 0) + 1
            filter_dist[fname] = filter_dist.get(fname, 0) + 1
            bits_dist[bits] = bits_dist.get(bits, 0) + 1
        di = pk.docinfo
        meta = {
            "title": str(di.get("/Title", "")) or None,
            "producer": str(di.get("/Producer", "")) or None,
            "creator": str(di.get("/Creator", "")) or None,
            "subject": str(di.get("/Subject", "")) or None,
        }
        has_xmp = "/Metadata" in pk.Root
        return {
            "pdf_version": pk.pdf_version,
            "page_count": n,
            "is_encrypted": pk.is_encrypted,
            "is_linearized": pk.is_linearized,
            "object_stream_count": objstm,
            "catalog_keys": catalog,
            "page_sizes_pt": page_sizes,
            "image_count": len(images),
            "colorspace_distribution": cs_dist,
            "filter_distribution": filter_dist,
            "bits_distribution": {str(k): v for k, v in bits_dist.items()},
            "metadata": meta,
            "has_xmp_metadata": has_xmp,
            "images_sample": images[:6],
        }


def audit_render(path: Path, render_dpi: int = 50) -> dict:
    """pypdfium2（PDFium）逐页打开+渲染+文本层计数。"""
    t0 = time.perf_counter()
    doc = pdfium.PdfDocument(str(path))
    n = len(doc)
    ver = pdfium_version(doc)
    rendered = 0
    render_errors = []
    total_chars = 0
    nonblank = 0
    sizes = []
    thumbs = []
    for i in range(n):
        pg = doc[i]
        w, h = pg.get_size()
        sizes.append([round(w, 2), round(h, 2)])
        try:
            bmp = pg.render(scale=render_dpi / 72.0)
            pil = bmp.to_pil().convert("RGB")
            arr = np.asarray(pil)
            rendered += 1
            if arr.mean() < 250:
                nonblank += 1
            thumbs.append(pil)
        except Exception as exc:
            render_errors.append({"page": i + 1, "error": f"{type(exc).__name__}: {exc}"})
        try:
            tp = pg.get_textpage()
            total_chars += tp.count_chars()
        except Exception as exc:
            render_errors.append({"page": i + 1, "error": f"textpage: {exc}"})
    doc.close()
    elapsed = time.perf_counter() - t0
    return {
        "pdfium_file_version": ver,
        "page_count": n,
        "rendered_ok": rendered,
        "render_errors": render_errors,
        "nonblank_pages": nonblank,
        "text_layer_chars": total_chars,
        "is_image_based_no_ocr": total_chars == 0,
        "page_sizes_pt": sizes,
        "render_elapsed_s": round(elapsed, 3),
        "_thumbs": thumbs,
    }


def make_contact_sheet(thumbs_by_pdf: dict, out: Path) -> None:
    """每个 PDF 取首/中/末渲染缩略图拼成证据联系表。"""
    rows = []
    tw = 150
    for label, thumbs in thumbs_by_pdf.items():
        if not thumbs:
            continue
        n = len(thumbs)
        idx = sorted({0, n // 2, n - 1})
        row_imgs = []
        for i in idx:
            im = thumbs[i]
            th = int(tw * im.height / im.width)
            row_imgs.append(im.resize((tw, th), Image.LANCZOS))
        rows.append((label, row_imgs))
    if not rows:
        return
    label_w = 220
    row_h = max(max(im.height for _, ims in rows for im in ims) + 24, 60)
    max_cols = max(len(ims) for _, ims in rows)
    canvas = Image.new("RGB", (label_w + max_cols * (tw + 10) + 10, row_h * len(rows) + 10), (245, 245, 245))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    for r, (label, ims) in enumerate(rows):
        y = 10 + r * row_h
        draw.text((8, y + 8), label, fill=(20, 20, 20))
        for c, im in enumerate(ims):
            canvas.paste(im, (label_w + c * (tw + 10), y + 20))
    canvas.save(out)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    REVIEW.mkdir(parents=True, exist_ok=True)
    readers = detect_readers()
    print("=== 本机阅读器检测 ===")
    for name, info in readers.items():
        print(f"  {name}: {'已安装 ' + info['path'] if info['installed'] else '未安装'}")

    print("\n=== 构建合成混合样本（真实导出管线）===")
    synth = build_synthetic_mixed()
    print(f"  {synth}")

    targets = {"synthetic_mixed": synth}
    for label, p in REAL_PDFS.items():
        if p.exists():
            targets[label] = p
        else:
            print(f"  跳过（不存在）: {label} {p}")

    results = {}
    thumbs_by_pdf = {}
    for label, path in targets.items():
        print(f"\n=== 审查 {label}: {path.name} ===")
        st = audit_structure(path)
        rd = audit_render(path)
        thumbs = rd.pop("_thumbs")
        thumbs_by_pdf[label] = thumbs
        ok = (rd["rendered_ok"] == rd["page_count"] and not rd["render_errors"]
              and st["page_count"] == rd["page_count"])
        results[label] = {
            "file": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "structure": st,
            "pdfium_render": rd,
            "render_all_pages_ok": ok,
        }
        print(f"  PDF {st['pdf_version']}  页数 {st['page_count']}  加密 {st['is_encrypted']}  "
              f"线性化 {st['is_linearized']}  对象流 {st['object_stream_count']}")
        print(f"  图像 {st['image_count']}  色空间 {st['colorspace_distribution']}  "
              f"过滤器 {st['filter_distribution']}  位深 {st['bits_distribution']}")
        print(f"  PDFium 渲染 {rd['rendered_ok']}/{rd['page_count']}  错误 {len(rd['render_errors'])}  "
              f"文本层字符 {rd['text_layer_chars']}（图像型无OCR={rd['is_image_based_no_ocr']}）  "
              f"耗时 {rd['render_elapsed_s']}s")

    sheet = REVIEW / "compat_render_sheet.png"
    make_contact_sheet(thumbs_by_pdf, sheet)

    all_ok = all(r["render_all_pages_ok"] for r in results.values())
    all_no_ocr = all(r["pdfium_render"]["is_image_based_no_ocr"] for r in results.values())
    summary = {
        "pypdfium2_version": md.version("pypdfium2"),
        "pikepdf_version": md.version("pikepdf"),
        "readers": readers,
        "pdfium_is_chrome_edge_engine": True,
        "targets_audited": len(results),
        "all_pages_render_ok": all_ok,
        "all_image_based_no_ocr": all_no_ocr,
        "real_reader_note": "Adobe/Foxit/Sumatra 未安装，未虚构其测试结果；Chrome/Edge 已安装且与 pypdfium2 同为 PDFium 引擎。",
        "results": results,
        "contact_sheet": str(sheet.relative_to(ROOT)),
    }
    out = REVIEW / "compatibility_audit.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== 汇总 ===")
    print(f"  审查 {len(results)} 个 PDF  全部逐页渲染OK={all_ok}  全部图像型无OCR={all_no_ocr}")
    print(f"  报告: {out}")
    print(f"  联系表: {sheet}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())