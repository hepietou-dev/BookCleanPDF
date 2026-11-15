"""M4.7.7 灰度长任务：复用已有 M1–M3 任务目录，导出 grayscale × {lossless,balanced,compact}。

不重跑 M1–M3。每模式独立子目录（报告/联系表不互相覆盖）。
记录：总时间 / 峰值内存 / 文件大小 / 页数 / 灰度数 / pikepdf 结构验证 / PDFium 逐页渲染验证。
产出：grayscale_{mode}.pdf + export_report.json/txt + 联系表 jpg + 色彩空间/渲染验证报告。
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pikepdf
import pypdfium2 as pdfium
from PIL import Image

from bookclean.config import PDFExportConfig
from bookclean.pdf_exporter import PDFExporter
from bookclean.pdf_verifier import _classify_colorspace, _iter_image_xobjects


class _MEM(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def working_set_mb():
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    try:
        get_info = kernel32.K32GetProcessMemoryInfo
    except AttributeError:
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_MEM), wintypes.DWORD]
    get_info.restype = wintypes.BOOL
    m = _MEM(); m.cb = ctypes.sizeof(_MEM)
    if not get_info(kernel32.GetCurrentProcess(), ctypes.byref(m), m.cb):
        return 0.0
    return m.WorkingSetSize / 1e6


def structure_check(path):
    gray = rgb = other = bits8 = 0
    with pikepdf.open(path) as pdf:
        for pno, name, obj in _iter_image_xobjects(pdf):
            label, n, is_gray = _classify_colorspace(obj.get("/ColorSpace"))
            bits = int(obj.get("/BitsPerComponent", 0))
            if is_gray:
                gray += 1
                if bits == 8:
                    bits8 += 1
            elif label in ("/DeviceRGB", "/CalRGB", "/Lab") or (label.startswith("ICCBased") and n == 3):
                rgb += 1
            else:
                other += 1
    return {"device_gray_images": gray, "unexpected_rgb_images": rgb,
            "other_images": other, "gray_8bit_images": bits8}


def render_check(path, plan):
    doc = pdfium.PdfDocument(str(path))
    n = len(doc)
    max_diff = 0
    bad = []
    size_bad = []
    for i in range(n):
        w, h = doc[i].get_size()
        exp = plan[i] if i < len(plan) else {}
        if abs(w - exp.get("output_page_width_pt", w)) > 0.5 or abs(h - exp.get("output_page_height_pt", h)) > 0.5:
            size_bad.append(i + 1)
        rgb = np.asarray(doc[i].render(scale=60 / 72.0).to_pil().convert("RGB")).astype(int)
        d = int(max(np.abs(rgb[:, :, 0] - rgb[:, :, 1]).max(), np.abs(rgb[:, :, 1] - rgb[:, :, 2]).max()))
        max_diff = max(max_diff, d)
        if d > 1:
            bad.append(i + 1)
    doc.close()
    return {"page_count": n, "max_channel_diff": max_diff,
            "channel_failed_pages": bad, "size_failed_pages": size_bad}


import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--modes", default="lossless,balanced,compact")
    args = ap.parse_args()
    task = Path(args.task_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    modes = args.modes.split(",")

    metrics = {"task_dir": str(task), "modes": {}}
    for mode in modes:
        mdir = out / mode; mdir.mkdir(parents=True, exist_ok=True)
        pdf_path = mdir / f"grayscale_{mode}.pdf"
        peak = [0.0]

        def cb(rec):
            peak[0] = max(peak[0], working_set_mb())

        def vcb(i, n):
            peak[0] = max(peak[0], working_set_mb())

        cfg = PDFExportConfig(compression=mode, color_mode="grayscale", verify=True, overwrite=True)
        t0 = time.monotonic()
        info = PDFExporter(cfg).export(task, pdf_path, progress_cb=cb, verify_progress_cb=vcb)
        elapsed = time.monotonic() - t0
        peak[0] = max(peak[0], working_set_mb())

        st = structure_check(pdf_path)
        rd = render_check(pdf_path, info["pages"])
        s = info["summary"]
        gv = info["grayscale_validation"]
        metrics["modes"][mode] = {
            "file": str(pdf_path), "size_bytes": pdf_path.stat().st_size,
            "pages": s["total_pages"], "grayscale_pages": s["grayscale_pages"],
            "color_pages": s["color_pages"], "jpeg_pages": s["jpeg_pages"], "png_pages": s["png_pages"],
            "export_seconds": round(elapsed, 2), "peak_working_set_mb": round(peak[0], 1),
            "structure": st, "render": rd,
            "grayscale_validation_status": gv["status"],
            "conversion": info["grayscale_conversion"],
        }
        print(f"[{mode}] pages={s['total_pages']} size={pdf_path.stat().st_size} "
              f"time={elapsed:.1f}s peak={peak[0]:.1f}MB gray={st['device_gray_images']} "
              f"rgb={st['unexpected_rgb_images']} render_max={rd['max_channel_diff']} "
              f"gv={gv['status']}", flush=True)

    # 联系表 jpg（取 balanced 的验证联系表）
    src_sheet = out / "balanced" / "verify_contact_sheet.png"
    if src_sheet.exists():
        Image.open(src_sheet).convert("RGB").save(out / "grayscale_contact_sheet.jpg", quality=85)

    (out / "m4_7_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # 色彩空间验证报告
    cs_lines = ["# PDF 色彩空间验证报告（pikepdf 内部图像对象）", ""]
    for mode, m in metrics["modes"].items():
        st = m["structure"]
        cs_lines.append(f"## {mode}: DeviceGray={st['device_gray_images']} 8bit={st['gray_8bit_images']} "
                        f"意外RGB={st['unexpected_rgb_images']} 其它={st['other_images']} "
                        f"-> {'PASS' if st['unexpected_rgb_images'] == 0 and st['other_images'] == 0 and st['gray_8bit_images'] == m['pages'] else 'FAIL'}")
    (out / "colorspace_validation.md").write_text("\n".join(cs_lines) + "\n", encoding="utf-8")

    # PDFium 渲染验证报告
    rd_lines = ["# PDFium 渲染验证报告（逐页 RGB 通道一致性 + 尺寸）", ""]
    for mode, m in metrics["modes"].items():
        rd = m["render"]
        rd_lines.append(f"## {mode}: pages={rd['page_count']} max_channel_diff={rd['max_channel_diff']} "
                        f"channel_failed={rd['channel_failed_pages']} size_failed={rd['size_failed_pages']} "
                        f"-> {'PASS' if not rd['channel_failed_pages'] and not rd['size_failed_pages'] and rd['max_channel_diff'] <= 1 else 'FAIL'}")
    (out / "pdfium_render_validation.md").write_text("\n".join(rd_lines) + "\n", encoding="utf-8")

    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()