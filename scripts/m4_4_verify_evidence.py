"""M4.4 真实书证据脚本（只读原始/处理结果，绝不修改任务目录）。

产出 temp/m4_4_export/：
- real_balanced.pdf：20 页真实书，balanced，verify=True，原子输出。
- export_report.json / export_report.txt：导出报告（含逐页验证明细）。
- verify_contact_sheet.png + verify_preview_pXXX.jpg：全页联系表与高清预览。
- compare_pXX_before_after.jpg：源处理结果(before) vs PDF 渲染(after) 同像素对比。
- m4_4_metrics.json：验证指标 + 原子/冲突/输入守卫/篡改检测演示结果。
"""

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bookclean.config import PDFExportConfig
from bookclean.pdf_exporter import ExportError, PDFExporter
from bookclean.pdf_verifier import PDFVerifier

TASK = ROOT / "temp/m3_real_book/tasks/task_20261111_230829_24bdec"
OUT = ROOT / "temp/m4_4_export"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("real_balanced*"):
        f.unlink()
    manifest = json.loads((TASK / "manifest.json").read_text(encoding="utf-8"))
    metrics = {"task_id": manifest.get("task_id"), "task_dir": str(TASK)}

    pdf_path = OUT / "real_balanced.pdf"
    cfg = PDFExportConfig(compression="balanced", verify=True, title="BookClean M4.4 真实书验证")
    t0 = time.monotonic()
    info = PDFExporter(cfg).export(TASK, pdf_path)
    metrics["export_seconds"] = round(time.monotonic() - t0, 3)
    metrics["output_size_bytes"] = info["output_size_bytes"]
    metrics["page_count"] = info["summary"]["total_pages"]
    metrics["validation_status"] = info["validation_status"]
    v = info["validation"]
    metrics["validation_passed"] = v["passed"]
    metrics["validation_issues"] = v["issues"]
    metrics["contact_sheet"] = v["contact_sheet"]
    metrics["previews"] = v["previews"]
    metrics["summary"] = info["summary"]

    per = v["per_page"]
    diffs = [p["content_diff"] for p in per if p["content_diff"] is not None]
    metrics["verify_per_page"] = {
        "all_size_ok": all(p["size_ok"] for p in per),
        "all_blank_ok": all(p["blank_ok"] for p in per),
        "pages_with_content_diff": len(diffs),
        "content_diff_min": round(min(diffs), 2) if diffs else None,
        "content_diff_max": round(max(diffs), 2) if diffs else None,
        "content_diff_mean": round(sum(diffs) / len(diffs), 2) if diffs else None,
    }

    # ---- 原子 / 冲突 / 输入守卫 / 篡改检测演示 ----
    checks = {}
    checks["no_partial_left"] = not (OUT / "real_balanced.partial.pdf").exists()

    before_bytes = pdf_path.read_bytes()
    try:
        PDFExporter(PDFExportConfig(compression="balanced", verify=False)).export(TASK, pdf_path)
        checks["conflict_blocked"] = False
    except ExportError:
        checks["conflict_blocked"] = True
    checks["conflict_pdf_unchanged"] = pdf_path.read_bytes() == before_bytes

    inp = Path(manifest["pages"][0]["source_path"])
    try:
        PDFExporter(PDFExportConfig(verify=False, overwrite=True)).export(TASK, inp)
        checks["input_guard_blocked"] = False
        checks["input_guard_msg"] = ""
    except ExportError as e:
        checks["input_guard_blocked"] = True
        checks["input_guard_msg"] = str(e)

    # 篡改计划：把第 1 页源图换成亮度差异最大的页 + 尺寸偏移 -> verifier 应报错
    from bookclean.pdf_verifier import _load_source_sig
    sigs = [_load_source_sig(p["source_image_path"]) for p in info["pages"]]
    means = [s.mean() if s is not None else 255.0 for s in sigs]
    j = max(range(len(means)), key=lambda k: abs(means[k] - means[0]))
    tampered = copy.deepcopy(info)
    tampered["pages"][0]["source_image_path"] = info["pages"][j]["source_image_path"]
    tampered["pages"][0]["output_page_width_pt"] += 30
    tv = PDFVerifier().verify(pdf_path, tampered, contact_dir=OUT / "tamper")
    checks["tamper_detected"] = not tv["passed"]
    metrics["tamper_demo"] = {"swapped_with_page": j + 1, "issues": tv["issues"][:8]}
    metrics["safety_checks"] = checks

    # ---- 前后对比（源处理结果 vs PDF 渲染，同像素尺寸）----
    doc = pdfium.PdfDocument(str(pdf_path))
    picks = []
    for want in ("pure_text", "photo_dominated", "grayscale_illustration"):
        k = next((idx for idx, p in enumerate(manifest["pages"]) if p.get("page_type") == want), None)
        if k is not None and k not in picks:
            picks.append(k)
    compares = []
    for k in picks:
        rec = info["pages"][k]
        with Image.open(rec["source_image_path"]) as im:
            before = im.convert("RGB")
        scale = rec["image_width_px"] / rec["output_page_width_pt"]
        after = doc[k].render(scale=scale).to_pil().convert("RGB")
        after = after.crop((0, 0, before.width, before.height))
        a = np.asarray(before, dtype=np.int16)
        b = np.asarray(after, dtype=np.int16)
        mad = float(np.abs(a - b).mean())
        canvas = Image.new("RGB", (before.width * 2 + 24, before.height), (255, 255, 255))
        canvas.paste(before, (0, 0))
        canvas.paste(after, (before.width + 24, 0))
        cp = OUT / f"compare_p{k + 1:02d}_before_after.jpg"
        canvas.save(cp, quality=92)
        compares.append({"page": k + 1, "page_type": manifest["pages"][k].get("page_type"),
                         "mean_abs_diff": round(mad, 3), "file": str(cp)})
    doc.close()
    metrics["before_after"] = compares

    (OUT / "m4_4_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in
                      ("page_count", "output_size_bytes", "validation_passed", "validation_issues",
                       "verify_per_page", "safety_checks")}, ensure_ascii=False, indent=2))
    print("before_after:", json.dumps(compares, ensure_ascii=False))
    print("OK ->", OUT)


if __name__ == "__main__":
    main()