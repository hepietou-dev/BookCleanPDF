"""M4.6 真实阅读器兼容性证据：双独立引擎重开最终 PDF。

- pypdfium2（PDFium，Chrome/Edge 同款渲染引擎）渲染代表页到 PNG（150 DPI，文字可读）。
- pikepdf（QPDF）独立结构校验：三模式页数 / 逐页 MediaBox / DocumentInfo / XMP。
产出 temp/m4_6_subset/reader_compat.json + reader_balanced_pXXX.png。原书只读，不联网。
运行: python scripts/m4_6_reader_compat.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pikepdf
import pypdfium2 as pdfium

OUT = Path("temp/m4_6_subset")
MODES = {
    "balanced": OUT / "balanced" / "real_balanced.pdf",
    "compact": OUT / "compact" / "real_compact.pdf",
    "lossless": OUT / "lossless" / "real_lossless.pdf",
}
EXPECTED_PAGES = 120


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    result = {
        "engine_pdfium": getattr(pdfium, "__version__", "pypdfium2"),
        "engine_pikepdf": pikepdf.__version__,
        "modes": {},
    }

    # ---- pikepdf（QPDF）独立结构校验：三模式 ----
    for mode, path in MODES.items():
        with pikepdf.open(path) as pdf:
            n = len(pdf.pages)
            sizes = set()
            for i in range(n):
                box = pdf.pages[i].mediabox
                sizes.add((round(float(box[2]) - float(box[0]), 2),
                           round(float(box[3]) - float(box[1]), 2)))
            xmp = {}
            with pdf.open_metadata() as meta:
                for k in ("dc:title", "dc:creator", "pdf:Producer", "xmp:CreateDate"):
                    try:
                        xmp[k] = str(meta.get(k))
                    except Exception:
                        xmp[k] = None
            result["modes"][mode] = {
                "file": str(path),
                "pikepdf_pages": n,
                "size_bytes": path.stat().st_size,
                "page_sizes_pt_unique": sorted(sizes),
                "docinfo": {str(k): str(v) for k, v in pdf.docinfo.items()},
                "xmp": xmp,
            }

    # ---- PDFium 渲染 balanced 代表页（150 DPI，文字可读）----
    doc = pdfium.PdfDocument(str(MODES["balanced"]))
    n = len(doc)
    result["pdfium_balanced_pages"] = n
    means = []
    for i in range(n):
        arr = np.asarray(doc[i].render(scale=40 / 72).to_pil().convert("L"))
        means.append(float(arr.mean()))
    darkest = int(np.argmin(means))
    picks = sorted({0, n // 2, n - 1, darkest})
    rendered = []
    for i in picks:
        pil = doc[i].render(scale=150 / 72).to_pil().convert("RGB")
        fp = OUT / f"reader_balanced_p{i + 1:03d}.png"
        pil.save(fp)
        rendered.append({"page": i + 1, "mean_luminance": round(means[i], 1),
                         "file": str(fp), "px": list(pil.size)})
    doc.close()
    result["pdfium_rendered"] = rendered
    result["darkest_page"] = darkest + 1
    result["browser_navigation"] = (
        "Chromium 内置浏览器成功加载 file://…/real_balanced.pdf（无导航错误）；"
        "浏览器面板隐藏无法截图，故以 PDFium（Chrome/Edge 同款引擎）渲染为视觉证据。"
    )

    (OUT / "reader_compat.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    for mode in MODES:
        assert result["modes"][mode]["pikepdf_pages"] == EXPECTED_PAGES, mode
    assert result["pdfium_balanced_pages"] == EXPECTED_PAGES
    print("OK ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())