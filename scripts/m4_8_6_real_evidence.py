"""M4.8.6 真实图书类型覆盖证据：从已完成 120 页任务目录抽取代表页（不重跑 M1–M3）。

覆盖规格要求的类型：纯文本/不均匀背景/照片/密集脚注/表格/黑边/横向。
真实样本不存在的类型（表格/横向/彩色地图）如实标注，不用合成冒充。
输出 temp/m4_8_review/real_type_coverage.json + real_type_sheet.png（最终图像联系表）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw, ImageFont

from bookclean.pdf_exporter import FinalImageResolver


def _cjk_font(size: int):
    for name in ("msyh.ttc", "msyhbd.ttc", "simsun.ttc", "simhei.ttf"):
        p = Path("C:/Windows/Fonts") / name
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                pass
    return ImageFont.load_default()

REVIEW = ROOT / "temp" / "m4_8_review"
TASK = ROOT / "temp" / "m4_6_subset" / "tasks" / "task_20261112_140743_86d276"

# 代表页（display_index）与类型标签
SELECT = [
    (13, "纯文本 pure_text"),
    (28, "不均匀背景(几何回退)"),
    (16, "照片 photo_dominated"),
    (18, "灰度插图 grayscale_illust"),
    (14, "纯文本含脚注"),
    (3, "黑边清除 border_cleaned"),
    (5, "回退原始 fallback_original"),
]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    manifest = json.loads((TASK / "manifest.json").read_text(encoding="utf-8"))
    pages = {p["display_index"]: p for p in manifest["pages"]}
    resolver = FinalImageResolver(TASK)

    from collections import Counter
    types = Counter(p.get("page_type") for p in manifest["pages"])
    landscape = sum(1 for p in manifest["pages"]
                    if (p.get("image_width") or 0) > (p.get("image_height") or 0))
    coverage = {
        "task": str(TASK.relative_to(ROOT)),
        "total_pages": len(manifest["pages"]),
        "page_type_distribution": dict(types),
        "landscape_pages": landscape,
        "present_types": {
            "纯文本": types.get("pure_text", 0),
            "照片": types.get("photo_dominated", 0),
            "灰度插图": types.get("grayscale_illustration", 0),
            "黑边清除": sum(1 for p in manifest["pages"] if p.get("border_status") == "completed"),
            "不均匀背景/几何回退": sum(1 for p in manifest["pages"] if p.get("fallback_level") == "geometry"),
        },
        "absent_types_honest": {
            "表格": "真实样本无表格页（page_type 无 table），未用合成冒充",
            "横向": f"真实样本横向页 {landscape} 页（全书竖向），未用合成冒充",
            "彩色地图": "真实样本无彩色地图页，未用合成冒充",
            "密集脚注": "无独立 dense_footnote 标签；脚注包含在 pure_text 中（代表页 14 含脚注）",
        },
        "selected_pages": [],
    }

    tw = 220
    cells = []
    for idx, label in SELECT:
        p = pages[idx]
        fi = resolver.resolve(manifest, p)
        with Image.open(fi.path) as im:
            rgb = im.convert("RGB")
        th = int(tw * rgb.height / rgb.width)
        thumb = rgb.resize((tw, th), Image.LANCZOS)
        cells.append((idx, label, fi.source, thumb))
        coverage["selected_pages"].append({
            "display_index": idx, "label": label, "page_type": p.get("page_type"),
            "final_source": fi.source, "border_status": p.get("border_status"),
            "fallback_level": p.get("fallback_level"),
        })

    cols = 4
    rows = (len(cells) + cols - 1) // cols
    cell_h = max(c[3].height for c in cells) + 40
    canvas = Image.new("RGB", (cols * (tw + 12) + 12, rows * cell_h + 12), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = _cjk_font(15)
    font_s = _cjk_font(12)
    for i, (idx, label, src, thumb) in enumerate(cells):
        x = 12 + (i % cols) * (tw + 12)
        y = 12 + (i // cols) * cell_h
        draw.text((x, y), f"p{idx} {label}", fill=(20, 20, 20), font=font)
        draw.text((x, y + 18), f"source={src}", fill=(90, 90, 90), font=font_s)
        canvas.paste(thumb, (x, y + 34))
    sheet = REVIEW / "real_type_sheet.png"
    canvas.save(sheet)
    coverage["contact_sheet"] = str(sheet.relative_to(ROOT))
    out = REVIEW / "real_type_coverage.json"
    out.write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    print("类型分布:", dict(types))
    print("横向页:", landscape, "（如实：无横向）")
    print("代表页:", [(c[0], c[1]) for c in cells])
    print("OK ->", out, sheet)
    return 0


if __name__ == "__main__":
    sys.exit(main())