"""实测 PyInstaller dist 体积构成（说明书 §13.1，不以估算代替）。

报告：dist 总体积/文件数、最大 20 文件、按顶层组件的体积占比、是否混入开发/测试/缓存/模型。
用法：.venv/Scripts/python.exe scripts/measure_dist.py [dist_dir]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist" / "BookCleanPDF"

_LEAK_MARKERS = ("pytest", "_pytest", "pygments", "pip/", "pip-", "__pycache__",
                 "/tests/", "site-packages", ".pyc", "numpy/f2py", "models/")


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} GB"


def main() -> int:
    if not DIST.is_dir():
        print(f"dist 目录不存在: {DIST}")
        return 1
    files = [p for p in DIST.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    top = sorted(files, key=lambda p: p.stat().st_size, reverse=True)[:20]

    # 按顶层组件归类（onedir 6.x 资源在 _internal/ 下）
    comp = defaultdict(int)
    comp_count = defaultdict(int)
    for p in files:
        rel = p.relative_to(DIST)
        parts = rel.parts
        if parts[0] == "_internal" and len(parts) > 1:
            key = parts[1]
        else:
            key = parts[0]
        comp[key] += p.stat().st_size
        comp_count[key] += 1
    comp_sorted = sorted(comp.items(), key=lambda kv: kv[1], reverse=True)

    leaks = []
    for p in files:
        rel = p.relative_to(DIST).as_posix().lower()
        if any(m in rel for m in _LEAK_MARKERS):
            leaks.append(p.relative_to(DIST).as_posix())

    report = {
        "dist_dir": str(DIST),
        "total_bytes": total,
        "total_human": _human(total),
        "file_count": len(files),
        "top20_files": [
            {"path": p.relative_to(DIST).as_posix(), "bytes": p.stat().st_size,
             "human": _human(p.stat().st_size)} for p in top
        ],
        "components": [
            {"name": k, "bytes": v, "human": _human(v), "files": comp_count[k],
             "pct": round(100.0 * v / total, 2)} for k, v in comp_sorted
        ],
        "leak_candidates": leaks[:200],
        "leak_count": len(leaks),
        "over_200mb": total > 200 * 1024 * 1024,
    }
    out = ROOT / "temp" / "m5_review" / "dist_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"dist 总体积: {_human(total)} ({total} bytes), 文件数 {len(files)}")
    print(f"超 200MB 目标: {report['over_200mb']}")
    print("最大 20 文件:")
    for t in report["top20_files"]:
        print(f"  {t['human']:>12}  {t['path']}")
    print("体积构成（前 15 组件）:")
    for c in report["components"][:15]:
        print(f"  {c['human']:>12}  {c['pct']:5.1f}%  {c['name']} ({c['files']} files)")
    print(f"开发/测试/缓存/模型泄漏候选: {len(leaks)}")
    for l in leaks[:20]:
        print(f"  {l}")
    print(f"报告: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())