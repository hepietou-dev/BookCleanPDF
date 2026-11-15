"""从实际安装的 wheel 收集第三方许可证文本到 THIRD_PARTY_LICENSES/（M5.5）。

以实际 wheel 内容为准（AGENT.md §13），不仅凭库名判断。逐项合规核对属 M7 发布验证。
用法：.venv/Scripts/python.exe scripts/collect_licenses.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import importlib.metadata as md

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "THIRD_PARTY_LICENSES"
LOCK = ROOT / "requirements-lock.txt"

_NAME_RE = re.compile(r"^(LICENSE|COPYING|NOTICE|LicenseRef)", re.IGNORECASE)
_TEXT_SUFFIX = {".txt", ".md", ""}


def _runtime_deps() -> list[str]:
    deps = []
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        deps.append(re.split(r"[=<>!]", line)[0])
    return deps


def _license_files(dist):
    seen = set()
    for f in dist.files or []:
        s = str(f).replace("\\", "/")
        low = s.lower()
        if "__pycache__" in low or low.endswith((".py", ".pyc", ".xml", ".pyi")):
            continue
        parts = s.split("/")
        fname = parts[-1]
        is_distinfo_lic = ".dist-info/" in s and (
            "/licenses/" in s or _NAME_RE.match(fname))
        is_top_lic = _NAME_RE.match(fname) and Path(fname).suffix.lower() in _TEXT_SUFFIX
        if not (is_distinfo_lic or is_top_lic):
            continue
        if "license" not in low and "copying" not in low and "notice" not in low and "licenseref" not in low:
            continue
        if s in seen:
            continue
        seen.add(s)
        yield f, s


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for dep in _runtime_deps():
        try:
            dist = md.distribution(dep)
        except md.PackageNotFoundError:
            print(f"跳过（未安装）: {dep}")
            continue
        meta = dist.metadata
        target = OUT / dep
        target.mkdir(parents=True, exist_ok=True)
        collected = []
        for f, rel in _license_files(dist):
            src = Path(dist.locate_file(f))
            if not src.is_file():
                continue
            flat = rel.replace("/", "__")
            dst = target / flat
            shutil.copyfile(src, dst)
            collected.append(flat)
        entry = {
            "name": dep,
            "version": dist.version,
            "license_metadata": meta.get("License", "") or "",
            "license_expression": meta.get("License-Expression", "") or "",
            "homepage": meta.get("Home-page", "") or "",
            "collected_files": collected,
        }
        manifest.append(entry)
        print(f"{dep} {dist.version}: {len(collected)} 个许可证文件")

    # Python 运行时（PyInstaller 打包 python312.dll）PSF 许可
    py_lic = Path(sys.base_prefix) / "LICENSE.txt"
    if py_lic.is_file():
        pydir = OUT / "python-runtime"
        pydir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(py_lic, pydir / "PSF-LICENSE.txt")
        manifest.append({"name": "python-runtime", "version": sys.version.split()[0],
                         "license_metadata": "PSF License", "collected_files": ["PSF-LICENSE.txt"]})
        print(f"python-runtime {sys.version.split()[0]}: PSF LICENSE")

    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"清单: {OUT / 'manifest.json'}（{len(manifest)} 项）")
    print("注意：Tcl/Tk 许可与逐项合规核对属 M7 发布验证。")
    return 0


if __name__ == "__main__":
    sys.exit(main())