# -*- coding: utf-8 -*-
"""M6.5 插件分发包构建：ocr-rapid 插件目录 → 离线 zip + sha256 清单（Q4：独立可选下载）。

基础安装包不含引擎；本脚本把已验证的插件目录（plugin.json + runtime/）打成单个
离线 zip，并生成同名 .json 清单（zip 的 sha256/尺寸/安装命令），供
install_ocr_plugin.py 校验安装。打包前逐模型 sha256 复核，损坏即拒绝打包。

用法:
  python scripts/build_ocr_plugin.py [--plugin-dir temp/plugins/ocr-rapid/1.4.4]
         [--out temp/m6_review/dist/ocr-rapid-1.4.4-plugin.zip]
退出码: 0 成功 / 1 校验失败 / 2 参数或目录无效。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bookclean.ocr.plugin_manager import PluginManager


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dir_size(path: Path) -> tuple[int, int]:
    total = files = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
            files += 1
    return total, files


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="构建 OCR 插件离线分发包（zip + sha256 清单）")
    parser.add_argument("--plugin-dir", default="temp/plugins/ocr-rapid/1.4.4",
                        help="已安装且验证过的插件目录（含 plugin.json）")
    parser.add_argument("--out", default=None, help="输出 zip 路径（默认 temp/m6_review/dist/<engine>-<ver>-plugin.zip）")
    args = parser.parse_args(argv)

    plugin_dir = Path(args.plugin_dir).resolve()
    mf = plugin_dir / "plugin.json"
    if not mf.is_file():
        print(f"错误: 插件清单不存在: {mf}", file=sys.stderr)
        return 2
    pm = PluginManager(plugin_dir.parent.parent)
    plugin = pm._load_manifest(mf)
    problems = PluginManager.verify_models(plugin)
    if problems:
        for p in problems:
            print("错误: " + p, file=sys.stderr)
        return 1
    print(f"模型校验通过: {plugin.engine_id} {plugin.engine_version}（{len(plugin.models)} 个模型）")

    out = Path(args.out) if args.out else (
        Path("temp/m6_review/dist") / f"{plugin.engine_id}-{plugin.engine_version}-plugin.zip")
    out.parent.mkdir(parents=True, exist_ok=True)

    raw_size, file_count = _dir_size(plugin_dir)
    tmp = out.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(plugin_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(plugin_dir).as_posix())
    tmp.replace(out)
    zip_size = out.stat().st_size
    zip_sha = _sha256(out)

    sidecar = {
        "engine_id": plugin.engine_id,
        "engine_version": plugin.engine_version,
        "file": out.name,
        "size_bytes": zip_size,
        "sha256": zip_sha,
        "uncompressed_bytes": raw_size,
        "file_count": file_count,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "install": f"python scripts/install_ocr_plugin.py {out.name}",
    }
    out.with_suffix(".zip.json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("---- 体积审计 ----")
    print(f"插件目录（未压缩）: {raw_size / 1e6:.2f}MB / {file_count} 文件")
    print(f"分发包（zip）:      {zip_size / 1e6:.2f}MB（压缩率 {zip_size / raw_size * 100:.1f}%）")
    print(f"sha256: {zip_sha}")
    print(f"输出: {out}")
    print(f"清单: {out.with_suffix('.zip.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())