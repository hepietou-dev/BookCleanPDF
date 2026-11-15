# -*- coding: utf-8 -*-
"""M6.5 插件离线安装：zip → 校验 sha256 → 解压到插件目录 → 发现确认（Q4）。

安全：zip 成员路径防穿越（zip-slip）；有同名 .json 清单或 --sha256 时强制校验；
目标目录已存在时拒绝覆盖（插件可整目录删除后重装，或 --force 先移走旧目录为 .bak）。

用法:
  python scripts/install_ocr_plugin.py ocr-rapid-1.4.4-plugin.zip
         [--plugins-root temp/plugins] [--sha256 <期望值>] [--force]
退出码: 0 安装成功 / 1 校验失败 / 2 参数或包无效。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bookclean.ocr.plugin_manager import PluginManager


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        if not str(target).startswith(str(dest) + "\\") and target != dest \
                and not str(target).startswith(str(dest) + "/"):
            raise ValueError(f"zip 成员路径越界（zip-slip）: {member.filename}")
    zf.extractall(dest)


def _robust_rename(src: Path, dst: Path, timeout: float = 30.0) -> None:
    """staging→final 原子重命名，对 Windows 瞬时锁退避重试。

    自包含 runtime 含真实 python.exe 与大量 DLL，解压后杀毒软件（Defender）会扫描
    并短暂锁定目录，立即 os.rename 会抛 WinError 5/32。实测锁约数秒后释放，故重试。
    仅对访问拒绝/共享 violation 重试；其它错误（如跨卷）立即抛出。
    """
    deadline = time.monotonic() + timeout
    delay = 0.5
    while True:
        try:
            os.rename(src, dst)
            return
        except OSError as exc:
            winerr = getattr(exc, "winerror", None)
            transient = isinstance(exc, PermissionError) or winerr in (5, 32)
            if not transient or time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 1.5, 3.0)


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="离线安装 OCR 插件 zip 包")
    parser.add_argument("zip_path", help="插件 zip 包（build_ocr_plugin.py 产物）")
    parser.add_argument("--plugins-root", default=None, help="插件根目录（默认用户数据目录/plugins）")
    parser.add_argument("--sha256", default=None, help="期望的 zip sha256（默认读同名 .json 清单）")
    parser.add_argument("--force", action="store_true", help="目标已存在时先移走为 .bak 再安装")
    args = parser.parse_args(argv)

    zip_path = Path(args.zip_path).resolve()
    if not zip_path.is_file():
        print(f"错误: 分发包不存在: {zip_path}", file=sys.stderr)
        return 2

    expected = args.sha256
    sidecar = zip_path.with_suffix(".zip.json")
    if expected is None and sidecar.is_file():
        expected = json.loads(sidecar.read_text(encoding="utf-8")).get("sha256")
        print(f"使用清单校验: {sidecar.name}")
    if expected:
        actual = _sha256(zip_path)
        if actual != expected:
            print(f"错误: sha256 不匹配（包可能损坏或被替换）\n  期望 {expected}\n  实际 {actual}",
                  file=sys.stderr)
            return 1
        print("sha256 校验通过")
    else:
        print("警告: 无清单且未提供 --sha256，跳过完整性校验（不推荐）", file=sys.stderr)

    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        print(f"错误: 不是合法 zip: {exc}", file=sys.stderr)
        return 2
    with zf:
        if "plugin.json" not in zf.namelist():
            print("错误: zip 根部缺少 plugin.json（不是 BookClean 插件包）", file=sys.stderr)
            return 2
        manifest = json.loads(zf.read("plugin.json").decode("utf-8"))
        engine_id = str(manifest.get("engine_id") or "")
        version = str(manifest.get("engine_version") or "")
        if not engine_id or not version:
            print("错误: plugin.json 缺少 engine_id/engine_version", file=sys.stderr)
            return 2

        plugins_root = Path(args.plugins_root) if args.plugins_root else PluginManager().plugins_root
        final_dir = plugins_root / engine_id / version
        if final_dir.exists():
            if not args.force:
                print(f"错误: 目标已存在: {final_dir}（删除该目录后重装，或 --force 移走为 .bak）",
                      file=sys.stderr)
                return 2
            bak = final_dir.with_name(version + ".bak")
            if bak.exists():
                shutil.rmtree(bak)
            final_dir.rename(bak)
            print(f"旧目录已移走: {bak}")

        staging = plugins_root / f".staging-{engine_id}-{version}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            try:
                _safe_extract(zf, staging)
            except ValueError as exc:
                print(f"错误: {exc}", file=sys.stderr)
                return 2
            pm = PluginManager(plugins_root)
            info = pm._load_manifest(staging / "plugin.json")
            problems = PluginManager.verify_models(info)
            if problems:
                for p in problems:
                    print("错误: " + p, file=sys.stderr)
                return 1
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            _robust_rename(staging, final_dir)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise

    status = PluginManager(plugins_root).check_available(engine_id)
    if not status.installed:
        print(f"错误: 安装后未发现插件: {status.reason}", file=sys.stderr)
        return 1
    print(f"安装成功: {engine_id} {version} → {final_dir}")
    print("在 BookClean 设置中点击“检测引擎”确认状态为“已就绪”。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())