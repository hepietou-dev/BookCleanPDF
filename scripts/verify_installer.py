"""M5.5 安装程序实测：静默安装→体积→运行→静默卸载→核验用户数据/输出不被删除。

per-user 安装（PrivilegesRequired=lowest），无需 UAC/命令行。
哨兵文件代表"用户生成的输出 PDF"与"用户任务数据"，卸载后必须仍在。
结果写 temp/m5_review/installer_verify.json。
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bookclean import __version__  # noqa: E402

INSTALLER = ROOT / "dist" / "installer" / f"BookCleanPDF-{__version__}-setup.exe"
LOCAL = Path(os.environ["LOCALAPPDATA"])
INSTALL_DIR = LOCAL / "Programs" / "BookCleanPDF"
DATA_DIR = LOCAL / "BookCleanPDF"
OUT_DIR = Path.home() / "Documents" / "BookCleanPDF"


def _dir_size(p: Path):
    total = count = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
            count += 1
    return total, count


def main() -> int:
    report = {}
    report["installer_size_bytes"] = INSTALLER.stat().st_size

    created_out = not OUT_DIR.exists()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sentinel_pdf = OUT_DIR / "_m55_uninstall_sentinel.pdf"
    sentinel_data = DATA_DIR / "_m55_uninstall_sentinel.txt"
    sentinel_pdf.write_bytes(b"%PDF-1.4 m55 uninstall sentinel\n%%EOF")
    sentinel_data.write_text("m55 user-data sentinel", encoding="utf-8")

    log = ROOT / "temp" / "m5_review" / "install.log"
    p = subprocess.run([str(INSTALLER), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NOICONS",
                        "/TASKS=", f"/LOG={log}"], timeout=300, capture_output=True)
    report["install_rc"] = p.returncode
    report["install_dir"] = str(INSTALL_DIR)
    report["install_dir_exists"] = INSTALL_DIR.exists()

    if INSTALL_DIR.exists():
        size, count = _dir_size(INSTALL_DIR)
        report["installed_size_bytes"] = size
        report["installed_size_mb"] = round(size / 1024 / 1024, 2)
        report["installed_file_count"] = count
        exe = INSTALL_DIR / "BookCleanPDF.exe"
        v = subprocess.run([str(exe), "--version"], timeout=60, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        report["installed_version_rc"] = v.returncode
        report["installed_version_stdout"] = v.stdout.strip()
        d = subprocess.run([str(exe), "--diagnose"], timeout=120, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        report["installed_diagnose_rc"] = d.returncode

    unins = INSTALL_DIR / "unins000.exe"
    ulog = ROOT / "temp" / "m5_review" / "uninstall.log"
    u = subprocess.run([str(unins), "/VERYSILENT", "/SUPPRESSMSGBOXES", f"/LOG={ulog}"],
                       timeout=300, capture_output=True)
    report["uninstall_rc"] = u.returncode
    # 卸载器在主进程退出后约 0.5s 才重试删除 {app} 目录（error 145），需短暂轮询。
    deadline = time.monotonic() + 15
    while INSTALL_DIR.exists() and time.monotonic() < deadline:
        time.sleep(0.5)
    report["install_dir_removed"] = not INSTALL_DIR.exists()
    report["user_output_preserved"] = sentinel_pdf.exists()
    report["user_data_preserved"] = sentinel_data.exists()

    sentinel_pdf.unlink(missing_ok=True)
    sentinel_data.unlink(missing_ok=True)
    if created_out:
        try:
            OUT_DIR.rmdir()
        except OSError:
            pass

    ok = (report["install_rc"] == 0 and report.get("installed_diagnose_rc") == 0
          and report["uninstall_rc"] == 0 and report["install_dir_removed"]
          and report["user_output_preserved"] and report["user_data_preserved"])
    report["ok"] = ok
    out = ROOT / "temp" / "m5_review" / "installer_verify.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())