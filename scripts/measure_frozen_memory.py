"""实测 frozen 进程长任务峰值内存（ctypes GetProcessMemoryInfo，不依赖 psutil）。

用法: python scripts/measure_frozen_memory.py [pages]
以 BOOKCLEAN_SMOKE_PAGES=pages 启动 dist/BookCleanPDF/BookCleanPDF.exe --smoke，
运行期间采样工作集/私有字节峰值，结束后读取 smoke_result.json 汇总写
temp/m5_review/frozen_memory.json。默认 40 页/任务 × 4 组合 = 长任务。
"""

import ctypes
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "dist" / "BookCleanPDF" / "BookCleanPDF.exe"


class _PMC_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t),
    ]


def main() -> int:
    pages = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    env = dict(os.environ, BOOKCLEAN_SMOKE_PAGES=str(pages))
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)

    t0 = time.monotonic()
    proc = subprocess.Popen([str(EXE), "--smoke"], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    handle = kernel32.OpenProcess(0x0400 | 0x0010, False, proc.pid)
    if not handle:
        handle = kernel32.OpenProcess(0x1000, False, proc.pid)

    peak_os = peak_sampled = peak_private = 0
    samples = 0
    while proc.poll() is None:
        if handle:
            counters = _PMC_EX()
            counters.cb = ctypes.sizeof(_PMC_EX)
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                peak_os = max(peak_os, counters.PeakWorkingSetSize)
                peak_sampled = max(peak_sampled, counters.WorkingSetSize)
                peak_private = max(peak_private, counters.PrivateUsage)
                samples += 1
        time.sleep(0.4)
    rc = proc.returncode
    elapsed = round(time.monotonic() - t0, 1)
    if handle:
        kernel32.CloseHandle(handle)
    proc.communicate()

    data_root = Path(os.environ["LOCALAPPDATA"]) / "BookCleanPDF"
    result = json.loads((data_root / "smoke_result.json").read_text(encoding="utf-8"))
    report = {
        "pages_per_task": pages,
        "matrix_tasks": len(result["runs"]),
        "total_page_processings": sum(r.get("processed", 0) for r in result["runs"]),
        "smoke_ok": result["ok"],
        "exit_code": rc,
        "elapsed_seconds": elapsed,
        "peak_working_set_mb": round(peak_os / 1024 / 1024, 1),
        "peak_sampled_working_set_mb": round(peak_sampled / 1024 / 1024, 1),
        "peak_private_mb": round(peak_private / 1024 / 1024, 1),
        "samples": samples,
        "runs_ok": [r["ok"] for r in result["runs"]],
        "resume_reused": result["resume_reuse"].get("reused"),
        "originals_unchanged": result["originals_unchanged"],
    }
    out = ROOT / "temp" / "m5_review" / "frozen_memory.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if (result["ok"] and rc == 0) else 1


if __name__ == "__main__":
    sys.exit(main())