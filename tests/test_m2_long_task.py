"""M2.4 长任务：50 页混合样本整流水线处理，成功率/页序/内存/耗时/Manifest/回退。"""

import ctypes
import json
from ctypes import wintypes

import numpy as np
from PIL import Image

from bookclean.presets import get_preset
from bookclean.processing_pipeline import ProcessingRunner
from bookclean.task_manager import ImportTaskManager
from scripts.make_m2_samples import (
    black_border, color_content, perspective, rotate, shadow, specks, text_page, yellow,
)

PAGE_COUNT = 50


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def working_set_mb() -> float:
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    try:
        get_info = kernel32.K32GetProcessMemoryInfo
    except AttributeError:
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
    get_info.restype = wintypes.BOOL
    counters = _PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    assert get_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return counters.WorkingSetSize / 1024**2


def _mixed_pages(n):
    base, m = text_page()
    builders = [
        lambda: base,
        lambda: yellow(base, m, paper=(232, 210, 150), noise=2.0)[0],
        lambda: rotate(base, m, 3.0)[0],
        lambda: black_border(base, m)[0],
        lambda: shadow(base, m, "spine", 0.35)[0],
        lambda: color_content(base, m)[0],
        lambda: specks(base, m, n=80)[0],
        lambda: perspective(base, m)[0],
        lambda: rotate(base, m, -4.0)[0],
        lambda: yellow(base, m, paper=(248, 240, 214))[0],
    ]
    return [builders[i % len(builders)]() for i in range(n)]


def test_50_page_mixed_pipeline(tmp_path):
    in_dir = tmp_path / "inputs"
    in_dir.mkdir()
    files = []
    for i, rgb in enumerate(_mixed_pages(PAGE_COUNT)):
        p = in_dir / f"page_{i + 1:03d}.png"
        Image.fromarray(rgb).save(p, dpi=(150, 150))
        files.append(p)
    imported = ImportTaskManager(tmp_path / "tasks").run(files, dpi=150)
    task_dir = imported["task_dir"]

    samples = [working_set_mb()]

    def progress(entry):
        if entry["page_index"] % 10 == 0:
            samples.append(working_set_mb())

    runner = ProcessingRunner()
    res = runner.run(task_dir, get_preset("auto"), progress_cb=progress)
    samples.append(working_set_mb())
    s = res["summary"]

    assert s["processed"] == PAGE_COUNT, s
    assert s["failed"] == 0
    manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
    assert [p["page_index"] for p in manifest["pages"]] == list(range(PAGE_COUNT))
    # 页序：processed 文件与 display_index 一一对应
    for entry in manifest["pages"]:
        assert entry["processed_image_path"] == f"processed_pages/page_{entry['display_index']:06d}.png"
        assert (task_dir / entry["processed_image_path"]).exists()
    assert len(list((task_dir / "processed_pages").glob("*.png"))) == PAGE_COUNT
    assert len(list((task_dir / "original_pages").glob("*.png"))) == PAGE_COUNT
    assert manifest["processing_status"] == "completed"

    growth = samples[-1] - samples[0]
    peak = max(samples)
    print(f"\n[M2 长任务] 内存样本(MB): {[round(x, 1) for x in samples]}")
    print(f"[M2 长任务] 增长 {growth:.1f} MB, 峰值 {peak:.1f} MB")
    print(f"[M2 长任务] 成功 {s['processed']}/{PAGE_COUNT}, 回退 {s['fallbacks']}, "
          f"决策 {s['decisions']}, 耗时 {s['elapsed_seconds']}s, 平均 {s['avg_page_ms']}ms/页")
    assert growth < 100, f"内存持续增长 {growth:.1f} MB"
    assert peak - samples[0] < 250, f"内存峰值异常 {peak:.1f} MB"