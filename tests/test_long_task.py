"""E. 数百页长任务：逐页渲染、页序、内存稳定性（Windows 工作集实测）。"""

import ctypes
from ctypes import wintypes

import numpy as np
from PIL import Image

from bookclean.task_manager import ImportTaskManager

PAGE_COUNT = 300


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def working_set_mb() -> float:
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    try:
        get_info = kernel32.K32GetProcessMemoryInfo  # Win7+ kernel32 导出名
    except AttributeError:
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
    get_info.restype = wintypes.BOOL
    counters = _PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    ok = get_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    assert ok, "GetProcessMemoryInfo 调用失败"
    return counters.WorkingSetSize / 1024**2


def test_300_page_import_memory_and_order(tmp_path, pdf_factory, solid_pdf_pages):
    pdf = pdf_factory(solid_pdf_pages(PAGE_COUNT), name="long.pdf")
    samples = []

    def on_page(entry):
        if entry["page_index"] % 30 == 0:
            samples.append(working_set_mb())

    result = ImportTaskManager(temp_root=tmp_path / "temp").run([pdf], progress_cb=on_page)
    samples.append(working_set_mb())
    manifest = result["manifest"]

    assert manifest["status"] == "completed"
    assert manifest["total_pages"] == PAGE_COUNT
    assert manifest["completed_pages"] == PAGE_COUNT
    assert [p["page_index"] for p in manifest["pages"]] == list(range(PAGE_COUNT))

    for i, entry in enumerate(manifest["pages"]):
        arr = np.asarray(Image.open(result["task_dir"] / entry["image_path"]))
        expected = (i % 256, (i * 7) % 256, (i * 13) % 256)
        assert tuple(arr[280, 200]) == expected, f"第 {i + 1} 页颜色不符，页序可能错乱"

    growth = samples[-1] - samples[0]
    peak = max(samples)
    print(f"\n[长任务内存] 样本(MB): {[round(s, 1) for s in samples]}")
    print(f"[长任务内存] 增长 {growth:.1f} MB, 峰值 {peak:.1f} MB, 耗时 {manifest['elapsed_seconds']}s")
    assert growth < 100, f"内存持续增长 {growth:.1f} MB，疑似泄漏"
    assert peak - samples[0] < 200, f"内存峰值异常 {peak:.1f} MB"