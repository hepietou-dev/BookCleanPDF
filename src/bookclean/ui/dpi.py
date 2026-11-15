"""Windows 高 DPI 适配。

set_dpi_awareness() 必须在创建 Tk root **之前**调用，否则 Windows 会按 96DPI 虚拟化拉伸导致模糊。
apply_scaling() 在 root 创建后按实际 DPI 设置 tk scaling，使字体/控件尺寸与系统一致。
非 Windows 或调用失败时静默降级（不影响功能）。
"""

from __future__ import annotations

import ctypes
import sys


def set_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def apply_scaling(root) -> None:
    import tkinter

    try:
        dpi = root.winfo_fpixels("1i")
        root.tk.call("tk", "scaling", dpi / 72.0)
    except tkinter.TclError:
        pass