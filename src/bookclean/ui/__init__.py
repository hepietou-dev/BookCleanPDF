"""Tkinter / ttk 界面层（M5 实现）。UI 不直接执行图像算法，只经 AppController 调用内核。"""

from __future__ import annotations


def run_gui() -> int:
    """启动桌面 GUI。返回进程退出码。DPI 感知必须在创建 Tk root 之前声明。"""
    import tkinter as tk

    from ..logging_setup import setup_logging
    from ..user_paths import UserPaths
    from .dpi import set_dpi_awareness
    from .main_window import MainWindow

    paths = UserPaths()
    paths.ensure_dirs()
    setup_logging(paths.logs_dir)
    set_dpi_awareness()
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()
    return 0