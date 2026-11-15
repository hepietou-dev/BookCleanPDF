"""零依赖 Windows 文件拖拽（WM_DROPFILES）。

不引入 windnd / tkinterdnd2（未安装，且项目偏好零新运行时依赖）。做法：
用 ctypes 子类化 toplevel 窗口过程，接收 shell32 的 WM_DROPFILES，解析拖入路径，
再用 widget.after(0, ...) 把回调投递回 Tk 事件循环（窗口过程本就在 Tk 主线程触发，
after 仅为避免消息处理期间的重入）。任何一步失败都优雅降级为“仅按钮选择”。

注意：DragAcceptFiles/DragQueryFileW/DragFinish 由 **shell32** 导出（不是 user32）。
回调对象必须被持久引用，否则被 GC 回收后窗口过程指针悬空会导致崩溃。
"""

from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes

logger = logging.getLogger("bookclean")

WM_DROPFILES = 0x0233
GWLP_WNDPROC = -4

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)


def _query_drop_paths(hdrop) -> list[str]:
    shell32 = ctypes.windll.shell32
    shell32.DragQueryFileW.argtypes = [wintypes.HANDLE, wintypes.UINT, wintypes.LPWSTR, wintypes.UINT]
    shell32.DragQueryFileW.restype = wintypes.UINT
    count = shell32.DragQueryFileW(hdrop, 0xFFFFFFFF, None, 0)
    paths: list[str] = []
    for i in range(count):
        length = shell32.DragQueryFileW(hdrop, i, None, 0)
        buf = ctypes.create_unicode_buffer(length + 1)
        shell32.DragQueryFileW(hdrop, i, buf, length + 1)
        if buf.value:
            paths.append(buf.value)
    shell32.DragFinish(hdrop)
    return paths


class DropTarget:
    """持有一个已子类化的拖拽目标；disable() 还原窗口过程并释放引用。"""

    def __init__(self, widget, hwnd, old_proc, new_proc):
        self._widget = widget
        self._hwnd = hwnd
        self._old_proc = old_proc
        self._new_proc = new_proc  # 持久引用，防止 GC 回收回调
        self._enabled = True

    def disable(self) -> None:
        if not self._enabled:
            return
        self._enabled = False
        try:
            user32 = ctypes.windll.user32
            user32.SetWindowLongPtrW(self._hwnd, GWLP_WNDPROC, self._old_proc)
            ctypes.windll.shell32.DragAcceptFiles(self._hwnd, False)
        except (AttributeError, OSError) as exc:
            logger.warning("还原拖拽窗口过程失败: %s", exc)
        self._new_proc = None


def enable_drop(widget, on_files) -> DropTarget | None:
    """为 widget 所属 toplevel 启用文件拖拽。on_files(list[str]) 在 Tk 主线程被调用。

    成功返回 DropTarget（调用方需持有引用，关闭时 disable()）；失败返回 None（降级为按钮选择）。
    """
    if sys.platform != "win32":
        return None
    try:
        user32 = ctypes.windll.user32
        shell32 = ctypes.windll.shell32

        inner = widget.winfo_id()
        hwnd = user32.GetParent(inner) or inner  # Tk toplevel 的真实 HWND 是其父窗口
        if not hwnd:
            return None

        user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.GetWindowLongPtrW.restype = LRESULT
        user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, LRESULT]
        user32.SetWindowLongPtrW.restype = LRESULT
        user32.CallWindowProcW.argtypes = [LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.CallWindowProcW.restype = LRESULT
        shell32.DragAcceptFiles.argtypes = [wintypes.HWND, wintypes.BOOL]

        old_proc = user32.GetWindowLongPtrW(hwnd, GWLP_WNDPROC)

        def _wndproc(h, msg, wparam, lparam):
            if msg == WM_DROPFILES:
                try:
                    paths = _query_drop_paths(wparam)
                    if paths:
                        widget.after(0, lambda: on_files(paths))
                except (OSError, ValueError) as exc:
                    logger.warning("解析拖拽文件失败: %s", exc)
                return 0
            return user32.CallWindowProcW(old_proc, h, msg, wparam, lparam)

        new_proc = WNDPROC(_wndproc)
        shell32.DragAcceptFiles(hwnd, True)
        user32.SetWindowLongPtrW(hwnd, GWLP_WNDPROC, ctypes.cast(new_proc, ctypes.c_void_p).value)
        logger.info("已启用文件拖拽 (hwnd=%s)", hwnd)
        return DropTarget(widget, hwnd, old_proc, new_proc)
    except (AttributeError, OSError, ValueError) as exc:
        logger.warning("启用拖拽失败，降级为按钮选择: %s", exc)
        return None