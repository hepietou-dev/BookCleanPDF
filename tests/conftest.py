"""M1 测试夹具：全部样本由程序离线生成，不下载任何在线文件。"""

import gc
import io

import img2pdf
import pytest
from PIL import Image

_TK_ROOT = None  # 由 session 级 tk_root 登记，供 autouse 清理判断解释器是否存活


def png_bytes(im: Image.Image, dpi: int = 300) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG", dpi=(dpi, dpi))
    return buf.getvalue()


@pytest.fixture
def pdf_factory(tmp_path):
    def make(images, name="sample.pdf", dpi=300):
        path = tmp_path / name
        path.write_bytes(img2pdf.convert(*[png_bytes(im, dpi) for im in images]))
        return path

    return make


@pytest.fixture
def img_factory(tmp_path):
    def make(name, mode="RGB", size=(40, 20), color=(255, 0, 0), **save_kwargs):
        im = Image.new(mode, size, color)
        path = tmp_path / name
        im.save(path, **save_kwargs)
        return path

    return make


@pytest.fixture
def solid_pdf_pages():
    """生成指定数量的纯色页，颜色由页索引决定，用于验证页序。"""

    def make(count, size=(400, 560)):
        return [
            Image.new("RGB", size, (i % 256, (i * 7) % 256, (i * 13) % 256))
            for i in range(count)
        ]

    return make


@pytest.fixture(scope="session")
def tk_root():
    """单一 Tk 解释器（session 级），供所有 GUI 测试模块共享。

    Windows 上反复创建 Tk() 解释器会偶发 init.tcl/tcl_findLibrary 初始化抖动，
    故整个测试会话只创建一个；各用例用 Toplevel。无可用显示环境时跳过（不伪造通过）。
    """
    global _TK_ROOT
    tk = pytest.importorskip("tkinter")
    from bookclean.ui.dpi import set_dpi_awareness
    set_dpi_awareness()
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"无可用显示环境，跳过 GUI 测试: {exc}")
    root.withdraw()
    _TK_ROOT = root
    yield root
    _TK_ROOT = None
    try:
        root.destroy()
    except tk.TclError:
        pass


@pytest.fixture(autouse=True)
def _collect_tk_garbage():
    """每个用例后在 Tcl 事件处理内回收悬挂的 Tk Variable。

    MainWindow 的 StringVar 随 Toplevel 销毁后仍被 Python 引用，若留待后续用例
    任意分配点触发 __del__，会因不在事件循环内而抛 'main thread is not in main loop'。
    此处在 after 回调内 gc.collect（update 驱动），使 __del__ 在事件处理中安全执行。
    非 GUI 用例（无解释器）直接跳过，不创建 Tk。
    """
    yield
    root = _TK_ROOT
    if root is None:
        return
    tk = pytest.importorskip("tkinter")
    try:
        if not root.winfo_exists():
            return
        root.after(0, gc.collect)
        root.update()
    except tk.TclError:
        pass