# -*- mode: python ; coding: utf-8 -*-
"""BookClean PDF — PyInstaller onedir 打包规格（M5.5）。

策略（说明书 §13.2）：onedir、windowed GUI、用 .spec 明确收集资源与依赖。
- 收集 bookclean 全部子模块（含惰性导入的 ui/frozen_smoke）、tcl/tk、运行时二进制
  （cv2/numpy/pikepdf/pypdfium2/lxml/PIL/img2pdf 由各 hook 自动收集）、assets、第三方许可证。
- 排除开发/测试依赖（pytest/_pytest/pygments/pip/wheel）与 stdlib 测试包。
- 体积优化（经代码审计正当排除，非盲删 DLL）：内核无任何 VideoCapture/VideoWriter/
  CascadeClassifier/cv2.data 使用（grep 核实），故排除 cv2 的 videoio ffmpeg DLL（~30MB）
  与 cv2/data 级联数据（~9MB）。排除后必须经 --smoke 全链路 + 全量回归验证不破坏运行。
- 不启用 UPX（可能破坏 cv2/pikepdf DLL；说明书 §8.4 谨慎评估）。
"""

import os

from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

hiddenimports = collect_submodules("bookclean") + [
    "pikepdf", "lxml.etree", "lxml._elementtree",
    "pypdfium2", "pypdfium2_raw", "PIL", "img2pdf", "cv2", "numpy", "packaging",
]

a = Analysis(
    [os.path.join(ROOT, "app.py")],
    pathex=[os.path.join(ROOT, "src")],
    binaries=[],
    datas=[
        (os.path.join(ROOT, "assets"), "assets"),
        (os.path.join(ROOT, "THIRD_PARTY_LICENSES"), "THIRD_PARTY_LICENSES"),
        # M7.5：OCR worker 脚本必须作为真实文件落盘，供插件独立 Python 子进程运行
        # （rapid_provider.WORKER_SCRIPT = Path(__file__).parent / "rapid_worker.py"，
        #  frozen 下 __file__ 位于 _internal/bookclean/ocr/，故 dest 同为 bookclean/ocr）。
        (os.path.join(ROOT, "src", "bookclean", "ocr", "rapid_worker.py"), "bookclean/ocr"),
        (os.path.join(ROOT, "src", "bookclean", "ocr", "paddle_bridge_worker.py"), "bookclean/ocr"),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest", "_pytest", "pygments", "pip", "wheel",
        "tkinter.test", "test", "unittest", "doctest", "pdb", "pydoc",
        "numpy.f2py", "numpy.distutils",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)


def _excluded_cv2(item) -> bool:
    """正当排除 cv2 videoio ffmpeg DLL 与 cv2/data（内核无视频 I/O、无级联分类器）。"""
    name = str(item[0]).replace("\\", "/").lower()
    path = str(item[1]).replace("\\", "/").lower()
    if "opencv_videoio_ffmpeg" in name or "opencv_videoio_ffmpeg" in path:
        return True
    if "/cv2/data/" in path or name.startswith("cv2/data/") or "/cv2/data/" in name:
        return True
    return False


a.binaries = [b for b in a.binaries if not _excluded_cv2(b)]
a.datas = [d for d in a.datas if not _excluded_cv2(d)]

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BookCleanPDF",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="BookCleanPDF",
)