"""原子文件写入：先写同目录临时文件，再 os.replace，避免异常退出留下半个 JSON。"""

import io
import json
import os
import tempfile
import time
from pathlib import Path


def _replace_with_retry(tmp_name: str, path: Path, attempts: int = 6) -> None:
    # Windows 上 os.replace 可能因杀毒实时扫描/索引器短暂占用目标而抛 PermissionError；
    # 短退避重试（仅针对该瞬态错误），避免逐页频繁写 manifest 时偶发失败。
    for i in range(attempts):
        try:
            os.replace(tmp_name, path)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.05 * (i + 1))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _replace_with_retry(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))


def atomic_save_png(im, path: Path, dpi: float | None = None) -> None:
    """原子保存 PNG：先在内存完整编码，再临时文件 + fsync + os.replace。

    中断/磁盘满时不会留下半个 PNG（要么旧文件、要么完整新文件）。任意时刻仅一页像素。
    """
    buf = io.BytesIO()
    kwargs = {"format": "PNG"}
    if dpi is not None:
        kwargs["dpi"] = (dpi, dpi)
    im.save(buf, **kwargs)
    atomic_write_bytes(path, buf.getvalue())