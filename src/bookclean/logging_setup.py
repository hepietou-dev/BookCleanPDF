"""本地日志。只记录状态/耗时/错误，不记录图像像素数据，不上传任何信息。"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("bookclean")
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter(_FORMAT)
    file_handler = RotatingFileHandler(
        log_dir / "bookclean.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # 打包 --windowed（pythonw）下 sys.stderr 可能为 None，此时不挂控制台句柄，避免日志发射崩溃。
    if sys.stderr is not None:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger