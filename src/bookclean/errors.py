"""统一异常类型。所有输入错误必须带明确上下文（文件、页码、原因）。"""


class BookCleanError(Exception):
    """项目异常基类。"""


class InputError(BookCleanError):
    """输入文件层面的错误。"""


class UnsupportedFormatError(InputError):
    """不支持的文件格式。"""


class PDFError(InputError):
    """PDF 打开、结构或渲染错误。"""


class PDFPasswordError(PDFError):
    """PDF 受密码保护或密码错误。"""


class PDFCorruptError(PDFError):
    """PDF 损坏或无法解析。"""


class ImageLoadError(InputError):
    """图片读取或解码错误。"""