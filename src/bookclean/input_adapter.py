"""统一输入入口：区分 PDF/图片，文件夹自然排序，显式多文件保留用户顺序。

页序约定：PageData.page_index 为整个任务内 0-based 全局页序；
metadata["source_page_index"] 为源文件内页索引（PDF 页 / TIFF 帧，0-based）。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

from .errors import InputError, UnsupportedFormatError
from .image_loader import SUPPORTED_IMAGE_EXTENSIONS, ImageLoader
from .models import PageData
from .pdf_reader import PDFReader
from .utils.natural_sort import natural_key

OnError = Callable[[int | None, Exception], None]


@dataclass
class InputFile:
    path: Path
    kind: str  # "pdf" / "image"
    page_count: int
    error: str | None = None


def classify_path(path: Path) -> str:
    ext = path.suffix.casefold()
    if ext == ".pdf":
        return "pdf"
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    raise UnsupportedFormatError(f"不支持的文件格式: {path.name}（支持 PDF/JPG/JPEG/PNG/TIFF/BMP）")


class InputAdapter:
    def __init__(self, pdf_reader: PDFReader | None = None, image_loader: ImageLoader | None = None):
        self.pdf_reader = pdf_reader or PDFReader()
        self.image_loader = image_loader or ImageLoader()

    def collect(self, paths: Sequence[str | Path]) -> list[InputFile]:
        """展开输入列表。文件夹内只收支持格式并自然排序；显式文件保留给定顺序。"""
        files: list[Path] = []
        for raw in paths:
            path = Path(raw)
            if path.is_dir():
                children = [
                    p for p in path.iterdir()
                    if p.is_file() and p.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS | {".pdf"}
                ]
                files.extend(sorted(children, key=lambda p: natural_key(p.name)))
            elif path.is_file():
                classify_path(path)
                files.append(path)
            else:
                raise InputError(f"输入路径不存在: {path}")

        inputs: list[InputFile] = []
        for path in files:
            kind = classify_path(path)
            try:
                if kind == "pdf":
                    count = self.pdf_reader.get_page_count(path)
                else:
                    count = self.image_loader.get_image_info(path)["n_frames"]
                inputs.append(InputFile(path=path, kind=kind, page_count=count))
            except InputError as exc:
                inputs.append(InputFile(path=path, kind=kind, page_count=0, error=str(exc)))
        return inputs

    def iter_pages(
        self,
        inputs: Sequence[InputFile],
        dpi: int = 300,
        on_error: OnError | None = None,
    ) -> Iterator[PageData]:
        """按输入顺序逐页产出标准化 PageData，全局页序连续编号。

        单页失败：on_error(全局页索引, 异常) 后继续；
        文件级失败：on_error(None, 异常) 后跳到下一个文件。
        """
        base = 0
        for input_file in inputs:
            if input_file.error is not None:
                if on_error is not None:
                    on_error(None, InputError(f"{input_file.path}: {input_file.error}"))
                continue

            def report(local_index: int, exc: Exception, _base: int = base) -> None:
                if on_error is not None:
                    on_error(_base + local_index, exc)

            if input_file.kind == "pdf":
                source = self.pdf_reader.iter_pages(input_file.path, dpi=dpi, on_error=report)
            else:
                source = self.image_loader.iter_image_frames(input_file.path, on_error=report)
            try:
                for page in source:
                    page.page_index = base + page.metadata["source_page_index"]
                    yield page
            except InputError as exc:
                if on_error is not None:
                    on_error(None, exc)
            base += input_file.page_count