"""图片读取（Pillow）：EXIF 像素级方向校正、色彩统一、多帧 TIFF、超大图保护。

只做输入标准化，不做任何图像增强。
"""

import logging
import warnings
from pathlib import Path
from typing import Callable, Iterator

from PIL import Image, ImageOps

from .errors import ImageLoadError
from .models import PageData
from .utils.image_utils import normalize_to_rgb, to_rgb_array

logger = logging.getLogger("bookclean")

OnError = Callable[[int, Exception], None]

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class ImageLoader:
    def _open(self, image_path: str | Path) -> Image.Image:
        path = Path(image_path)
        if not path.is_file():
            raise ImageLoadError(f"图片文件不存在: {path}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                im = Image.open(path)
                im.load()
            return im
        except Image.DecompressionBombWarning as exc:
            raise ImageLoadError(
                f"图片像素尺寸异常大，已拒绝读取（防止解压炸弹）: {path} ({exc})"
            ) from exc
        except Image.DecompressionBombError as exc:
            raise ImageLoadError(f"图片像素尺寸超出安全上限: {path} ({exc})") from exc
        except Exception as exc:
            raise ImageLoadError(f"图片无法读取或已损坏: {path} ({type(exc).__name__}: {exc})") from exc

    def get_image_info(self, image_path: str | Path) -> dict:
        im = self._open(image_path)
        try:
            dpi = im.info.get("dpi")
            return {
                "format": im.format,
                "mode": im.mode,
                "width": im.width,
                "height": im.height,
                "n_frames": int(getattr(im, "n_frames", 1)),
                "exif_orientation": ImageLoader._safe_orientation(im),
                "has_icc": bool(im.info.get("icc_profile")),
                "dpi": (float(dpi[0]), float(dpi[1])) if dpi else None,
            }
        finally:
            im.close()

    def load_image(self, image_path: str | Path) -> PageData:
        """读取首帧。多帧 TIFF 请用 iter_image_frames。"""
        for page in self.iter_image_frames(image_path):
            return page
        raise ImageLoadError(f"图片没有任何可读帧: {image_path}")

    def iter_image_frames(
        self,
        image_path: str | Path,
        on_error: OnError | None = None,
    ) -> Iterator[PageData]:
        """逐帧读取（多页 TIFF 每帧作为独立页面）。单帧失败报告后继续下一帧。"""
        path = Path(image_path)
        im = self._open(path)
        try:
            n_frames = int(getattr(im, "n_frames", 1))
            for frame_index in range(n_frames):
                try:
                    yield self._read_frame(im, path, frame_index)
                except Exception as exc:
                    logger.error("图片帧读取失败 %s 帧 %d: %s", path.name, frame_index, exc)
                    if on_error is not None:
                        on_error(frame_index, exc)
        finally:
            im.close()

    @staticmethod
    def _safe_orientation(im: Image.Image) -> int | None:
        try:
            value = im.getexif().get(274)
            return int(value) if value is not None else None
        except Exception:
            return None

    @staticmethod
    def _read_frame(im: Image.Image, path: Path, frame_index: int) -> PageData:
        if frame_index > 0:
            im.seek(frame_index)
        raw_width, raw_height = im.size
        orientation = ImageLoader._safe_orientation(im)
        mode_before = im.mode

        if orientation in (2, 3, 4, 5, 6, 7, 8):
            oriented = ImageOps.exif_transpose(im)
        else:
            oriented = im
        rgb, color_warnings = normalize_to_rgb(oriented)
        arr = to_rgb_array(rgb)

        dpi = im.info.get("dpi")
        source_dpi = float(dpi[0]) if dpi else None
        height, width = arr.shape[:2]
        metadata = {
            "source_page_index": frame_index,
            "format": im.format,
            "mode_before": mode_before,
            "raw_width": raw_width,
            "raw_height": raw_height,
            "exif_orientation": orientation,
            "icc_profile": "applied" if im.info.get("icc_profile") else "none",
            "n_frames": int(getattr(im, "n_frames", 1)),
        }
        if color_warnings:
            metadata["warnings"] = color_warnings
            metadata["icc_profile"] = "failed" if any("ICC" in w for w in color_warnings) else metadata["icc_profile"]
        return PageData(
            page_index=frame_index,
            image=arr,
            source_path=str(path),
            source_kind="image",
            original_width=width,
            original_height=height,
            source_dpi=source_dpi,
            color_mode="RGB",
            metadata=metadata,
        )