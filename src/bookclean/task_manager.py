"""任务管理（M1 范围：导入与页面标准化）。

逐页流水线：读取一页 -> 标准化 -> 保存 PNG -> 缩略图 -> 更新 manifest -> 释放内存。
原始文件只读；所有产物写入独立任务目录；manifest 原子写入。
M4 将在此基础上扩展处理队列、取消与断点恢复。
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image, ImageDraw, ImageFont

from .errors import BookCleanError
from .input_adapter import InputAdapter, InputFile
from .models import PageData
from .utils.atomic_file import atomic_write_json
from .utils.image_utils import make_thumbnail

logger = logging.getLogger("bookclean")

PAGE_STATUS = ("pending", "processing", "completed", "failed", "skipped")


class ImportTaskManager:
    def __init__(self, temp_root: str | Path, adapter: InputAdapter | None = None):
        self.temp_root = Path(temp_root)
        self.adapter = adapter or InputAdapter()

    def run(
        self,
        paths: Sequence[str | Path],
        dpi: int = 300,
        progress_cb: Callable[[dict], None] | None = None,
    ) -> dict:
        """执行导入任务，返回 {"task_dir", "manifest_path", "manifest"}。"""
        started = time.monotonic()
        inputs = self.adapter.collect(paths)
        task_dir = self._create_task_dir()
        manifest_path = task_dir / "manifest.json"
        manifest = self._new_manifest(task_dir, inputs, dpi)
        atomic_write_json(manifest_path, manifest)

        def on_error(global_index: int | None, exc: Exception) -> None:
            if global_index is None:
                manifest["file_errors"].append(str(exc))
                logger.error("文件级导入失败: %s", exc)
            else:
                entry = manifest["pages"][global_index]
                entry["status"] = "failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
                logger.error("页面导入失败 第 %d 页: %s", global_index + 1, exc)
            atomic_write_json(manifest_path, manifest)

        for page in self.adapter.iter_pages(inputs, dpi=dpi, on_error=on_error):
            entry = manifest["pages"][page.page_index]
            entry["status"] = "processing"
            try:
                self._save_page(task_dir, page, entry, dpi)
                entry["status"] = "completed"
            except Exception as exc:
                entry["status"] = "failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
                logger.error("页面保存失败 第 %d 页: %s", page.page_index + 1, exc)
            atomic_write_json(manifest_path, manifest)
            if progress_cb is not None:
                progress_cb(entry)
            page.image = None
            del page

        for entry in manifest["pages"]:
            if entry["status"] == "pending":
                entry["status"] = "failed"
                entry["error"] = entry["error"] or "文件级错误，页面未能处理"
        self._finalize(task_dir, manifest, started)
        atomic_write_json(manifest_path, manifest)
        return {"task_dir": task_dir, "manifest_path": manifest_path, "manifest": manifest}

    def _create_task_dir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_dir = self.temp_root / f"task_{stamp}_{uuid.uuid4().hex[:6]}"
        try:
            for sub in ("original_pages", "thumbnails", "logs"):
                (task_dir / sub).mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise BookCleanError(f"无法创建临时任务目录（权限或磁盘问题）: {task_dir} ({exc})") from exc
        return task_dir

    @staticmethod
    def _new_manifest(task_dir: Path, inputs: list[InputFile], dpi: int) -> dict:
        pages = []
        index = 0
        for input_file in inputs:
            for local in range(input_file.page_count):
                pages.append({
                    "page_index": index,
                    "display_index": index + 1,
                    "source_path": str(input_file.path),
                    "source_kind": input_file.kind,
                    "source_page_index": local,
                    "status": "pending",
                    "image_path": None,
                    "thumbnail_path": None,
                    "error": input_file.error,
                })
                index += 1
        return {
            "task_id": task_dir.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "render_dpi": dpi,
            "inputs": [
                {"path": str(f.path), "kind": f.kind, "page_count": f.page_count, "error": f.error}
                for f in inputs
            ],
            "total_pages": len(pages),
            "pages": pages,
            "file_errors": [],
            "contact_sheet": None,
        }

    @staticmethod
    def _save_page(task_dir: Path, page: PageData, entry: dict, dpi: int) -> None:
        im = Image.fromarray(page.image)
        name = f"page_{entry['display_index']:06d}.png"
        image_path = task_dir / "original_pages" / name
        save_dpi = page.source_dpi or dpi
        im.save(image_path, format="PNG", dpi=(save_dpi, save_dpi))

        thumb = make_thumbnail(im, max_edge=300)
        thumb_name = f"thumb_{entry['display_index']:06d}.jpg"
        thumb_path = task_dir / "thumbnails" / thumb_name
        thumb.save(thumb_path, format="JPEG", quality=85)

        entry.update({
            "image_path": f"original_pages/{name}",
            "thumbnail_path": f"thumbnails/{thumb_name}",
            "image_width": page.original_width,
            "image_height": page.original_height,
            "dpi": save_dpi,
            "color_mode": page.color_mode,
            "metadata": page.metadata,
        })

    @staticmethod
    def _finalize(task_dir: Path, manifest: dict, started: float) -> None:
        statuses = [p["status"] for p in manifest["pages"]]
        failed = [p["display_index"] for p in manifest["pages"] if p["status"] == "failed"]
        manifest["completed_pages"] = statuses.count("completed")
        manifest["failed_pages"] = failed
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 2)
        if manifest["total_pages"] == 0 or manifest["completed_pages"] == 0:
            manifest["status"] = "failed"
        elif failed or manifest["file_errors"]:
            manifest["status"] = "completed_with_errors"
        else:
            manifest["status"] = "completed"
        if manifest["completed_pages"] > 0:
            sheet = ImportTaskManager._contact_sheet(task_dir, manifest["pages"])
            manifest["contact_sheet"] = sheet

    @staticmethod
    def _contact_sheet(task_dir: Path, pages: list[dict]) -> str:
        completed = [p for p in pages if p["status"] == "completed"]
        cols, cell_w, cell_h, label_h, pad = 5, 210, 290, 22, 8
        rows = (len(completed) + cols - 1) // cols
        sheet = Image.new(
            "RGB",
            (cols * (cell_w + pad) + pad, rows * (cell_h + label_h + pad) + pad),
            (245, 245, 245),
        )
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype("msyh.ttc", 14)
            label = "第 {} 页"
        except OSError:
            font = ImageFont.load_default(size=14)
            label = "Page {}"
        for i, entry in enumerate(completed):
            thumb = Image.open(task_dir / entry["thumbnail_path"])
            thumb.thumbnail((cell_w, cell_h), Image.LANCZOS)
            x = pad + (i % cols) * (cell_w + pad)
            y = pad + (i // cols) * (cell_h + label_h + pad)
            sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y))
            draw.text((x, y + cell_h + 3), label.format(entry["display_index"]), fill=(60, 60, 60), font=font)
            thumb.close()
        rel = "contact_sheet.jpg"
        sheet.save(task_dir / rel, format="JPEG", quality=85)
        return rel