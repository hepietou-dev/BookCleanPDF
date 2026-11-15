"""应用服务层（Adapter / Application Service）：GUI 与内核之间的唯一桥梁。

设计约束（M5）：
- 本模块 **不导入 Tkinter**，不复制任何处理算法；只编排现有内核接口。
- 单向依赖 ui → app_controller → kernel(M1–M4.8)，无循环依赖。
- M5.1 职责：输入元信息检查、用户设置→内核配置映射、开始处理前的预检、
  输出目录/详情打开。
- M5.2 职责：start_task 编排后台 worker（单线程顺序调用内核），cancel_task 协作取消，
  poll_events 供 UI 主线程经 root.after 排空消费；本层仍不触碰 Tkinter。
- 原始输入永远只读；预检不创建任务目录、不写任何用户文件（无副作用）。
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from .config import (
    EXPORT_COLOR_MODES,
    EXPORT_COMPRESSION_MODES,
    BorderCleanConfig,
    OcrConfig,
    PDFExportConfig,
    TaskConfig,
)
from .errors import InputError
from .image_loader import SUPPORTED_IMAGE_EXTENSIONS
from .input_adapter import InputAdapter
from .presets import get_preset
from . import task_history
from .task_result import build_result_view
from .task_worker import TaskWorker
from .user_paths import UserPaths

logger = logging.getLogger("bookclean")


@dataclass
class InputInfo:
    """单个输入文件的展示信息。path 用 str 以便 JSON 序列化（日志/历史）。"""

    path: str
    name: str
    kind: str  # "pdf" / "image" / "folder"
    page_count: int
    size_bytes: int
    error: str | None = None


@dataclass
class TaskSettings:
    """普通用户可见的设置，全部映射到既有内核特性，不暴露复杂内部参数。

    preset       页面处理预设（默认 auto=自动）
    color_mode   输出类型：preserve=保留彩色/灰度，grayscale=整本灰度
    compression  输出质量：lossless=无损 / balanced=均衡 / compact=紧凑
    border_clean 是否清除扫描黑边（M4.8.1）
    output_dir   输出目录（空=用户 Documents/BookCleanPDF）
    overwrite    是否允许覆盖同名输出（默认否，保护既有成果）
    ocr_enabled  是否生成可搜索 PDF（OCR 文字层，M6.3，默认关；关=一切如旧）
    ocr_engine   OCR 引擎（auto=按推荐序选已装插件）
    """

    preset: str = "auto"
    color_mode: str = "preserve"
    compression: str = "balanced"
    border_clean: bool = True
    output_dir: str = ""
    overwrite: bool = False
    ocr_enabled: bool = False
    ocr_engine: str = "auto"


@dataclass
class ValidationResult:
    """开始处理前的预检结果。ok=True 表示可以进入 M5.2 后台执行。"""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    output_path: str = ""


def _size_of(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


class AppController:
    """GUI 应用服务。持有当前输入列表与用户设置，编排内核接口。"""

    def __init__(self, paths: UserPaths | None = None, adapter: InputAdapter | None = None):
        self.paths = paths or UserPaths()
        self.adapter = adapter or InputAdapter()
        self._inputs: list[InputInfo] = []
        self._settings = TaskSettings(output_dir=str(self.paths.default_output_dir))
        self._active: dict | None = None

    # ---- 输入与设置状态 -------------------------------------------------

    @property
    def inputs(self) -> list[InputInfo]:
        return list(self._inputs)

    @property
    def settings(self) -> TaskSettings:
        return self._settings

    def set_settings(self, settings: TaskSettings) -> None:
        """替换用户设置并校验枚举取值（非法立即报错，不静默失败）。"""
        if settings.preset not in ("auto", "old_book", "phone_photo", "faithful"):
            raise ValueError(f"未知的页面处理预设: {settings.preset!r}")
        if settings.color_mode not in EXPORT_COLOR_MODES:
            raise ValueError(f"输出类型必须是 {EXPORT_COLOR_MODES} 之一: {settings.color_mode!r}")
        if settings.compression not in EXPORT_COMPRESSION_MODES:
            raise ValueError(f"输出质量必须是 {EXPORT_COMPRESSION_MODES} 之一: {settings.compression!r}")
        self._settings = settings

    def inspect_inputs(self, paths: Sequence[str | Path]) -> list[InputInfo]:
        """读取输入元信息（页数/大小），逐文件捕获错误，绝不中断整批导入。

        复用 InputAdapter.collect（文件夹自然排序、损坏文件内部降级为 error）。
        对不支持格式/路径不存在/权限不足等会抛出的情况，逐路径捕获并记录为 error 行。
        按解析后的绝对路径去重，保留首次出现顺序。
        """
        seen = {Path(i.path).resolve() for i in self._inputs}
        added: list[InputInfo] = []
        for raw in paths:
            p = Path(raw)
            try:
                collected = self.adapter.collect([p])
            except (InputError, OSError) as exc:
                info = InputInfo(
                    path=str(p), name=p.name, kind=self._guess_kind(p),
                    page_count=0, size_bytes=_size_of(p), error=str(exc),
                )
                added.append(info)
                logger.warning("导入失败 %s: %s", p, exc)
                continue

            if not collected and p.is_dir():
                added.append(InputInfo(
                    path=str(p), name=p.name, kind="folder", page_count=0,
                    size_bytes=0, error="文件夹中没有支持的 PDF/图片文件",
                ))
                continue

            for f in collected:
                resolved = f.path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                added.append(InputInfo(
                    path=str(f.path), name=f.path.name, kind=f.kind,
                    page_count=f.page_count, size_bytes=_size_of(f.path), error=f.error,
                ))
                if f.error:
                    logger.warning("输入文件有问题 %s: %s", f.path, f.error)

        self._inputs.extend(added)
        logger.info("导入检查完成：新增 %d 项，当前共 %d 项", len(added), len(self._inputs))
        return self.inputs

    def remove_input(self, path: str | Path) -> None:
        target = Path(path).resolve()
        self._inputs = [i for i in self._inputs if Path(i.path).resolve() != target]

    def clear_inputs(self) -> None:
        self._inputs = []

    @staticmethod
    def _guess_kind(p: Path) -> str:
        if p.is_dir():
            return "folder"
        ext = p.suffix.casefold()
        if ext == ".pdf":
            return "pdf"
        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            return "image"
        return "file"

    # ---- 设置 → 内核配置映射 -------------------------------------------

    def resolve_output_dir(self) -> Path:
        out = self._settings.output_dir.strip()
        return Path(out) if out else self.paths.default_output_dir

    def build_configs(self) -> tuple[TaskConfig, PDFExportConfig, BorderCleanConfig | None]:
        """把用户设置映射为内核配置对象并逐一 validate()。不执行任何处理。"""
        cfg = get_preset(self._settings.preset)
        cfg.output_dir = str(self.resolve_output_dir())
        cfg.validate()

        export = PDFExportConfig(
            compression=self._settings.compression,
            color_mode=self._settings.color_mode,
            overwrite=self._settings.overwrite,
            verify=True,
        )
        export.validate()

        border: BorderCleanConfig | None = None
        if self._settings.border_clean:
            border = BorderCleanConfig()
            border.validate()
        return cfg, export, border

    def build_ocr_config(self) -> OcrConfig | None:
        """把用户 OCR 设置映射为 OcrConfig（M6.3）。未启用返回 None（导出与 M5 一致）。

        独立于 build_configs（保持其三元组签名向后兼容）。不导入任何 OCR 引擎代码。
        """
        if not self._settings.ocr_enabled:
            return None
        ocr = OcrConfig(enabled=True, engine=self._settings.ocr_engine)
        ocr.validate()
        return ocr

    # ---- M6.4 OCR 引擎状态 / 管理（惰性导入：OCR 关时不触碰任何 OCR 代码） ----

    def ocr_plugin_status(self, engine: str | None = None) -> dict:
        """查询所选引擎的插件可用性（供设置组状态行）。仅在用户交互时调用。

        惰性导入 PluginManager：构造 GUI/OCR 关时绝不导入 bookclean.ocr。
        返回 Availability.to_dict()；目录不可读等异常降级为 installed=False。
        """
        from .ocr.plugin_manager import PluginManager
        eng = engine if engine is not None else self._settings.ocr_engine
        try:
            return PluginManager().check_available(eng).to_dict()
        except OSError as exc:
            return {"installed": False, "engine_id": "", "engine_version": "",
                    "models_ok": False, "reason": f"插件目录不可读: {exc}", "plugin_dir": ""}

    def open_plugins_dir(self) -> bool:
        """打开用户插件根目录（供“安装/管理引擎…”）。惰性导入，失败只返回 False。"""
        from .ocr.plugin_manager import default_plugins_root
        return self.open_path(default_plugins_root())

    @staticmethod
    def ocr_install_help() -> str:
        """离线安装指引（静态文本，不导入 OCR 代码）。插件独立可选，基础包不含引擎。"""
        return (
            "OCR 引擎为独立可选插件，基础安装包不含引擎运行时。\n\n"
            "方式一（推荐，自动校验）：\n"
            "  python scripts/install_ocr_plugin.py ocr-rapid-1.4.4-plugin.zip\n"
            "  zip 与同名 .json 清单一起获取；安装时自动校验 sha256，防损坏/替换。\n\n"
            "方式二（手动解压）：\n"
            "  1. 获取 ocr-rapid 插件离线包（含独立 Python 运行时 + PP-OCRv4 模型）。\n"
            "  2. 解压到插件目录下的 ocr-rapid/<版本>/ ，使该目录含 plugin.json 与 runtime/。\n"
            "  3. 回到本面板点击“检测引擎”，状态显示“已就绪”即可勾选生成可搜索 PDF。\n\n"
            "PP-OCRv5 高精度桥接（可选，需本机已安装 DeepSeekOCR_v2）：\n"
            "  python scripts/ocr_bridge_setup.py\n"
            "  只读复用既有安装，0 下载；峰值内存约 5.7GB、初始化约 11s，建议密集中文页选用。\n\n"
            "插件可整目录删除/独立更新；未安装时普通 PDF 导出完全不受影响。\n"
            "点击下方“打开插件目录”可直接定位安装位置。"
        )

    def default_output_path(self) -> Path:
        """预检/展示用的目标 PDF 路径：<首个有效输入名>_<质量>_<类型>.pdf。"""
        stem = "bookclean"
        for info in self._inputs:
            if info.error is None and info.kind in ("pdf", "image"):
                stem = Path(info.path).stem
                break
        s = self._settings
        return self.resolve_output_dir() / f"{stem}_{s.compression}_{s.color_mode}.pdf"

    # ---- 开始处理前预检（无副作用） ------------------------------------

    def validate_start(self) -> ValidationResult:
        """M5.1 的“开始处理”仅做预检与就绪判断；真正执行在 M5.2。

        检查：有有效输入 / 配置合法 / 输出目录可写 / 输出≠输入 / 覆盖冲突。
        预检不创建目录、不写文件；可写性为尽力而为（真实写入错误在 M5.2 导出时暴露）。
        """
        errors: list[str] = []
        warnings: list[str] = []
        output_path = self.default_output_path()

        valid = [i for i in self._inputs if i.error is None and i.page_count > 0]
        invalid = [i for i in self._inputs if i.error is not None]
        if not self._inputs:
            errors.append("请先导入 PDF 或图片文件。")
        elif not valid:
            errors.append("没有可处理的有效文件，请检查导入列表中的错误项。")
        if invalid:
            names = "、".join(i.name for i in invalid[:5])
            more = f" 等 {len(invalid)} 个" if len(invalid) > 5 else ""
            warnings.append(f"以下文件有问题，将被跳过：{names}{more}。")

        try:
            self.build_configs()
        except ValueError as exc:
            errors.append(f"设置不合法：{exc}")

        if self._settings.ocr_enabled:
            try:
                self.build_ocr_config()
            except ValueError as exc:
                errors.append(f"OCR 设置不合法：{exc}")
            else:
                # 惰性导入：仅启用 OCR 时才触碰插件发现代码（OCR 关⇒不导入任何 OCR 代码）
                from .ocr.plugin_manager import PluginManager
                avail = PluginManager().check_available(self._settings.ocr_engine)
                if not avail.installed:
                    warnings.append("已勾选生成可搜索 PDF，但未安装 OCR 插件：本次将跳过文字层，"
                                    "普通 PDF 导出不受影响。如需 OCR 请先安装 ocr-rapid 插件。")

        writable_err = self._check_output_writable(self.resolve_output_dir())
        if writable_err:
            errors.append(writable_err)

        try:
            out_resolved = output_path.resolve()
            for info in self._inputs:
                if Path(info.path).resolve() == out_resolved:
                    errors.append("输出文件不能与输入文件相同，请更改输出目录或文件名。")
                    break
            if output_path.exists() and not self._settings.overwrite:
                errors.append(f"输出文件已存在：{output_path}（可更改输出目录，或勾选“允许覆盖”）。")
        except OSError as exc:
            errors.append(f"无法解析输出路径：{exc}")

        ok = not errors
        logger.info("开始处理预检：%s（errors=%d, warnings=%d）", "通过" if ok else "未通过", len(errors), len(warnings))
        return ValidationResult(ok=ok, errors=errors, warnings=warnings, output_path=str(output_path))

    @staticmethod
    def _check_output_writable(out_dir: Path) -> str | None:
        """检查最近的存在祖先目录是否可写（不创建目录，避免预检副作用）。"""
        target = out_dir
        try:
            while not target.exists():
                parent = target.parent
                if parent == target:
                    break
                target = parent
            if not os.access(target, os.W_OK):
                return f"输出目录不可写：{out_dir}"
        except OSError as exc:
            return f"无法访问输出目录：{out_dir}（{exc}）"
        return None

    # ---- 打开输出/详情（Windows shell，带守卫） ------------------------

    def open_output_dir(self) -> bool:
        return self.open_path(self.resolve_output_dir())

    def open_path(self, path: str | Path) -> bool:
        """用系统默认方式打开文件或目录。失败只记录日志并返回 False，不抛出到 UI。"""
        target = Path(path)
        try:
            if target.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                logger.warning("打开失败：路径不存在 %s", target)
                return False
            os.startfile(str(target))  # type: ignore[attr-defined]  # Windows-only
            return True
        except (OSError, AttributeError) as exc:
            logger.warning("打开失败 %s: %s", target, exc)
            return False

    # ---- 详情摘要（供“查看任务详情”对话框与日志） ----------------------

    def describe(self) -> dict:
        valid = [i for i in self._inputs if i.error is None]
        invalid = [i for i in self._inputs if i.error is not None]
        return {
            "input_count": len(self._inputs),
            "valid_count": len(valid),
            "invalid_count": len(invalid),
            "total_pages": sum(i.page_count for i in valid),
            "total_size_bytes": sum(i.size_bytes for i in self._inputs),
            "settings": asdict(self._settings),
            "output_dir": str(self.resolve_output_dir()),
            "output_path": str(self.default_output_path()),
            "inputs": [asdict(i) for i in self._inputs],
        }

    # ---- M5.2 后台任务编排（线程/队列/取消；本层不触碰 Tkinter） --------

    def start_task(self) -> str | None:
        """预检→建配置→spawn 单个 worker 线程→立即返回目标输出路径。

        已有活跃任务或预检未通过则返回 None（不抛异常到 UI）。同一时刻仅一个活跃任务，
        防止重复启动；窗口刷新只 poll_events，绝不重新触发处理。
        """
        if self.is_running():
            logger.warning("已有活跃任务，拒绝重复启动")
            return None
        result = self.validate_start()
        if not result.ok:
            return None
        cfg, export, border = self.build_configs()
        ocr = self.build_ocr_config()
        valid = [i for i in self._inputs if i.error is None and i.page_count > 0]
        input_paths = [i.path for i in valid]
        total_pages = sum(i.page_count for i in valid)
        output_path = Path(result.output_path)

        q: queue.Queue = queue.Queue()
        cancel = threading.Event()
        label = Path(input_paths[0]).name if input_paths else "BookClean 任务"
        worker = TaskWorker(
            input_paths=input_paths, total_pages=total_pages,
            task_config=cfg, export_config=export, border_config=border,
            output_path=output_path, tasks_root=self.paths.tasks_root,
            event_queue=q, cancel_event=cancel,
            history_path=self.paths.history_path, settings=asdict(self._settings), label=label,
            ocr_config=ocr,
        )
        thread = threading.Thread(target=worker.run, name="bookclean-task", daemon=True)
        self._active = {"thread": thread, "cancel": cancel, "queue": q,
                        "worker": worker, "output_path": str(output_path)}
        thread.start()
        self.save_settings()
        logger.info("后台任务已启动：%d 文件 / %d 页 → %s", len(input_paths), total_pages, output_path)
        return str(output_path)

    def cancel_task(self) -> bool:
        """置协作取消信号。处理/黑边逐页生效；导入/导出/验证经回调钩子在下一边界中止。"""
        if not self.is_running():
            return False
        self._active["cancel"].set()
        logger.info("已请求取消后台任务")
        return True

    def is_running(self) -> bool:
        return self._active is not None and self._active["thread"].is_alive()

    def poll_events(self) -> list:
        """排空事件队列（UI 主线程经 root.after 周期调用）。只读，绝不触发处理。"""
        if self._active is None:
            return []
        q = self._active["queue"]
        events = []
        while True:
            try:
                events.append(q.get_nowait())
            except queue.Empty:
                break
        return events

    def wait(self, timeout: float | None = None) -> bool:
        """等待 worker 结束（关闭前取消后调用）。返回是否已结束。"""
        if self._active is None:
            return True
        self._active["thread"].join(timeout)
        return not self._active["thread"].is_alive()

    def clear_active(self) -> None:
        """终止事件后释放活跃任务注册表（worker 线程已自行结束）。"""
        self._active = None

    # ---- M5.3 结果视图与日志（只读派生，不改任务目录） ------------------

    def load_result(self, task_dir: str | Path, output_path: str | Path | None = None) -> dict:
        """读取任务目录 + 导出报告派生结果视图，并附日志路径。供结果页展示。

        只读既有 Manifest/export_report，绝不修改任务目录结构；回退页不计为完全优化，
        仅 os.replace 后的正式 PDF 计完成（详见 task_result.build_result_view）。
        """
        view = build_result_view(task_dir, output_path)
        view["log_path"] = str(self.paths.logs_dir / "bookclean.log")
        return view

    def open_log(self) -> bool:
        """打开错误日志文件（供失败/结果页“查看日志”）。失败只记录并返回 False。"""
        return self.open_path(self.paths.logs_dir / "bookclean.log")

    # ---- M5.4 任务恢复 / 历史索引 / 设置持久化（只读派生，不改任务目录） ----

    def load_task(self, task_dir: str | Path, output_path: str | Path | None = None) -> dict:
        """载入既有任务派生视图（结果页/恢复判定）。含 state/resumable，只读不改任务目录。"""
        return self.load_result(task_dir, output_path)

    def get_history(self) -> list[dict]:
        """读 GUI 历史索引（最新在前），并按 Manifest 派生准确 state/resumable。

        索引 last_state 仅作线索（崩溃时可能停在 running）；权威状态以任务目录现有字段
        派生（架构 §4.2）。缺失任务目录标 state=missing，UI 据此禁用恢复。
        """
        result: list[dict] = []
        for e in task_history.load_history(self.paths.history_path):
            td = e.get("task_dir", "")
            view = build_result_view(td, e.get("output_path") or None)
            item = dict(e)
            item["state"] = view["state"]
            item["resumable"] = view["resumable"]
            item["total_pages"] = view["total_pages"]
            item["label"] = e.get("label") or Path(td).name
            result.append(item)
        return result

    def resume_task(self, task_dir: str | Path, settings: dict | None = None) -> str | None:
        """续跑既有任务：复用已验证页（resume=True），绝不重跑 M1 导入。

        守卫：已有活跃任务→None；任务不可恢复（completed/missing/无已导入页）→None。
        处理配置取自历史快照（或传入 settings），保证与逐页记录匹配→可复用页直接跳过。
        输出路径沿用历史记录的目标 PDF；返回该路径，失败返回 None（不抛到 UI）。
        """
        if self.is_running():
            logger.warning("已有活跃任务，拒绝重复启动/恢复")
            return None
        task_dir = Path(task_dir)
        entry = task_history.find_entry(self.paths.history_path, task_dir)
        # 先取历史目标输出，以便正确定位其旁的导出报告来判定 completed（否则会误判可恢复）
        output_hint = (entry.get("output_path") if entry else "") or None
        view = build_result_view(task_dir, output_hint)
        if not view["resumable"]:
            logger.warning("任务不可恢复（state=%s）: %s", view["state"], task_dir)
            return None

        snap = settings or (entry.get("settings") if entry else None) or asdict(self._settings)
        if not self._apply_settings_snapshot(snap):
            return None
        try:
            cfg, export, border = self.build_configs()
            ocr = self.build_ocr_config()
        except ValueError as exc:
            logger.warning("恢复配置不合法: %s", exc)
            return None

        output_path = output_hint or view["output_path"] or str(self.default_output_path())
        output_path = Path(output_path)

        q: queue.Queue = queue.Queue()
        cancel = threading.Event()
        worker = TaskWorker(
            input_paths=[], total_pages=view["total_pages"],
            task_config=cfg, export_config=export, border_config=border,
            output_path=output_path, tasks_root=self.paths.tasks_root,
            event_queue=q, cancel_event=cancel,
            resume_task_dir=task_dir, history_path=self.paths.history_path,
            settings=asdict(self._settings), label=(entry.get("label") if entry else "") or task_dir.name,
            ocr_config=ocr,
        )
        thread = threading.Thread(target=worker.run, name="bookclean-resume", daemon=True)
        self._active = {"thread": thread, "cancel": cancel, "queue": q,
                        "worker": worker, "output_path": str(output_path)}
        thread.start()
        logger.info("任务续跑已启动：%s → %s（复用已验证页，不重跑导入）", task_dir, output_path)
        return str(output_path)

    def _apply_settings_snapshot(self, snap: dict) -> bool:
        """把设置快照应用到当前设置（仅取 TaskSettings 已知字段）。非法→False。"""
        fields = TaskSettings.__dataclass_fields__
        known = {k: snap[k] for k in fields if k in snap}
        if not known.get("output_dir"):
            known["output_dir"] = str(self.paths.default_output_dir)
        try:
            self.set_settings(TaskSettings(**known))
        except (ValueError, TypeError) as exc:
            logger.warning("恢复设置快照不合法: %s", exc)
            return False
        return True

    def save_settings(self) -> None:
        """持久化当前用户设置快照（下次启动恢复）。失败只记录，不抛到 UI。"""
        try:
            task_history.save_config(self.paths.config_path, asdict(self._settings))
        except OSError as exc:
            logger.warning("保存设置失败: %s", exc)

    def load_settings(self) -> bool:
        """启动时恢复上次设置。无配置/非法→False（保留默认）。"""
        cfg = task_history.load_config(self.paths.config_path)
        if not cfg:
            return False
        return self._apply_settings_snapshot(cfg)