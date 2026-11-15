"""主窗口视图：在给定 root（Tk 或 Toplevel）上组装导入区/配置区/操作区/进度区/状态栏。

不子类化 Tk：窗口生命周期（创建/mainloop/销毁）由调用方掌握，视图只负责构建与编排，
便于测试用单一解释器 + 每用例 Toplevel（避免反复创建 Tk 解释器在 Windows 上的初始化抖动）。

M5.2 范围：开始处理→后台 worker；root.after(80ms) 排空事件队列更新真实进度；
运行中锁定导入/配置/开始并启用取消；终止事件（完成/失败/取消）复位 UI；
关闭时若仍在运行则确认→协作取消→等待→销毁。进度全部来自内核回调，无定时假进度。
"""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import messagebox, ttk

from ..app_controller import AppController
from ..error_messages import describe_error
from ..task_events import (
    PageProgress,
    StageChanged,
    StageSummary,
    TaskCancelled,
    TaskCompleted,
    TaskFailed,
    TaskState,
)
from .dpi import apply_scaling
from .dnd import enable_drop
from .history_panel import show_history
from .import_panel import ImportPanel, format_size
from .result_panel import show_result
from .settings_panel import SettingsPanel

_PRESET_LABEL = {"auto": "自动", "old_book": "旧书翻新", "phone_photo": "手机照片", "faithful": "最大保真"}
_COLOR_LABEL = {"preserve": "彩色（保留原色）", "grayscale": "灰度"}
_QUALITY_LABEL = {"lossless": "无损", "balanced": "均衡", "compact": "紧凑"}
_RUNNING = (TaskState.IMPORTING, TaskState.PROCESSING, TaskState.BORDER_CLEANING,
            TaskState.OCR_RECOGNIZING, TaskState.EXPORTING, TaskState.VALIDATING)
_STATE_LABEL = {TaskState.PENDING: "空闲", TaskState.READY: "就绪", TaskState.IMPORTING: "导入中",
                TaskState.PROCESSING: "处理中", TaskState.BORDER_CLEANING: "清黑边",
                TaskState.OCR_RECOGNIZING: "OCR 识别",
                TaskState.EXPORTING: "导出中", TaskState.VALIDATING: "验证中",
                TaskState.COMPLETED: "完成", TaskState.FAILED: "失败", TaskState.CANCELLED: "已取消"}
_VIEW_TERMINAL = {"completed": TaskState.COMPLETED, "failed": TaskState.FAILED,
                  "cancelled": TaskState.CANCELLED}
_POLL_MS = 80


def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


class MainWindow:
    def __init__(self, root, controller: AppController | None = None):
        self.root = root
        self.controller = controller or AppController()
        self.state = TaskState.PENDING
        self._drop = None
        self._running = False
        self._poll_id = None
        self._start_time = 0.0
        self._counters: dict = {}
        self._last_result: dict | None = None
        self._last_terminal = None
        self._result_win = None
        self._history_win = None

        root.title("BookClean PDF")
        root.geometry("880x760")
        root.minsize(740, 620)
        apply_scaling(root)
        self.controller.load_settings()
        self._build()
        self._update_readiness()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._drop = enable_drop(root, self._on_drop)

    def _build(self):
        try:
            style = ttk.Style(self.root)
            if "vista" in style.theme_names():
                style.theme_use("vista")
        except tk.TclError:
            pass

        header = ttk.Label(self.root, text="BookClean PDF — 把扫描书/手机照片清理成干净的图像型 PDF",
                           font=("", 12, "bold"), padding=(10, 8))
        header.pack(side="top", fill="x")

        status_frame = ttk.Frame(self.root, padding=(10, 4))
        status_frame.pack(side="bottom", fill="x")
        self.state_label = ttk.Label(status_frame, text="空闲", width=8, anchor="w")
        self.state_label.pack(side="left")
        self.status_var = tk.StringVar(value="请导入 PDF 或图片文件。")
        ttk.Label(status_frame, textvariable=self.status_var, anchor="w").pack(side="left", fill="x", expand=True)

        ops = ttk.Frame(self.root, padding=(10, 4))
        ops.pack(side="bottom", fill="x")
        self.start_btn = ttk.Button(ops, text="开始处理", command=self._on_start)
        self.start_btn.pack(side="left")
        self.cancel_btn = ttk.Button(ops, text="取消", command=self._on_cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=4)
        ttk.Button(ops, text="打开输出目录", command=self._on_open_output).pack(side="left", padx=4)
        ttk.Button(ops, text="查看任务详情", command=self._on_details).pack(side="left")
        self.result_btn = ttk.Button(ops, text="查看结果", command=self._on_show_result, state="disabled")
        self.result_btn.pack(side="left", padx=4)
        ttk.Button(ops, text="历史任务", command=self._on_history).pack(side="left", padx=4)

        progress = ttk.LabelFrame(self.root, text="任务进度", padding=(10, 6))
        progress.pack(side="bottom", fill="x", padx=10, pady=4)
        top = ttk.Frame(progress)
        top.pack(fill="x")
        self.stage_var = tk.StringVar(value="尚未开始")
        ttk.Label(top, textvariable=self.stage_var, anchor="w").pack(side="left")
        self.elapsed_var = tk.StringVar(value="00:00")
        ttk.Label(top, textvariable=self.elapsed_var, anchor="e").pack(side="right")
        self.progress_bar = ttk.Progressbar(progress, mode="determinate", maximum=100, value=0)
        self.progress_bar.pack(fill="x", pady=(4, 2))
        self.counts_var = tk.StringVar(value="")
        ttk.Label(progress, textvariable=self.counts_var, anchor="w", foreground="#333").pack(fill="x")

        self.settings_panel = SettingsPanel(self.root, self.controller, on_change=self._update_readiness)
        self.settings_panel.pack(side="bottom", fill="x")
        self.import_panel = ImportPanel(self.root, self.controller, on_change=self._update_readiness)
        self.import_panel.pack(side="top", fill="both", expand=True)

    # ---- 状态与就绪 -----------------------------------------------------

    def _update_readiness(self):
        if self._running or self.state in _RUNNING:
            return
        valid = [i for i in self.controller.inputs if i.error is None and i.page_count > 0]
        if valid:
            self.state = TaskState.READY
            self.start_btn.state(["!disabled"])
            pages = sum(i.page_count for i in valid)
            self.status_var.set(f"已导入 {len(valid)} 个有效文件 / {pages} 页，可点击“开始处理”。")
        else:
            self.state = TaskState.PENDING
            self.start_btn.state(["disabled"])
            self.status_var.set("请导入 PDF 或图片文件。")
        self._refresh_state_label()

    def _refresh_state_label(self):
        self.state_label.configure(text=_STATE_LABEL.get(self.state, str(self.state)))

    # ---- 操作区 ---------------------------------------------------------

    def _on_start(self):
        if self._running:
            return
        result = self.controller.validate_start()
        if not result.ok:
            self.state = TaskState.PENDING
            self._refresh_state_label()
            self.status_var.set("预检未通过：" + " ".join(result.errors))
            messagebox.showerror("无法开始", "\n".join(result.errors))
            return
        output = self.controller.start_task()
        if output is None:
            messagebox.showerror("无法开始", "任务已在运行或预检未通过，请检查导入与设置。")
            return
        msg = f"任务已启动，目标输出：{output}"
        if result.warnings:
            msg += "（注意：" + " ".join(result.warnings) + "）"
        self._enter_running(TaskState.IMPORTING, "导入与渲染输入", msg)

    def _enter_running(self, stage: TaskState, stage_label: str, status_msg: str):
        """进入运行态（开始/续跑共用）：锁定 UI、复位进度、启动轮询。只由显式操作触发。"""
        self._running = True
        self._start_time = time.monotonic()
        self._counters = {}
        self._last_result = None
        self._last_terminal = None
        self.result_btn.state(["disabled"])
        self._set_running_ui(True)
        self.state = stage
        self._refresh_state_label()
        self.stage_var.set(stage_label)
        self.progress_bar.configure(value=0)
        self.counts_var.set("")
        self.elapsed_var.set("00:00")
        self.status_var.set(status_msg)
        self._schedule_poll()

    def _on_cancel(self):
        if not self._running:
            return
        if self.controller.cancel_task():
            self.cancel_btn.state(["disabled"])
            self.status_var.set("正在取消…处理/清黑边在下一页边界停止；导入/导出/验证在当前阶段边界中止。")

    def _on_open_output(self):
        if not self.controller.open_output_dir():
            messagebox.showerror("打开失败", "无法打开输出目录，请检查路径或权限。")

    def _on_drop(self, paths):
        if self._running:
            return
        self.import_panel.add_paths(paths)

    # ---- 历史任务与恢复（M5.4） ------------------------------------------

    def _on_history(self):
        if self._history_win is not None and self._history_win.winfo_exists():
            self._history_win.destroy()
        self._history_win = show_history(
            self.root, self.controller, on_resume=self._on_resume, on_reopen=self._on_reopen)

    def _on_resume(self, task_dir):
        if self._running:
            messagebox.showerror("无法恢复", "已有任务在运行，请先完成或取消后再恢复。")
            return
        output = self.controller.resume_task(task_dir)
        if output is None:
            messagebox.showerror("无法恢复", "该任务不可恢复（已完成、已丢失或无已导入页面），或已有任务在运行。")
            return
        if self._history_win is not None and self._history_win.winfo_exists():
            self._history_win.destroy()
            self._history_win = None
        self._enter_running(TaskState.PROCESSING, "逐页图像修复（续跑复用已验证页）",
                            f"续跑已启动，复用已验证页（不重跑导入），目标输出：{output}")

    def _on_reopen(self, task_dir, output_path):
        result = self.controller.load_task(task_dir, output_path)
        self._last_result = result
        self._last_terminal = _VIEW_TERMINAL.get(result.get("state"))
        self.result_btn.state(["!disabled"])
        self._open_result_window(result, self._last_terminal)

    def _refresh_history(self):
        if self._history_win is not None and self._history_win.winfo_exists():
            self._history_win.destroy()
            self._history_win = show_history(
                self.root, self.controller, on_resume=self._on_resume, on_reopen=self._on_reopen)

    # ---- 后台事件轮询（主线程，只读事件，绝不触发处理） ------------------

    def _schedule_poll(self):
        self._poll_id = self.root.after(_POLL_MS, self._poll)

    def _cancel_poll(self):
        if self._poll_id is not None:
            self.root.after_cancel(self._poll_id)
            self._poll_id = None

    def _poll(self):
        self._poll_id = None
        if not self._running:
            return
        for ev in self.controller.poll_events():
            self._handle_event(ev)
        if self._running:
            self.elapsed_var.set(_fmt_elapsed(time.monotonic() - self._start_time))
            self._schedule_poll()

    def _handle_event(self, ev):
        if isinstance(ev, StageChanged):
            self.state = ev.state
            self._refresh_state_label()
            self.stage_var.set(ev.label or _STATE_LABEL.get(ev.state, str(ev.state)))
            self.progress_bar.configure(value=0)
        elif isinstance(ev, PageProgress):
            self.state = ev.state
            if ev.total > 0:
                self.progress_bar.configure(value=100.0 * ev.done / ev.total)
            if ev.counters:
                self._counters = ev.counters
            self._update_counts(ev)
        elif isinstance(ev, StageSummary):
            self._counters = ev.summary
            self._update_summary_counts(ev)
        elif isinstance(ev, TaskCompleted):
            self._finish(TaskState.COMPLETED, f"完成：{ev.output_path}")
            size = ev.summary.get("output_size_bytes")
            extra = f"\n大小：{format_size(size)}" if size else ""
            messagebox.showinfo("处理完成", f"已生成通过验证的 PDF：\n{ev.output_path}{extra}")
            self._show_result(ev.task_dir, ev.output_path, TaskState.COMPLETED)
        elif isinstance(ev, TaskFailed):
            self._finish(TaskState.FAILED, "失败：" + ev.error)
            info = describe_error(ev.exc_type, ev.error)
            messagebox.showerror(
                "处理失败",
                f"{info.message}\n\n建议：{info.action}\n\n阶段：{_STATE_LABEL.get(ev.state, ev.state)}")
            self._show_result(ev.task_dir, None, TaskState.FAILED)
        elif isinstance(ev, TaskCancelled):
            self._finish(TaskState.CANCELLED, "已取消（原件未改动；已处理页保留在任务目录）。")
            messagebox.showinfo("已取消", "任务已取消。原始文件未改动，已处理的页面保留在任务目录，可稍后恢复。")
            self._show_result(ev.task_dir, None, TaskState.CANCELLED)

    def _update_counts(self, ev: PageProgress):
        c = self._counters
        if ev.state == TaskState.PROCESSING:
            self.counts_var.set(
                f"优化 {c.get('processed', 0)} · 回退 {c.get('fallbacks', 0)} · 失败 {c.get('failed', 0)}"
                f"（{ev.done}/{ev.total} 页）")
        elif ev.state == TaskState.BORDER_CLEANING:
            self.counts_var.set(
                f"清除 {c.get('cleaned', 0)} · 跳过 {c.get('skipped', 0)} · 失败 {c.get('failed', 0)}"
                f"（{ev.done}/{ev.total} 页）")
        elif ev.state == TaskState.OCR_RECOGNIZING:
            self.counts_var.set(
                f"OCR 识别 {c.get('completed', 0)} · 失败 {c.get('failed', 0)} · 跳过 {c.get('skipped', 0)}"
                f"（{ev.done}/{ev.total} 页）")
        else:
            self.counts_var.set(f"{ev.done}/{ev.total} 页")

    def _update_summary_counts(self, ev: StageSummary):
        s = ev.summary
        if ev.state == TaskState.IMPORTING:
            self.counts_var.set(f"导入 {s.get('imported', 0)}/{s.get('total_pages', 0)} 页 · 文件错误 {s.get('file_errors', 0)}")
        elif ev.state == TaskState.PROCESSING:
            # 内核 summary.processed 含回退页；完全优化页 = processed - fallbacks（回退不算完全成功）
            optimized = max(0, s.get("processed", 0) - s.get("fallbacks", 0))
            self.counts_var.set(
                f"优化 {optimized} · 回退 {s.get('fallbacks', 0)} · 失败 {s.get('failed', 0)} · 复用 {s.get('reused', 0)}")
        elif ev.state == TaskState.BORDER_CLEANING:
            self.counts_var.set(f"清除 {s.get('cleaned', 0)} · 跳过 {s.get('skipped', 0)} · 失败 {s.get('failed', 0)}")
        elif ev.state == TaskState.OCR_RECOGNIZING:
            self.counts_var.set(
                f"OCR 识别 {s.get('completed', 0)} · 失败 {s.get('failed', 0)} · 跳过 {s.get('skipped', 0)}"
                f" · 复用已识别 {s.get('reused', 0)} 页 · 文字 {s.get('total_text_chars', 0)} 字符")
        elif ev.state == TaskState.EXPORTING:
            self.counts_var.set(
                f"合成 {s.get('total_pages', 0)} 页 · 原始回退 {s.get('fallback_original_pages', 0)} · 几何回退 {s.get('fallback_geometry_pages', 0)}")

    def _finish(self, state: TaskState, status_msg: str):
        self._running = False
        self.state = state
        self._refresh_state_label()
        self.stage_var.set(_STATE_LABEL.get(state, str(state)))
        self.elapsed_var.set(_fmt_elapsed(time.monotonic() - self._start_time))
        self._cancel_poll()
        self._set_running_ui(False)
        self.status_var.set(status_msg)
        self.controller.clear_active()
        self._refresh_history()

    def _set_running_ui(self, running: bool):
        self.start_btn.state(["disabled" if running else "!disabled"])
        self.cancel_btn.state(["!disabled" if running else "disabled"])
        self.import_panel.set_enabled(not running)
        self.settings_panel.set_enabled(not running)

    # ---- 结果展示（M5.3） ------------------------------------------------

    def _show_result(self, task_dir: str, output_path, terminal_state):
        if not task_dir:
            return
        result = self.controller.load_result(task_dir, output_path)
        self._last_result = result
        self._last_terminal = terminal_state
        self.result_btn.state(["!disabled"])
        self._open_result_window(result, terminal_state)

    def _on_show_result(self):
        if self._last_result is not None:
            self._open_result_window(self._last_result, self._last_terminal)

    def _open_result_window(self, result, terminal_state):
        if self._result_win is not None and self._result_win.winfo_exists():
            self._result_win.destroy()
        self._result_win = show_result(self.root, self.controller, result, terminal_state,
                                       on_resume=self._on_resume)

    # ---- 详情与关闭 -----------------------------------------------------

    def _on_details(self):
        d = self.controller.describe()
        s = d["settings"]
        lines = [
            "任务详情（当前导入与设置）",
            "=" * 40,
            f"输入项：{d['input_count']}（有效 {d['valid_count']} / 问题 {d['invalid_count']}）",
            f"总页数：{d['total_pages']}",
            f"总大小：{format_size(d['total_size_bytes'])}",
            f"输出目录：{d['output_dir']}",
            f"目标 PDF：{d['output_path']}",
            "",
            "设置：",
            f"  页面处理：{_PRESET_LABEL.get(s['preset'], s['preset'])}",
            f"  输出类型：{_COLOR_LABEL.get(s['color_mode'], s['color_mode'])}",
            f"  输出质量：{_QUALITY_LABEL.get(s['compression'], s['compression'])}",
            f"  清除黑边：{'是' if s['border_clean'] else '否'}",
            f"  允许覆盖：{'是' if s['overwrite'] else '否'}",
            f"  可搜索 PDF（OCR）：{'开' if s.get('ocr_enabled') else '关'}"
            + (f"（引擎 {s.get('ocr_engine', 'auto')}）" if s.get('ocr_enabled') else ""),
            "",
            "文件清单：",
        ]
        for info in d["inputs"]:
            if info["error"]:
                lines.append(f"  [问题] {info['name']}：{info['error']}")
            else:
                lines.append(f"  [可处理] {info['name']}（{info['kind']}，{info['page_count']} 页，{format_size(info['size_bytes'])}）")
        result = self.controller.validate_start()
        lines += ["", "预检：" + ("通过" if result.ok else "未通过")]
        for e in result.errors:
            lines.append(f"  错误：{e}")
        for w in result.warnings:
            lines.append(f"  注意：{w}")

        win = tk.Toplevel(self.root)
        win.title("任务详情")
        win.geometry("640x520")
        win.transient(self.root)
        text = tk.Text(win, wrap="word", padx=8, pady=8)
        vsb = ttk.Scrollbar(win, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=vsb.set)
        text.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")
        ttk.Button(win, text="关闭", command=win.destroy).pack(side="bottom", pady=6)

    def _on_close(self):
        if self._running:
            if not messagebox.askyesno(
                "任务正在运行",
                "后台任务仍在运行。确定要取消任务并退出吗？\n\n"
                "原始文件不受影响；已处理的页面保留在任务目录，可稍后恢复。",
            ):
                return
            self.controller.cancel_task()
            self.controller.wait(timeout=10)
        self._cancel_poll()
        if self._drop is not None:
            self._drop.disable()
        self.root.destroy()