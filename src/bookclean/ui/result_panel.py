"""结果页（M5.3）：结果汇总 + 失败/回退页列表 + 页详情一键定位 + 打开 PDF/文件夹/报告/日志。

只展示 controller.load_result 派生的事实（回退不计完全优化、仅正式 PDF 计完成）。
所有打开操作经 controller.open_path / open_log（带守卫，失败只提示不抛出）。
不在 UI 内加载全分辨率图像；“定位到该页”以系统默认方式外部打开图像文件（说明书 §9.2）。
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from ..task_events import TaskState
from .import_panel import format_size

_STATE_LABEL = {TaskState.COMPLETED: "完成", TaskState.FAILED: "失败", TaskState.CANCELLED: "已取消"}
_LEVEL_LABEL = {"background": "背景回退", "geometry": "几何回退", "original": "原始回退"}
_STAGE_LABEL = {"import": "导入", "processing": "处理", "border": "黑边", "ocr": "OCR"}
_COLOR_LABEL = {"preserve": "彩色（保留原色）", "grayscale": "灰度"}
_QUALITY_LABEL = {"lossless": "无损", "balanced": "均衡", "compact": "紧凑"}


def _summary_lines(result: dict, terminal_state) -> list[str]:
    state_label = _STATE_LABEL.get(terminal_state, "已结束")
    out_exists = result.get("output_exists")
    size = result.get("output_size_bytes", 0)
    if out_exists:
        out_line = f"已生成正式 PDF（{format_size(size)}）"
    else:
        out_line = "未生成正式 PDF（.partial 不计完成；原件未改动）"
    validation = result.get("validation_status") or "未验证"
    issues = result.get("validation_issues", [])
    val_line = f"{validation}（问题 {len(issues)}）"
    bd = result.get("fallback_breakdown", {})
    color = _COLOR_LABEL.get(result.get("color_mode", ""), result.get("color_mode", "") or "—")
    quality = _QUALITY_LABEL.get(result.get("compression_mode", ""), result.get("compression_mode", "") or "—")
    border = result.get("border", {})
    lines = [
        f"状态：{state_label}",
        f"任务：{result.get('task_id', '')}",
        f"输出 PDF：{result.get('output_path') or '—'}",
        f"输出文件：{out_line}",
        f"输出模式：{color} · {quality}",
        f"PDF 验证：{val_line}",
        "",
        f"总页数：{result.get('total_pages', 0)}",
        f"完全优化：{result.get('optimized', 0)} 页（回退页不计入）",
        f"回退：{result.get('fallbacks', 0)} 页"
        f"（背景 {bd.get('background', 0)} · 几何 {bd.get('geometry', 0)} · 原始 {bd.get('original', 0)}）",
        f"失败：{result.get('failed', 0)} 页",
        f"黑边清除：清除 {border.get('cleaned', 0)} · 跳过 {border.get('skipped', 0)} · 失败 {border.get('failed', 0)}",
        f"总耗时：{result.get('elapsed_seconds', 0)} s",
        f"输入文件：{len(result.get('inputs', []))} 个",
    ]
    ocr = result.get("ocr", {}) or {}
    if ocr.get("enabled"):
        engine = f"{ocr.get('engine_id', '')} {ocr.get('engine_version', '')}".strip() or "—"
        lines += [
            "",
            f"OCR 文字层：{ocr.get('status', '') or '—'}",
            f"  引擎：{engine}",
            f"  识别 {ocr.get('completed', 0)} · 失败 {ocr.get('failed', 0)} · "
            f"跳过 {ocr.get('skipped', 0)} · 复用 {ocr.get('reused', 0)}",
            f"  文字 {ocr.get('total_text_chars', 0)} 字符 · "
            f"覆盖率 {ocr.get('coverage', 0) * 100:.0f}% · 耗时 {ocr.get('elapsed_seconds', 0)} s",
        ]
        if ocr.get("reason"):
            lines.append(f"  说明：{ocr['reason']}")
    else:
        lines += ["", "OCR 文字层：关（普通图像型 PDF，导出与 M5 一致）"]
    warnings = result.get("warnings", [])
    if warnings:
        lines += ["", "警告："] + [f"  - {w}" for w in warnings]
    if issues:
        lines += ["", "验证问题："] + [f"  - {i}" for i in issues]
    return lines


def show_result(parent, controller, result: dict, terminal_state=None, on_resume=None) -> tk.Toplevel:
    """打开结果窗口（非模态）。返回 Toplevel，便于测试检查与销毁。

    on_resume(task_dir) 可选：当结果可恢复（resumable）时提供“恢复续跑”按钮（M5.4）。
    """
    win = tk.Toplevel(parent)
    win.title("处理结果")
    win.geometry("780x620")
    win.minsize(640, 480)
    win.transient(parent)

    # ---- 汇总 -----------------------------------------------------------
    summary_frame = ttk.LabelFrame(win, text="结果汇总", padding=(8, 6))
    summary_frame.pack(side="top", fill="x", padx=10, pady=(10, 4))
    summary = tk.Text(summary_frame, height=12, wrap="word", padx=6, pady=4)
    summary.insert("1.0", "\n".join(_summary_lines(result, terminal_state)))
    summary.configure(state="disabled")
    summary.pack(fill="x")
    win.summary_text = summary

    # ---- 失败/回退页列表 ------------------------------------------------
    nb = ttk.Notebook(win)
    nb.pack(side="top", fill="both", expand=True, padx=10, pady=4)
    failed = result.get("failed_pages", [])
    fallback = result.get("fallback_pages", [])

    failed_tab = ttk.Frame(nb)
    nb.add(failed_tab, text=f"失败页（{len(failed)}）")
    win.failed_tree = _build_tree(
        failed_tab, ("display_index", "stage", "reason"),
        ("页码", "阶段", "原因"), (60, 80, 480),
        [(p["display_index"], _STAGE_LABEL.get(p["stage"], p["stage"]), p["reason"]) for p in failed])

    fallback_tab = ttk.Frame(nb)
    nb.add(fallback_tab, text=f"回退页（{len(fallback)}）")
    win.fallback_tree = _build_tree(
        fallback_tab, ("display_index", "level", "reason", "source"),
        ("页码", "回退级别", "原因", "采用来源"), (60, 90, 360, 120),
        [(p["display_index"], _LEVEL_LABEL.get(p["level"], p["level"]), p["reason"], p["source"]) for p in fallback])

    # ---- 页详情（一键定位） --------------------------------------------
    detail_frame = ttk.LabelFrame(win, text="页面详情（选中上方列表行）", padding=(8, 6))
    detail_frame.pack(side="top", fill="x", padx=10, pady=4)
    detail = tk.Text(detail_frame, height=5, wrap="word", padx=6, pady=4)
    detail.insert("1.0", "在“失败页”或“回退页”列表中选择一行，查看该页详情并定位。")
    detail.configure(state="disabled")
    detail.pack(fill="x")
    win.detail_text = detail

    selected: dict = {}

    def _show_detail(page: dict, kind: str):
        selected.clear()
        selected.update(page)
        if kind == "failed":
            head = f"失败页 · 第 {page['display_index']} 页 · 阶段：{_STAGE_LABEL.get(page['stage'], page['stage'])}"
        else:
            head = f"回退页 · 第 {page['display_index']} 页 · 级别：{_LEVEL_LABEL.get(page['level'], page['level'])}"
        lines = [head, f"原因：{page.get('reason') or '—'}"]
        if page.get("source"):
            lines.append(f"采用来源：{page['source']}")
        lines.append(f"该页图像：{page.get('image_path') or '—'}")
        lines.append(f"原始输入：{page.get('source_path') or '—'}")
        detail.configure(state="normal")
        detail.delete("1.0", "end")
        detail.insert("1.0", "\n".join(lines))
        detail.configure(state="disabled")

    def _on_select(_event=None):
        for tree, pages, kind in ((win.failed_tree, failed, "failed"), (win.fallback_tree, fallback, "fallback")):
            sel = tree.selection()
            if sel:
                idx = tree.index(sel[0])
                if 0 <= idx < len(pages):
                    _show_detail(pages[idx], kind)
                return

    win.failed_tree.bind("<<TreeviewSelect>>", _on_select)
    win.fallback_tree.bind("<<TreeviewSelect>>", _on_select)

    detail_btns = ttk.Frame(detail_frame)
    detail_btns.pack(fill="x", pady=(4, 0))

    def _open_page_image():
        path = selected.get("image_path") or selected.get("source_path")
        if not path:
            messagebox.showinfo("未选择页面", "请先在列表中选择一页。", parent=win)
            return
        if not controller.open_path(path):
            messagebox.showerror("打开失败", f"无法打开该页图像：\n{path}", parent=win)

    def _open_page_folder():
        path = selected.get("image_path") or selected.get("source_path")
        if not path:
            messagebox.showinfo("未选择页面", "请先在列表中选择一页。", parent=win)
            return
        if not controller.open_path(Path(path).parent):
            messagebox.showerror("打开失败", f"无法打开所在文件夹：\n{Path(path).parent}", parent=win)

    ttk.Button(detail_btns, text="打开该页图像", command=_open_page_image).pack(side="left")
    ttk.Button(detail_btns, text="打开所在文件夹", command=_open_page_folder).pack(side="left", padx=4)

    # ---- 底部操作 -------------------------------------------------------
    bar = ttk.Frame(win, padding=(10, 6))
    bar.pack(side="bottom", fill="x")

    def _open_output_pdf():
        path = result.get("output_path")
        if not result.get("output_exists") or not path:
            messagebox.showinfo("无正式输出", "尚未生成通过验证的正式 PDF。", parent=win)
            return
        if not controller.open_path(path):
            messagebox.showerror("打开失败", f"无法打开 PDF：\n{path}", parent=win)

    def _open_output_folder():
        path = result.get("output_path")
        target = Path(path).parent if path else controller.resolve_output_dir()
        if not controller.open_path(target):
            messagebox.showerror("打开失败", f"无法打开输出文件夹：\n{target}", parent=win)

    def _open_report():
        path = result.get("report_path")
        if not path or not Path(path).exists():
            messagebox.showinfo("无报告", "本次任务未生成导出报告（可能未完成导出）。", parent=win)
            return
        if not controller.open_path(path):
            messagebox.showerror("打开失败", f"无法打开报告：\n{path}", parent=win)

    def _open_log():
        if not controller.open_log():
            messagebox.showerror("打开失败", "无法打开错误日志，请检查日志目录。", parent=win)

    win.open_pdf_btn = ttk.Button(bar, text="打开 PDF", command=_open_output_pdf)
    win.open_pdf_btn.pack(side="left")
    if not result.get("output_exists"):
        win.open_pdf_btn.state(["disabled"])
    win.open_folder_btn = ttk.Button(bar, text="打开输出文件夹", command=_open_output_folder)
    win.open_folder_btn.pack(side="left", padx=4)
    win.open_report_btn = ttk.Button(bar, text="查看详细报告", command=_open_report)
    win.open_report_btn.pack(side="left", padx=4)
    win.open_log_btn = ttk.Button(bar, text="查看错误日志", command=_open_log)
    win.open_log_btn.pack(side="left", padx=4)

    def _resume():
        if on_resume is not None:
            on_resume(result.get("task_dir", ""))
            win.destroy()

    win.resume_btn = None
    if result.get("resumable") and on_resume is not None:
        win.resume_btn = ttk.Button(bar, text="恢复续跑", command=_resume)
        win.resume_btn.pack(side="left", padx=4)
    ttk.Button(bar, text="关闭", command=win.destroy).pack(side="right")

    return win


def _build_tree(parent, columns, headings, widths, rows) -> ttk.Treeview:
    tree = ttk.Treeview(parent, columns=columns, show="headings", height=8)
    for col, head, width in zip(columns, headings, widths):
        tree.heading(col, text=head)
        tree.column(col, width=width, anchor="w")
    for row in rows:
        tree.insert("", "end", values=row)
    vsb = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    tree.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")
    return tree