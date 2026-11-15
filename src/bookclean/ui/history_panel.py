"""历史任务面板（M5.4）：最近任务索引 + 恢复续跑 / 重开结果。

只展示 controller.get_history() 派生的事实（state/resumable 以任务目录 Manifest 为准，
索引 last_state 仅作线索）。恢复经 controller.resume_task（复用已验证页，绝不重跑导入）；
重开经 controller.load_task 派生结果视图。本面板不执行任何处理，不加载页面像素。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

_STATE_LABEL = {"completed": "已完成", "cancelled": "已取消", "incomplete": "未完成",
                "failed": "失败", "missing": "已丢失", "running": "运行中"}


def _fmt_created(value: str) -> str:
    return (value or "")[:19].replace("T", " ")


def show_history(parent, controller, on_resume=None, on_reopen=None) -> tk.Toplevel:
    """打开历史任务窗口（非模态）。返回 Toplevel，便于测试检查与销毁。

    on_resume(task_dir) / on_reopen(task_dir, output_path) 由主窗口提供，分别触发续跑与重开结果。
    """
    win = tk.Toplevel(parent)
    win.title("历史任务")
    win.geometry("820x460")
    win.minsize(680, 360)
    win.transient(parent)

    ttk.Label(win, text="最近任务（状态以任务目录为准；“未完成/已取消/失败”可恢复续跑，复用已验证页）",
              padding=(10, 8)).pack(side="top", fill="x")

    body = ttk.Frame(win, padding=(10, 0))
    body.pack(side="top", fill="both", expand=True)
    columns = ("label", "state", "pages", "ocr", "created", "output")
    headings = ("任务", "状态", "页数", "OCR", "创建时间", "输出 PDF")
    widths = (190, 76, 56, 50, 140, 240)
    tree = ttk.Treeview(body, columns=columns, show="headings", height=12)
    for col, head, width in zip(columns, headings, widths):
        tree.heading(col, text=head)
        tree.column(col, width=width, anchor="w")
    vsb = ttk.Scrollbar(body, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    tree.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")
    win.history_tree = tree

    bar = ttk.Frame(win, padding=(10, 6))
    bar.pack(side="bottom", fill="x")
    resume_btn = ttk.Button(bar, text="恢复续跑", state="disabled")
    resume_btn.pack(side="left")
    reopen_btn = ttk.Button(bar, text="查看结果", state="disabled")
    reopen_btn.pack(side="left", padx=4)
    ttk.Button(bar, text="刷新", command=lambda: _refresh()).pack(side="left", padx=4)
    ttk.Button(bar, text="关闭", command=win.destroy).pack(side="right")
    win.resume_btn = resume_btn
    win.reopen_btn = reopen_btn

    entries: list[dict] = []

    def _selected() -> dict | None:
        sel = tree.selection()
        if not sel:
            return None
        idx = tree.index(sel[0])
        return entries[idx] if 0 <= idx < len(entries) else None

    def _update_buttons(_event=None):
        e = _selected()
        if e is None:
            resume_btn.state(["disabled"])
            reopen_btn.state(["disabled"])
            return
        resume_btn.state(["!disabled"] if e.get("resumable") else ["disabled"])
        reopen_btn.state(["disabled"] if e.get("state") == "missing" else ["!disabled"])

    def _refresh():
        nonlocal entries
        entries = controller.get_history()
        tree.delete(*tree.get_children())
        for e in entries:
            ocr_on = bool((e.get("settings") or {}).get("ocr_enabled"))
            tree.insert("", "end", values=(
                e.get("label", ""), _STATE_LABEL.get(e.get("state", ""), e.get("state", "")),
                e.get("total_pages", 0), "开" if ocr_on else "关",
                _fmt_created(e.get("created_at", "")),
                e.get("output_path", "") or "—"))
        _update_buttons()

    def _do_resume():
        e = _selected()
        if e is None:
            return
        if not e.get("resumable"):
            messagebox.showinfo("不可恢复", "该任务已完成或已丢失，无法续跑。", parent=win)
            return
        if on_resume is not None:
            on_resume(e["task_dir"])

    def _do_reopen():
        e = _selected()
        if e is None:
            return
        if e.get("state") == "missing":
            messagebox.showinfo("任务已丢失", "任务目录不存在，无法查看结果。", parent=win)
            return
        if on_reopen is not None:
            on_reopen(e["task_dir"], e.get("output_path") or None)

    resume_btn.configure(command=_do_resume)
    reopen_btn.configure(command=_do_reopen)
    tree.bind("<<TreeviewSelect>>", _update_buttons)
    tree.bind("<Double-1>", lambda _e: _do_reopen())

    _refresh()
    return win