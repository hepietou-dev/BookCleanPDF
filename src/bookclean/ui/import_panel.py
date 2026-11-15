"""文件导入区域：选择 PDF/图片/文件夹、拖拽导入、文件信息列表、输出目录显示与更改。

只负责展示与采集用户输入，所有元信息检查/配置映射委托 AppController（不复制内核逻辑）。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk

from ..app_controller import AppController

_KIND_LABELS = {"pdf": "PDF", "image": "图片", "folder": "文件夹", "file": "文件"}
_IMAGE_TYPES = [("图片", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"), ("所有文件", "*.*")]
_PDF_TYPES = [("PDF 文件", "*.pdf"), ("所有文件", "*.*")]


def format_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


class ImportPanel(ttk.Frame):
    def __init__(self, parent, controller: AppController, on_change=None):
        super().__init__(parent, padding=8)
        self.controller = controller
        self.on_change = on_change

        bar = ttk.Frame(self)
        bar.pack(fill="x")
        self._buttons = [
            ttk.Button(bar, text="选择 PDF", command=self._choose_pdf),
            ttk.Button(bar, text="选择图片", command=self._choose_images),
            ttk.Button(bar, text="选择文件夹", command=self._choose_folder),
            ttk.Button(bar, text="移除选中", command=self._remove_selected),
            ttk.Button(bar, text="清空", command=self._clear),
        ]
        for i, b in enumerate(self._buttons):
            b.pack(side="left", padx=(0 if i == 0 else 4, 0))

        self.drop_hint = ttk.Label(
            self, text="提示：可将 PDF / 图片 / 文件夹直接拖拽到窗口导入", foreground="#555"
        )
        self.drop_hint.pack(fill="x", pady=(6, 2))

        list_frame = ttk.Frame(self)
        list_frame.pack(fill="both", expand=True)
        cols = ("name", "kind", "pages", "size", "status")
        self.tree = ttk.Treeview(list_frame, columns=cols, show="headings", height=8)
        headers = {"name": ("文件名", 240), "kind": ("类型", 60), "pages": ("页数", 50),
                   "size": ("大小", 80), "status": ("状态", 260)}
        for c in cols:
            text, width = headers[c]
            self.tree.heading(c, text=text)
            self.tree.column(c, width=width, anchor="w" if c in ("name", "status") else "center")
        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.tree.tag_configure("error", foreground="#b00020")
        self.tree.tag_configure("ok", foreground="#1b5e20")

        self.summary = ttk.Label(self, text="尚未导入文件")
        self.summary.pack(fill="x", pady=(4, 2))

        out_row = ttk.Frame(self)
        out_row.pack(fill="x", pady=(4, 0))
        ttk.Label(out_row, text="输出目录：").pack(side="left")
        self.out_var = tk.StringVar(value=str(controller.resolve_output_dir()))
        self.out_entry = ttk.Entry(out_row, textvariable=self.out_var, state="readonly")
        self.out_entry.pack(side="left", fill="x", expand=True, padx=4)
        self._change_btn = ttk.Button(out_row, text="更改…", command=self._change_output_dir)
        self._change_btn.pack(side="left")

        self.refresh()

    def set_enabled(self, enabled: bool):
        """运行中锁定导入操作（列表仍可查看，但禁止增删/改输出目录）。"""
        state = "!disabled" if enabled else "disabled"
        for b in self._buttons:
            b.state([state])
        self._change_btn.state([state])

    # ---- 导入动作 -------------------------------------------------------

    def _choose_pdf(self):
        paths = filedialog.askopenfilenames(title="选择 PDF", filetypes=_PDF_TYPES)
        if paths:
            self.add_paths(paths)

    def _choose_images(self):
        paths = filedialog.askopenfilenames(title="选择图片（可多选）", filetypes=_IMAGE_TYPES)
        if paths:
            self.add_paths(paths)

    def _choose_folder(self):
        path = filedialog.askdirectory(title="选择文件夹")
        if path:
            self.add_paths([path])

    def add_paths(self, paths):
        """导入入口：按钮选择与拖拽共用此方法（拖拽回调即调用这里）。"""
        self.controller.inspect_inputs(paths)
        self.refresh()
        self._notify()

    def _remove_selected(self):
        for iid in self.tree.selection():
            info = self.controller.inputs[int(iid)]
            self.controller.remove_input(info.path)
        self.refresh()
        self._notify()

    def _clear(self):
        self.controller.clear_inputs()
        self.refresh()
        self._notify()

    def _change_output_dir(self):
        path = filedialog.askdirectory(title="选择输出目录")
        if not path:
            return
        settings = self.controller.settings
        settings.output_dir = path
        self.controller.set_settings(settings)
        self.out_var.set(path)
        self._notify()

    def _notify(self):
        if self.on_change is not None:
            self.on_change()

    # ---- 展示刷新 -------------------------------------------------------

    def refresh(self):
        self.tree.delete(*self.tree.get_children())
        infos = self.controller.inputs
        for idx, info in enumerate(infos):
            if info.error:
                status = info.error
                tag = "error"
            else:
                status = "可处理"
                tag = "ok"
            self.tree.insert(
                "", "end", iid=str(idx),
                values=(info.name, _KIND_LABELS.get(info.kind, info.kind),
                        info.page_count if info.page_count else "-",
                        format_size(info.size_bytes), status),
                tags=(tag,),
            )
        valid = [i for i in infos if i.error is None]
        pages = sum(i.page_count for i in valid)
        self.summary.configure(
            text=f"共 {len(infos)} 项，有效 {len(valid)} 项 / {pages} 页，问题 {len(infos) - len(valid)} 项"
        )
        self.out_var.set(str(self.controller.resolve_output_dir()))