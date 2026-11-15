"""任务配置区域：仅暴露既有内核特性对应的普通用户选项，不展示复杂内部参数。

页面处理=preset，输出类型=color_mode，输出质量=compression，
清除扫描黑边=border_clean(M4.8.1)，允许覆盖=overwrite。输出目录在导入区域管理。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from ..app_controller import AppController, TaskSettings

_PRESETS = [("自动（推荐）", "auto"), ("旧书翻新", "old_book"),
            ("手机照片", "phone_photo"), ("最大保真", "faithful")]
_COLORS = [("彩色（保留原色）", "preserve"), ("灰度", "grayscale")]
_QUALITY = [("无损（体积最大）", "lossless"), ("均衡（推荐）", "balanced"), ("紧凑（体积最小）", "compact")]
# M6.5：auto + RapidOCR + PP-OCRv5 桥接（复用本机既有 DeepSeekOCR_v2 安装，0 下载）。
_OCR_ENGINES = [("自动（推荐）", "auto"), ("RapidOCR", "ocr-rapid"),
                ("PP-OCRv5 桥接（高精度）", "ocr-paddle-bridge")]


class SettingsPanel(ttk.Frame):
    def __init__(self, parent, controller: AppController, on_change=None):
        super().__init__(parent, padding=8)
        self.controller = controller
        self.on_change = on_change

        self.preset_var = tk.StringVar()
        self.color_var = tk.StringVar()
        self.quality_var = tk.StringVar()
        self.border_var = tk.BooleanVar(value=controller.settings.border_clean)
        self.overwrite_var = tk.BooleanVar(value=controller.settings.overwrite)
        self._combos: list = []
        self._checks: list = []

        self._preset_map = self._add_combo("页面处理：", _PRESETS, self.preset_var, controller.settings.preset)
        self._color_map = self._add_combo("输出类型：", _COLORS, self.color_var, controller.settings.color_mode)
        self._quality_map = self._add_combo("输出质量：", _QUALITY, self.quality_var, controller.settings.compression)

        self._checks.append(ttk.Checkbutton(self, text="清除扫描黑边（保守，宁可留边不切内容）",
                                            variable=self.border_var, command=self._sync))
        self._checks.append(ttk.Checkbutton(self, text="允许覆盖同名输出文件",
                                            variable=self.overwrite_var, command=self._sync))
        for c in self._checks:
            c.pack(anchor="w", pady=2)

        self._build_ocr_group()

    def _build_ocr_group(self):
        """M6.4 可搜索 PDF（OCR 文字层）设置组。默认关；关=一切如旧。

        引擎状态经控制器惰性查询（仅在勾选/点击检测时导入插件发现代码），
        故 OCR 关时构造 GUI 不导入任何 OCR 代码。
        """
        s = self.controller.settings
        self.ocr_enabled_var = tk.BooleanVar(value=s.ocr_enabled)
        self.ocr_engine_var = tk.StringVar()
        self.ocr_status_var = tk.StringVar(value="引擎状态：勾选后检测")

        group = ttk.LabelFrame(self, text="可搜索 PDF（OCR 文字层，可选）", padding=6)
        group.pack(fill="x", pady=(6, 2))
        self._ocr_group = group

        self._ocr_check = ttk.Checkbutton(
            group, text="生成可搜索 PDF（在图像优化后独立运行；失败页不影响 PDF 导出）",
            variable=self.ocr_enabled_var, command=self._on_ocr_toggle)
        self._ocr_check.pack(anchor="w")

        row = ttk.Frame(group)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="引擎：", width=12).pack(side="left")
        reverse = {key: disp for disp, key in _OCR_ENGINES}
        self.ocr_engine_var.set(reverse.get(s.ocr_engine, _OCR_ENGINES[0][0]))
        self._ocr_engine_map = {disp: key for disp, key in _OCR_ENGINES}
        self.ocr_combo = ttk.Combobox(row, textvariable=self.ocr_engine_var,
                                      values=[d for d, _ in _OCR_ENGINES],
                                      state="readonly", width=24)
        self.ocr_combo.pack(side="left")
        self.ocr_combo.bind("<<ComboboxSelected>>", lambda _e: self._sync())
        self._ocr_detect_btn = ttk.Button(row, text="检测引擎", command=self._refresh_ocr_status)
        self._ocr_detect_btn.pack(side="left", padx=6)
        self._ocr_manage_btn = ttk.Button(row, text="安装/管理引擎…", command=self._on_manage_engines)
        self._ocr_manage_btn.pack(side="left")

        ttk.Label(group, textvariable=self.ocr_status_var, anchor="w",
                  foreground="#333").pack(fill="x", pady=(2, 0))
        ttk.Label(group, text="ⓘ OCR 为独立可选插件，未安装时普通导出完全不受影响；全程离线，不上传文件。",
                  anchor="w", foreground="#666", wraplength=760).pack(fill="x")
        self._ocr_widgets = [self.ocr_combo, self._ocr_detect_btn, self._ocr_manage_btn]
        self._update_ocr_widgets()

    def _on_ocr_toggle(self):
        self._update_ocr_widgets()
        if self.ocr_enabled_var.get():
            self._refresh_ocr_status()
        self._sync()

    def _update_ocr_widgets(self):
        enabled = bool(self.ocr_enabled_var.get())
        self.ocr_combo.configure(state="readonly" if enabled else "disabled")
        for btn in (self._ocr_detect_btn, self._ocr_manage_btn):
            btn.state(["!disabled"] if enabled else ["disabled"])

    def _refresh_ocr_status(self):
        engine = self._ocr_engine_map.get(self.ocr_engine_var.get(), "auto")
        status = self.controller.ocr_plugin_status(engine)
        if status.get("installed"):
            text = (f"引擎状态：{status.get('engine_id', '')} "
                    f"{status.get('engine_version', '')} 已就绪")
            if status.get("engine_id") == "ocr-paddle-bridge":
                text += "　⚠ 高精度引擎：峰值内存约 5.7GB、初始化约 11s（实测），建议密集中文页选用"
            self.ocr_status_var.set(text)
        else:
            reason = status.get("reason", "") or "请安装 ocr-rapid 插件"
            if engine == "ocr-paddle-bridge":
                reason = ("需本机已安装 DeepSeekOCR_v2；运行 scripts/ocr_bridge_setup.py "
                          "配置桥接（复用既有安装，0 下载）")
            self.ocr_status_var.set(f"引擎状态：未安装（{reason}）")

    def _on_manage_engines(self):
        messagebox.showinfo("安装/管理 OCR 引擎", self.controller.ocr_install_help(), parent=self)
        self.controller.open_plugins_dir()
        self._refresh_ocr_status()

    def set_enabled(self, enabled: bool):
        """运行中锁定配置（防止处理途中改设置导致与 worker 配置不一致）。"""
        combo_state = "readonly" if enabled else "disabled"
        for combo in self._combos:
            combo.configure(state=combo_state)
        check_state = "!disabled" if enabled else "disabled"
        for c in self._checks:
            c.state([check_state])
        self._ocr_check.state([check_state])
        if enabled:
            self._update_ocr_widgets()
        else:
            self.ocr_combo.configure(state="disabled")
            for btn in (self._ocr_detect_btn, self._ocr_manage_btn):
                btn.state(["disabled"])

    def _add_combo(self, label, options, var, current_key):
        row = ttk.Frame(self)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=12).pack(side="left")
        mapping = {disp: key for disp, key in options}
        reverse = {key: disp for disp, key in options}
        var.set(reverse.get(current_key, options[0][0]))
        combo = ttk.Combobox(row, textvariable=var, values=[d for d, _ in options],
                             state="readonly", width=24)
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", lambda _e: self._sync())
        self._combos.append(combo)
        return mapping

    def _sync(self):
        s = self.controller.settings
        s.preset = self._preset_map[self.preset_var.get()]
        s.color_mode = self._color_map[self.color_var.get()]
        s.compression = self._quality_map[self.quality_var.get()]
        s.border_clean = bool(self.border_var.get())
        s.overwrite = bool(self.overwrite_var.get())
        s.ocr_enabled = bool(self.ocr_enabled_var.get())
        s.ocr_engine = self._ocr_engine_map.get(self.ocr_engine_var.get(), "auto")
        self.controller.set_settings(s)
        if self.on_change is not None:
            self.on_change()

    def collect(self) -> TaskSettings:
        self._sync()
        return self.controller.settings