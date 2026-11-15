"""集中管理的任务参数。算法文件不得各自写死默认值。"""

from dataclasses import dataclass, fields

OUTPUT_MODES = ("clean_gray", "adaptive_bw", "mixed")
PRESETS = ("auto", "old_book", "phone_photo", "faithful")
COMPRESSION_PRESETS = ("high_quality", "balanced", "compact")

DPI_MIN, DPI_MAX = 72, 600

# M4 PDF 导出参数取值
EXPORT_COMPRESSION_MODES = ("lossless", "balanced", "compact")
EXPORT_COLOR_MODES = ("preserve", "grayscale")
EXPORT_PAGE_SIZE_MODES = ("preserve", "auto", "A4", "A5", "B5", "Letter", "custom")
EXPORT_FIT_MODES = ("contain", "cover")
EXPORT_TARGET_DPIS = (200, 300, 400)

# M6.3 可选本地 OCR 引擎选择（auto=按推荐序选已装插件）
OCR_ENGINES = ("auto", "ocr-rapid", "ocr-paddle-bridge")


@dataclass
class TaskConfig:
    preset: str = "auto"
    render_dpi: int = 300
    output_dpi: int = 300
    output_mode: str = "clean_gray"
    background_strength: float = 0.5
    yellow_removal_strength: float = 0.5
    denoise_strength: float = 0.2
    sharpen_strength: float = 0.15
    enable_deskew: bool = True
    enable_crop: bool = True
    enable_perspective: bool = False
    enable_dewarp: bool = False
    enable_binarization: bool = False
    compression_preset: str = "balanced"
    output_dir: str = ""
    # M2 几何校正阈值（启发式门槛，不是统计置信概率）
    min_skew_confidence: float = 0.5
    deskew_max_angle: float = 8.0
    crop_safety_margin: float = 0.01
    min_perspective_confidence: float = 0.7
    min_perspective_score: float = 0.12
    binarization_k: float = 0.2  # Sauvola 局部阈值参数 k
    # M3.2 区域化背景清理门槛：低于此置信度/纸张面积不强制区域漂白（保留几何结果）
    min_paper_confidence: float = 0.6
    min_paper_area_ratio: float = 0.15
    # M3.3 分区质量阈值（启发式，经合成样本校准；非 OCR 准确率）
    quality_bg_std_ratio_max: float = 1.3      # 纸张背景 std 相对原图允许上升倍率
    quality_text_core_min: float = 0.90        # 文字核心保留率下限（低于此值=严重文字损伤）
    quality_color_chroma_min: float = 0.90     # 彩色色度保留率下限（低于此值=严重褪色）
    quality_color_area_min: float = 0.50       # 彩色面积保留率下限
    quality_illustration_std_min: float = 0.70  # 插图灰度动态范围保留下限
    quality_photo_blank_std_min: float = 5.0   # 照片页判“被抹空”的灰度 std 下限（照片前景占比指标不适用）
    quality_risk_accept_max: float = 30.0      # 风险评分(0..100)高于此值即使无严重项也升级重试

    def validate(self) -> None:
        """在配置入口统一校验，非法参数立即报错，不允许传入算法后静默失败。"""
        if self.preset not in PRESETS:
            raise ValueError(f"preset 必须是 {PRESETS} 之一，收到: {self.preset!r}")
        if self.output_mode not in OUTPUT_MODES:
            raise ValueError(f"output_mode 必须是 {OUTPUT_MODES} 之一，收到: {self.output_mode!r}")
        if self.compression_preset not in COMPRESSION_PRESETS:
            raise ValueError(
                f"compression_preset 必须是 {COMPRESSION_PRESETS} 之一，收到: {self.compression_preset!r}"
            )
        for name in ("render_dpi", "output_dpi"):
            value = getattr(self, name)
            if not isinstance(value, int) or not DPI_MIN <= value <= DPI_MAX:
                raise ValueError(f"{name} 必须是 {DPI_MIN}..{DPI_MAX} 的整数，收到: {value!r}")
        for f in fields(self):
            if f.name.endswith("_strength"):
                value = getattr(self, f.name)
                if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"{f.name} 必须在 0.0..1.0 之间，收到: {value!r}")
        if self.enable_dewarp:
            raise ValueError("页面展平属于 P2 可选扩展，基础版不得启用 (enable_dewarp=True)")
        for name in ("min_skew_confidence", "min_perspective_confidence", "min_perspective_score",
                     "min_paper_confidence", "min_paper_area_ratio"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须在 0.0..1.0 之间，收到: {value!r}")
        if not isinstance(self.deskew_max_angle, (int, float)) or not 0.0 <= self.deskew_max_angle <= 15.0:
            raise ValueError(f"deskew_max_angle 必须在 0.0..15.0 之间，收到: {self.deskew_max_angle!r}")
        if not isinstance(self.crop_safety_margin, (int, float)) or not 0.0 <= self.crop_safety_margin <= 0.1:
            raise ValueError(f"crop_safety_margin 必须在 0.0..0.1 之间，收到: {self.crop_safety_margin!r}")
        if not isinstance(self.binarization_k, (int, float)) or not 0.0 <= self.binarization_k <= 1.0:
            raise ValueError(f"binarization_k 必须在 0.0..1.0 之间，收到: {self.binarization_k!r}")
        for name in ("quality_text_core_min", "quality_color_chroma_min", "quality_color_area_min",
                     "quality_illustration_std_min"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须在 0.0..1.0 之间，收到: {value!r}")
        if not isinstance(self.quality_bg_std_ratio_max, (int, float)) or not 1.0 <= self.quality_bg_std_ratio_max <= 5.0:
            raise ValueError(f"quality_bg_std_ratio_max 必须在 1.0..5.0 之间，收到: {self.quality_bg_std_ratio_max!r}")
        if not isinstance(self.quality_photo_blank_std_min, (int, float)) or not 0.0 <= self.quality_photo_blank_std_min <= 100.0:
            raise ValueError(f"quality_photo_blank_std_min 必须在 0.0..100.0 之间，收到: {self.quality_photo_blank_std_min!r}")
        if not isinstance(self.quality_risk_accept_max, (int, float)) or not 0.0 <= self.quality_risk_accept_max <= 100.0:
            raise ValueError(f"quality_risk_accept_max 必须在 0.0..100.0 之间，收到: {self.quality_risk_accept_max!r}")


@dataclass
class BorderCleanConfig:
    """M4.8.1 扫描黑边清除参数（保守策略：宁可留边，不可切内容）。

    检测在降采样灰度图上进行；所有门槛为启发式阈值，不是统计置信概率。
    仅当某条边同时通过覆盖率/纹理/亮内容/跨越/纸面确认五重守卫，
    且置信度 >= min_confidence 时才裁切该边，并保留安全余量薄边。
    """

    min_confidence: float = 0.6        # 单边执行裁切的最低置信度
    max_edge_fraction: float = 0.12    # 单边黑边宽度上限（占该方向尺寸比例），超过视为异常不裁
    min_paper_fraction: float = 0.20   # 中央区域亮像素占比下限，低于此值判为照片/暗页跳过
    safety_margin_fraction: float = 0.004  # 裁切线内侧保留的安全薄边（占该方向尺寸比例）
    dark_delta: float = 60.0           # 黑边判定：亮度低于 纸张亮度 - dark_delta
    dark_abs_max: float = 140.0        # 黑边绝对亮度上限（与 dark_delta 取较小阈值）
    min_coverage: float = 0.85         # 边带内暗像素覆盖率下限
    analysis_max_dim: int = 1000       # 检测用降采样最长边
    texture_std_max: float = 10.0      # 边带局部标准差上限（超过=有纹理/文字）
    max_texture_fraction: float = 0.03  # 边带内高纹理像素占比上限
    max_bright_fraction: float = 0.005  # 边带内亮像素占比上限（超过=黑底白字等内容）
    bright_cc_min_area: int = 15       # 全分辨率边带内最大亮连通域面积上限（超过=字形/内容，灰尘小点放行）
    straddle_guard_fraction: float = 0.02  # 裁切线内侧守卫条带宽度（占该方向尺寸比例）
    straddle_cc_fraction: float = 0.20  # 守卫条带内最大暗连通域占比上限（超过=插图/照片压边）
    min_crop_area_fraction: float = 0.30  # 裁切后面积占原图下限，低于此值放弃裁切

    def validate(self) -> None:
        for name in ("min_confidence", "min_paper_fraction", "min_coverage",
                     "max_texture_fraction", "max_bright_fraction", "straddle_cc_fraction",
                     "min_crop_area_fraction"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须在 0.0..1.0 之间，收到: {value!r}")
        if not isinstance(self.max_edge_fraction, (int, float)) or not 0.0 < self.max_edge_fraction <= 0.5:
            raise ValueError(f"max_edge_fraction 必须在 0.0..0.5 之间，收到: {self.max_edge_fraction!r}")
        if not isinstance(self.safety_margin_fraction, (int, float)) or not 0.0 <= self.safety_margin_fraction <= 0.05:
            raise ValueError(f"safety_margin_fraction 必须在 0.0..0.05 之间，收到: {self.safety_margin_fraction!r}")
        if not isinstance(self.straddle_guard_fraction, (int, float)) or not 0.0 < self.straddle_guard_fraction <= 0.2:
            raise ValueError(f"straddle_guard_fraction 必须在 0.0..0.2 之间，收到: {self.straddle_guard_fraction!r}")
        for name in ("dark_delta", "dark_abs_max", "texture_std_max"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0.0 < value <= 255.0:
                raise ValueError(f"{name} 必须在 0.0..255.0 之间，收到: {value!r}")
        if not isinstance(self.analysis_max_dim, int) or not 200 <= self.analysis_max_dim <= 4000:
            raise ValueError(f"analysis_max_dim 必须是 200..4000 的整数，收到: {self.analysis_max_dim!r}")
        if not isinstance(self.bright_cc_min_area, int) or not 1 <= self.bright_cc_min_area <= 100000:
            raise ValueError(f"bright_cc_min_area 必须是 1..100000 的整数，收到: {self.bright_cc_min_area!r}")


@dataclass
class PDFExportConfig:
    """M4 PDF 导出参数。默认：balanced / preserve / contain / 验证开 / 不覆盖 / 不线性化。"""

    compression: str = "balanced"
    color_mode: str = "preserve"  # preserve=保留原彩色/灰度；grayscale=整本转 8-bit 单通道灰度
    target_dpi: int | None = None  # None=保留源分辨率；只允许下采样，不上采样
    page_size_mode: str = "preserve"  # preserve=PDF 保留原尺寸+照片自动；auto=全部按长边估算
    fit_mode: str = "contain"
    auto_long_edge_mm: float = 250.0  # 照片 Auto 模式长边参考尺寸
    custom_width_mm: float | None = None
    custom_height_mm: float | None = None
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    overwrite: bool = False
    linearize: bool = False
    verify: bool = True

    def validate(self) -> None:
        if self.compression not in EXPORT_COMPRESSION_MODES:
            raise ValueError(f"compression 必须是 {EXPORT_COMPRESSION_MODES} 之一，收到: {self.compression!r}")
        if self.color_mode not in EXPORT_COLOR_MODES:
            raise ValueError(f"color_mode 必须是 {EXPORT_COLOR_MODES} 之一，收到: {self.color_mode!r}")
        if self.page_size_mode not in EXPORT_PAGE_SIZE_MODES:
            raise ValueError(f"page_size_mode 必须是 {EXPORT_PAGE_SIZE_MODES} 之一，收到: {self.page_size_mode!r}")
        if self.fit_mode not in EXPORT_FIT_MODES:
            raise ValueError(f"fit_mode 必须是 {EXPORT_FIT_MODES} 之一（禁止拉伸），收到: {self.fit_mode!r}")
        if self.target_dpi is not None and self.target_dpi not in EXPORT_TARGET_DPIS:
            raise ValueError(f"target_dpi 必须是 None 或 {EXPORT_TARGET_DPIS} 之一，收到: {self.target_dpi!r}")
        if not isinstance(self.auto_long_edge_mm, (int, float)) or not 50.0 <= self.auto_long_edge_mm <= 1000.0:
            raise ValueError(f"auto_long_edge_mm 必须在 50..1000 mm 之间，收到: {self.auto_long_edge_mm!r}")
        if self.page_size_mode == "custom":
            for name in ("custom_width_mm", "custom_height_mm"):
                value = getattr(self, name)
                if not isinstance(value, (int, float)) or not 10.0 <= value <= 2000.0:
                    raise ValueError(f"custom 模式要求 {name} 在 10..2000 mm 之间，收到: {value!r}")
        for name in ("overwrite", "linearize", "verify"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} 必须是布尔值，收到: {getattr(self, name)!r}")


@dataclass
class OcrConfig:
    """M6.3 可选本地 OCR 文字层参数。默认关；关⇒导出与 M5 完全一致（不导入任何 OCR 代码）。

    OCR 引擎在独立插件 venv 子进程运行，基础包零新运行时依赖。本类仅含原始参数，
    不导入 bookclean.ocr（避免 config 被导出内核导入时连带引入 OCR 代码）；
    到 OcrParams 的转换在 ocr_stage 内完成。
    """

    enabled: bool = False
    engine: str = "auto"               # auto=按推荐序选已装插件
    threads: int = 4                   # ORT intra_op 线程
    lang: str = "auto"
    text_score: float = 0.5            # 识别文本置信度阈值
    page_timeout_s: float = 120.0      # 单页超时（操作参数，不入缓存键）
    restart_every_n_pages: int = 200   # 长文档每 N 页重启 worker 释放碎片（0=不重启）
    plugins_root: str = ""             # 空=默认用户插件目录

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(f"enabled 必须是布尔值，收到: {self.enabled!r}")
        if self.engine not in OCR_ENGINES:
            raise ValueError(f"engine 必须是 {OCR_ENGINES} 之一，收到: {self.engine!r}")
        if not isinstance(self.threads, int) or not 1 <= self.threads <= 64:
            raise ValueError(f"threads 必须是 1..64 的整数，收到: {self.threads!r}")
        if not isinstance(self.text_score, (int, float)) or not 0.0 <= self.text_score <= 1.0:
            raise ValueError(f"text_score 必须在 0.0..1.0 之间，收到: {self.text_score!r}")
        if not isinstance(self.page_timeout_s, (int, float)) or not 1.0 <= self.page_timeout_s <= 3600.0:
            raise ValueError(f"page_timeout_s 必须在 1..3600 秒之间，收到: {self.page_timeout_s!r}")
        if not isinstance(self.restart_every_n_pages, int) or not 0 <= self.restart_every_n_pages <= 100000:
            raise ValueError(f"restart_every_n_pages 必须是 0..100000 的整数，收到: {self.restart_every_n_pages!r}")