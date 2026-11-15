# BookClean PDF

完全本地运行的 Windows 桌面工具：把纸质图书扫描件和手机拍摄页面修复为背景干净、页面端正、文字清楚的图像型 PDF。以内容保真为最高优先级，不损伤文字、脚注、页码和插图。

- 目标平台：Windows 10/11 64 位，Python 3.12
- 完全离线：不上传文件、不联网检查更新、不自动下载模型
- 基础版不含 PyTorch / PaddlePaddle / TensorFlow / CUDA / 大型 OCR 运行时
- 原始输入文件永远只读，输出与临时文件独立管理

需求与开发规范以《BookClean PDF 项目说明书》为准（见 AGENT.md 摘要）。

## 当前状态

M0–M4.8 已完成：输入/PDF 渲染、图像修复、质量检查与安全回退、扫描黑边保守清除、
PDF 导出与验证、断点续跑、阅读器兼容审查（既有 501 项测试全部保留）。
M5（桌面 GUI）已完成：M5.0 架构、M5.1 主界面与导入、M5.2 后台任务与真实进度、
M5.3 结果与异常、M5.4 恢复与稳定、M5.5 打包/验收（全量 656 项测试通过）。
默认 `python app.py` 启动 Tkinter 桌面界面（导入/配置/开始处理/真实进度/取消/输出目录/任务详情/历史恢复）；
后台 worker 单线程顺序调用内核，进度来自真实回调，取消为协作式（处理/黑边逐页，导入/导出/验证阶段边界）。
M5.5 产出 PyInstaller onedir 构建 + Inno 安装包（离线、免装 Python、卸载保留用户数据），
并经打包冒烟/诊断/GUI/长任务内存/安装卸载实测。命令行入口与 `--diagnose`/`--version`/`--smoke` 保持兼容。
正式发布验收（干净 Windows、断网、许可证逐项核对）属 M7。

## 开发环境搭建

```bat
cd BookCleanPDF
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

依赖精确版本见 `requirements-lock.txt`（已在 Windows x64 + Python 3.12.10 实测验证）。
如直连 PyPI 超时，可为 pip 指定本地代理：`--proxy http://127.0.0.1:7897`。

## 运行

```bat
scripts\run_dev.bat               rem 启动桌面 GUI（M5.2：导入/配置/后台处理/真实进度/取消/详情）
scripts\run_dev.bat --diagnose    rem 环境诊断（依赖/功能链/磁盘/写权限）
scripts\run_dev.bat --version     rem 显示版本
.venv\Scripts\python.exe -m pytest rem 运行测试
.venv\Scripts\python.exe scripts\inspect_input.py "文件或文件夹" --dpi 300  rem 导入检查
.venv\Scripts\python.exe scripts\process_pages.py "任务目录" --preset auto --compare 5  rem M2 图像处理
.venv\Scripts\python.exe scripts\export_pdf.py "任务目录" -o out.pdf  rem M4 PDF 导出
```

`process_pages.py` 读取 M1 任务目录的 `original_pages/`，输出到 `processed_pages/`（原件只读），
兼容扩展 `manifest.json` 并生成 `processing_report.json` 与处理前后联系表。
预设：`auto`（Auto Clean，默认）/ `old_book`（Old Book）/ `phone_photo`（Photo）/ `faithful`（Preserve）。

## 打包与安装（M5.5）

```bat
scripts\build_windows.bat                 rem PyInstaller onedir 构建 + 体积实测（dist/BookCleanPDF）
"C:\Program Files\Inno Setup 7\ISCC.exe" packaging\installer.iss   rem 编译安装包（dist/installer）
dist\BookCleanPDF\BookCleanPDF.exe --smoke      rem 打包全链路冒烟（headless，写 smoke_result.json）
dist\BookCleanPDF\BookCleanPDF.exe --diagnose   rem 打包环境诊断
.venv\Scripts\python.exe scripts\measure_frozen_memory.py 40   rem frozen 长任务峰值内存实测
.venv\Scripts\python.exe scripts\verify_installer.py           rem 静默安装/卸载 + 用户数据保留核验
```

- 交付为 **onedir**（`dist/BookCleanPDF/`，运行时 Python 已捆绑，用户无需另装 Python）+ Inno 安装包
  `dist/installer/BookCleanPDF-0.0.1-setup.exe`（per-user 安装，无需 UAC/命令行，离线）。
- 实测体积：dist **160.53 MB**（1082 文件，<200MB 目标）；安装包 **42.31 MB**；安装后 **165.05 MB**。
- 卸载仅移除安装目录；用户数据 `%LOCALAPPDATA%\BookCleanPDF` 与输出 PDF（`文档\BookCleanPDF`）**不删除**。
- 第三方许可证随包分发于 `_internal/THIRD_PARTY_LICENSES/`（M5.5 收集，M7 逐项合规核对）。
- 干净 Windows / 断网 / 许可证逐项核对等正式发布验收属 **M7**，M5.5 不声称完成。

## 目录结构

```text
app.py                  程序入口（默认启动 GUI；--diagnose/--version 保持兼容）
src/bookclean/          核心包
  config.py             TaskConfig / BorderCleanConfig / PDFExportConfig 集中参数与统一校验
  models.py             PageData / PageFeatures / ProcessResult
  errors.py             统一异常类型
  logging_setup.py      本地日志（不记录图像内容）
  pdf_reader.py         PDF 逐页渲染（pypdfium2，密码/损坏/旋转处理）
  image_loader.py       图片读取（EXIF 像素校正、多帧 TIFF、色彩统一）
  input_adapter.py      统一输入入口（文件夹自然排序、全局页序）
  task_manager.py       导入任务（独立任务目录、manifest、缩略图、联系表）
  page_analyzer.py      M2 页面分析（倾斜/四边形/背景/泛黄/噪声/彩色/前景）
  geometry_corrector.py M2 几何校正（纠斜/保守裁边/高置信透视，可跳过）
  background_cleaner.py M2 背景清理（背景估计/均一化/Lab 去黄/去噪/前景保护）
  text_enhancer.py      M2 文字增强（Clean Gray/Sauvola 二值/自动保色）
  quality_checker.py    M2 质量检查与自动回退决策（传统指标，非 OCR）
  presets.py            M2 统一预设（Auto Clean/Old Book/Photo/Preserve）
  processing_pipeline.py M2 统一流水线 + 批量 runner（兼容扩展 Manifest，支持取消/续跑）
  border_detector.py    M4.8.1 扫描黑边保守清除（五重守卫+置信度门控）
  pdf_exporter.py       M4 PDF 导出（逐页编码、.partial+验证+replace、回退解析）
  pdf_verifier.py       M4 PDF 验证（尺寸/渲染/灰度色空间）
  recovery.py           M4.8.4 断点续跑 8 维校验
  user_paths.py         M5.1 开发/打包路径解析（不依赖开发绝对路径）
  task_events.py        M5.2 GUI 状态机词汇 + worker→UI 不可变事件协议
  task_worker.py        M5.2/M5.4 后台 worker（单线程顺序调内核，回调只入队，协作取消；resume 续跑+导入即登记历史）
  app_controller.py     M5.1–M5.4 应用服务层（GUI↔内核唯一桥梁，不导入 Tkinter；start/cancel/poll/resume/history/load_result）
  error_messages.py     M5.3 异常→友好提示纯映射（不导入 Tkinter/内核，未知类型保留详情）
  task_result.py        M5.3/M5.4 只读 manifest 派生结果视图（逐页总数、回退不计成功、仅正式 PDF 计完成、state/resumable）
  task_history.py       M5.4 历史索引 + GUI 配置持久化（不复制页面数据/不改 Manifest，去重置顶上限 50）
  frozen_smoke.py       M5.5 打包冒烟 harness（headless 复用 AppController→worker→内核；矩阵/重开/续跑/异常/原件只读）
  ocr/                  M6.1–M6.5 可选 OCR 插件基础设施（Provider/PluginManager/WorkerClient/rapid worker/fake + PP-OCRv5 只读桥接 worker/行重建 + 分发打包/离线安装校验；不被 M1–M5 导入，引擎在独立插件 runtime 子进程运行，桥接只读复用用户既有 DeepSeekOCR_v2 安装、0 修改）
  utils/                自然排序、原子写入、图像标准化
  ui/                   Tkinter 界面（main_window 进度/轮询/取消/关闭守卫/历史/恢复 + import/settings/result/history/dnd/dpi）
tests/                  pytest 测试
assets/  models/optional/
packaging/              pyinstaller.spec（onedir）+ installer.iss（Inno，卸载保留用户数据）
scripts/                开发/构建/实测脚本（build_windows / measure_dist / measure_frozen_memory / verify_installer / collect_licenses / export_pdf / ocr_run）
THIRD_PARTY_LICENSES/   第三方许可证（M5.5 已收集随包分发；M7 逐项合规核对）
logs/  temp/            本地日志与临时文件
```

后续模块（input_adapter / page_analyzer / geometry / background / quality / exporter 等）按里程碑 M1–M4 加入，保持 UI、任务管理、图像算法、PDF 输出职责分离。

## 里程碑

- [x] M0 环境与项目骨架
- [x] M1 输入与 PDF 渲染
- [x] M2 基础图像修复核心
- [x] M3 质量检查和安全回退
- [x] M4 PDF 输出与批量任务（含 M4.8 工程质量完善）
- [x] M5 桌面 UI（M5.0 架构、M5.1 主界面与导入、M5.2 后台任务与真实进度、M5.3 结果展示与异常处理、M5.4 任务恢复与稳定、M5.5 打包/验收 全部完成）
- [x] M6 可选本地 OCR 与可搜索 PDF（M6.0 方案确认、M6.1 插件基础设施 + RapidOCR Provider、M6.2 可搜索文字层 W2、M6.3 OCR 任务集成、M6.4 OCR GUI 集成、M6.5 PP-OCRv5 只读桥接 + 插件分发 全部完成：843 测试通过；W2 pikepdf 不嵌字体文字层，像素差 0、提取/搜索全对、坐标 ≤3pt、OCR 关导出与 M5 内容字节一致；M6.3 OcrStageRunner 接入流水线（border 后/export 前），缓存键=图像sha+引擎指纹+识别参数+schema（布局不入键），Manifest 只增 ocr_*，失败隔离/续跑/取消/插件降级，300 页+取消+续跑+强杀恢复全过、failed 页不毁导出、worker 峰值 <1GB；M6.4 设置 OCR 组/预检/进度阶段/结果统计/历史快照，真实 GUI 全流程+未装插件指引截图验证，GUI/控制器 import 期不导入 OCR 代码；M6.5 桥接只读复用用户既有 DeepSeekOCR_v2 安装（0 下载/0 修改，SHA256+src+__pycache__ 只读证明），词级碎行重建为 PDFium 提取不动点序（提取==OCR 全 10 页），引擎切换缓存失效/二次全复用/取消续跑/插件移除降级全过，worker 峰值≈5.7GB GUI 警示，分发 zip 106.31MB sha256 校验+离线安装往返，基础安装包仅 +0.055MB（零新运行时依赖）；OCR 为独立可选插件，缺插件时导出行为与 M5 完全一致）
- [ ] M7 打包与发布验证（进行中：M7.0 审查计划、M7.1 许可证合规、M7.2 干净 Windows 安装运行（本机）、M7.3 完全断网验证（本机，四项验收通过）、M7.4 全面功能与稳定性回归（本机通过：843 测试全保留全过 0 跳过/xfail；12 组合功能矩阵八项检查全过、OCR 开关渲染逐对差 0；OCR 专项 12 类全覆盖、清洁合成页 max CER 0.0698≤0.15、真实页保留原始文件+页索引仅作功能覆盖；文字层一致性像素差 0、坐标偏差 2.4pt≤3、搜索/复制/提取全对；稳定性/性能 all_ok，桥接 worker 峰值 5231MB 与 GUI 5.7GB 警示一致、D1 无插件/D2 初始化失败安全降级导出仍 passed、取消续跑/强杀恢复全过；零产品代码改动、规格最低要求未减；真实 OOM 强制与干净机专属门未验证）已完成；M7.5 正式发布构建已完成（本机）：三件分发物重建（基础 onedir 161.56MiB／安装包 44.46MB／OCR 插件 zip 103.42MB），修复三个发布阻断——R1 插件自包含独立 Python（无 pyvenv.cfg，prefix==base_prefix==runtime）、R2 frozen GUI OCR worker 落盘（ocr_smoke ok，153 字符识别/161 提取/可搜索/failed 0）、R3 安装重命名抗杀毒瞬时锁（_robust_rename 仅退避重试 WinError 5/32）；真实隔离安装→运行→卸载 ok（用户数据/输出保留）、损坏 zip/模型篡改/不完整安装负面测试全部正确拒绝、发布包污染扫描 ok、SHA256 re-verify all_match、全量回归 843 passed/0 failed/0 skipped/0 xfail（PYTEST_EXIT=0，与 M7.4 基线完全一致，未降标准/未删测试）；发布目录 release/1.0.0/（149.66MB，93 文件，含安装包+插件 zip+SHA256SUMS+README+VERSION+第三方许可证）；干净机专属门按 R4 标未验证；M7.6 最终报告待做）