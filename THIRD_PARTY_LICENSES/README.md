# 第三方许可证

本目录在 M7 打包阶段收集随安装包分发的全部第三方许可证文本与版权声明。

需逐项核对（以实际 wheel 内容为准，不能仅凭库名判断）：

- numpy：BSD-3-Clause
- opencv-python-headless：OpenCV 本体 Apache-2.0；wheel 构建脚本 MIT；需检查 wheel 内 FFmpeg 等第三方组件许可
- Pillow：MIT-CMU（HPND）
- pypdfium2 / PDFium：Apache-2.0 / BSD-3-Clause；以 wheel 内附带的许可清单为准
- img2pdf：LGPL-3.0，分发时必须履行相应义务
- Python 运行时（PyInstaller 打包时）：PSF License

当前状态：M7.5 已收集并随 1.0.0 安装包/发布包分发。逐项合规审查见 docs/m7_1_license_audit.md
与 manifest.json（组件清单）。本目录含基础依赖、Python 运行时、OCR 插件及模型权重的许可证文本。