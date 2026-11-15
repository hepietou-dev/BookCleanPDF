@echo off
rem BookClean PDF — Windows onedir 构建脚本（M5.5）。
rem 用法：scripts\build_windows.bat  （在仓库根或任意目录均可）
rem 产物：dist\BookCleanPDF\（onedir，windowed GUI）；体积报告 temp\m5_review\dist_metrics.json
setlocal
cd /d "%~dp0.."

echo [1/2] PyInstaller onedir 构建...
call .venv\Scripts\python.exe -m PyInstaller --noconfirm --distpath dist --workpath build packaging\pyinstaller.spec
if errorlevel 1 (
  echo PyInstaller 构建失败。
  exit /b 1
)

echo [2/2] 实测 dist 体积构成...
call .venv\Scripts\python.exe scripts\measure_dist.py
endlocal