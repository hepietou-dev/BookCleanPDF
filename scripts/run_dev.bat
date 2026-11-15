@echo off
rem Dev launcher: run app with the project virtualenv (ASCII only, cmd-safe)
chcp 65001 >nul
cd /d "%~dp0.."
".venv\Scripts\python.exe" app.py %*