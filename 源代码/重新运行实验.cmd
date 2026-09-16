@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist "D:\anaconda\python.exe" (
  "D:\anaconda\python.exe" -u run_all.py
) else (
  python -u run_all.py
)
pause
