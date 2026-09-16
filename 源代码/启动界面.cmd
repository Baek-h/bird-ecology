@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist "D:\anaconda\python.exe" (
  "D:\anaconda\python.exe" app.py
) else (
  python app.py
)
if errorlevel 1 pause
