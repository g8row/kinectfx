@echo off
REM Launch KinectFX on Windows.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo No virtualenv found. Run setup.ps1 first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" main.py
