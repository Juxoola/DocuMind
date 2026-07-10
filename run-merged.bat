@echo off
setlocal
chcp 65001 >nul 2>&1
title DocuMind
set PYTHONIOENCODING=utf-8

echo ============================================
echo   DocuMind -- one console
echo ============================================
echo.
echo Starting backend + frontend in one window...
echo Press Ctrl+C to stop.
echo.

:: run-merged.ps1 must be UTF-8 with BOM for PowerShell 5.1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-merged.ps1"

echo.
echo ============================================
echo   Server stopped.
echo   Window will close in 5 seconds.
echo ============================================
ping -n 6 127.0.0.1 >nul
exit /b
