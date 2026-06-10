@echo off
chcp 65001 >nul 2>&1
title DocuMind Backend

echo ============================================
echo   DocuMind - Backend only
echo ============================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-merged.ps1" -NoFrontend
echo.
pause
