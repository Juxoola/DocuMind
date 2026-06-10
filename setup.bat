@echo off
chcp 65001 >nul 2>&1
title DocuMind Setup

echo ============================================
echo   DocuMind - Setup
echo ============================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"

echo.
echo ============================================
echo   Setup finished.
echo ============================================
echo.
pause
