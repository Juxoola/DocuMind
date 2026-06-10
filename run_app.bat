@echo off
title NotebookLM Local Clone

echo ============================================
echo   NotebookLM Local Clone
echo ============================================
echo.

:: Launch PowerShell script — всё в одном окне, без всплывающих консолей
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"

echo.
echo Server stopped.
timeout /t 2 >nul
