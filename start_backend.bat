@echo off
chcp 65001 >nul
title NotebookLM Backend
echo Запуск backend-сервера...
if not exist "logs" mkdir logs
python main.py
pause
