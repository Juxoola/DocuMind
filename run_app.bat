@echo off
chcp 65001 >nul
title NotebookLM Local Clone
echo ==========================================
echo    Starting NotebookLM Local Clone
echo ==========================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    pause
    exit /b
)

:: Check Node.js
node -v >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found!
    pause
    exit /b
)

echo [1/2] Starting Backend (FastAPI)...
start /min "Backend" cmd /c "python main.py"

echo [2/2] Starting Frontend (Vite)...
cd frontend
start /min "Frontend" cmd /c "npm run dev"

echo.
echo ==========================================
echo All services are starting in background!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173
echo ==========================================
echo.
timeout /t 5
exit
