@echo off
setlocal enabledelayedexpansion
title NotebookLM Local Clone

chcp 65001 >nul 2>&1

echo ============================================
echo   NotebookLM Local Clone
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Install Python 3.11+
    pause
    exit /b
)

:: Detect venv (setup.ps1 creates .venv, use it if exists)
set PYTHON_CMD=python
if exist ".venv\Scripts\python.exe" (
    set PYTHON_CMD=.venv\Scripts\python.exe
    echo [OK] Using venv: .venv
)

:: Node.js check
node -v >nul 2>&1
if %errorlevel% neq 0 set NO_FRONTEND=1

:: Logs
if not exist "logs" mkdir logs

:: Kill old llama-server processes
tasklist /FI "IMAGENAME eq llama-server.exe" 2>NUL | findstr "llama-server.exe" >NUL
if !errorlevel! equ 0 (
    echo [INFO] Cleaning up old llama-server...
    taskkill /F /IM llama-server.exe >nul 2>&1
)

echo.
echo [1/2] Starting backend on http://localhost:8000
start "NB-Backend" /min cmd /c "%PYTHON_CMD% main.py"

:: Healthcheck via PowerShell
set READY=
for /l %%i in (1,1,30) do (
    ping -n 2 127.0.0.1 >nul
    powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/llm-status' -TimeoutSec 2 -UseBasicParsing; if ($r.StatusCode -eq 200) { exit 0 } } catch { exit 1 }" >nul 2>&1
    if !errorlevel! equ 0 set READY=1 & goto ready
)
:ready
if defined READY (echo [OK] Backend ready) else (echo [WARN] Backend healthcheck timed out)

if not defined NO_FRONTEND (
    echo.
    echo [2/2] Starting frontend on http://localhost:5173
    if exist "frontend\node_modules" (
        start "NB-Frontend" /min cmd /c "cd /d frontend && npm run dev"
        echo [OK] Frontend ready
    ) else (
        echo [WARN] Run "cd frontend ^&^& npm install" first
    )
)

echo.
echo ============================================
echo   Server is running!
echo ============================================
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:5173
echo   API docs: http://localhost:8000/docs
echo.
echo   Close this window to stop the server.
echo.
pause >nul

:: Graceful shutdown
echo.
echo Shutting down...
powershell -Command "try { Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/gguf-kill-all' -Method POST -TimeoutSec 5 -UseBasicParsing } catch {}" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq NB-Backend" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq NB-Frontend" >nul 2>&1
taskkill /F /IM llama-server.exe >nul 2>&1
echo Server stopped.
timeout /t 2 >nul
