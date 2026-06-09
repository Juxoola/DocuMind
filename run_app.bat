@echo off
chcp 65001 >nul
title NotebookLM Local Clone
setlocal enabledelayedexpansion

:: Цвета (ANSI)
set "ESC=["
set "GREEN=%ESC%32m"
set "CYAN=%ESC%36m"
set "YELLOW=%ESC%33m"
set "RED=%ESC%31m"
set "RESET=%ESC%0m"
set "BOLD=%ESC%1m"

echo %BOLD%%CYAN%============================================%RESET%
echo %BOLD%%CYAN%   NotebookLM Local Clone%RESET%
echo %BOLD%%CYAN%============================================%RESET%
echo.

:: ── Проверка Python ──
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%[ERROR] Python не найден!%RESET%
    echo %YELLOW%Установите Python 3.11+: https://www.python.org/downloads/%RESET%
    echo %YELLOW%ВАЖНО: при установке отметьте "Add Python to PATH"%RESET%
    pause
    exit /b
)
for /f "tokens=*" %%a in ('python --version 2^>^&1') do set "PY_VER=%%a"
echo %GREEN%[OK] %PY_VER%%RESET%

:: ── Проверка Node.js (для frontend) ──
node -v >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%[WARN] Node.js не найден. Frontend не запустится.%RESET%
    echo %YELLOW%       Установите: https://nodejs.org/%RESET%
    set "NO_FRONTEND=1"
) else (
    for /f "tokens=*" %%a in ('node -v 2^>^&1') do set "NODE_VER=%%a"
    echo %GREEN%[OK] %NODE_VER%%RESET%
)

:: ── Проверка логов ──
if not exist "logs" mkdir logs

:: ── Очистка старых процессов llama-server ──
echo %CYAN%[INFO] Проверка остаточных процессов...%RESET%
tasklist /FI "IMAGENAME eq llama-server.exe" 2>NUL | find /I "llama-server.exe" >NUL
if !errorlevel! equ 0 (
    echo %YELLOW%[WARN] Найдены старые процессы llama-server. Завершаю...%RESET%
    taskkill /F /IM llama-server.exe >nul 2>&1
    timeout /t 2 /nobreak >nul
)

:: ── Запуск backend ──
echo.
echo %BOLD%%CYAN%[1/2] Запуск backend (FastAPI)...%RESET%
start "Backend" /min cmd /c "python main.py" > logs\server.log 2>&1

:: Ждём готовности backend
echo %CYAN%[INFO] Ожидание готовности сервера...%RESET%
set "READY="
for /l %%i in (1,1,30) do (
    timeout /t 1 /nobreak >nul
    curl -s http://127.0.0.1:8000/api/llm-status >nul 2>&1
    if not errorlevel 1 set "READY=1" & goto :backend_ready
)
:backend_ready
if defined READY (
    echo %GREEN%[OK] Backend запущен: http://localhost:8000%RESET%
) else (
    echo %YELLOW%[WARN] Backend запущен, но не отвечает. Проверьте logs\server.log%RESET%
)

:: ── Запуск frontend ──
if not defined NO_FRONTEND (
    echo.
    echo %BOLD%%CYAN%[2/2] Запуск frontend (Vite)...%RESET%

    if exist "frontend\node_modules" (
        start "Frontend" /min cmd /c "cd frontend && npm run dev"
        echo %GREEN%[OK] Frontend запущен: http://localhost:5173%RESET%
    ) else (
        echo %YELLOW%[WARN] node_modules не найдены. Выполните npm install в папке frontend\%RESET%
        echo %YELLOW%       или запустите setup.ps1 для полной установки.%RESET%
    )
)

echo.
echo %BOLD%%GREEN%============================================%RESET%
echo %BOLD%%GREEN%   Сервер запущен!%RESET%
echo %BOLD%%GREEN%============================================%RESET%
echo.
echo   Backend:  %CYAN%http://localhost:8000%RESET%
echo   Frontend: %CYAN%http://localhost:5173%RESET%
echo   API docs: %CYAN%http://localhost:8000/docs%RESET%
echo.
echo %YELLOW%  Закройте это окно для остановки сервера.%RESET%
echo.

:: Ожидание закрытия окна
pause >nul

:: ── Graceful shutdown ──
echo.
echo %CYAN%[INFO] Остановка сервера...%RESET%

:: Выгружаем GGUF-модели через API
curl -s -X POST http://127.0.0.1:8000/api/gguf-kill-all >nul 2>&1
if !errorlevel! equ 0 (
    echo %GREEN%[OK] Модели выгружены%RESET%
) else (
    echo %YELLOW%[WARN] API недоступен, завершаю процессы принудительно%RESET%
)

:: Убиваем процессы
taskkill /F /FI "WINDOWTITLE eq Backend" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Frontend" >nul 2>&1
taskkill /F /IM llama-server.exe >nul 2>&1

echo %GREEN%[OK] Сервер остановлен%RESET%
timeout /t 2 /nobreak >nul
