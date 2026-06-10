<#
.SYNOPSIS
    NotebookLM Local Clone — запуск сервера
.DESCRIPTION
    Запускает backend (FastAPI) и frontend (Vite) с цветным выводом,
    healthcheck и graceful shutdown.
.PARAMETER NoFrontend
    Запустить только backend без frontend
.PARAMETER Dev
    Режим разработки: reload=True, frontend в отдельном окне
#>
param(
    [switch]$NoFrontend,
    [switch]$Dev
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $ScriptDir "logs"

# ── Цвета ──
$Colors = @{
    Info = "Cyan"
    Ok   = "Green"
    Warn = "Yellow"
    Err  = "Red"
}

function Write-Color($Color, $Text) {
    Write-Host "[$(Get-Date -Format HH:mm:ss)] " -NoNewline
    Write-Host $Text -ForegroundColor $Color
}

# ── Проверка зависимостей ──
Write-Color $Colors.Info "=== NotebookLM Local Clone ==="

# Python
try {
    $pyVersion = python --version 2>&1
    Write-Color $Colors.Ok "Python: $pyVersion"
} catch {
    Write-Color $Colors.Err "Python не найден! Установите Python 3.11+"
    exit 1
}

# Node.js
if (-not $NoFrontend) {
    try {
        $nodeVersion = node --version 2>&1
        Write-Color $Colors.Ok "Node.js: $nodeVersion"
    } catch {
        Write-Color $Colors.Err "Node.js не найден! Установите Node.js 18+"
        exit 1
    }
}

# llama-server.exe
$serverExe = Join-Path $ScriptDir "bin" "llama-server.exe"
if (-not (Test-Path $serverExe)) {
    Write-Color $Colors.Warn "llama-server.exe не найден в bin/. Установите вручную."
}

# ── Проверка логов ──
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# ── Очистка orphan-процессов ──
Write-Color $Colors.Info "Проверка остаточных процессов llama-server..."
$orphans = Get-Process "llama-server" -ErrorAction SilentlyContinue
if ($orphans) {
    Write-Color $Colors.Warn "Найдены процессы llama-server. Завершаю..."
    $orphans | Stop-Process -Force
}

# ── Запуск backend ──
$env:PYTHONPATH = $ScriptDir
$logFile = Join-Path $LogDir "server.log"

Write-Color $Colors.Info "Запуск backend (FastAPI)..."
$backendJob = Start-Job -Name "Backend" -ScriptBlock {
    param($dir, $logFile)
    Set-Location $dir
    python main.py *>> $logFile
} -ArgumentList $ScriptDir, $logFile

# Ждём готовности backend
Start-Sleep -Seconds 2
$backendReady = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8000/llm-status" -Method GET -TimeoutSec 2 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            $backendReady = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

if ($backendReady) {
    Write-Color $Colors.Ok "Backend запущен: http://localhost:8000"
} else {
    Write-Color $Colors.Warn "Backend запущен, но healthcheck не прошёл. Проверьте логи: $logFile"
}

# ── Запуск frontend ──
$frontendJob = $null
if (-not $NoFrontend) {
    $frontendDir = Join-Path $ScriptDir "frontend"
    if (Test-Path (Join-Path $frontendDir "node_modules" ".package-lock.json")) {
        Write-Color $Colors.Info "Запуск frontend (Vite)..."
        $frontendJob = Start-Job -Name "Frontend" -ScriptBlock {
            param($dir)
            Set-Location $dir
            npm run dev
        } -ArgumentList $frontendDir
        Write-Color $Colors.Ok "Frontend запущен: http://localhost:5173"
    } else {
        Write-Color $Colors.Warn "node_modules не найдены. Выполните setup.ps1 или 'npm install' в frontend/"
    }
} else {
    Write-Color $Colors.Info "Frontend пропущен (--NoFrontend)"
}

Write-Color $Colors.Ok "=== Сервер запущен ==="
Write-Color $Colors.Info "Логи: $logFile"
Write-Color $Colors.Info "Нажмите Ctrl+C для остановки"

# ── Graceful shutdown ──
try {
    while ($true) {
        Start-Sleep -Seconds 1
        # Проверяем что backend жив
        if ($backendJob.State -eq "Failed") {
            Write-Color $Colors.Err "Backend упал. Логи: $logFile"
            break
        }
    }
} finally {
    Write-Color $Colors.Info "Остановка сервера..."

    if ($frontendJob) {
        Stop-Job $frontendJob -ErrorAction SilentlyContinue
        Remove-Job $frontendJob -ErrorAction SilentlyContinue
    }

    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/gguf-kill-all" -Method POST -TimeoutSec 5 -ErrorAction SilentlyContinue | Out-Null
        Write-Color $Colors.Ok "GGUF-серверы остановлены"
    } catch {
        Write-Color $Colors.Warn "Не удалось остановить GGUF-серверы через API"
    }

    Stop-Job $backendJob -ErrorAction SilentlyContinue
    Remove-Job $backendJob -ErrorAction SilentlyContinue

    # Финальная зачистка
    Get-Process "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

    Write-Color $Colors.Info "Сервер остановлен."
}
