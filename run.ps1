<#
.SYNOPSIS
    NotebookLM Local Clone — запуск сервера (одно окно)
.DESCRIPTION
    Запускает backend (FastAPI) и frontend (Vite) в одном окне.
    Вывод backend → logs\server.log, frontend → logs\frontend.log.
    Нажмите q + Enter для остановки.
.PARAMETER NoFrontend
    Запустить только backend без frontend
#>
param(
    [switch]$NoFrontend
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $ScriptDir "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

# ── Кодировка: UTF-8 ──
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ── Цвета ──
$Colors = @{ Info = "Cyan"; Ok = "Green"; Warn = "Yellow"; Err = "Red" }
function Write-Color($Color, $Text) {
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "[$ts] $Text" -ForegroundColor $Color
}

try {

# ── Проверка зависимостей ──
Write-Color $Colors.Info "=== NotebookLM Local Clone ==="
try { Write-Color $Colors.Ok "Python: $(python --version 2>&1)" } catch { Write-Color $Colors.Err "Python не найден!"; Read-Host; exit 1 }
if (-not $NoFrontend) {
    try { Write-Color $Colors.Ok "Node.js: $(node --version 2>&1)" } catch { Write-Color $Colors.Err "Node.js не найден!"; Read-Host; exit 1 }
}

# ── Чистка остатков ──
Get-Process "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force

# ── Запуск backend ──
$logFile = Join-Path $LogDir "server.log"
$env:PYTHONPATH = $ScriptDir

Write-Color $Colors.Info "Запуск backend (FastAPI)..."
$backendJob = Start-Job -Name "Backend" -ScriptBlock {
    param($dir, $log)
    $env:PYTHONPATH = $dir
    Set-Location $dir
    python main.py 2>&1 >> $log
} -ArgumentList $ScriptDir, $logFile

# ── Healthcheck с прогрессом ──
$backendReady = $false
Write-Host "  Ждём backend" -NoNewline
for ($i = 1; $i -le 60; $i++) {
    Write-Host "." -NoNewline

    # Проверяем, жив ли процесс
    if ($backendJob.State -eq "Failed") {
        Write-Host ""
        Write-Color $Colors.Err "Backend упал! Логи: $logFile"
        if (Test-Path $logFile) { Get-Content $logFile -Tail 20 | ForEach-Object { Write-Host "  $_" } }
        break
    }

    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/llm-status" -Method GET -TimeoutSec 2 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            $backendReady = $true
            Write-Host " OK ($i сек)"
            break
        }
    } catch {
        Start-Sleep -Milliseconds 1000
    }
}

if ($backendReady) {
    Write-Color $Colors.Ok "Backend: http://localhost:8000"
} else {
    Write-Host ""
    Write-Color $Colors.Warn "Backend не ответил за 60 секунд. Логи: $logFile"
}

# ── Запуск frontend ──
$frontendJob = $null
if (-not $NoFrontend) {
    $frontendDir = Join-Path $ScriptDir "frontend"
    if (Test-Path (Join-Path $frontendDir "node_modules")) {
        Write-Color $Colors.Info "Запуск frontend (Vite)..."
        $frontendJob = Start-Job -Name "Frontend" -ScriptBlock {
            param($dir)
            Set-Location $dir
            npm run dev
        } -ArgumentList $frontendDir
        Write-Color $Colors.Ok "Frontend: http://localhost:5173"
    } else {
        Write-Color $Colors.Warn "node_modules не найдены. Запустите: cd frontend && npm install"
    }
}

Write-Color $Colors.Ok "=== Сервер запущен ==="
Write-Color $Colors.Info "Логи: $logFile"
Write-Color $Colors.Info "Нажмите q + Enter для остановки"

# ── Ожидание команды остановки ──
while ($true) {
    $input = Read-Host
    if ($input -eq "q") { break }
}

} catch {
    Write-Color $Colors.Err "Ошибка: $_"
    Read-Host "Нажмите Enter для выхода"
    exit 1
}

# ── Shutdown ──
Write-Color $Colors.Info "Остановка сервера..."
try { Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/gguf-kill-all" -Method POST -TimeoutSec 5 -ErrorAction SilentlyContinue | Out-Null } catch {}

if ($frontendJob) {
    Stop-Job $frontendJob -ErrorAction SilentlyContinue
    Remove-Job $frontendJob -ErrorAction SilentlyContinue
}

Stop-Job $backendJob -ErrorAction SilentlyContinue
Remove-Job $backendJob -ErrorAction SilentlyContinue

Get-Process "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Color $Colors.Info "Сервер остановлен."
Start-Sleep -Seconds 2
