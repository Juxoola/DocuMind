<#
.SYNOPSIS
    NotebookLM Local Clone — запуск сервера (одно окно)
.DESCRIPTION
    Запускает backend (FastAPI) и frontend (Vite) в одном окне.
    Вывод в лог-файлы. Напишите q + Enter для остановки.
#>
param([switch]$NoFrontend)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $ScriptDir "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

# ── UTF-8 ──
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Color($c, $t) {
    Write-Host "$(Get-Date -Format "HH:mm:ss") $t" -ForegroundColor $c
}

try {

Color "Cyan" "=== NotebookLM Local Clone ==="
Color "Green" "Python: $(python --version 2>&1)"
if (-not $NoFrontend) { Color "Green" "Node.js: $(node --version 2>&1)" }

# ── Чистка ──
Get-Process "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force

# ── Запуск backend ──
$logFile = Join-Path $LogDir "server.log"
$env:PYTHONPATH = $ScriptDir

Color "Cyan" "Запуск backend (FastAPI)..."
$backend = Start-Process -WindowStyle Hidden -PassThru `
    -FilePath "python" -ArgumentList "main.py" `
    -WorkingDirectory $ScriptDir `
    -RedirectStandardOutput $logFile -RedirectStandardError $logFile

# ── Healthcheck ──
$backendReady = $false
Write-Host "  Ждём backend" -NoNewline
for ($i = 1; $i -le 60; $i++) {
    Start-Sleep -Seconds 1
    Write-Host "." -NoNewline

    if ($backend.HasExited) {
        Write-Host " FAIL"
        Color "Red" "Backend упал! (exit: $($backend.ExitCode)) Логи: $logFile"
        if (Test-Path $logFile) { Get-Content -Tail 30 $logFile | ForEach-Object { Write-Host "  $_" } }
        break
    }

    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/llm-status" -TimeoutSec 1 -UseBasicParsing -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $backendReady = $true; Write-Host " OK ($i сек)"; break }
    } catch {}
}

if ($backendReady) {
    Color "Green" "Backend: http://localhost:8000"
} else {
    Write-Host ""
    Color "Yellow" "Backend не ответил за 60 сек. Логи: $logFile"
    if (Test-Path $logFile) { Get-Content -Tail 30 $logFile | ForEach-Object { Write-Host "  $_" } }
}

# ── Запуск frontend (даже если backend не ответил) ──
if (-not $NoFrontend) {
    $fDir = Join-Path $ScriptDir "frontend"
    if (Test-Path (Join-Path $fDir "node_modules")) {
        $frontendLog = Join-Path $LogDir "frontend.log"
        Color "Cyan" "Запуск frontend (Vite)..."
        $frontend = Start-Process -WindowStyle Hidden -PassThru `
            -FilePath "node" -ArgumentList "run dev" `
            -WorkingDirectory $fDir `
            -RedirectStandardOutput $frontendLog -RedirectStandardError $frontendLog
        Start-Sleep -Seconds 3
        Color "Green" "Frontend: http://localhost:5173"
    } else {
        Color "Yellow" "node_modules не найдены. cd frontend && npm install"
    }
}

Color "Green" "=== Сервер запущен ==="
Color "Cyan" "Логи: $logFile"
Color "Cyan" "Напишите q + Enter для остановки"

while ($true) {
    if ((Read-Host) -eq "q") { break }
}

} catch {
    Color "Red" "Ошибка: $_"
    Read-Host "Нажмите Enter"
} finally {
    Color "Cyan" "Остановка сервера..."
    try { Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/gguf-kill-all" -Method POST -TimeoutSec 5 -UseBasicParsing -ErrorAction SilentlyContinue } catch {}
    if ($backend -and !$backend.HasExited) { $backend.Kill() }
    if ($frontend -and !$frontend.HasExited) { $frontend.Kill() }
    Get-Process "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force
    Color "Cyan" "Сервер остановлен."
}
