<#
.SYNOPSIS
    DocuMind — установка в 1 клик
.DESCRIPTION
    Проверяет и устанавливает все зависимости для работы проекта:
    - Python виртуальное окружение + pip install
    - Node.js модули для frontend
    - .env файл (если отсутствует)
    - Проверка llama-server.exe
#>
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

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

Write-Color $Colors.Info "=== DocuMind — Установка ==="
Write-Color $Colors.Info "Директория: $ScriptDir"

# ── 1. Проверка Python ──
Write-Color $Colors.Info "[1/5] Проверка Python..."
try {
    $pyVersion = python --version 2>&1
    Write-Color $Colors.Ok "  $pyVersion"
} catch {
    Write-Color $Colors.Err "Python не найден! Скачайте: https://www.python.org/downloads/"
    Write-Color $Colors.Err "Установите Python 3.11+ и ОБЯЗАТЕЛЬНО отметьте 'Add Python to PATH'"
    exit 1
}

# ── 2. Виртуальное окружение ──
$venvDir = Join-Path $ScriptDir ".venv"
if (-not (Test-Path $venvDir)) {
    Write-Color $Colors.Info "[2/5] Создание виртуального окружения..."
    python -m venv $venvDir
    Write-Color $Colors.Ok "  Виртуальное окружение создано: $venvDir"
} else {
    Write-Color $Colors.Ok "[2/5] Виртуальное окружение уже существует"
}

# Активируем и устанавливаем зависимости
$pip = Join-Path $venvDir "Scripts" "pip.exe"
if (-not (Test-Path $pip)) {
    $pip = Join-Path $venvDir "Scripts" "python.exe"
}

Write-Color $Colors.Info "[3/5] Установка Python-зависимостей..."
try {
    if (Test-Path (Join-Path $ScriptDir "pyproject.toml")) {
        & "$(Join-Path $venvDir 'Scripts' 'python.exe')" -m pip install -e "$ScriptDir" --quiet 2>&1 | Out-Null
    } else {
        & "$(Join-Path $venvDir 'Scripts' 'python.exe')" -m pip install -r "$ScriptDir\requirements.txt" --quiet 2>&1 | Out-Null
    }
    Write-Color $Colors.Ok "  Зависимости установлены"
} catch {
    Write-Color $Colors.Warn "  Ошибка установки зависимостей: $_"
    Write-Color $Colors.Info "  Попробуйте вручную: pip install -r requirements.txt"
}

# ── 3. Node.js и frontend ──
Write-Color $Colors.Info "[4/5] Проверка Node.js..."
try {
    $nodeVersion = node --version 2>&1
    Write-Color $Colors.Ok "  $nodeVersion"

    $frontendDir = Join-Path $ScriptDir "frontend"
    $nodeModules = Join-Path $frontendDir "node_modules"

    if (-not (Test-Path $nodeModules)) {
        Write-Color $Colors.Info "  Установка frontend-зависимостей (npm install)..."
        Push-Location $frontendDir
        npm install --silent 2>&1 | Out-Null
        Pop-Location
        Write-Color $Colors.Ok "  Frontend-зависимости установлены"
    } else {
        Write-Color $Colors.Ok "  Frontend-зависимости уже установлены"
    }
} catch {
    Write-Color $Colors.Warn "  Node.js не найден. Frontend не будет работать без Node.js 18+"
}

# ── 4. .env файл ──
Write-Color $Colors.Info "[5/5] Проверка .env..."
$envFile = Join-Path $ScriptDir ".env"
$envExample = Join-Path $ScriptDir ".env.example"
if (-not (Test-Path $envFile)) {
    if (Test-Path $envExample) {
        Copy-Item $envExample $envFile
        Write-Color $Colors.Ok "  .env создан из .env.example"
    } else {
        Write-Color $Colors.Warn "  .env.example не найден. Создаю .env с умолчаниями..."
        @"
HOST=0.0.0.0
PORT=8000
LM_STUDIO_URL=http://localhost:1234/v1
GGUF_SEARCH_DIRS=F:/llm
UPLOAD_MAX_SIZE_MB=500
"@ | Out-File -FilePath $envFile -Encoding utf8
        Write-Color $Colors.Ok "  .env создан с настройками по умолчанию"
    }
} else {
    Write-Color $Colors.Ok "  .env уже существует"
}

# ── 5. Проверка llama-server.exe ──
$serverExe = Join-Path $ScriptDir "bin" "llama-server.exe"
if (-not (Test-Path $serverExe)) {
    Write-Color $Colors.Warn ""
    Write-Color $Colors.Warn "⚠  llama-server.exe не найден в bin/"
    Write-Color $Colors.Warn "   Скачайте последний релиз llama.cpp:"
    Write-Color $Colors.Warn "   https://github.com/ggml-org/llama.cpp/releases"
    Write-Color $Colors.Warn "   Распакуйте llama-server.exe + .dll файлы в: $ScriptDir\bin"
} else {
    Write-Color $Colors.Ok "  llama-server.exe найден"
}

# ── Итог ──
Write-Color $Colors.Ok ""
Write-Color $Colors.Ok "=== Установка завершена! ==="
Write-Color $Colors.Info ""
Write-Color $Colors.Info "Для запуска выполните:"
Write-Color $Colors.Info "  .\run.ps1"
Write-Color $Colors.Info ""
Write-Color $Colors.Info "Или вручную:"
Write-Color $Colors.Info "  python main.py           # Backend"
Write-Color $Colors.Info "  cd frontend && npm run dev   # Frontend"
Write-Color $Colors.Info ""

# Активация venv (если пользователь хочет)
Write-Color $Colors.Info "Для активации виртуального окружения:"
Write-Color $Colors.Info "  .venv\Scripts\activate"
