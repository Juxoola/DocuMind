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
Write-Color $Colors.Info "[1/6] Проверка Python..."
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

Write-Color $Colors.Info "[3/6] Установка Python-зависимостей..."
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
Write-Color $Colors.Info "[4/6] Проверка Node.js..."
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

    Write-Color $Colors.Info "  Сборка frontend (npm run build)..."
    Push-Location $frontendDir
    npm run build 2>&1 | Out-Null
    Pop-Location
    Write-Color $Colors.Ok "  Frontend собран (dist/)"
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

# ── 5. Проверка / скачивание llama-server.exe ──
$binDir = Join-Path $ScriptDir "bin"
$serverExe = Join-Path $binDir "llama-server.exe"

if (-not (Test-Path $serverExe)) {
    Write-Color $Colors.Info "[5/6] llama-server.exe не найден — скачиваю с GitHub..."

    try {
        # Получаем последний релиз из GitHub API
        $release = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest" -UseBasicParsing

        # Ищем CUDA-сборку для Windows x64 (приоритет: CUDA 12, потом любая CUDA)
        $cudaAsset = $release.assets | Where-Object { $_.name -match "cuda.*win.*x64" -and $_.name -match "\.zip$" } |
            Sort-Object { $_.name } -Descending | Select-Object -First 1

        if (-not $cudaAsset) {
            Write-Color $Colors.Warn "  CUDA-сборка не найдена в релизе $($release.tag_name)"
            Write-Color $Colors.Warn "  Скачайте вручную: https://github.com/ggml-org/llama.cpp/releases"
            Write-Color $Colors.Warn "  Распакуйте llama-server.exe + .dll файлы в: $binDir"
        } else {
            Write-Color $Colors.Info "  Релиз: $($release.tag_name)"
            Write-Color $Colors.Info "  Файл: $($cudaAsset.name) ($([math]::Round($cudaAsset.size / 1MB, 1)) MB)"

            $zipPath = Join-Path $env:TEMP "llama-cpp-cuda.zip"
            Write-Color $Colors.Info "  Скачивание..."
            Invoke-WebRequest -Uri $cudaAsset.browser_download_url -OutFile $zipPath -UseBasicParsing

            if (-not (Test-Path $binDir)) { New-Item -ItemType Directory -Path $binDir -Force | Out-Null }

            Write-Color $Colors.Info "  Распаковка в bin/..."
            Expand-Archive -Path $zipPath -DestinationPath $binDir -Force

            # Если архив содержит подпапду — перемещаем содержимое в корень bin/
            $subDirs = Get-ChildItem -Path $binDir -Directory
            if ($subDirs) {
                foreach ($sd in $subDirs) {
                    Get-ChildItem -Path $sd.FullName | Move-Item -Destination $binDir -Force
                    Remove-Item $sd.FullName -Recurse -Force
                }
            }

            Remove-Item $zipPath -Force -ErrorAction SilentlyContinue

            if (Test-Path $serverExe) {
                Write-Color $Colors.Ok "  llama-server.exe установлен!"
            } else {
                Write-Color $Colors.Warn "  Архив распакован, но llama-server.exe не найден в bin/"
                Write-Color $Colors.Warn "  Проверьте содержимое: $binDir"
            }
        }
    } catch {
        Write-Color $Colors.Err "  Ошибка скачивания: $_"
        Write-Color $Colors.Warn "  Скачайте вручную: https://github.com/ggml-org/llama.cpp/releases"
        Write-Color $Colors.Warn "  Нужна CUDA-сборка, распакуйте в: $binDir"
    }
} else {
    Write-Color $Colors.Ok "[5/6] llama-server.exe найден"
}

# ── 6. Скачивание моделей ──
$modelsDir = Join-Path $ScriptDir "models"
if (-not (Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null
}

$models = @(
    @{
        Url = "https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/resolve/main/qwen3-reranker-0.6b-q8_0.gguf?download=true"
        FileName = "qwen3-reranker-0.6b-q8_0.gguf"
        Label = "Reranker (0.6B)"
    }
    @{
        Url = "https://huggingface.co/yomir/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf?download=true"
        FileName = "Qwen3-Embedding-0.6B-Q8_0.gguf"
        Label = "Embedding (0.6B)"
    }
    @{
        Url = "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-UD-Q4_K_XL.gguf?download=true"
        FileName = "Qwen3.5-4B-UD-Q4_K_XL.gguf"
        Label = "LLM (4B Q4_K_XL)"
    }
)

Write-Color $Colors.Info "[6/6] Проверка моделей GGUF..."

$allPresent = $true
foreach ($m in $models) {
    $modelPath = Join-Path $modelsDir $m.FileName
    if (Test-Path $modelPath) {
        $size = (Get-Item $modelPath).Length
        Write-Color $Colors.Ok "  [$($m.Label)] найден ($([math]::Round($size / 1MB, 1)) MB)"
    } else {
        $allPresent = $false
    }
}

if (-not $allPresent) {
    Write-Color $Colors.Info "  Некоторые модели отсутствуют — скачиваю..."
    foreach ($m in $models) {
        $modelPath = Join-Path $modelsDir $m.FileName
        if (Test-Path $modelPath) {
            continue
        }
        Write-Color $Colors.Info "  [6/6] Скачиваю $($m.Label)..."
        try {
            $tempFile = Join-Path $env:TEMP $m.FileName
            Invoke-WebRequest -Uri $m.Url -OutFile $tempFile -UseBasicParsing
            Move-Item $tempFile $modelPath -Force
            $size = (Get-Item $modelPath).Length
            Write-Color $Colors.Ok "    Готово ($([math]::Round($size / 1MB, 1)) MB)"
        } catch {
            Write-Color $Colors.Err "    Ошибка: $_"
        }
    }
} else {
    Write-Color $Colors.Ok "  Все модели на месте"
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
