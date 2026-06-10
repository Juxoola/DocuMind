<#
.SYNOPSIS
    DocuMind — merged backend + frontend in one console
.DESCRIPTION
    Runs backend (FastAPI) and frontend (Vite) in one PowerShell window
    with colored [Backend] / [Frontend] prefixes, real-time output,
    healthcheck, and graceful shutdown via Ctrl+C.
.PARAMETER NoFrontend
    Backend only, skip frontend
#>
param(
    [switch]$NoFrontend
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $ScriptDir "logs"

# ── UTF-8 for correct Cyrillic output from Python ──
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# ── Colours (info messages only) ──
$C = @{
    Info   = "Cyan"
    Ok     = "Green"
    Warn   = "Yellow"
    Err    = "Red"
}

function Write-Color($Color, $Text) {
    Write-Host "[$(Get-Date -Format HH:mm:ss)] " -NoNewline
    Write-Host $Text -ForegroundColor $Color
}

$script:Jobs = @{}
$script:CtrlCPressed = $false

function Cleanup-All {
    Write-Color $C.Info "Stopping server..."

    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/gguf-kill-all" -Method POST -TimeoutSec 5 -ErrorAction SilentlyContinue | Out-Null
    } catch {}

    foreach ($name in $script:Jobs.Keys) {
        $jb = $script:Jobs[$name]
        if ($jb.State -eq "Running") {
            Stop-Job $jb -ErrorAction SilentlyContinue
        }
        Remove-Job $jb -ErrorAction SilentlyContinue
    }

    # Kill orphan processes
    Get-Process "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    # Kill orphan python main.py via WMI (CommandLine not available in PS5 Get-Process)
    Get-CimInstance -ClassName Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*main.py*" } |
        Invoke-CimMethod -MethodName Terminate -ErrorAction SilentlyContinue | Out-Null

    Write-Color $C.Info "Server stopped."
}

# ── Ctrl+C handler via .NET event API (PS5.1 compatible) ──
[Console]::add_CancelKeyPress({
    param($sender, $e)
    $e.Cancel = $true          # prevent immediate termination
    $script:CtrlCPressed = $true
})

# ── Dependency checks ──
Write-Color $C.Info "=== DocuMind (merged) ==="

try {
    $pyVer = python --version 2>&1
    Write-Color $C.Ok "Python: $pyVer"
} catch {
    Write-Color $C.Err "Python not found! Install Python 3.11+"
    exit 1
}

if (-not $NoFrontend) {
    try {
        $nodeVer = node --version 2>&1
        Write-Color $C.Ok "Node.js: $nodeVer"
    } catch {
        Write-Color $C.Err "Node.js not found! Install Node.js 18+"
        exit 1
    }
}

$serverExe = Join-Path (Join-Path $ScriptDir "bin") "llama-server.exe"
if (-not (Test-Path $serverExe)) {
    Write-Color $C.Warn "llama-server.exe not found in bin/. Install manually."
}

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# ── Clean old processes before start ──
Get-Process "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-CimInstance -ClassName Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*main.py*" } |
    Invoke-CimMethod -MethodName Terminate -ErrorAction SilentlyContinue | Out-Null

# ── Start backend ──
$env:PYTHONPATH = $ScriptDir
Write-Color $C.Info "Starting backend (FastAPI)..."
$script:Jobs["Backend"] = Start-Job -Name "Backend" -ScriptBlock {
    param($dir)
    # Force UTF-8 in this job process so Python Cyrillic decodes correctly
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $env:PYTHONIOENCODING = "utf-8"
    Set-Location $dir
    python -u main.py 2>&1
} -ArgumentList $ScriptDir

# ── Healthcheck ──
Start-Sleep -Seconds 2
$backendReady = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/llm-status" -Method GET -TimeoutSec 2 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            $backendReady = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}
if ($backendReady) {
    Write-Color $C.Ok "Backend running: http://localhost:8000"
} else {
    Write-Color $C.Warn "Backend started but healthcheck failed."
}

# ── Start frontend ──
if (-not $NoFrontend) {
    $frontendDir = Join-Path $ScriptDir "frontend"
    if (Test-Path (Join-Path $frontendDir "node_modules")) {
        Write-Color $C.Info "Starting frontend (Vite)..."
        $script:Jobs["Frontend"] = Start-Job -Name "Frontend" -ScriptBlock {
            param($dir)
            Set-Location $dir
            npm run dev 2>&1
        } -ArgumentList $frontendDir
        Write-Color $C.Ok "Frontend running: http://localhost:5173"
    } else {
        Write-Color $C.Warn "node_modules not found. Run setup.ps1 first."
    }
}

Write-Color $C.Ok "=== Server is running ==="
Write-Color $C.Info "Press Ctrl+C to stop"
Write-Host ""

# ── Main loop — Receive-Job in real time ──
try {
    while ($true) {
        # Ctrl+C pressed?
        if ($script:CtrlCPressed) {
            Write-Host ""
            Write-Color $C.Info "Ctrl+C detected, shutting down..."
            break
        }

        if ($script:Jobs["Backend"].State -eq "Failed") {
            Write-Color $C.Err "Backend crashed (see log above)."
            break
        }

        $allDone = $true
        foreach ($name in $script:Jobs.Keys) {
            $jb = $script:Jobs[$name]
            if ($jb.State -eq "Running" -or $jb.State -eq "NotStarted") {
                $allDone = $false
            }
        }
        if ($allDone) {
            Write-Color $C.Info "All processes finished."
            break
        }

        foreach ($name in $script:Jobs.Keys) {
            $jb = $script:Jobs[$name]

            $lines = Receive-Job $jb
            if ($lines) {
                foreach ($line in $lines) {
                    $text = "$line"
                    if ($text.Trim()) {
                        Write-Host "[$name] $text"
                    }
                }
            }
        }

        Start-Sleep -Milliseconds 200
    }
} finally {
    Cleanup-All
}
