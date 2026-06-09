@echo off
title Create NotebookLM Shortcut
echo Creating shortcut on Desktop...

powershell -NoProfile -ExecutionPolicy Bypass -Command "$d=[Environment]::GetFolderPath('Desktop'); $w=New-Object -ComObject WScript.Shell; $s=$w.CreateShortcut($d+'\NotebookLM.lnk'); $s.TargetPath='%~dp0run_app.bat'; $s.WorkingDirectory='%~dp0'; $s.Description='NotebookLM Local Clone'; $s.WindowStyle=1; $s.Save(); Write-Host OK"

if %errorlevel% equ 0 (
    echo [OK] Shortcut created on Desktop!
) else (
    echo [ERROR] Failed to create shortcut
)
echo.
pause
