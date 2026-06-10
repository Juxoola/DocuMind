@echo off
chcp 65001 >nul 2>&1
title Create Shortcuts

echo ============================================
echo   Creating shortcuts on Desktop...
echo ============================================
echo.

set SCRIPT_DIR=%~dp0
set DESKTOP=%USERPROFILE%\Desktop

:: ---- Start App (full app) ----
set SHORTCUT_NAME_APP=DocuMind
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$d=[Environment]::GetFolderPath('Desktop');" ^
  "$w=New-Object -ComObject WScript.Shell;" ^
  "$s=$w.CreateShortcut($d+'\\%SHORTCUT_NAME_APP%.lnk');" ^
  "$s.TargetPath='%SCRIPT_DIR%run-merged.bat';" ^
  "$s.WorkingDirectory='%SCRIPT_DIR%';" ^
  "$s.Description='DocuMind - backend + frontend';" ^
  "$s.WindowStyle=1;" ^
  "$s.Save()"
if %errorlevel% equ 0 ( echo [OK] Shortcut: "%SHORTCUT_NAME_APP%" ^-^> run-merged.bat
) else ( echo [ERR] Failed to create "%SHORTCUT_NAME_APP%" )

:: ---- Start Backend (backend only) ----
set SHORTCUT_NAME_BACKEND=DocuMind Backend
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$d=[Environment]::GetFolderPath('Desktop');" ^
  "$w=New-Object -ComObject WScript.Shell;" ^
  "$s=$w.CreateShortcut($d+'\\%SHORTCUT_NAME_BACKEND%.lnk');" ^
  "$s.TargetPath='%SCRIPT_DIR%start_backend.bat';" ^
  "$s.WorkingDirectory='%SCRIPT_DIR%';" ^
  "$s.Description='DocuMind - backend only';" ^
  "$s.WindowStyle=1;" ^
  "$s.Save()"
if %errorlevel% equ 0 ( echo [OK] Shortcut: "%SHORTCUT_NAME_BACKEND%" ^-^> start_backend.bat
) else ( echo [ERR] Failed to create "%SHORTCUT_NAME_BACKEND%" )

echo.
echo ============================================
echo   Done. Shortcuts on Desktop:
echo     "DocuMind"          - full app
echo     "DocuMind Backend"  - backend only
echo ============================================
echo.
pause
