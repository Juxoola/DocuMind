# create_shortcut.ps1 — создаёт ярлык NotebookLM на рабочем столе
$desktop = [Environment]::GetFolderPath('Desktop')
$target = Join-Path $PSScriptRoot 'run_app.bat'
$icon = Join-Path $PSScriptRoot 'bin\llama-server.exe'

$wshell = New-Object -ComObject WScript.Shell
$shortcut = $wshell.CreateShortcut([System.IO.Path]::Combine($desktop, 'NotebookLM.lnk'))
$shortcut.TargetPath = $target
$shortcut.WorkingDirectory = $PSScriptRoot
$shortcut.Description = 'NotebookLM Local Clone'
$shortcut.WindowStyle = 1
if (Test-Path $icon) {
    $shortcut.IconLocation = "$icon, 0"
}
$shortcut.Save()
Write-Host 'OK'
