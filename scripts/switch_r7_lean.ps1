# PREPARED, RUN ONLY ON THE OWNER'S WORD: switch r7 to the lean resume (fruit-fly-sim-r7-lean + --vram-readout-lean 1).
# 1 stop the current resume queue + the exact-code watcher, 2 clear gpu.lock, 3 resume from last.pt with the lean code
# (viewer window), 4 start auto_resume_r7_lean.ps1 detached (Win32_Process.Create) watching the lean queue (Max 2 fresh).
param([string]$CurrentQueue = "blob25d_r7_resume1")
$root = Split-Path $PSScriptRoot -Parent
$qdir = Join-Path $root "runs\_queues"
$lean = Join-Path $root "fruit-fly-sim-r7-lean"
$py = Join-Path $root ".venv\Scripts\python.exe"
$Name = "world_blob25d_12k"
$last = Join-Path $root "runs\$Name\last.pt"
# 1
$wpid = [int](Get-Content (Join-Path $qdir "auto_resume_r7.pid") -TotalCount 1)
if (Get-Process -Id $wpid -ErrorAction SilentlyContinue) { Stop-Process -Id $wpid -Force; "stopped watcher $wpid" }
& (Join-Path $PSScriptRoot "stop_run.ps1") -Queue $CurrentQueue
Start-Sleep 8
# 2
$lock = Join-Path $root "runs\gpu.lock"; if (Test-Path $lock) { Remove-Item $lock; "cleared gpu.lock" }
# 3
$a = (Get-Content (Join-Path $PSScriptRoot "auto_resume_r7_lean.ps1") | Select-String -Pattern '^\$a = "(.*)"$').Matches[0].Groups[1].Value
$q = "blob25d_r7_lean"
$qcmd = Join-Path $qdir "$q.cmd"; $qlog = Join-Path $qdir "$q.log"
Set-Content -Path $qcmd -Encoding ascii -Value @("@echo off", "cd /d `"$lean`"",
    "echo ===== %date% %time% starting $Name (lean resume from $last)",
    "`"$py`" -u `"$lean\train_lm.py`" --run-name $Name $a --resume `"$last`"",
    "echo ===== %date% %time% finished $Name (exit %errorlevel%)", "echo ===== queue $q done")
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$res = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "cmd.exe /c `"`"$qcmd`" >> `"$qlog`" 2>&1`""; CurrentDirectory = $lean; ProcessStartupInformation = $startup }
Set-Content -Path (Join-Path $qdir "$q.pid") -Value $res.ProcessId
"lean queue $q started, pid $($res.ProcessId)"
& (Join-Path $PSScriptRoot "watch_run.ps1") -Queue $q
# 4
$st2 = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]7 }
$w = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$PSScriptRoot\auto_resume_r7_lean.ps1`" -WatchQueue $q";
    CurrentDirectory = $root; ProcessStartupInformation = $st2 }
Set-Content -Path (Join-Path $qdir "auto_resume_r7_lean.pid") -Value $w.ProcessId
"lean watcher pid $($w.ProcessId)"
