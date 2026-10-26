# Stop a training queue (and the run it is on), or a single run.
#   .\scripts\stop_run.ps1 -Queue stage0
#   .\scripts\stop_run.ps1 -Name s0_ssd      (stops only that run; a queue moves on to its next run)
param([string]$Queue, [string]$Name)
$root = Split-Path $PSScriptRoot -Parent
if ($Queue) { $pidFile = Join-Path $root "runs\_queues\$Queue.pid" }
elseif ($Name) { $pidFile = Join-Path $root "runs\$Name\run.pid" }
else { throw "give -Queue or -Name" }
if (-not (Test-Path $pidFile)) { throw "no pid file: $pidFile" }
$procId = [int](Get-Content $pidFile -TotalCount 1)
if (-not (Get-Process -Id $procId -ErrorAction SilentlyContinue)) { Write-Host "pid $procId is not running"; return }
taskkill.exe /PID $procId /T /F
Write-Host "stopped pid $procId and its children"
