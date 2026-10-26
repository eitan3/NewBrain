# Open a window that follows a run's log. Closing the window does NOT stop training.
#   .\scripts\watch_run.ps1 -Queue stage0      (whole queue)
#   .\scripts\watch_run.ps1 -Name s0_ssd       (one run)
param([string]$Queue, [string]$Name)
$root = Split-Path $PSScriptRoot -Parent
if ($Name) { $log = Join-Path $root "runs\$Name\train.log"; $title = "NewBrain run $Name" }
elseif ($Queue) { $log = Join-Path $root "runs\_queues\$Queue.log"; $title = "NewBrain queue $Queue" }
else { throw "give -Queue or -Name" }
$cmd = "`$host.UI.RawUI.WindowTitle = '$title'; " +
       "while (-not (Test-Path -LiteralPath '$log')) { Start-Sleep 1 }; " +
       "Get-Content -LiteralPath '$log' -Wait -Tail 60"
Start-Process powershell.exe -ArgumentList "-NoExit", "-NoProfile", "-Command", $cmd
