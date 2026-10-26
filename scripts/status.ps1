# Show every run: running or not, and the last log line.   .\scripts\status.ps1
$root = Split-Path $PSScriptRoot -Parent
Get-ChildItem (Join-Path $root "runs") -Directory | Where-Object Name -ne "_queues" | ForEach-Object {
    $pidFile = Join-Path $_.FullName "run.pid"
    $alive = $false
    if (Test-Path $pidFile) { $alive = [bool](Get-Process -Id ([int](Get-Content $pidFile)) -ErrorAction SilentlyContinue) }
    $log = Join-Path $_.FullName "train.log"
    $last = if (Test-Path $log) { Get-Content $log -Tail 1 } else { "" }
    [pscustomobject]@{ Run = $_.Name; Running = $alive; Last = $last }
} | Format-Table -AutoSize -Wrap
