# Open a window that follows a RUN across crashes and auto-resumes (unlike watch_run.ps1, which follows one queue).
# It shows runs\<Name>\train.log (appended by every resume) plus, as banner lines, the queue logs' start / finish /
# error lines for this run (exit codes, tracebacks, guard refusals, the watcher's resume decisions).
# Closing the window does NOT stop training.
#   .\scripts\watch_run_follow.ps1 -Name world_fastmem_change_12k
param([Parameter(Mandatory)][string]$Name, [switch]$Inner)
$root = Split-Path $PSScriptRoot -Parent
if (-not $Inner) {
    $me = $MyInvocation.MyCommand.Path
    Start-Process powershell.exe -ArgumentList "-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$me`"", "-Name", $Name, "-Inner"
    return
}
$host.UI.RawUI.WindowTitle = "NewBrain run $Name (follows restarts)"
$qdir = Join-Path $root "runs\_queues"
$train = Join-Path $root "runs\$Name\train.log"
$pos = @{}                                   # file -> lines already shown
$pat = "starting $Name|finished $Name|guard REFUSED|Traceback|Error|NAN ABORT|$Name failed|resume \d+ of|started queue"
function Show-New($file, [switch]$All, [switch]$Banner) {
    if (-not (Test-Path -LiteralPath $file)) { return }
    $lines = @(Get-Content -LiteralPath $file -ErrorAction SilentlyContinue)
    $n = if ($pos.ContainsKey($file)) { $pos[$file] } else { [Math]::Max(0, $lines.Count - $(if ($All) { 60 } else { 0 })) }
    for ($i = $n; $i -lt $lines.Count; $i++) {
        $l = $lines[$i]
        if ($Banner) {
            if ($l -match $pat) { Write-Host ">>> [$(Split-Path $file -Leaf)] $l" -ForegroundColor Yellow }
        } else { Write-Host $l }
    }
    $pos[$file] = $lines.Count
}
Write-Host "following run $Name (train.log + its queue logs); closing this window does not stop training" -ForegroundColor Cyan
Show-New $train -All
foreach ($f in Get-ChildItem $qdir -Filter *.log | Where-Object { Select-String -LiteralPath $_.FullName -Pattern "starting $Name" -Quiet }) {
    $pos[$f.FullName] = @(Get-Content -LiteralPath $f.FullName).Count   # (old queue logs: only new lines from now on)
}
while ($true) {
    Show-New $train
    foreach ($f in Get-ChildItem $qdir -Filter *.log -ErrorAction SilentlyContinue) {
        if ($pos.ContainsKey($f.FullName) -or (Select-String -LiteralPath $f.FullName -Pattern "starting $Name|watching runs .*$Name" -Quiet)) {
            Show-New $f.FullName -Banner
        }
    }
    Start-Sleep 3
}
