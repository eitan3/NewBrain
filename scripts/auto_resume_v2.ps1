# Auto-resume watcher for a launch_queue_v2.ps1 queue (started by it, detached; see scripts\README_launch.md).
# Waits until the queue's log says "queue <q> done"; then for every run of the queue:
#   exit 0                          -> done
#   guard refused (exit 98)         -> not resumed (the owner decides: add savers / new run)
#   NaN abort (nan_abort.pt newer than last.pt, or "NAN ABORT" in the tail of its train.log / the queue log)
#                                   -> not resumed (it would recur: same data / RNG)
#   no last.pt, or -Max resumes used -> not resumed
#   otherwise                       -> resumed: a new queue <Queue>_resume<k> via launch_queue_v2.ps1 with the run's
#                                      launch_args.txt + --resume runs\<name>\last.pt, from its code_snapshot
# and waits for that queue's "done" in turn. Exits when nothing is left to resume (e.g. every run finished cleanly).
#   .\scripts\auto_resume_v2.ps1 -Queue q -Names run_a[,run_b] [-Max 2]
param(
    [Parameter(Mandatory)][string]$Queue,
    [Parameter(Mandatory)][string[]]$Names,
    [int]$Max = 2,
    [string]$Root = "",          # (dry-run hooks)
    [string]$Py = "",
    [int]$PollSec = 30
)
$root = if ($Root) { $Root } else { Split-Path $PSScriptRoot -Parent }
$runsDir = Join-Path $root "runs"
$qdir = Join-Path $runsDir "_queues"
$Names = @($Names | ForEach-Object { $_ -split "," } | Where-Object { $_ })
function Say($m) { Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [watcher $Queue] $m" }
Say "watching runs $($Names -join ', ') (max $Max resumes each)"
$used = @{}
foreach ($n in $Names) { $used[$n] = 0 }
$q = $Queue
$k = 0
while ($true) {
    $qlog = Join-Path $qdir "$q.log"
    while (-not (Test-Path $qlog) -or -not (Select-String -Path $qlog -SimpleMatch "queue $q done" -Quiet -ErrorAction SilentlyContinue)) {
        Start-Sleep $PollSec
    }
    Say "queue $q done"
    $todo = @()
    foreach ($name in $Names) {
        $e = [regex]::Escape($name)
        $fin = Select-String -Path $qlog -Pattern "finished $e \(exit (-?\d+)\)" | Select-Object -Last 1
        if (-not $fin) { continue }                                   # (not in this queue)
        $code = [int]$fin.Matches[0].Groups[1].Value
        $rdir = Join-Path $runsDir $name
        $last = Join-Path $rdir "last.pt"
        $nanpt = Join-Path $rdir "nan_abort.pt"
        $tl = Join-Path $rdir "train.log"
        $nanlog = (($Names.Count -eq 1) -and (Get-Content $qlog -Tail 40 -ErrorAction SilentlyContinue | Select-String -SimpleMatch "NAN ABORT" -Quiet)) -or
                  ((Test-Path $tl) -and (Get-Content $tl -Tail 40 -ErrorAction SilentlyContinue | Select-String -SimpleMatch "NAN ABORT" -Quiet))
        $nanpt_new = (Test-Path $nanpt) -and (-not (Test-Path $last) -or (Get-Item $nanpt).LastWriteTime -gt (Get-Item $last).LastWriteTime)
        if ($nanpt_new -or $nanlog) { Say "$name stopped on a NaN abort (exit $code); not resuming (left for the owner)"; continue }
        if ($code -eq 0) { Say "$name finished cleanly"; continue }
        if (Select-String -Path $qlog -SimpleMatch "guard REFUSED $name" -Quiet) {
            Say "${name}: the VRAM guard refused it; not resuming (see $qlog)"; continue
        }
        if (-not (Test-Path $last)) { Say "$name failed (exit $code) and has no last.pt; not resuming"; continue }
        if ($used[$name] -ge $Max) { Say "$name failed (exit $code); resume limit $Max reached, not resuming"; continue }
        $argf = Join-Path $rdir "launch_args.txt"
        if (-not (Test-Path $argf)) { Say "$name failed (exit $code) but has no launch_args.txt; not resuming"; continue }
        $used[$name] += 1
        $a = (Get-Content $argf -Raw).Trim()
        $todo += "$name | $a --resume `"$last`""
        Say "$name failed (exit $code); resume $($used[$name]) of $Max from $last"
    }
    if (-not $todo) { Say "nothing left to resume; watcher exits"; break }
    $lock = Join-Path $runsDir "gpu.lock"
    if (Test-Path $lock) { Remove-Item $lock }       # (the queue is done: its lock is stale)
    $k += 1
    $q = "$Queue" + "_resume$k"
    $lq = @{ Queue = $q; Runs = $todo; NoWatch = $true; NoWatcher = $true; SkipGuard = $true; Root = $root }
    if ($Py) { $lq.Py = $Py }
    & (Join-Path $PSScriptRoot "launch_queue_v2.ps1") @lq
    Say "started queue $q"
}
exit 0
