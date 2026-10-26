# launch_queue.ps1 with a per-run code snapshot, a launch-time VRAM guard and an auto-resume watcher (for the next
# runs; launch_queue.ps1 itself is unchanged). See scripts\README_launch.md.
#   .\scripts\launch_queue_v2.ps1 -Queue q -Runs "my_run | --flags ..."
# * code snapshot: at launch every <Src>\*.py (default fruit-fly-sim-next) is copied to runs\<name>\code_snapshot\
#   (+ SHA256SUMS.txt) and the run executes from that snapshot, so later edits can never change a running or resumed
#   run's code. A fresh run refuses an existing snapshot (pick a new name); a run whose arguments contain --resume
#   reuses its snapshot (created only if missing).
# * guard (fresh runs, unless -SkipGuard): right before the run, in the queue (so it waits for -After like the run),
#   vram_guard.py from the snapshot runs ~20 steps at the FINAL size with the run's exact flags on the GPU (one eval,
#   one IVF refit, K = 16); peak reserved > cap - GuardMarginGB (cap = --mem-fraction x GPU memory) refuses the run:
#   the queue log gets "guard REFUSED <name>" with the savers that would fit, and "finished <name> (exit 98)".
# * watcher (unless -NoWatcher): scripts\auto_resume_v2.ps1 for this queue, detached (Win32_Process.Create); its pid
#   in runs\_queues\<queue>.watcher.pid, its log in runs\_queues\<queue>.watcher.log.
param(
    [Parameter(Mandatory)][string]$Queue,
    [Parameter(Mandatory)][string[]]$Runs,
    [string]$After = "",
    [switch]$NoWatch,
    [switch]$Prune,
    [string]$Src = "fruit-fly-sim-next",      # code folder (relative to the root, or absolute) snapshotted for fresh runs
    [switch]$SkipGuard,
    [double]$GuardMarginGB = 2.0,
    [switch]$NoWatcher,
    [int]$MaxResume = 2,
    [string]$Root = "",                       # (dry-run hooks: another tree / python / poll interval)
    [string]$Py = "",
    [int]$PollSec = 30
)
if ($Prune) { & (Join-Path $PSScriptRoot "prune_runs.ps1") }
$root = if ($Root) { $Root } else { Split-Path $PSScriptRoot -Parent }
$py = if ($Py) { $Py } else { Join-Path $root ".venv\Scripts\python.exe" }
$src = if ([System.IO.Path]::IsPathRooted($Src)) { $Src } else { Join-Path $root $Src }
$runsDir = Join-Path $root "runs"
$qdir = Join-Path $runsDir "_queues"
New-Item -ItemType Directory -Force $qdir | Out-Null
$qlog = Join-Path $qdir "$Queue.log"
$qcmd = Join-Path $qdir "$Queue.cmd"
if (Test-Path $qcmd) { throw "queue $Queue already exists ($qcmd): pick a new queue name" }

$lines = @("@echo off")
if ($After) {
    $afterLog = Join-Path $qdir "$After.log"
    $lines += "echo ===== %date% %time% waiting for queue $After to finish"
    $lines += ":wait"
    $lines += "findstr /c:`"queue $After done`" `"$afterLog`" >nul 2>&1 || (ping -n 31 127.0.0.1 >nul & goto wait)"
}
$names = @()
$i = 0
foreach ($r in $Runs) {
    $i += 1
    $name, $trainArgs = $r.Split("|", 2) | ForEach-Object { $_.Trim() }
    $names += $name
    $rdir = Join-Path $runsDir $name
    $snap = Join-Path $rdir "code_snapshot"
    $resume = $trainArgs -match "--resume"
    if (-not ($resume -and (Test-Path (Join-Path $snap "train_lm.py")))) {
        if ((Test-Path $snap) -and -not $resume) { throw "code snapshot already exists for $name (pick a new run name): $snap" }
        New-Item -ItemType Directory -Force $snap | Out-Null
        Copy-Item (Join-Path $src "*.py") $snap
        Get-ChildItem $snap -Filter *.py | Get-FileHash -Algorithm SHA256 |
            ForEach-Object { "{0} *{1}" -f $_.Hash.ToLower(), (Split-Path $_.Path -Leaf) } |
            Set-Content (Join-Path $snap "SHA256SUMS.txt") -Encoding ascii
    }
    if (-not $resume) {  # (the watcher resumes with these arguments + --resume <last.pt>)
        Set-Content -Path (Join-Path $rdir "launch_args.txt") -Value $trainArgs -Encoding ascii
    }
    $script = Join-Path $snap "train_lm.py"
    if ($trainArgs.StartsWith("@")) {  # "name | @other_script.py args": another script, also from the snapshot
        $s, $trainArgs = $trainArgs.Substring(1).Split(" ", 2)
        $script = Join-Path $snap $s
    }
    $lines += "cd /d `"$snap`""
    if (-not $SkipGuard -and -not $resume -and -not $r.Split("|", 2)[1].Trim().StartsWith("@")) {
        $argFile = Join-Path $rdir "guard_args.txt"
        Set-Content -Path $argFile -Value $trainArgs -Encoding ascii
        $lines += "echo ===== %date% %time% guard $name (vram_guard.py, final size, margin $GuardMarginGB GB)"
        $lines += "`"$py`" -u `"$snap\vram_guard.py`" --argv-file `"$argFile`" --runs `"$runsDir`" --margin-gb $GuardMarginGB --workdir `"$rdir\guard`""
        $lines += "if errorlevel 1 goto refused_$i"
    }
    $lines += "echo ===== %date% %time% starting $name (code snapshot $snap)"
    $lines += "`"$py`" -u `"$script`" --run-name $name $trainArgs"
    $lines += "echo ===== %date% %time% finished $name (exit %errorlevel%)"
    $lines += "goto next_$i"
    $lines += ":refused_$i"
    $lines += "echo ===== %date% %time% guard REFUSED $name (see the guard's lines above; the run was not started)"
    $lines += "echo ===== %date% %time% finished $name (exit 98)"
    $lines += ":next_$i"
}
$lines += "echo ===== queue $Queue done"
Set-Content -Path $qcmd -Value $lines -Encoding ascii

$cmdline = "cmd.exe /c `"`"$qcmd`" >> `"$qlog`" 2>&1`""
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$res = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $cmdline; CurrentDirectory = $root; ProcessStartupInformation = $startup }
if ($res.ReturnValue -ne 0) { throw "could not start queue (code $($res.ReturnValue))" }
Set-Content -Path (Join-Path $qdir "$Queue.pid") -Value $res.ProcessId
Write-Host "queue '$Queue' started, pid $($res.ProcessId), $($Runs.Count) run(s) (each from runs\<name>\code_snapshot; fresh snapshots from $src); log $qlog"

if (-not $NoWatcher) {
    $wlog = Join-Path $qdir "$Queue.watcher.log"
    $ps = Join-Path $PSHOME "powershell.exe"
    $wargs = "-NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $PSScriptRoot 'auto_resume_v2.ps1')`" -Queue $Queue " +
             "-Names $($names -join ',') -Max $MaxResume -Root `"$root`" -Py `"$py`" -PollSec $PollSec"
    $wcmd = "cmd.exe /c `"`"$ps`" $wargs >> `"$wlog`" 2>&1`""
    $wres = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = $wcmd; CurrentDirectory = $root; ProcessStartupInformation = $startup }
    if ($wres.ReturnValue -ne 0) { throw "could not start the watcher (code $($wres.ReturnValue))" }
    Set-Content -Path (Join-Path $qdir "$Queue.watcher.pid") -Value $wres.ProcessId
    Write-Host "auto-resume watcher started, pid $($wres.ProcessId) (max $MaxResume resumes per run); log $wlog"
}
if (-not $NoWatch) {  # one window per run that keeps following it across crashes / auto-resumes
    foreach ($n in $names) { & (Join-Path $PSScriptRoot "watch_run_follow.ps1") -Name $n }
}
