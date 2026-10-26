# Start one or more training runs, one after another, as a detached background job.
# The job survives closing this window and the viewer window; stop it with stop_run.ps1.
#
#   .\scripts\launch_queue.ps1 -Queue stage0 -Runs "s0_attn | --mixer attn", "s0_ssd | --mixer ssd"
#   .\scripts\launch_queue.ps1 -Queue one -Runs "my_run | --mixer ssd --tokens 50e6"
#
# Each run writes runs\<name>\train.log (+ metrics.jsonl, final.pt, sample.txt).
# The whole queue's console goes to runs\_queues\<Queue>.log, and its PID to runs\_queues\<Queue>.pid.
param(
    [Parameter(Mandatory)][string]$Queue,
    [Parameter(Mandatory)][string[]]$Runs,
    [string]$After = "",   # wait until queue <After> has finished before starting (one GPU job at a time)
    [switch]$NoWatch,
    [switch]$Prune         # first run prune_runs.ps1 (old runs' *.pt -> Recycle Bin; the 5 newest runs untouched)
)
if ($Prune) { & (Join-Path $PSScriptRoot "prune_runs.ps1") }
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root ".venv\Scripts\python.exe"
$train = Join-Path $root "fruit-fly-sim\train_lm.py"
$qdir = Join-Path $root "runs\_queues"
New-Item -ItemType Directory -Force $qdir | Out-Null
$qlog = Join-Path $qdir "$Queue.log"
$qcmd = Join-Path $qdir "$Queue.cmd"

$lines = @("@echo off", "cd /d `"$root\fruit-fly-sim`"")
if ($After) {
    $afterLog = Join-Path $qdir "$After.log"
    $lines += "echo ===== %date% %time% waiting for queue $After to finish"
    $lines += ":wait"
    $lines += "findstr /c:`"queue $After done`" `"$afterLog`" >nul 2>&1 || (ping -n 31 127.0.0.1 >nul & goto wait)"
}
foreach ($r in $Runs) {
    $name, $trainArgs = $r.Split("|", 2) | ForEach-Object { $_.Trim() }
    $script = $train
    if ($trainArgs.StartsWith("@")) {  # "name | @other_script.py args" runs another script in fruit-fly-sim
        $s, $trainArgs = $trainArgs.Substring(1).Split(" ", 2)
        $script = Join-Path $root "fruit-fly-sim\$s"
    }
    $lines += "echo ===== %date% %time% starting $name"
    $lines += "`"$py`" -u `"$script`" --run-name $name $trainArgs"
    $lines += "echo ===== %date% %time% finished $name (exit %errorlevel%)"
}
$lines += "echo ===== queue $Queue done"
Set-Content -Path $qcmd -Value $lines -Encoding ascii

# Win32_Process.Create starts the job outside this shell's process tree, so it keeps running
# when this window (or the Claude session that ran it) closes.
$cmdline = "cmd.exe /c `"`"$qcmd`" >> `"$qlog`" 2>&1`""
# The job's own console window is hidden (ShowWindow = 0): closing a console would kill training.
# Watch progress in the separate viewer window instead; closing that one is always safe.
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$res = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $cmdline; CurrentDirectory = "$root\fruit-fly-sim"; ProcessStartupInformation = $startup }
if ($res.ReturnValue -ne 0) { throw "could not start queue (code $($res.ReturnValue))" }
Set-Content -Path (Join-Path $qdir "$Queue.pid") -Value $res.ProcessId
Write-Host "queue '$Queue' started, pid $($res.ProcessId), $($Runs.Count) run(s); log $qlog"
if (-not $NoWatch) { & (Join-Path $PSScriptRoot "watch_run.ps1") -Queue $Queue }
