# PREPARED, NOT RUNNING: the auto-resume watcher for r7 with the readout-lean fix (needs the owner's "use the fix").
# Same queue / run / args as auto_resume_r7.ps1, plus --vram-readout-lean 1, and the resume runs from the frozen
# folder fruit-fly-sim-r7-lean\ (r7's code snapshot + only the readout-lean change).
param([string]$Root = "", [string]$TrainCmd = "", [switch]$NoView, [string]$WatchQueue = "blob25d_r7")
# (-WatchQueue: the queue to watch first, e.g. the lean queue started by switch_r7_lean.ps1)
# (-Root / -TrainCmd / -NoView: dry-run hooks only; defaults = the real setup)
$root = if ($Root) { $Root } else { Split-Path $PSScriptRoot -Parent }
$lean = Join-Path $root "fruit-fly-sim-r7-lean"
$py = Join-Path $root ".venv\Scripts\python.exe"
$a = "--arch loop --spike graded --homeo 0.01 --homeo-mode floor --opt brain --batch 16 --accum 2 --eval-batch 16 --route-aux 0.02 --capacity 1.5 --moe 768,6,192,384,auxfree --heads 36 --start heads=6,experts=192 --neuro --neuro-every 25 --neuro-t 0.05,0.6 --head-group 6 --head-groups 6 --state none --waves --slots 16 --ws-direct --anneal 0.05,0.2 --birth-mode split --birth-delta 0.1 --route-z 1e-4 --route-noise 0.3 --route-noise-decay 0 --deep-sup 0.3 --halt 1 --think-w 0.1 --think-fb 1 --think-target latent --think-n 16 --think-use 1 --data X:\Datasets\wiki_world_mix --tokens 200e6 --concepts 16,32,64,128 --gm 1 --gm-preset 25d --gm-probation-dtype int8 --gm-qcap -64 --gm-private 256 --gm-global 196608 --chunk-ce 2048 --ckpt-route 1 --mem-fraction 0.896 --frag-gb 3 --nan-abort 1 --save-at 1000 --keep-ckpts 5 --save-report-ckpts 1 --diag-every 250 --runs C:\Eitan\Python\NewBrain\runs --vram-readout-lean 1"
$log = Join-Path $root "runs\_queues\auto_resume_r7_lean.log"
$Name = "world_blob25d_12k"; $Queue = $WatchQueue; $Max = 2; $n = 0; $q = $Queue
while ($true) {
    $qlog = Join-Path $root "runs\_queues\$q.log"
    while (-not (Test-Path $qlog) -or -not (Select-String -Path $qlog -Pattern "queue $q done" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 30 }
    $fin = Select-String -Path $qlog -Pattern "finished $Name \(exit (\d+)\)" | Select-Object -Last 1
    $code = if ($fin) { [int]$fin.Matches[0].Groups[1].Value } else { -1 }
    if ($code -eq 0) { "$Name finished cleanly" | Tee-Object -FilePath $log -Append; break }
    $last = Join-Path $root "runs\$Name\last.pt"
    if ($n -ge $Max -or -not (Test-Path $last)) { "$Name failed (exit $code); not resuming (limit or no last.pt)" | Tee-Object -FilePath $log -Append; break }
    $nanpt = Join-Path $root "runs\$Name\nan_abort.pt"
    if (((Test-Path $nanpt) -and ((Get-Item $nanpt).LastWriteTime -gt (Get-Item $last).LastWriteTime)) -or
        (Get-Content $qlog -Tail 40 | Select-String -Pattern "NAN ABORT" -CaseSensitive -Quiet)) { "NaN abort; not resuming" | Tee-Object -FilePath $log -Append; break }
    $lock = Join-Path $root "runs\gpu.lock"; if (Test-Path $lock) { Remove-Item $lock }
    $n += 1; $q = "$Queue" + "_lean_resume$n"
    "$(Get-Date -Format HH:mm:ss) $Name failed (exit $code); lean resume $n of $Max from $last as queue $q" | Tee-Object -FilePath $log -Append
    # launch from the frozen lean folder (launch_queue.ps1 runs fruit-fly-sim\, so the queue cmd is written here directly)
    $qdir = Join-Path $root "runs\_queues"; $qcmd = Join-Path $qdir "$q.cmd"; $qlog2 = Join-Path $qdir "$q.log"
    Set-Content -Path $qcmd -Encoding ascii -Value @("@echo off", "cd /d `"$lean`"",
        "echo ===== %date% %time% starting $Name (lean resume from $last)",
        $(if ($TrainCmd) { $TrainCmd } else { "`"$py`" -u `"$lean\train_lm.py`" --run-name $Name $a --resume `"$last`"" }),
        "echo ===== %date% %time% finished $Name (exit %errorlevel%)", "echo ===== queue $q done")
    $startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
    $res = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = "cmd.exe /c `"`"$qcmd`" >> `"$qlog2`" 2>&1`""; CurrentDirectory = $lean; ProcessStartupInformation = $startup }
    Set-Content -Path (Join-Path $qdir "$q.pid") -Value $res.ProcessId
    if (-not $NoView) { & (Join-Path $PSScriptRoot "watch_run.ps1") -Queue $q }
}
