# Watch a training queue; if its run exits with an error, resume it from runs\<Name>\last.pt (at most -Max times).
# NOT started automatically: run it only when the owner has agreed to auto-resume.
#   .\scripts\auto_resume.ps1 -Queue blob25d_r5 -Name world_blob25d_12k -Max 2 -RunArgs "<the run's flags>"
param(
    [Parameter(Mandatory)][string]$Queue,
    [Parameter(Mandatory)][string]$Name,
    [Parameter(Mandatory)][string]$RunArgs,
    [int]$Max = 2
)
$root = Split-Path $PSScriptRoot -Parent
$n = 0
$q = $Queue
while ($true) {
    $log = Join-Path $root "runs\_queues\$q.log"
    while (-not (Test-Path $log) -or -not (Select-String -Path $log -Pattern "queue $q done" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 30 }
    $fin = Select-String -Path $log -Pattern "finished $Name \(exit (\d+)\)" | Select-Object -Last 1
    $code = if ($fin) { [int]$fin.Matches[0].Groups[1].Value } else { -1 }
    if ($code -eq 0) { Write-Host "$Name finished cleanly"; break }
    $last = Join-Path $root "runs\$Name\last.pt"
    if ($n -ge $Max) { Write-Host "$Name failed (exit $code); resume limit $Max reached, stopping"; break }
    if (-not (Test-Path $last)) { Write-Host "$Name failed (exit $code) and has no last.pt; stopping"; break }
    # a NaN abort would recur on resume (same data / RNG): leave it for the owner
    $nanpt = Join-Path $root "runs\$Name\nan_abort.pt"
    $nanlog = Get-Content $log -Tail 40 -ErrorAction SilentlyContinue | Select-String -Pattern "NAN ABORT" -CaseSensitive -Quiet
    if ((Test-Path $nanpt) -and ((Get-Item $nanpt).LastWriteTime -gt (Get-Item $last).LastWriteTime) -or $nanlog) {
        Write-Host "$Name stopped on a NaN abort; not resuming (left for the owner)"; break
    }
    $lock = Join-Path $root "runs\gpu.lock"
    if (Test-Path $lock) { Remove-Item $lock }
    $n += 1
    $q = "$Queue" + "_resume$n"
    Write-Host "$(Get-Date -Format HH:mm:ss) $Name failed (exit $code); resume $n of $Max from $last as queue $q"
    & (Join-Path $PSScriptRoot "launch_queue.ps1") -Queue $q -Runs "$Name | $RunArgs --resume $last"   # (opens the viewer window)
}
