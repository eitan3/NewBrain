# PREPARED, RUN ONLY ON THE OWNER'S WORD: resume world_fastmem_change_12k from its last.pt (step 2,000) with the
# packed-scan fix (fruit-fly-sim-fastmem-b). The run's code_snapshot is swapped for the fixed code; the old snapshot
# is kept as code_snapshot_orig. Launch args unchanged (the fix's --fastmem-scan-budget defaults to 2^22).
# Opens the run-follow viewer and starts a fresh auto-resume watcher (Max 2).
$root = Split-Path $PSScriptRoot -Parent
$run = Join-Path $root "runs\world_fastmem_change_12k"
$snap = Join-Path $run "code_snapshot"
if (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'train_lm' }) {
    throw "a train_lm process is running; not resuming" }
if (-not (Test-Path (Join-Path $run "code_snapshot_orig"))) { Rename-Item $snap "code_snapshot_orig" }
New-Item -ItemType Directory -Force $snap | Out-Null
Copy-Item (Join-Path $root "fruit-fly-sim-fastmem-b\*.py") $snap
Get-ChildItem $snap -Filter *.py | Get-FileHash -Algorithm SHA256 |
    ForEach-Object { "{0} *{1}" -f $_.Hash.ToLower(), (Split-Path $_.Path -Leaf) } |
    Set-Content (Join-Path $snap "SHA256SUMS.txt") -Encoding ascii
$lock = Join-Path $root "runs\gpu.lock"; if (Test-Path $lock) { Remove-Item $lock }
$a = (Get-Content (Join-Path $run "launch_args.txt") -Raw).Trim()
& (Join-Path $PSScriptRoot "launch_queue_v2.ps1") -Queue fastmem2 -Runs "world_fastmem_change_12k | $a --resume `"$run\last.pt`""
