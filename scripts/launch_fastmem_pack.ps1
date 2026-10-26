# PREPARED, RUN ONLY ON THE OWNER'S WORD: world_fastmem_pack_12k from step 0. Code fruit-fly-sim-fastmem-c (snapshotted
# by launch_queue_v2), data X:\Datasets\world_change_mix_v2, world_fastmem_b's flags + --fastmem-pack 4. The queue runs
# its in-queue vram_guard first (run flags, final size), then the run; a run-follow viewer window; auto-resume watcher
# (MaxResume 2). The CPU on/off eval chain (2K, 4K, 6K, 8K, 10K, final; --pack 4 + --pack 1 rows; in-context split) is a
# separate script: world_change\eval_chain_fmpack.sh (started after this).
param([string]$Extra = "")                   # extra train_lm flags, e.g. "--vram-ckpt-concept 1 --vram-ckpt-norm 1"
$root = Split-Path $PSScriptRoot -Parent
if (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'train_lm|vram_guard' }) {
    throw "a train_lm / vram_guard process is running; not launching" }
$name = "world_fastmem_pack_12k"
$a = "--arch loop --spike graded --homeo 0.01 --homeo-mode floor --opt brain --batch 16 --accum 2 --eval-batch 16 " +
    "--route-aux 0.02 --capacity 1.5 --moe 192,6,192,384,auxfree --heads 36 --start heads=6,experts=48 --neuro " +
    "--neuro-every 25 --neuro-t 0.05,0.6 --head-group 6 --head-groups 6 --state none --waves --slots 16 --ws-direct " +
    "--anneal 0.05,0.2 --birth-mode split --birth-delta 0.1 --route-z 1e-4 --route-noise 0.3 --route-noise-decay 0 " +
    "--deep-sup 0.3 --halt 1 --think-w 0.1 --think-fb 1 --think-target latent --think-n 16 --think-use 1 " +
    "--data X:\Datasets\world_change_mix_v2 --data-mode stream --tokens 200e6 --concepts 16,32,64,128 --chunk-ce 2048 " +
    "--ckpt-route 1 --mem-fraction 0.896 --frag-gb 3 --nan-abort 1 --save-at 1000 --keep-ckpts 5 --save-report-ckpts 1 " +
    "--diag-every 250 --runs $root\runs --memory fastmem --fastmem-size lite --fastmem-stream 1 --fastmem-write-grad full " +
    "--fastmem-ckpt 1 --memory-save 1 --fastmem-pack 4"
if ($Extra) { $a += " $Extra" }
& (Join-Path $PSScriptRoot "launch_queue_v2.ps1") -Queue fmpack -Src "fruit-fly-sim-fastmem-c" -MaxResume 2 -Runs "$name | $a"
