# PREPARED, RUN ONLY ON THE OWNER'S WORD: the GPU check for the fastmem-c pack run (one GPU job; nothing else may use the
# GPU). vram_guard.py from fruit-fly-sim-fastmem-c: the run's exact flags (world_fastmem_b's + --fastmem-pack 4, data
# world_change_mix_v2) at the FINAL size (192 experts, 36 heads), K = 16 on every step, worst-case routing over each
# 2,048-token pack (every token of a pack to the same k experts / gk global heads), one evaluate(); capped at the RUN cap
# (--mem-fraction 0.896 of 25.65 GB = ~23 GB, a hard per-process cap: it OOMs, never spills; ABOVE the usual 16 GB
# side-job cap, so it needs the owner's explicit OK), 2 CPU threads. Pass: worst-case peak reserved <= 21.4 - 2 GB. Then the same without worst-case routing at K 6
# for the s/step comparison with world_fastmem_b (its train.log: ~9-10k tok/s = ~1.7 s/step).
#   .\scripts\gpu_check_fastmem_c.ps1            -> runs\_gpu_check_fastmem_c\{worst,speed}\vram_guard.json + .log
param([string]$Extra = "")                   # extra train_lm flags, e.g. "--vram-ckpt-concept 1 --vram-ckpt-norm 1"
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root ".venv\Scripts\python.exe"
$src = Join-Path $root "fruit-fly-sim-fastmem-c"
$out = Join-Path $root "runs\_gpu_check_fastmem_c"
if (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'train_lm|vram_guard' }) {
    throw "a train_lm / vram_guard process is running; not starting" }
if (Test-Path (Join-Path $root "runs\gpu.lock")) { throw "runs\gpu.lock exists: the GPU is in use" }
New-Item -ItemType Directory -Force $out | Out-Null
$argv = "--arch loop --spike graded --homeo 0.01 --homeo-mode floor --opt brain --batch 16 --accum 2 --eval-batch 16 " +
    "--route-aux 0.02 --capacity 1.5 --moe 192,6,192,384,auxfree --heads 36 --start heads=6,experts=48 --neuro " +
    "--neuro-every 25 --neuro-t 0.05,0.6 --head-group 6 --head-groups 6 --state none --waves --slots 16 --ws-direct " +
    "--anneal 0.05,0.2 --birth-mode split --birth-delta 0.1 --route-z 1e-4 --route-noise 0.3 --route-noise-decay 0 " +
    "--deep-sup 0.3 --halt 1 --think-w 0.1 --think-fb 1 --think-target latent --think-n 16 --think-use 1 " +
    "--data X:\Datasets\world_change_mix_v2 --data-mode stream --tokens 200e6 --concepts 16,32,64,128 --chunk-ce 2048 " +
    "--ckpt-route 1 --mem-fraction 0.896 --frag-gb 3 --nan-abort 1 --memory fastmem --fastmem-size lite " +
    "--fastmem-stream 1 --fastmem-write-grad full --fastmem-ckpt 1 --memory-save 1 --fastmem-pack 4"
if ($Extra) { $argv += " $Extra" }
Set-Content (Join-Path $out "argv.txt") $argv -Encoding ascii
$env:OMP_NUM_THREADS = "2"; $env:MKL_NUM_THREADS = "2"; $env:CUDA_VISIBLE_DEVICES = "0"
Push-Location $src
foreach ($c in @(@{n = "worst"; a = @("--worst-route", "1", "--K", "16")}, @{n = "speed"; a = @("--worst-route", "0", "--K", "6")})) {
    $w = Join-Path $out $c.n
    New-Item -ItemType Directory -Force $w | Out-Null
    & $py -B -u (Join-Path $src "vram_guard.py") --argv-file (Join-Path $out "argv.txt") --runs (Join-Path $root "runs") `
        --margin-gb 2 --workdir $w @($c.a) *>&1 | Tee-Object (Join-Path $w "guard.log")
    "===== $($c.n) exit $LASTEXITCODE"
}
Pop-Location
