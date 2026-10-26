# PREPARED, RUN ONLY ON THE OWNER'S WORD: move old NewBrain run folders / captures from C: to X:\NewBrain_archive.
# Per item: robocopy (throttled with /IPG so the live run's reads from X: are not starved), then verify (file count,
# total bytes, SHA256 of every *.pt), and only if all match, remove the C: copy. Any mismatch: stop, keep both copies.
#   .\scripts\archive_to_x.ps1            (dry run: lists items, sizes, target paths)
#   .\scripts\archive_to_x.ps1 -Go        (copy + verify + remove from C:)
param([switch]$Go, [int]$Ipg = 20)
$root = "C:\Eitan\Python\NewBrain"
$dst = "X:\NewBrain_archive"
$items = @(
    "runs\world_wave26_12k", "runs\world_wave27c_small_12k", "runs\world_wave27b_small_12k",
    "runs\world_wave27_small_12k", "runs\world_blob25d_12k", "runs\world_blob25d_12k_r5_stopped",
    "runs\world_blob25d_12k_r6_stopped", "runs\world_mem_life_12k", "runs\wiki_mem_12k",
    "runs\wiki_mem_12k_read0_aborted", "speed_wip\splat\realcheck", "speed_wip\mem26", "fruit-fly-sim-wave\cap"
)
function Stat($p) {
    $f = Get-ChildItem -LiteralPath $p -Recurse -File -ErrorAction Stop
    [pscustomobject]@{ n = $f.Count; bytes = ($f | Measure-Object Length -Sum).Sum }
}
foreach ($it in $items) {
    $src = Join-Path $root $it; $tgt = Join-Path $dst $it
    if (-not (Test-Path -LiteralPath $src)) { "skip (missing): $it"; continue }
    $s = Stat $src
    "{0}: {1} files, {2:N1} GB -> {3}" -f $it, $s.n, ($s.bytes / 1e9), $tgt
    if (-not $Go) { continue }
    robocopy $src $tgt /E /COPY:DAT /R:2 /W:5 /IPG:$Ipg /NP /NFL /NDL | Out-Null
    if ($LASTEXITCODE -ge 8) { "  ROBOCOPY FAILED ($LASTEXITCODE); stopping, nothing removed"; break }
    $t = Stat $tgt
    if ($t.n -ne $s.n -or $t.bytes -ne $s.bytes) { "  MISMATCH (count/bytes); stopping, nothing removed"; break }
    $bad = 0
    foreach ($pt in Get-ChildItem -LiteralPath $src -Recurse -File -Filter *.pt) {
        $rel = $pt.FullName.Substring($src.Length)
        if ((Get-FileHash $pt.FullName).Hash -ne (Get-FileHash (Join-Path $tgt $rel)).Hash) { $bad += 1; "  HASH MISMATCH $rel" }
    }
    if ($bad) { "  stopping, nothing removed"; break }
    Remove-Item -LiteralPath $src -Recurse -Force
    "  verified and removed from C:"
}
