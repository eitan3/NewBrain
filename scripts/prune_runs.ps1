# Free disk space in runs\: keep everything of the newest -Keep run folders (by the newest file inside); for the older runs
# move only the *.pt checkpoints to the Recycle Bin (logs, metrics, config, samples stay).
# Never touches runs\_queues (or any folder starting with "_") or a folder whose run.pid is a live train_lm process.
#
#   .\scripts\prune_runs.ps1 -DryRun          # list what would move, and the GB total
#   .\scripts\prune_runs.ps1 -Keep 8
param(
    [int]$Keep = 5,
    [switch]$DryRun
)
Add-Type -AssemblyName Microsoft.VisualBasic
$root = Split-Path $PSScriptRoot -Parent
$runs = Join-Path $root "runs"

function Test-LiveRun($dir) {
    $pidFile = Join-Path $dir.FullName "run.pid"
    if (-not (Test-Path $pidFile)) { return $false }
    $txt = (Get-Content $pidFile -Raw -ErrorAction SilentlyContinue)
    if (-not $txt) { return $false }
    $id = 0
    if (-not [int]::TryParse($txt.Trim(), [ref]$id)) { return $false }
    $p = Get-CimInstance Win32_Process -Filter "ProcessId = $id" -ErrorAction SilentlyContinue
    if (-not $p) { return $false }
    # a stale pid reused by another process (e.g. svchost, whose command line is unreadable) is not a live run
    if ($p.Name -notmatch "^python") { return $false }
    if ($p.CommandLine -and $p.CommandLine -notmatch "train_lm") { return $false }
    return $true
}

# rank by the newest file anywhere inside the run (a folder's own timestamp only changes when entries are added/removed)
function Get-RunTime($dir) {
    $t = Get-ChildItem $dir.FullName -Recurse -File -ErrorAction SilentlyContinue |
        Measure-Object LastWriteTime -Maximum
    if ($t.Maximum) { return $t.Maximum } else { return $dir.LastWriteTime }
}
$dirs = Get-ChildItem $runs -Directory | Where-Object { -not $_.Name.StartsWith("_") } |
    Sort-Object { Get-RunTime $_ } -Descending
$kept = $dirs | Select-Object -First ([Math]::Max($Keep, 0))
$old = $dirs | Select-Object -Skip ([Math]::Max($Keep, 0))
Write-Host "keeping the $($kept.Count) newest runs: $(($kept | ForEach-Object Name) -join ', ')"

$total = 0L
$n = 0
foreach ($d in $old) {
    if (Test-LiveRun $d) { Write-Host "skip $($d.Name): run.pid is a live train_lm process"; continue }
    $pts = Get-ChildItem $d.FullName -Recurse -File -Filter *.pt -ErrorAction SilentlyContinue
    foreach ($f in $pts) {
        $total += $f.Length
        $n++
        $rel = $f.FullName.Substring($runs.Length + 1)
        if ($DryRun) {
            Write-Host ("would move  {0,9:N1} MB  {1}" -f ($f.Length / 1MB), $rel)
        } else {
            try {
                [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($f.FullName,
                    [Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs,
                    [Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin)
                Write-Host ("recycled    {0,9:N1} MB  {1}" -f ($f.Length / 1MB), $rel)
            } catch {
                Write-Host "FAILED $rel : $($_.Exception.Message)"
            }
        }
    }
}
$verb = if ($DryRun) { "would move" } else { "moved" }
Write-Host ("{0} {1} .pt files, {2:N2} GB, to the Recycle Bin" -f $verb, $n, ($total / 1GB))
