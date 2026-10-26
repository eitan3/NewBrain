# "Whole organism" + 3-seed repeats (the recommended set, ~5 hours on the RTX 5090 laptop).
#   powershell -ExecutionPolicy Bypass -File C:\Eitan\Python\NewBrain\scripts\launch_organism.ps1
# Runs detached (survives closing windows); a viewer window opens. Stop: .\scripts\stop_run.ps1 -Queue organism
$org = "--mixer ssd --spike graded --homeo 0.01 --homeo-mode floor --redo-every 500 " +
       "--start layers=3,heads=3,mlp=768 --grow-at 0.25,0.5 --grow-where spread --grow-init copy --tokens 277.7e6"
$g2 = "--mixer ssd --start layers=3,heads=3,mlp=768 --grow-at 0.25,0.5 --grow-where spread --grow-init copy --tokens 277.7e6"
$sf = "--mixer ssd --spike graded --homeo 0.01 --homeo-mode floor --redo-every 500 --tokens 200e6"
& (Join-Path $PSScriptRoot "launch_queue.ps1") -Queue organism -Runs @(
    "organism_s0 | $org --seed 0",
    "organism_s1 | $org --seed 1",
    "organism_s2 | $org --seed 2",
    "s0_ssd_s1 | --mixer ssd --tokens 200e6 --seed 1",
    "s0_ssd_s2 | --mixer ssd --tokens 200e6 --seed 2",
    "s0g2_grow_s1 | $g2 --seed 1",
    "s0g2_grow_s2 | $g2 --seed 2",
    "s1f_graded_s1 | $sf --seed 1",
    "s1f_graded_s2 | $sf --seed 2"
)
