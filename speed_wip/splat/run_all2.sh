#!/bin/sh
cd /c/Eitan/Python/NewBrain/speed_wip/splat
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4
PY=/c/Eitan/Python/NewBrain/.venv/Scripts/python.exe
$PY -B -u run_toy.py main rot > log_main_rot.txt 2>&1 &
$PY -B -u run_toy.py main warp > log_main_warp.txt 2>&1
wait
$PY -B -u run_toy.py width rot > log_width_rot.txt 2>&1 &
$PY -B -u run_toy.py width warp > log_width_warp.txt 2>&1
wait
$PY -B -u run_toy.py cap rot > log_cap_rot.txt 2>&1 &
$PY -B -u run_toy.py cap warp > log_cap_warp.txt 2>&1
wait
$PY -B -u run_toy.py interf rot > log_interf_rot.txt 2>&1 &
$PY -B -u run_toy.py fade rot > log_fade_rot.txt 2>&1
wait
echo ALLDONE > rest_done.txt
