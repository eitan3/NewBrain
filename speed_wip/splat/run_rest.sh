#!/bin/sh
cd /c/Eitan/Python/NewBrain/speed_wip/splat
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4
PY=/c/Eitan/Python/NewBrain/.venv/Scripts/python.exe
until [ -s log_main_rot.txt ] && [ -s log_main_warp.txt ]; do sleep 10; done
for e in width cap; do
  $PY -B -u run_toy.py $e rot > log_${e}_rot.txt 2>&1 &
  $PY -B -u run_toy.py $e warp > log_${e}_warp.txt 2>&1
  wait
done
$PY -B -u run_toy.py interf rot > log_interf_rot.txt 2>&1 &
$PY -B -u run_toy.py fade rot > log_fade_rot.txt 2>&1
wait
echo ALLDONE > rest_done.txt
