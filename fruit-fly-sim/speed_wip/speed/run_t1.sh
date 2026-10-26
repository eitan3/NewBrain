#!/bin/bash
# Task 1 runs, one GPU process at a time
PY=../../../.venv/Scripts/python.exe
$PY accum_run.py --batch 16 --accum 2 --tag A   > log_t1_A.txt 2>&1
$PY accum_run.py --batch 16 --accum 2 --tag A2  > log_t1_A2.txt 2>&1
$PY accum_run.py --batch 8 --accum 4 --tag B    > log_t1_B.txt 2>&1
$PY accum_run.py --batch 16 --accum 2 --fix all --tag C > log_t1_C.txt 2>&1
$PY accum_run.py --batch 8 --accum 4 --fix all --tag D  > log_t1_D.txt 2>&1
