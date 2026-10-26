#!/bin/bash
PY=../../../.venv/Scripts/python.exe
$PY save_init.py > log_t1_init.txt 2>&1
$PY accum_run.py --batch 16 --accum 2 --fix all --dropless 1 --tag E > log_t1_E.txt 2>&1
$PY accum_run.py --batch 8 --accum 4 --fix all --dropless 1 --tag F  > log_t1_F.txt 2>&1
$PY accum_run.py --batch 16 --accum 2 --fix all --tag C2 > log_t1_C2.txt 2>&1
$PY accum_run.py --batch 16 --accum 2 --fix all --dropless 1 --tag E2 > log_t1_E2.txt 2>&1
