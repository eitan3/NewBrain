#!/bin/bash
PY=../../../.venv/Scripts/python.exe
run() { tag=$1; shift; $PY speed_time.py --tag $tag "$@" > log_speed_$tag.txt 2>&1; tail -n 1 log_speed_$tag.txt; }
run base
run topk --variant topk_nosync
run nosync0 --variant topk_nosync,route_nosync0
run static --variant topk_nosync,route_static
run rms --variant rms_fused
run spikefn --variant spike_fn
run combo --variant topk_nosync,route_static,rms_fused,spike_fn
run base16 --batch 16 --accum 2 --variant chunk_ce
