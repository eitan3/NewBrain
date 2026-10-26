#!/bin/sh
# The VRAM study's GPU runs, one process at a time (each loads the step-7000 checkpoint and grows it to full size).
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
run() { name=$1; shift; echo "=== $name: $*"; timeout 2400 $PY -u "$@" > "log_$name.txt" 2>&1; grep -v "^{" "log_$name.txt" | tail -12; }
for v in "$@"; do
  case $v in
    mb8) run mb8 profile_step.py --batch 8 --accum 4 --ks 6,10 --time 2,6,10,16 --out step_profile_mb8.json ;;
    mb8ce) run mb8ce profile_step.py --variant chunk_ce --batch 8 --accum 4 --ks 6,10 --time 2,6,10,16 --out step_profile_mb8ce.json ;;
    ckpt) run ckpt profile_step.py --variant chunk_ce,ckpt_pass --ks 2,6,10,16 --time 2,6,10,16 --snapshot 10 --out step_profile_ckpt.json ;;
    ckpt32) run ckpt32 profile_step.py --variant chunk_ce,ckpt_pass --batch 32 --accum 1 --ks 6,10,16 --time 2,6,10,16 --out step_profile_ckpt32.json ;;
    ckptonly) run ckptonly profile_step.py --variant ckpt_pass --ks 6,10 --time 2,6,10,16 --out step_profile_ckptonly.json ;;
    b32ce) run b32ce profile_step.py --variant chunk_ce --batch 32 --accum 1 --ks 2,4 --out step_profile_b32ce.json ;;
    nohomeo) run nohomeo profile_step.py --variant nohomeo --ks 10 --time 2,6,10,16 --out step_profile_nohomeo.json ;;
    report_xfree) run report_xfree profile_report.py --extra-free 1 --report 0 --out report_profile_xfree.json ;;
    ptime) run ptime profile_time.py ;;
    expseg) echo "=== expseg"; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -c "import torch; torch.cuda.set_per_process_memory_fraction(0.9); x=torch.empty(2**28, device='cuda'); print('reserved', torch.cuda.memory_reserved()/1e9, 'segments', len(torch.cuda.memory_snapshot()))" 2>&1 | tail -5 ;;
    report) run report profile_report.py --out report_profile.json ;;
    report_ce) run report_ce profile_report.py --variant chunk_ce,ckpt_pass --out report_profile_ckpt.json ;;
    report_nofree) run report_nofree profile_report.py --free 0 --out report_profile_nofree.json ;;
  esac
done
