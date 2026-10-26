#!/bin/bash
# PREPARED, RUN ONLY AFTER world_fastmem_pack_12k IS LAUNCHED: CPU on/off changing-facts evals (2 threads, no GPU) on
# copies of last.pt at steps 2000, 4000, 6000, 8000, 10000 and final.pt. Each: a scratch copy (+ code_snapshot), the eval
# with --pack 4 (primary: training's pass layout) and --pack 1 (secondary), 8 streams x 48 seqs of world_change_mix_v2
# (cross-seq + in-context splits), then the copy is deleted. Progress lines: world_change/fmp_eval_progress.txt.
RUN=/c/Eitan/Python/NewBrain/runs/world_fastmem_pack_12k
L=/c/Eitan/Python/NewBrain/runs/_queues/fmpack.log
SP=/c/Users/eitan/AppData/Local/Temp/claude/C--Eitan-Python-NewBrain/de301b72-951d-49a4-9895-252fb20d74d6/scratchpad
WC=/c/Eitan/Python/NewBrain/world_change
PY=/c/Eitan/Python/NewBrain/.venv/Scripts/python.exe
cd $RUN || exit 1
for S in 2000 4000 6000 8000 10000 final; do
  if [ $S = final ]; then
    until grep -qE "finished world_fastmem_pack_12k" $L; do sleep 60; done
    SRC=final; [ -f final.pt ] || SRC=last
  else
    until grep -qE "^step +$S/" $L; do sleep 60; grep -qE "finished world_fastmem_pack_12k" $L && break; done
    T0=$(date +%s)
    until [ $(stat -c %Y last.memory.pt) -ge $((T0-180)) ] && [ $(stat -c %Y last.pt) -ge $((T0-180)) ]; do sleep 5; done
    SRC=last
  fi
  s1=0; s2=1
  while [ "$s1" != "$s2" ]; do s1="$(stat -c '%s %Y' $SRC.pt $SRC.memory.pt 2>/dev/null)"; sleep 20; s2="$(stat -c '%s %Y' $SRC.pt $SRC.memory.pt 2>/dev/null)"; done
  D=$SP/fmp$S; mkdir -p $D
  cp $SRC.pt $D/ckpt_$S.pt; cp $SRC.memory.pt $D/ckpt_$S.memory.pt; cp -r code_snapshot $D/; cp launch_args.txt $D/
  for P in 4 1; do
    (cd $WC; CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -B eval_change_fastmem.py --ckpt $D/ckpt_$S.pt \
      --data X:/Datasets/world_change_mix_v2 --max-streams 8 --max-seqs 48 --pack $P --tag fmp_cpu8x48_p${P}_s$S \
      > fmp_eval_${S}_p$P.log 2>&1; echo "eval $S pack $P exit $?" >> fmp_eval_progress.txt)
  done
  rm -rf $D
done
echo "all done" >> $WC/fmp_eval_progress.txt
