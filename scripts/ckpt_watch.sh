#!/bin/bash
# Copy runs/<run>/last.pt to ckpt_<step>.pt when it holds one of the wanted steps.
run=${RUNS:-/c/Eitan/Python/NewBrain/runs}/$1; shift; want=" $* "; last=""
while true; do
  if [ -f $run/final.pt ]; then echo "final present"; exit 0; fi
  if [ -f $run/last.pt ]; then
    m=$(stat -c %Y $run/last.pt)
    if [ "$m" != "$last" ]; then
      sleep 20; last=$(stat -c %Y $run/last.pt)
      s=$(grep -E "^step +[0-9]+/" $run/train.log | tail -1 | awk '{print $2}' | cut -d/ -f1)
      k=$(( s / 1000 * 1000 ))
      if [[ "$want" == *" $k "* ]] && [ ! -f $run/ckpt_$k.pt ]; then cp $run/last.pt $run/ckpt_$k.pt && echo "copied ckpt_$k"; fi
    fi
  fi
  sleep 60
done
