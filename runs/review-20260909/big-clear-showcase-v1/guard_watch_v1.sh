#!/bin/bash
# tf3090 (tmux guard-watch): for each armed run, when its trainer stops at the first retention-guard
# refusal (RUN/retention-guard-hit.json), snapshot the last accepted state and launch RUN-free, a fork
# with the acceptance limit lifted, in the same memory slot (MemoryMax 4500M, nice 10, oom 1000).
set -u
R=/home/ethan/.cache/drmc-rl/trainer-output/big-clear-showcase-v1
RUNS="turbo champ"
say() { echo "$(date -Is) $*" >> $R/guard-watch.log; }
say "armed: $RUNS"
while true; do
  for run in $RUNS; do
    [ -f $R/$run/retention-guard-hit.json ] && [ ! -f $R/$run/guard-handled ] || continue
    pgrep -f "train_controller_retention --config $R/$run/" >/dev/null && continue   # let it exit first
    touch $R/$run/guard-handled
    snap=$(cd $R/src-v3 && PYTHONPATH=. /home/ethan/dev/drmario/drmc-rl/.venv/bin/python $R/guard_fork.py $R/$run $R/$run-free 2>>$R/guard-watch.log)
    [ -n "$snap" ] || { say "guard_fork failed for $run"; continue; }
    say "$run hit the retention guard: $(tr -d '\n' < $R/$run/retention-guard-hit.json | cut -c1-400); snapshot $snap"
    tmux new -d -s showcase-$run-free "systemd-run --user --scope --quiet -p MemoryMax=4500M -p MemorySwapMax=0 nice -n 10 bash -c \"echo 1000 > /proc/self/oom_score_adj; cd $R/src-v3; exec env PATH=/home/ethan/dev/drmario/drmc-rl/.venv/bin:\\\$PATH PYTHONPATH=. OMP_NUM_THREADS=1 TMPDIR=$R/$run-free/tmp DRMARIO_REACH_LIB=$R/native/libdrm_reach_full.so DRMARIO_POOL_LIB=$R/native/libdrmario_pool.so python -m tools.program launch trainer-controller-retention --set controller_retention_config=$R/$run-free/finetune-free.json >> $R/$run-free/ppo.log 2>&1\""
    say "launched $run-free ($(cat $R/src-v3/COMMIT))"
  done
  a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$a" -lt 1500 ]; then  # one victim per minute: free runs, then turbo, then candidate 1
    for d in turbo-free champ-free turbo champ; do
      if pkill -INT -f "train_controller_retention --config $R/$d/"; then say "LOW MEMORY $a MB: SIGINT $d"; break; fi
    done
  fi
  sleep 60
done
