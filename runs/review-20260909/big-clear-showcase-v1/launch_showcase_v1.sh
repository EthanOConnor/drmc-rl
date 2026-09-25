#!/bin/bash
# One-shot launcher on tf3090 (tmux bigclear-showcase): waits, launches each candidate once, guards memory, exits.
#   candidate 1 (champ): when MemAvailable >= 7500 MB (>= 4 GB left after a ~3.5 GB trainer);
#   candidate 2 (armA):  when tmux gns-v1 has ended and MemAvailable >= 5500 MB.
# Containment: systemd scope MemoryMax (champ 5000M, armA 4500M), MemorySwapMax=0, nice 10,
# oom_score_adj 500 (champ) / 1000 (armA: the first OOM victim). Guard: below 1500 MB available
# SIGINT armA first, then champ; every checkpoint is kept, resume by setting "resume" by hand.
# Restart after candidate 1 is up: CHAMP_LAUNCHED=1. Hold candidate 2 while $R/hold-armA exists,
# at most until HOLD_UNTIL (epoch seconds; 0 = no hold).
set -u
R=/home/ethan/.cache/drmc-rl/trainer-output/big-clear-showcase-v1
LOG=$R/launch.log
say() { echo "$(date -Is) $*" >> $LOG; }
avail() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
pid_of() { pgrep -f "train_controller_retention --config $R/$1/" | head -1; }
launch() {
  local name=$1 cap=$2 adj=$3
  tmux new -d -s showcase-$name "systemd-run --user --scope --quiet -p MemoryMax=$cap -p MemorySwapMax=0 nice -n 10 bash -c \"echo $adj > /proc/self/oom_score_adj; cd $R/src; exec env PATH=/home/ethan/dev/drmario/drmc-rl/.venv/bin:\\\$PATH PYTHONPATH=. OMP_NUM_THREADS=1 TMPDIR=$R/$name/tmp DRMARIO_REACH_LIB=$R/native/libdrm_reach_full.so DRMARIO_POOL_LIB=$R/native/libdrmario_pool.so python -m tools.program launch trainer-controller-retention --set controller_retention_config=$R/$name/finetune-$name.json >> $R/$name/ppo.log 2>&1\""
  say "launched $name ($(cat $R/src/COMMIT)) avail_mb=$(avail)"
}
say "launcher started"
champ=${CHAMP_LAUNCHED:-0}; arma=0; HOLD_UNTIL=${HOLD_UNTIL:-0}
held() { [ -f $R/hold-armA ] && [ "$(date +%s)" -lt "$HOLD_UNTIL" ]; }
while true; do
  a=$(avail)
  if [ $champ = 0 ] && [ "$a" -ge 7500 ]; then launch champ 5000M 500; champ=1; sleep 240; continue; fi
  if [ $champ = 1 ] && [ $arma = 0 ] && ! held && ! tmux has-session -t gns-v1 2>/dev/null && [ "$a" -ge 5500 ]; then
    launch armA 4500M 1000; arma=1; sleep 240; continue; fi
  if [ "$a" -lt 1500 ]; then
    p=$(pid_of armA); [ -z "$p" ] && p=$(pid_of champ)
    [ -n "$p" ] && { say "LOW MEMORY $a MB: SIGINT $p"; kill -INT $p; sleep 120; continue; }
  fi
  echo "$(date -Is) avail_mb=$a champ=$(pid_of champ) armA=$(pid_of armA) gns=$(tmux has-session -t gns-v1 2>/dev/null && echo on || echo off)" >> $R/watch.log
  if [ $champ = 1 ] && [ $arma = 1 ] && [ -z "$(pid_of champ)" ] && [ -z "$(pid_of armA)" ]; then say "both trainers ended; launcher exits"; exit 0; fi
  sleep 60
done
