#!/bin/bash
# mombox (tmux guard-pool): register each guard-hit snapshot, then watch its free lineage's snapshots.
#   bigclear-<run>-guard-uNN (parent = the run's pool parent), bigclear-<run>-free-f... (parent = the guard-hit entrant)
M=/home/ethan/drmc-rl-data/trainer-output/big-clear-showcase-v1
cd ~/drmc-rl-pool/src || exit 1
export DRMC_POOL_URL=http://192.168.157.190:8097
PY=~/drmc-rl-pool/venv/bin/python
declare -A PARENT=([turbo]=bigclear-champ-f00100000000 [champ]=champion-retention-mixed-v2)
while true; do
  for run in turbo champ; do
    snap=$(ls $M/$run/core-guard-hit-u*.pt 2>/dev/null | head -1)
    [ -n "$snap" ] && [ ! -f $M/$run/guard-registered ] || continue
    sleep 120  # let the file settle
    u=$(basename $snap .pt | sed 's/core-guard-hit-u0*//')
    id=bigclear-$run-guard-u$u
    nice -n 15 $PY -m tools.rating_pool entrant add $id --checkpoint $snap --era style:big-combo \
      --parent ${PARENT[$run]} --run bigclear-$run --recipe big-clear-showcase-$run-guard-hit \
      --notes "last accepted update before the first retention-guard refusal" >> $M/guard-pool.log 2>&1 && touch $M/$run/guard-registered
    nice -n 15 $PY -m tools.rating_pool watch-run --dir $M/$run-free --pattern 'core-f*.pt' --exclude 'core-final*' \
      --run bigclear-$run-free --era style:big-combo --parent $id --recipe big-clear-showcase-$run-free >> $M/watch-pool-$run-free.log 2>&1 &
    echo "$(date -Is) registered $id; watching $run-free" >> $M/guard-pool.log
  done
  sleep 120
done
