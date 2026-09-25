#!/bin/bash
# mombox: register turbo's 12.5M-frame snapshots (era style:big-combo, parent = candidate 1's snapshot before the fork).
#   tmux new -d -s turbo-pool "bash ~/drmc-rl-data/trainer-output/big-clear-showcase-v1/watch_pool_turbo_v1.sh PARENT"
M=/home/ethan/drmc-rl-data/trainer-output/big-clear-showcase-v1
cd ~/drmc-rl-pool/src || exit 1
export DRMC_POOL_URL=http://192.168.157.190:8097
exec nice -n 15 ~/drmc-rl-pool/venv/bin/python -m tools.rating_pool watch-run --dir $M/turbo --pattern 'core-f*.pt' --exclude 'core-final*' \
  --run bigclear-turbo --era style:big-combo --parent "$1" --recipe big-clear-showcase-turbo-v1 >> $M/watch-pool-turbo.log 2>&1
