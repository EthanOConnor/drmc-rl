#!/bin/bash
# mombox: register each 25M snapshot of the showcase candidates (era style:big-combo).
#   tmux new -d -s showcase-pool "bash ~/drmc-rl-data/trainer-output/big-clear-showcase-v1/watch_pool_showcase_v1.sh"
M=/home/ethan/drmc-rl-data/trainer-output/big-clear-showcase-v1
cd ~/drmc-rl-pool/src || exit 1
export DRMC_POOL_URL=http://192.168.157.190:8097
nice -n 15 ~/drmc-rl-pool/venv/bin/python -m tools.rating_pool watch-run --dir $M/champ --pattern 'core-f*.pt' --exclude 'core-final*' \
  --run bigclear-champ --era style:big-combo --parent champion-retention-mixed-v2 --recipe big-clear-showcase-champ-v1 >> $M/watch-pool-champ.log 2>&1 &
nice -n 15 ~/drmc-rl-pool/venv/bin/python -m tools.rating_pool watch-run --dir $M/armA --pattern 'core-f*.pt' --exclude 'core-final*' \
  --run bigclear-armA --era style:big-combo --parent armA-ppo-v1-f00100000000 --recipe big-clear-showcase-armA-v1 >> $M/watch-pool-armA.log 2>&1 &
wait
