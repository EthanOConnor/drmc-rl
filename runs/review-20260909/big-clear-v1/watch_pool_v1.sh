#!/bin/bash
# Rating-pool hook for the big-clear arms, on mombox where the snapshots land (stdlib client,
# low priority). Each new core-f*.pt (every 25M frames) is registered as bigclear-<arm>-fNNNNNNNNNNN
# (era style:big-combo, lineage parent armA-ppo-v1-f00100000000; finals are registered by hand
# with tags style:big-combo,keep); the intention big-clear-setups then plays it against the parent (vs_parent, set:l14-spawn, 256 games per condition).
#   tmux new -d -s bigclear-pool "bash ~/drmc-rl-data/trainer-output/big-clear-v1/watch_pool_v1.sh"
set -u
M=/home/ethan/drmc-rl-data/trainer-output/big-clear-v1
cd ~/drmc-rl-pool/src || exit 1
export DRMC_POOL_URL=http://192.168.157.190:8097
for arm in shaped std; do
  nice -n 15 ~/drmc-rl-pool/venv/bin/python -m tools.rating_pool watch-run --dir $M/$arm --pattern 'core-f*.pt' --exclude 'core-final*' \
    --run bigclear-$arm --era style:big-combo --parent armA-ppo-v1-f00100000000 \
    --recipe big-clear-$arm-v1 >> $M/watch-pool-$arm.log 2>&1 &
done
wait
