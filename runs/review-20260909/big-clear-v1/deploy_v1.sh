#!/bin/bash
# Stage the big-clear fine-tunes (run from the Mac, in this worktree, at the committed HEAD).
# Everything is written on mombox under drmc-rl-data/trainer-output/big-clear-v1, which tf3090
# mounts at ~/.cache/drmc-rl/trainer-output/big-clear-v1 (nothing on tf3090's /dev/shm or root disk).
#   bash runs/review-20260909/big-clear-v1/deploy_v1.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
DATA=/Users/ethan/dev/drmario/drmc-rl-bigclear-data
M=drmc-rl-data/trainer-output/big-clear-v1                 # on mombox (relative to ~)
R=/home/ethan/.cache/drmc-rl/trainer-output/big-clear-v1    # the same directory on tf3090
COMMIT=$(git -C "$REPO" rev-parse --short HEAD)
[ -z "$(git -C "$REPO" status --porcelain -- drmc_rl tools runs/review-20260909/big-clear-v1)" ] || { echo "commit first"; exit 1; }

ssh mombox "mkdir -p $M/bank $M/native $M/src $M/tmp $M/shaped/tmp $M/std/tmp"
rsync -a "$DATA/bank/big-clear-train-v1.npz" "$DATA/bank/big-clear-train-v1.json" mombox:$M/bank/
git -C "$REPO" archive "$COMMIT" | ssh mombox "rm -rf $M/src && mkdir -p $M/src && tar -x -C $M/src && echo $COMMIT > $M/src/COMMIT"
ssh mombox "cp drmc-rl-data/trainer-output/afterstate-core-full-v1/native/libdrmario_pool.so drmc-rl-data/trainer-output/afterstate-core-full-v1/native/libdrm_reach_full.so $M/native/"
rsync -a "$HERE/bigclear_queue.sh" "$HERE/watch_pool_v1.sh" mombox:$M/

python3 "$HERE/../prepare_big_clear_finetune_v1.py"
for arm in shaped std; do
  rsync -a "$HERE/finetune-$arm.json" mombox:$M/$arm/finetune-$arm.json
done
ssh tf3090 "sha256sum $R/native/*.so /dev/shm/afterstate-core/native/*.so; cat $R/src/COMMIT"
echo "staged $COMMIT; start with: ssh tf3090 tmux new -d -s bigclear-queue 'bash $R/bigclear_queue.sh'"
