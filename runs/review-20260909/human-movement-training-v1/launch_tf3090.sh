#!/usr/bin/env bash
# Launch human-movement fine-tuning on tf3090 in tmux session human-movement-train.
# Usage (from the Mac, in this branch's checkout): launch_tf3090.sh mixed_v2|afterstate [CHECKPOINT]
# Only when the GPU slot is granted. The afterstate core also needs a commit that
# merges trainer/afterstate-core; pass that commit as COMMIT=... in the environment.
set -euo pipefail
CORE=$1; CKPT=${2:-}
COMMIT=${COMMIT:-$(git rev-parse --short=8 HEAD)}
REVIEW=/home/ethan/.cache/drmc-rl/trainer-output/review-20260909
SRC=$REVIEW/human-movement-$COMMIT-source
NAME=human-movement-$CORE-v1
HERE=$(cd "$(dirname "$0")" && pwd)
ssh tf3090 "tmux has-session -t human-movement-train 2>/dev/null && { echo session exists; exit 1; } || true"
git archive --format=tar "$COMMIT" | ssh tf3090 "mkdir -p $SRC && tar -x -C $SRC"
python3 "$HERE/prepare_human_movement_training_v1.py" --core "$CORE" ${CKPT:+--checkpoint "$CKPT"} \
  --output "$REVIEW/$NAME" /dev/stdout | python3 -c "import json,sys;c=json.load(sys.stdin);c['source_commit']='$COMMIT';print(json.dumps(c,indent=1))" \
  | ssh tf3090 "cat > $REVIEW/configs/$NAME.json"
NATIVE=$REVIEW/controller-core-4717a03-source
ssh tf3090 "cd $SRC && tmux new-session -d -s human-movement-train \"env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  DRMARIO_POOL_LIB=$NATIVE/vendor/drmario_native/build/libdrmario_pool.so DRMC_FRAME_LIBRARY=$NATIVE/vendor/drmario_native/build/libdrmario_pool.so \
  DRMARIO_REACH_LIB=$NATIVE/reach_native/build/libdrm_reach_full.so nice -n 5 /home/ethan/drmario/drmc-rl/.venv/bin/python \
  -m tools.program launch trainer-controller-retention --set controller_retention_config=$REVIEW/configs/$NAME.json \
  2>&1 | tee -a $REVIEW/$NAME.log\""
echo "launched $NAME from $COMMIT; progress: $REVIEW/$NAME/training.json"
