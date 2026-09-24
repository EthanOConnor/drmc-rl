#!/bin/bash
# Arm C outcome PPO on tf3090. Run inside tmux session afterstate-full-ppo:
#   tmux new -d -s afterstate-full-ppo "bash $ROOT/launch_afterstate_full_ppo_v1.sh"
# Waits (without touching the GPU) until arm A's PPO session afterstate-ppo-18h
# no longer exists, i.e. its PPO finished or its stop rule stopped it; arm A is
# never preempted. Source, natives, config and outputs live on the
# trainer-output mount, not the root disk.
set -u
ROOT=/home/ethan/.cache/drmc-rl/trainer-output/afterstate-core-full-v1
LOG=$ROOT/ppo.log
echo "$(date -Is) waiting for arm A (tmux afterstate-ppo-18h) to end" >> "$LOG"
while tmux has-session -t afterstate-ppo-18h 2>/dev/null; do sleep 60; done
# The session can end a moment before its trainer releases the GPU.
while pgrep -f "tools.train_controller_retention --config /dev/shm/afterstate-core/" >/dev/null; do sleep 30; done
echo "$(date -Is) arm A ended; launching arm C PPO ($(cat "$ROOT/src/COMMIT"))" >> "$LOG"
cd "$ROOT/src" || exit 1
exec env PATH=/home/ethan/dev/drmario/drmc-rl/.venv/bin:$PATH PYTHONPATH=. OMP_NUM_THREADS=1 \
  DRMARIO_REACH_LIB=$ROOT/native/libdrm_reach_full.so DRMARIO_POOL_LIB=$ROOT/native/libdrmario_pool.so \
  python -m tools.program launch trainer-controller-retention \
  --set controller_retention_config=$ROOT/ppo-full-v1.json >> "$LOG" 2>&1
