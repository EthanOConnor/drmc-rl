#!/bin/bash
# "Turbo": a short, deliberately aggressive fork of candidate 1 (bigclear-champ) for the showcase.
# Forks candidate 1's newest full update checkpoint (model, optimizer, sampler, adaptive lr, journal),
# doubles the bonus, shifts the start mix to 50% showiness / 15% stranded edge, snapshots every 12.5M frames.
# Runs in candidate 2's slot (paused at armA/core-u00046.pt): MemoryMax 4500M, nice 10, oom_score_adj 1000;
# SIGINT when MemAvailable < 1500 MB.
#   tmux new -d -s showcase-turbo-launch "bash $R/launch_turbo_v1.sh"
set -u
R=/home/ethan/.cache/drmc-rl/trainer-output/big-clear-showcase-v1
T=$R/turbo
LOG=$T/launch.log
mkdir -p $T/fork $T/tmp
say() { echo "$(date -Is) $*" >> $LOG; }
U=$(ls $R/champ/core-u*.pt | sort | tail -1)
cp $U $T/fork/ && cp $R/champ/training-games.jsonl $T/fork/training-games.jsonl
F=$T/fork/$(basename $U)
python3 - $R/champ/finetune-champ.json $T/finetune-turbo.json $F $(cat $R/src/COMMIT) <<'PY'
import hashlib, json, sys
src, dst, fork, commit = sys.argv[1:5]
c = json.load(open(src))
c.pop('resume', None)
c.update(output=dst.rsplit('/', 1)[0], source_commit=commit, checkpoint_every_frames=12_500_000,
         fork=dict(checkpoint=fork, sha256=hashlib.sha256(open(fork, 'rb').read()).hexdigest(),
                   journal=dst.rsplit('/', 1)[0] + '/fork/training-games.jsonl'),
         showiness_bonus=dict(steps=[[27.0, 0.10], [30.0, 0.30], [42.0, 0.60]], event_cap=0.60, game_cap=1.20,
                              horizontal=dict(per_clear=0.008, combo_extra=0.016, game_cap=0.20)))
mixes = c['start_mixes']
mixes[0]['fraction'], mixes[1]['fraction'] = 0.50, 0.15
json.dump(c, open(dst, 'w'), indent=1)
PY
say "forked $(basename $U); config written"
tmux new -d -s showcase-turbo "systemd-run --user --scope --quiet -p MemoryMax=4500M -p MemorySwapMax=0 nice -n 10 bash -c \"echo 1000 > /proc/self/oom_score_adj; cd $R/src; exec env PATH=/home/ethan/dev/drmario/drmc-rl/.venv/bin:\\\$PATH PYTHONPATH=. OMP_NUM_THREADS=1 TMPDIR=$T/tmp DRMARIO_REACH_LIB=$R/native/libdrm_reach_full.so DRMARIO_POOL_LIB=$R/native/libdrmario_pool.so python -m tools.program launch trainer-controller-retention --set controller_retention_config=$T/finetune-turbo.json >> $T/ppo.log 2>&1\""
say "launched turbo ($(cat $R/src/COMMIT)) avail_mb=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)"
sleep 120
while pgrep -f "train_controller_retention --config $T/" >/dev/null; do
  a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$a" -lt 1500 ]; then say "LOW MEMORY $a MB: SIGINT turbo"; pkill -INT -f "train_controller_retention --config $T/"; fi
  sleep 60
done
say "turbo ended"
