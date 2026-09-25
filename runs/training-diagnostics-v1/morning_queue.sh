#!/bin/bash
# GPU diagnostics queued for after the 09:30 PDT reservation on tf3090.
# Launch: tmux new-session -d -s diag-v1 "nice -n 19 systemd-run --user --scope -p MemoryMax=2800M ~/diag-scratch/morning_queue.sh > ~/diag-scratch/morning.log 2>&1"
# Scripts: copy tools/training_diagnostics.py and tools/gradient_noise_scale.py to ~/diag-scratch first.
set -uo pipefail
RUN=$HOME/.cache/drmc-rl/trainer-output/afterstate-core-v1/ppo-v1
OUT=$HOME/.cache/drmc-rl/trainer-output/diag-v1
PY=$HOME/dev/drmario/drmc-rl/.venv/bin/python
S=$HOME/diag-scratch/training_diagnostics.py
cd $HOME/.cache/drmc-rl/afterstate-core/src-0f36d78d
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
av() { echo "== $1 $(date -Is) avail=$(awk "/MemAvailable/{print \$2}" /proc/meminfo)kB"; }
# 1. Frozen-trunk value fitting (head only at 4 LRs, fresh MLP probe); ~20 min.
av value-fit
[ -f $OUT/value-fit-u58.json ] || $PY $S value-fit --run $RUN --checkpoint $RUN/core-u00058.pt \
  --train-updates 50 51 52 53 54 55 56 57 --test-updates 58 59 --out $OUT/value-fit-u58.json
# 2. Counterfactual pilot (timing + prefix-replay check), then the main branch rollouts.
av cf-pilot
[ -f $OUT/cf-pilot.jsonl ] || $PY $S counterfactual --run $RUN --checkpoint $RUN/core-u00058.pt --pace normal \
  --games 2 --games-per-batch 2 --points 3 --k 4 --m 4 --out $OUT/cf-pilot.jsonl
for spec in "sloth 16 3" "normal 10 5" "frame_perfect 8 5"; do
  set -- $spec
  av cf-$1
  [ -f $OUT/cf-$1.done ] || { $PY $S counterfactual --run $RUN --checkpoint $RUN/core-u00058.pt --pace $1 \
    --games $2 --games-per-batch 2 --points $3 --k 4 --m 8 --out $OUT/cf-$1.jsonl && touch $OUT/cf-$1.done; }
done
# 3. Epoch replay from the full u57 checkpoint (optimizer state) on update 58.
av epochs
[ -f $OUT/epochs-u57.json ] || { $PY $S epochs --run $RUN --checkpoint $HOME/gns-scratch/core-u00057.pt \
  --out $OUT/epochs-u57.json.partial --variants 128:2:1 128:4:1 512:4:1 1024:4:1 1024:4:4 \
  && mv $OUT/epochs-u57.json.partial $OUT/epochs-u57.json; }
# 4. Full-network value-only fitting at the trainer LR and 10x.
av value-full
[ -f $OUT/value-full-u58.json ] || $PY $S value-full --run $RUN --checkpoint $RUN/core-u00058.pt \
  --train-updates 55 56 57 --test-updates 59 --full-lrs 3e-6 3e-5 --steps 600 --out $OUT/value-full-u58.json
av done
