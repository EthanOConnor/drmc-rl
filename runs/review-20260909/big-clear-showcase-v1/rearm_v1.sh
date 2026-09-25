#!/bin/bash
# Restart a running showcase trainer on src-v3 with stop_on_retention_rejection, at an update boundary.
#   bash rearm_v1.sh RUN CONFIG_NAME MEMORYMAX OOM_ADJ     e.g.  rearm_v1.sh turbo finetune-turbo.json 4500M 1000
set -u
R=/home/ethan/.cache/drmc-rl/trainer-output/big-clear-showcase-v1
run=$1 cfg=$R/$1/$2 cap=$3 adj=$4
st() { python3 -c "import json;d=json.load(open('$R/$run/training.json'));print(d['updates'],d.get('phase'))"; }
set -- $(st); u0=$1
until set -- $(st) && [ "$1" -gt "$u0" ] && [ "$2" = collecting ]; do sleep 10; done
pkill -INT -f "train_controller_retention --config $R/$run/"
while pgrep -f "train_controller_retention --config $R/$run/" >/dev/null; do sleep 5; done
last=$(ls $R/$run/core-u*.pt | sort | tail -1)
python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); c["resume"]=sys.argv[2]; c["stop_on_retention_rejection"]=True; json.dump(c,open(sys.argv[1],"w"),indent=1)' $cfg "$last"
tmux kill-session -t showcase-$run 2>/dev/null
tmux new -d -s showcase-$run "systemd-run --user --scope --quiet -p MemoryMax=$cap -p MemorySwapMax=0 nice -n 10 bash -c \"echo $adj > /proc/self/oom_score_adj; cd $R/src-v3; exec env PATH=/home/ethan/dev/drmario/drmc-rl/.venv/bin:\\\$PATH PYTHONPATH=. OMP_NUM_THREADS=1 TMPDIR=$R/$run/tmp DRMARIO_REACH_LIB=$R/native/libdrm_reach_full.so DRMARIO_POOL_LIB=$R/native/libdrmario_pool.so python -m tools.program launch trainer-controller-retention --set controller_retention_config=$cfg >> $R/$run/ppo.log 2>&1\""
echo "$(date -Is) re-armed $run from $last on $(cat $R/src-v3/COMMIT)" >> $R/guard-watch.log
