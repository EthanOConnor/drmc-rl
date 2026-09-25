#!/bin/bash
# Big-clear fine-tunes on tf3090: a staged queue that only fills a free trainer slot.
#   tmux new -d -s bigclear-queue "bash $R/bigclear_queue.sh"
#
# Arms run one at a time, shaped (b) then standard (a), from the same parent and
# seed. An arm launches only when ALL hold:
#   * at most one other train_controller_retention process is running;
#   * MemAvailable >= 5000 MB (a fine-tune peaks near 3.5 GB; the 1.5 GB kill line stays clear);
#   * no committed launch is pending: arm C when arm A's session has ended but C
#     is not running yet, and C+S once its fork checkpoint exists (arm A and A+S
#     ended) until it has launched. A running arm is never preempted by them;
#     C+S's own launcher waits for memory, as it does today.
# Containment: systemd-run --scope MemoryMax=4500M MemorySwapMax=0, nice 10,
# oom_score_adj 1000 (the kernel's first OOM victim), TMPDIR on this mount.
# Guard, every minute: SIGINT the running arm (it keeps its last full update
# checkpoint) when MemAvailable < 1500 MB, or when arm A's mean frames/s over
# its last 3 updates falls below 50% of its 5125 solo baseline after the arm
# has overlapped it for 3 updates. A stopped arm resumes from its newest
# core-uNNNNN.pt once the launch conditions hold again.
set -u
R=/home/ethan/.cache/drmc-rl/trainer-output/big-clear-v1
W=/home/ethan/.cache/drmc-rl/trainer-output/afterstate-core-full-v1/ppo-v1-stranded
ARMS="shaped std"
BASE=5125
LOG=$R/queue.log
say() { echo "$(date -Is) $*" >> $LOG; }
avail() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
ours() { pgrep -f "train_controller_retention --config $R/" | head -1; }
others() { pgrep -f "train_controller_retention --config" | while read p; do
  grep -q "big-clear-v1" /proc/$p/cmdline 2>/dev/null || echo $p; done | wc -l; }
arm_a_fps() { grep '^{"updates' /dev/shm/afterstate-core/ppo.log 2>/dev/null | tail -3 | python3 -c 'import sys,json;v=[json.loads(l)["throughput"]["frames_per_second"] for l in sys.stdin];print(int(sum(v)/len(v)) if v else 0)'; }
done_arm() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get("status")=="Training complete" else 1)' $R/$1/training.json 2>/dev/null; }
pending_commit() {
  if ! tmux has-session -t afterstate-ppo-18h 2>/dev/null && ! pgrep -f "config .*/ppo-full-v1.json" >/dev/null \
     && tmux has-session -t afterstate-full-ppo 2>/dev/null; then echo "arm C launch"; return; fi
  if ! tmux has-session -t afterstate-ppo-18h 2>/dev/null && ! tmux has-session -t afterstate-ppo-stranded-A 2>/dev/null \
     && ls $W/fork/core-u*.pt >/dev/null 2>&1 && ! tmux has-session -t afterstate-full-ppo-stranded-C 2>/dev/null; then
    echo "C+S launch"; return; fi
}
launch() {
  local arm=$1 cfg=$R/$1/finetune-$1.json
  local last=$(ls $R/$arm/core-u*.pt 2>/dev/null | sort | tail -1)
  if [ -n "$last" ]; then
    python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); c["resume"]=sys.argv[2]; json.dump(c,open(sys.argv[1],"w"),indent=1)' $cfg "$last"
    say "resuming $arm from $last"
  fi
  tmux kill-session -t bigclear-$arm 2>/dev/null
  tmux new -d -s bigclear-$arm "systemd-run --user --scope --quiet -p MemoryMax=4500M -p MemorySwapMax=0 nice -n 10 bash -c \"echo 1000 > /proc/self/oom_score_adj; cd $R/src; exec env PATH=/home/ethan/dev/drmario/drmc-rl/.venv/bin:\\\$PATH PYTHONPATH=. OMP_NUM_THREADS=1 TMPDIR=$R/tmp DRMARIO_REACH_LIB=$R/native/libdrm_reach_full.so DRMARIO_POOL_LIB=$R/native/libdrmario_pool.so python -m tools.program launch trainer-controller-retention --set controller_retention_config=$cfg >> $R/$arm/ppo.log 2>&1\""
  say "launched $arm ($(cat $R/src/COMMIT)) avail_mb=$(avail) others=$(others)"
}
say "queue started: $ARMS"
while true; do
  pid=$(ours)
  a=$(avail)
  if [ -n "$pid" ]; then
    arm=$(tr '\0' ' ' < /proc/$pid/cmdline | grep -o 'big-clear-v1/[a-z]*' | head -1 | cut -d/ -f2)
    upd=$(grep -c '^{"updates' $R/$arm/ppo.log 2>/dev/null)
    fps=$(arm_a_fps)
    echo "$(date -Is) running=$arm pid=$pid avail_mb=$a others=$(others) armA_fps3=$fps gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d ' ')" >> $R/watch.log
    if [ "$a" -lt 1500 ]; then
      say "LOW MEMORY $a MB: stopping $arm ($pid)"; kill -INT $pid; sleep 120; continue
    fi
    if tmux has-session -t afterstate-ppo-18h 2>/dev/null && [ "${upd:-0}" -ge 3 ] && [ "$fps" -gt 0 ] && [ "$fps" -lt $((BASE/2)) ]; then
      say "ARM A THROUGHPUT $fps < 50% of $BASE: stopping $arm ($pid)"; kill -INT $pid; sleep 120; continue
    fi
  else
    next=""
    for arm in $ARMS; do done_arm $arm || { next=$arm; break; }; done
    if [ -z "$next" ]; then say "both arms complete; queue exits"; exit 0; fi
    why=$(pending_commit)
    if [ -n "$why" ]; then
      echo "$(date -Is) waiting: $why pending (next=$next)" >> $R/watch.log
    elif [ "$(others)" -gt 1 ]; then
      echo "$(date -Is) waiting: $(others) other trainers (next=$next)" >> $R/watch.log
    elif [ "$a" -lt 5000 ]; then
      echo "$(date -Is) waiting: avail_mb=$a < 5000 (next=$next)" >> $R/watch.log
    else
      launch $next; sleep 300; continue
    fi
  fi
  sleep 60
done
