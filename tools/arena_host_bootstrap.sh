#!/usr/bin/env bash
# Set up a Linux or WSL2 (Ubuntu 22.04/24.04) host as a trainer-study worker.
#
#   curl -fsSL https://raw.githubusercontent.com/EthanOConnor/drmc-rl/trainer/arena-distributed/tools/arena_host_bootstrap.sh \
#     | bash -s -- --coordinator http://192.168.157.114:8099 --token 'TOKEN' --gpu auto --start
#
# Steps (each is idempotent; rerun after a failure):
#   1. apt packages (git, build-essential, clang, curl) and uv
#   2. clone/fetch drmc-rl at --ref; init vendor/drmario_native (gh/ssh auth, or --native-bundle)
#      and check out --native-commit (default: the study's native_commit if the
#      coordinator is reachable, else the pinned submodule)
#   3. uv sync --locked --extra inference (CPU torch); with --gpu, replace torch
#      with a wheel that has kernels for this GPU (cu126 for Pascal sm_61)
#   4. build libdrm_reach_full + libdrmario_pool, run the engine/planner
#      self-test against the Mac reference and the distributed-study unit tests
#   5. write ~/.config/drmc-rl/study-worker.{token,env} and start-workers.sh;
#      --start launches the workers in the foreground
set -euo pipefail

REPO_URL=https://github.com/EthanOConnor/drmc-rl.git
DIR="$HOME/drmc-rl"
REF=trainer/arena-distributed
COORDINATOR=""
TOKEN=""
GPU=auto            # auto | cuda | cpu
NATIVE_COMMIT=""
NATIVE_BUNDLE=""
CPU_WORKERS=""      # default: (physical cores - 2 per GPU worker) / THREADS
GPU_WORKERS=3       # processes sharing the GPU; each runs its own Python frame loop
THREADS=2
PLANNER_WORKERS=""
START=0
TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu126

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift 2;;
    --ref) REF="$2"; shift 2;;
    --coordinator) COORDINATOR="$2"; shift 2;;
    --token) TOKEN="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --native-commit) NATIVE_COMMIT="$2"; shift 2;;
    --native-bundle) NATIVE_BUNDLE="$2"; shift 2;;
    --cpu-workers) CPU_WORKERS="$2"; shift 2;;
    --gpu-workers) GPU_WORKERS="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;
    --planner-workers) PLANNER_WORKERS="$2"; shift 2;;
    --torch-cuda-index) TORCH_CUDA_INDEX="$2"; shift 2;;
    --start) START=1; shift;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done

say() { printf '\n== %s\n' "$*"; }
CONFIG_DIR="$HOME/.config/drmc-rl"
mkdir -p "$CONFIG_DIR"

say "1/5 system packages and uv"
need=()
for pkg in git build-essential clang curl ca-certificates; do
  dpkg -s "$pkg" >/dev/null 2>&1 || need+=("$pkg")
done
if [ ${#need[@]} -gt 0 ]; then
  sudo apt-get update -y && sudo apt-get install -y "${need[@]}"
fi
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

say "2/5 source at $REF"
if [ ! -d "$DIR/.git" ]; then
  git clone "$REPO_URL" "$DIR"
fi
cd "$DIR"
git fetch origin "$REF"
git checkout -B "${REF//\//-}-worker" "origin/$REF" 2>/dev/null || git checkout --detach "$REF"
if [ -n "$NATIVE_BUNDLE" ]; then
  bundle="$NATIVE_BUNDLE"
  case "$bundle" in
    http://*|https://*) curl -fsSL "$bundle" -o /tmp/drmario_native.bundle; bundle=/tmp/drmario_native.bundle;;
  esac
  git config submodule.vendor/drmario_native.url "$bundle"
fi
git submodule update --init vendor/drmario_native
if [ -z "$NATIVE_COMMIT" ] && [ -n "$COORDINATOR" ] && [ -n "$TOKEN" ]; then
  NATIVE_COMMIT=$(curl -fsS -H "Authorization: Bearer $TOKEN" "$COORDINATOR/api/v1/study" 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["config"].get("native_commit",""))' || true)
fi
if [ -n "$NATIVE_COMMIT" ]; then
  git -C vendor/drmario_native fetch origin 2>/dev/null || true
  git -C vendor/drmario_native checkout --detach "$NATIVE_COMMIT"
fi
NATIVE_COMMIT=$(git -C vendor/drmario_native rev-parse --short=7 HEAD)
echo "drmc-rl $(git rev-parse --short HEAD), native $NATIVE_COMMIT"

say "3/5 Python environment"
uv sync --locked --extra inference --group dev
DEVICE=cpu
if [ "$GPU" != cpu ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  version=$(uv run --no-sync python -c 'import torch; print(torch.__version__.split("+")[0])')
  echo "GPU: $(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader)"
  uv pip install --python .venv/bin/python --reinstall-package torch \
    --index-url "$TORCH_CUDA_INDEX" --extra-index-url https://pypi.org/simple "torch==$version"
  if uv run --no-sync python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
major, minor = torch.cuda.get_device_capability(0)
arch = f"sm_{major}{minor}"
assert arch in torch.cuda.get_arch_list(), f"{arch} not in {torch.cuda.get_arch_list()}"
x = torch.randn(64, 64, device="cuda")
torch.testing.assert_close((x @ x).cpu(), x.cpu() @ x.cpu(), rtol=1e-3, atol=1e-3)
print("cuda ok", torch.__version__, torch.cuda.get_device_name(0), arch)
EOF
  then
    DEVICE=cuda
  else
    [ "$GPU" = cuda ] && { echo "requested --gpu cuda but this torch wheel cannot run here" >&2; exit 1; }
    echo "CUDA wheel unusable here; restoring the locked CPU torch"
    uv sync --locked --extra inference --group dev --reinstall-package torch
  fi
fi
echo "network device: $DEVICE"

say "4/5 native libraries and self-tests"
uv run --no-sync python -m tools.build_reach_native
uv run --no-sync python -m tools.build_drmario_pool
POOL="$DIR/vendor/drmario_native/build/libdrmario_pool.so"
REACH="$DIR/reach_native/build/libdrm_reach_full.so"
uv run --no-sync python -m tools.arena_host_selftest --native-library "$POOL" --reach-library "$REACH" \
  --native-commit "$NATIVE_COMMIT"
DRMARIO_REACH_LIB="$REACH" uv run --no-sync pytest -q tests/test_trainer_arena_distributed.py tests/test_reach_native_smoke.py

say "5/5 worker configuration"
CORES=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l)
[ "$CORES" -gt 0 ] || CORES=$(nproc)
if [ -z "$CPU_WORKERS" ]; then
  reserve=0; [ "$DEVICE" = cuda ] && reserve=$(( 2 * GPU_WORKERS ))
  CPU_WORKERS=$(( (CORES - reserve) / THREADS )); [ "$CPU_WORKERS" -ge 0 ] || CPU_WORKERS=0
fi
[ -n "$PLANNER_WORKERS" ] || PLANNER_WORKERS=3
[ -n "$TOKEN" ] && { umask 077; printf '%s\n' "$TOKEN" > "$CONFIG_DIR/study-worker.token"; }
cat > "$CONFIG_DIR/study-worker.env" <<EOF
DRMC_DIR=$DIR
DRMC_COORDINATOR=$COORDINATOR
DRMC_POOL_LIB=$POOL
DRMARIO_REACH_LIB=$REACH
DRMC_GPU_DEVICE=$DEVICE
DRMC_GPU_WORKERS=$GPU_WORKERS
DRMC_CPU_WORKERS=$CPU_WORKERS
DRMC_THREADS=$THREADS
DRMC_PLANNER_WORKERS=$PLANNER_WORKERS
EOF
cat > "$CONFIG_DIR/start-workers.sh" <<'EOF'
#!/usr/bin/env bash
# Foreground supervisor: one CUDA worker (if available) plus CPU workers. Ctrl-C stops all.
set -a; . "$HOME/.config/drmc-rl/study-worker.env"; set +a
export PATH="$HOME/.local/bin:$PATH" OMP_NUM_THREADS="$DRMC_THREADS" MKL_NUM_THREADS="$DRMC_THREADS"
cd "$DRMC_DIR"
LOGS="$HOME/.cache/drmc-rl/study-worker-logs"; mkdir -p "$LOGS"
pids=()
start() {  # start ID DEVICE THREADS
  uv run --no-sync python -m tools.trainer_arena_distributed worker --coordinator "$DRMC_COORDINATOR" \
    --worker-id "$(hostname)-$1" --device "$2" --threads "$3" --planner-workers "$DRMC_PLANNER_WORKERS" \
    --native-library "$DRMC_POOL_LIB" --reach-library "$DRMARIO_REACH_LIB" >>"$LOGS/$1.log" 2>&1 &
  pids+=($!)
}
trap 'kill "${pids[@]}" 2>/dev/null; wait' INT TERM EXIT
if [ "$DRMC_GPU_DEVICE" = cuda ]; then
  for i in $(seq 1 "${DRMC_GPU_WORKERS:-1}"); do start "cuda$i" cuda 1; done
fi
for i in $(seq 1 "${DRMC_CPU_WORKERS:-0}"); do start "cpu$i" cpu "$DRMC_THREADS"; done
echo "started ${#pids[@]} workers; logs in $LOGS"
wait
EOF
chmod +x "$CONFIG_DIR/start-workers.sh"
cat "$CONFIG_DIR/study-worker.env"
echo "start with: $CONFIG_DIR/start-workers.sh"
if [ "$START" = 1 ]; then
  [ -n "$COORDINATOR" ] && [ -s "$CONFIG_DIR/study-worker.token" ] || { echo "--start needs --coordinator and --token" >&2; exit 2; }
  exec "$CONFIG_DIR/start-workers.sh"
fi
