# Dr. Mario RL (drmc-rl)

`drmc-rl` trains and evaluates a single-player Dr. Mario placement policy.

Current default architecture:

- `python -m training.run`
- `algo: ppo_smdp`
- `env.id: DrMarioPlacementEnv-v0`
- `env.backend: cpp-pool`
- `env.state_repr: bitplane_bottle_conn_mask`
- `smdp_ppo.policy_type: candidate`
- `curriculum.mode: ln_hop_back`

The native `cpp-pool` path is the training path. It runs the Dr. Mario rules and
placement planner in-process through the `game_engine/` submodule, so no ROM or
emulator is needed for normal training. Libretro and Stable-Retro code remains
for parity, capture, and debugging only.

Clone/setup note:

```bash
git submodule update --init --recursive
```

Legal: ROMs are not included or distributed. Use your own legally obtained ROM
only for emulator parity/debug workflows.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,rl,viz]"
python -m tools.build_drmario_pool
python -m training.run --cfg training/configs/smdp_ppo.yaml --dry_run true
```

Start a small live training run:

```bash
python -m training.run --cfg training/configs/smdp_ppo.yaml \
  --ui tui --backend cpp-pool --num_envs 16
```

Run the annotated candidate-scoring experiment config:

```bash
python -m training.run --cfg training/configs/smdp_ppo_candidate.yaml \
  --ui headless --backend cpp-pool
```

If PyTorch installation is platform-specific, install the correct `torch`
packages first, then rerun `pip install -e ".[dev,rl,viz]"`.

## Current Data Flow

1. `game_engine/` builds `libdrmario_pool`, sourced from the standalone
   `drmario-native` project.
2. `envs/backends/drmario_pool.py` loads the native pool through ctypes.
3. `training/envs/drmario_pool_vec.py` exposes a vector env for placement-SMDP
   training.
4. `training/envs/dr_mario_vec.py` routes `DrMarioPlacementEnv-v0` +
   `backend=cpp-pool` directly to that vector env.
5. `training/algo/ppo_smdp.py` collects one transition per pill placement using
   masks, candidate costs, and SMDP frame durations.

## Backends

- `cpp-pool`: default training backend. In-process native pool, batched envs,
  no ROM, no Gymnasium async workers.
- `cpp-engine`: older subprocess/shared-memory backend. Keep for parity and
  compatibility checks.
- `libretro`: emulator oracle/debug path. Requires `DRMARIO_CORE_PATH` and
  `DRMARIO_ROM_PATH`.
- `stable-retro`: legacy compatibility path. Do not use as the default training
  setup.
- `mock`: hermetic smoke/dry-run dynamics.

## Repository Layout

- `training/`: runner, configs, PPO-SMDP, curriculum, diagnostics, vector envs.
- `models/policy/`: placement policy heads, candidate scorer, masked action
  distribution, candidate packing.
- `envs/backends/drmario_pool.py`: ctypes wrapper around `libdrmario_pool`.
- `training/envs/drmario_pool_vec.py`: current high-throughput placement vector
  env.
- `envs/retro/`: historical package name. The placement planner/reachability code
  here is still active; emulator wrappers are parity/debug support.
- `reach_native/`: native reachability helper used by planner paths.
- `game_engine/`: submodule mount for `drmario-native`.
- `dr-mario-disassembly/`: pinned disassembly submodule for reproducible
  references.
- `docs/`: current design, setup, placement, reward, and reference notes.
- `notes/`: durable worklog, memory, backlog, and scrutiny for future agents.

## Useful Commands

```bash
# Validate the active runner path without training.
python -m training.run --cfg training/configs/smdp_ppo.yaml --dry_run true

# Build native pool.
python -m tools.build_drmario_pool

# Benchmark the current native pool path.
python -m tools.bench_multienv --backend cpp-pool --vectorization sync \
  --num-envs 1,2,4,8,16 --repeats 3 --json-out runs/benchmarks/multienv.json

# Benchmark policy/network cost separately from simulator cost.
python -m tools.bench_policy --source cpp-pool --batch-size 16 --repeats 30 \
  --json-out runs/benchmarks/policy.json

# Emulator parity/debug, not the default training path.
python -m training.run --cfg training/configs/smdp_ppo.yaml --ui debug \
  --backend libretro --core quicknes --rom-path /path/to/DrMario.nes \
  --num_envs 1

# Curriculum report.
python tools/report_curriculum.py --confidence-table
```

## Documentation Map

- `docs/DESIGN.md`: current architecture and boundaries.
- `docs/PLACEMENT_POLICY.md`: policy heads, SMDP-PPO, candidate scoring.
- `docs/PLACEMENT_PLANNER.md`: 512-way placement planner and reachability.
- `docs/BENCHMARKS_2026-05-08.md`: current harness and policy benchmark notes.
- `docs/ENV_STANDUP_MAC_LINUX.md`: current setup path.
- `docs/RETRO_CORE_NOTES.md`: emulator parity/debug setup.
- `docs/PROJECT_DEEP_DIVE_2026-05-07.md`: dated audit of repo state and drift.

Project decisions and work history live under `notes/`. Older archived material
under `notes/archive/` is not authoritative unless a current doc points to it.
