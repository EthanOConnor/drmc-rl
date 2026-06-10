# drmc-rl Deep Dive - 2026-05-07

This is an evidence-backed orientation note for agents returning to `drmc-rl`
after a long pause. It describes the current checkout, what actually runs, where
docs drift from code, and the safest way to dive in.

Update later on 2026-05-07: root docs, setup docs, backlog, and agent guidance
were refreshed after this audit so `cpp-pool` placement-SMDP training is now the
forward-facing default. Treat the drift findings below as historical evidence
unless a current doc still shows the same issue.

Update on 2026-05-08: the local `.venv` was reinstalled from this checkout with
`.[dev,rl,viz]`, pytest is available, `pip check` is clean, and the dependency
stack is on NumPy `2.4.4` plus `opencv-python` `4.13.0.92`. The default
training config now uses candidate scoring; see
`docs/BENCHMARKS_2026-05-08.md`.

Update later on 2026-05-08: the default observation now uses
`bitplane_bottle_conn_mask`, which adds locked-capsule connection-edge planes
between the RGB/virus bottle facts and the planner feasibility planes.

## Bottom Line

`drmc-rl` is no longer the early Stable-Retro-first skeleton described by some
older docs. The active training path is now:

- `python -m training.run`
- `algo: ppo_smdp`
- `env.id: DrMarioPlacementEnv-v0`
- `env.backend: cpp-pool`
- `env.state_repr: bitplane_bottle_conn_mask`
- `curriculum.mode: ln_hop_back`

The fastest simulator path is the in-process native pool:

- Python wrapper: `envs/backends/drmario_pool.py`
- Vector env: `training/envs/drmario_pool_vec.py`
- Native source/build: `game_engine/`, now a `drmario-native` submodule
- Default library path: `game_engine/build/libdrmario_pool.dylib` on macOS

The codebase is usable, but the checkout needs care before broad work: there
are staged submodule-conversion changes, an unstaged submodule SHA drift, and a
stale editable install path in `.venv`. The documentation-drift section below
records issues found during this audit before the 2026-05-07 docs refresh.

## Current Checkout Reality

Repo root:

- `/Users/ethan/dev/drmario/drmc-rl`

Umbrella workspace:

- `/Users/ethan/dev/drmario`
- The umbrella root is not itself a git repo.
- Its `README.md` maps sibling projects and says cross-project tools should
  prefer shared `disassembly/` checkouts while project repos keep pinned
  submodules for reproducibility.

Git state observed on 2026-05-07:

- `main...origin/main [ahead 1]`
- `HEAD`: `44956e8 Expose native checkpoint reset in pool wrapper`
- `origin/main`: `5b04d46 chore(repo): remove external trainer stack and RE workspace`
- `game_engine/` is being converted from tracked files to a git submodule.
- Staged changes include `.gitmodules`, `README.md`, the submodule add/delete
  replacement for `game_engine/`, and notes updates.
- Unstaged changes include `AGENTS.md`, `docs/REFERENCES.md`,
  `training/configs/smdp_ppo_candidate.yaml`, and the `game_engine` submodule
  SHA.
- `CLAUDE.md` is untracked in this repo.

Submodule and sibling-engine state:

- Parent index records `game_engine` at `7ac0b875...`.
- The checked-out `game_engine/` submodule is at `19f297e` and is
  `main...origin/main [ahead 2]`.
- The sibling workspace repo `../drmario-native` is a separate checkout at
  `3401732`, dirty, and not identical to `game_engine/`.
- Before committing engine-related work, reconcile which native-engine checkout
  is authoritative, then update the parent submodule SHA intentionally.

Shared references:

- `dr-mario-disassembly/` submodule is clean at
  `ecb9899554d6d4c0c866be5dd6c0f40b7b117806`.
- `../disassembly/dr-mario-disassembly/` is at the same commit.
- `../disassembly/drmario/` is a separate ca65 disassembly checkout.
- For cross-project reference work, use the umbrella shared checkouts. For
  reproducible repo-local tests, use the pinned submodule.

## Architecture as It Actually Runs

### Runner

Canonical training entrypoint:

```bash
python -m training.run --cfg training/configs/smdp_ppo.yaml
```

Important behavior:

- `training/run.py` appends a unique run id under `cfg.logdir` unless
  `--logdir` is explicit.
- `--ui debug` defaults to sync vectorization and raw RAM/info payloads.
- `--ui headless` / `--ui tui` default to async for Gym vector envs.
- `cpp-pool` bypasses Gymnasium vector wrappers entirely and returns
  `DrMarioPoolVecEnv`, then the curriculum wrapper sits on top.

### Backends

Active backend layers:

- `cpp-pool`: fastest training path, in-process C ABI, no ROM required.
- `cpp-engine`: subprocess + shared-memory engine backend, useful for parity and
  older multi-env paths.
- `libretro`: emulator parity/debug path, requires a core and ROM.
- `stable-retro`: compatibility path, optional dependency.
- `mock`: special-cased in `DrMarioRetroEnv`; not a backend registry entry.

Forward-facing docs now point at `cpp-pool`; `cpp-engine` remains documented
only as a compatibility/parity lane.

### Placement SMDP

The macro env is spawn-latched:

- One decision per pill spawn.
- Action space is `4 x 16 x 8 = 512` macro placements.
- Feasibility and costs come from the planner/native engine.
- `placements/tau` is the SMDP duration in frames.
- `placements/feasible_mask` and `placements/cost_to_lock` are hot-path policy
  inputs for candidate scoring.

Main files:

- `envs/retro/placement_env.py`
- `envs/retro/placement_planner.py`
- `envs/retro/fast_reach.py`
- `envs/retro/reach_native.py`
- `reach_native/drm_reach_full.c`

### Policy and Training

Two policy families exist:

- Heatmap policy: dense fixed `4 x 16 x 8` logits via
  `models/policy/placement_heads.py`.
- Candidate policy: packed feasible actions plus explicit cost-to-lock via
  `models/policy/candidate_policy.py` and
  `models/policy/candidate_packing.py`.

After the 2026-05-08 follow-up, `training/configs/smdp_ppo.yaml` defaults to
candidate-scoring SMDP-PPO with `cpp-pool`.
`training/configs/smdp_ppo_heatmap.yaml` is the controlled heatmap baseline,
and `training/configs/smdp_ppo_candidate.yaml` remains the verbose annotated
candidate experiment config.

Current candidate config facts:

- `policy_type: candidate`
- `backend: cpp-pool`
- `state_repr: bitplane_bottle_conn_mask`
- `candidate_board_channels: 8`
- `candidate_board_encoder: cnn` in the working tree
- `pill_embed_type: ordered_pair`
- `aux_spec: v1`
- `curriculum.mode: ln_hop_back`
- `curriculum.start_level: -15`
- `curriculum.max_level: 20`

### Curriculum

The current curriculum is more advanced than the older docs suggest:

- Synthetic match levels: `-15..-4`, mapping to 1..12 match targets.
- Synthetic virus levels: `-3..0`, mapping to 1..4 viruses.
- `ln_hop_back` alternates new-level probes with tightened hop-backs.
- Confidence gates use EMA pseudo-counts plus a minimum decision floor.
- PPO rollout collection stops on curriculum advancement to keep updates
  stage-pure.
- Optional time/spawn budgets are soft constraints after mastery.

Main files:

- `training/envs/curriculum.py`
- `training/envs/dr_mario_vec.py`
- `training/envs/drmario_pool_vec.py`
- `envs/retro/drmario_env.py`

### Native Engine

The native engine has evolved into its own project:

- Standalone repo: `EthanOConnor/drmario-native`
- Mounted here as `game_engine/`
- Build engine binary: `make -C game_engine`
- Build pool library: `make -C game_engine libdrmario_pool`
- Python helper: `python -m tools.build_drmario_pool`

Current local `game_engine/` has build artifacts under `game_engine/build/`.
Artifacts are ignored in the native repo and should not be treated as source
truth.

## Verification Run in This Audit

Commands that succeeded:

```bash
.venv/bin/python --version
# Python 3.12.13

.venv/bin/python -m training.run --dry_run true

.venv/bin/python -m tools.build_drmario_pool --help

.venv/bin/python -c "from envs.backends.drmario_pool import is_library_present; print(is_library_present())"
# True
```

`training/configs/smdp_ppo.yaml` smoke:

- `make_vec_env(...)` returned `CurriculumVecEnv`.
- Reset shape with 16 envs: `(16, 8, 16, 8)`.

`cpp-pool` step smoke with 2 envs:

- Reset shape: `(2, 8, 16, 8)`.
- Feasible options: `[30, 30]`.
- Chosen first feasible actions: `[120, 120]`.
- Step rewards: `[0.0, 0.0]`.
- Done flags: both false.
- First `placements/tau`: `78`.

Candidate config smoke:

- Loaded as `ppo_smdp cpp-pool bitplane_bottle_conn_mask candidate ln_hop_back`.
- Reset shape with 1 env: `(1, 12, 16, 8)`.
- Curriculum env level: `-15`.
- Feasible options: `15`.

Candidate network smoke:

- `CandidatePlacementPolicyNet` forward pass produced logits `(2, 16)` and
  value `(2, 1)`.
- Valid candidate logits were finite; padding logits were masked below `-1e8`.

Historical command failure from the first audit:

```bash
.venv/bin/python -m pytest -q
# No module named pytest
```

This was fixed on 2026-05-08 for pytest, pyarrow, and matplotlib by
reinstalling this checkout with `.[dev,rl,viz]`. Stable-Retro remains optional
parity/debug tooling rather than part of the normal training path.

## Local Environment Drift

The editable install metadata was stale during the first audit:

- `pip show drmc-rl` reports editable project location
  `/Users/ethan/dev/drmc-rl`.
- That path does not exist.
- Imports work only when commands are run from `/Users/ethan/dev/drmario/drmc-rl`
  because the current working directory is first on `sys.path`.
- Running this venv from `/tmp` fails to import `training.run`.

The local `.venv` has since been repaired. If this regresses, reinstall from
this checkout:

```bash
python -m pip install -e ".[dev,rl,viz]"
```

There is no `retro` optional extra in `pyproject.toml` today. Earlier docs
referred to `.[retro,dev]`; the forward-facing setup docs now use the current
extras.

## Documentation Drift Found And Corrected

These are the main divergences found during the audit. Later on 2026-05-07,
the root README, root agent guide, core docs, setup docs, and backlog were
rewritten around the `cpp-pool` default. Keep this section as evidence of what
was stale before that cleanup, not as a current task list.

### README.md

- Previously steered fast training toward `cpp-engine`; refreshed to make
  `cpp-pool` the normal path.
- Previously recommended a removed optional extra; refreshed to use the current
  `dev`, `rl`, and `viz` extras.
- Previously described several substantial areas as early scaffolding; refreshed
  around the implemented placement training, native pool, candidate policy, and
  curriculum.
- Previously mentioned `mock` like a registered backend; refreshed so mock use
  does not confuse the active backend list.

### docs/DESIGN.md

- Previously reflected the October 2025 emulator-first architecture more than
  the current implementation.
- Refreshed around `cpp-pool`, candidate scoring, the native submodule split,
  modern curriculum gates, compressed artifacts, and the in-repo trainer.
- Older high-FPS plans are now listed as non-goals or history, not as the
  active simulator strategy.

### docs/PLACEMENT_POLICY.md

- Kept the useful placement-SMDP and candidate-scoring material.
- Refreshed examples to the `cpp-pool` path.
- Reconciled curriculum text around current synthetic levels and gate behavior.
- Clarified that parsed but unused candidate knobs should not be treated as live
  distribution behavior.

### docs/ENV_STANDUP_MAC_LINUX.md

- Corrected the demo CLI shape.
- Reframed emulator setup as parity/debug-only and made `cpp-pool` standup the
  first path.

### docs/TASKS_NEXT.md

- Removed early setup tasks from the forward queue.
- Reframed next work around `cpp-pool`, candidate policy, current curriculum,
  and the submodule split.

### docs/CONTRIBUTING.md

- Removed references to deleted simulator-layout experiments.
- Updated setup guidance around current extras and the default non-ROM training
  path.

### notes/BACKLOG.md

- Refocused the top section so current next work is not buried under completed
  multi-env/native-pool history.
- Kept historical entries where they explain why current architecture looks the
  way it does.

### notes/SCRUTINY.md

- Contains important risk records.
- Some old critical items are mitigated, but the file still has stale status
  like "64 tests, all passing" from an older environment. In this checkout,
  pytest is not installed.

### game_engine/AGENTS.md

- This now belongs to the native-engine submodule.
- Its "Remaining / To-Do" section is stale: Python interface and demo parity
  tests exist in the integrated project history.
- Update this in `drmario-native`, then bump the submodule here.

### Eval and Model Skeletons

Still genuinely skeletal or placeholder-heavy:

- `models/evaluator/train_qr.py`: explicitly says dataset loader is a skeleton.
- `models/pixel2state/model.py`: small UNet-like skeleton only.
- `envs/pettingzoo/parallel_env.py`: mock two-player ParallelEnv skeleton.
- `eval/harness/seed_sweep.py`: sample-action evaluator placeholder requiring
  `seed_sweep_env_ctor`.

## Agent Entry Path

For most future work:

1. Read `AGENTS.md`, this document, then `notes/MEMORY.md` from
   `2025-12-20` onward.
2. Check `git status --short --branch` before editing. This checkout already
   has staged and unstaged work.
3. If touching native engine behavior, resolve the `game_engine/` submodule
   versus `../drmario-native` split first.
4. If touching training, start from:
   - `training/run.py`
   - `training/envs/dr_mario_vec.py`
   - `training/envs/drmario_pool_vec.py`
   - `training/algo/ppo_smdp.py`
   - `training/configs/smdp_ppo.yaml`
   - `training/configs/smdp_ppo_candidate.yaml`
5. If touching placement legality/timing, start from:
   - `envs/retro/placement_env.py`
   - `envs/retro/placement_planner.py`
   - `envs/retro/fast_reach.py`
   - `reach_native/drm_reach_full.c`
   - `game_engine/GameLogic.cpp`
6. If touching RAM/state mapping, prefer:
   - `docs/STATE_OBS_AND_RAM_MAPPING.md`
   - `envs/specs/ram_to_state.py`
   - `envs/specs/ram_offsets.json`
   - `dr-mario-disassembly/` and shared `../disassembly/`
7. Use focused tests before broad tests:
   - `python -m pytest tests/test_cpp_pool_smoke.py -q`
   - `python -m pytest tests/test_candidate_policy.py -q`
   - `python -m pytest tests/test_curriculum_scheduler.py -q`
   - `python -m pytest tests/test_game_engine_demo.py -q`

## Remaining Cleanup After Forward-Docs Pass

The docs/backlog cleanup from later on 2026-05-07 handled the stale
`cpp-engine`, `.[retro,dev]`, Stable-Retro-first, and EnvPool-forward guidance.
Remaining cleanup is narrower:

1. Reconcile the `game_engine/` submodule SHA and sibling `../drmario-native`
   checkout.
2. Fix the stale editable install from this repo path and install dev test deps.
3. Decide whether old compatibility files should stay in place or move under a
   dedicated archive after import shims/tests are planned.
