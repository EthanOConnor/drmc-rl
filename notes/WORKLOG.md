# WORKLOG.md — drmc-rl

Chronological log of work done. Format: date, actor, brief summary.

---

## 2025-10-17 – Coding Agent (Codex CLI)

- Confirmed ROM revision (Dr. Mario Japan/USA rev0, CRC32 0xB1F7E3E9).
- Extracted and validated RAM map from disassembly.
- Implemented RAM→state mapping in `envs/specs/ram_to_state.py`.
- Added tests and CLI tooling (`tools/ram_planes_dump.py`).
- Documented RNG, virus placement, and state spec in `docs/`.

## 2025-12-19 – Coding Agent (Codex CLI)

- Fixed SMDP-PPO multi-env rollouts: batch actions per decision, per-env τ accounting, and per-env GAE with `env_id` tracking.
- Added `emit_raw_ram` env option to trim AsyncVectorEnv IPC payloads; debug runs keep raw RAM enabled.
- Reworked debug TUI for multi-env: grid/summary view, env selection hotkeys, restart-only env count changes.
- Added `tools/bench_multienv.py` scaling harness (sync vs async, fps/speedup/efficiency metrics).

## 2025-12-20 – Coding Agent (Codex CLI)

- Normalized Gymnasium vector `info` dicts into per-env info lists inside the debug UI wrapper to avoid array→scalar conversion errors when rendering multi-env boards (`training/envs/interactive.py`).
- Fixed nested info unbatching so dict-valued entries (like `preview_pill`) are split per-env rather than broadcast (`training/envs/dr_mario_vec.py`).
- Added a lightweight benchmark smoke test that runs the multienv harness for a short sync run when the C++ engine binary is available (`tests/test_bench_multienv_smoke.py`).
- Logged curriculum graduation events with frames/episodes totals and deltas in SMDP-PPO (`training/envs/curriculum.py`, `training/algo/ppo_smdp.py`).
- Improved debug TUI restart UX (status TTL + alt-screen) and ensured debug sessions always stop/close on exit (`training/ui/runner_debug_tui.py`, `training/run.py`).
- Added compact grid rendering for multi-env boards, auto-downshifted UI refresh with env count, and a unified timing (ms/frame) breakdown with a wider summary column for readability (`training/ui/board_viewer.py`, `training/ui/runner_debug_tui.py`).
- Hid the per-env reward panel while in summary/grid view to keep the UI focused (`training/ui/runner_debug_tui.py`).
- Debounced `[`/`]` env-count restarts, added a numeric env-count entry mode (`e` + digits + Enter), and expanded the footer to show pending restart state (`training/ui/runner_debug_tui.py`).
- Added a curriculum advancement report script (`tools/report_curriculum.py`).
- Added `tests/test_cpp_backend_multienv.py` for multi-instance C++ backend isolation.
- Removed noisy startup warnings by (a) switching environment package capture to `importlib.metadata` (no `pkg_resources` deprecation warning) and (b) lazy-importing W&B so its pydantic warnings don’t fire unless W&B is enabled (`training/utils/reproducibility.py`, `training/diagnostics/logger.py`).
- Defaulted `training.run` to write each invocation under a unique `run_id` subdirectory (unless `--logdir` is provided) and recorded `run_id`/`logdir` in run metadata (`training/run.py`).
- Emitted curriculum snapshots in SMDP-PPO `update_end` events and displayed curriculum level/goal/success-window and env-level distribution in the Rich TUI (`training/algo/ppo_smdp.py`, `training/ui/tui.py`, `training/ui/event_handler.py`).
- Added a new `ln_hop_back` curriculum mode (probe + ln-tightened hop-backs) and set it as the default for `training/configs/smdp_ppo.yaml`; extended synthetic match-count stages to `-15..-4` (1..12 matches) (`training/envs/curriculum.py`, `envs/retro/drmario_env.py`).
- Made task time budgets “soft”: allow play past budget exceedance and replace the terminal clear bonus with a smooth time-goal reward that’s positive under-budget and negative over-budget (`envs/retro/placement_env.py`).
- Relaxed curriculum confidence defaults to 1-sigma (stage pass) and 2-sigma (mastery), and slowed the ln-style pass-rate ramps via `pass_ramp_exponent_multiplier=1/3` (`training/envs/curriculum.py`, `training/configs/smdp_ppo.yaml`).
- Reworked curriculum gate stability: replaced tiny rolling windows with an EMA-based Wilson lower bound (min effective sample size) and added a `min_stage_decisions` floor; stopped SMDP-PPO rollouts immediately on curriculum advancement to keep PPO updates stage-pure (`training/envs/curriculum.py`, `training/algo/ppo_smdp.py`, `training/configs/smdp_ppo.yaml`).

## 2025-12-21 – Coding Agent (Codex CLI)

- Fixed a Gymnasium `AsyncVectorEnv` crash caused by returning `None` for sometimes-numeric info keys; unset optional fields are now omitted (`envs/retro/placement_env.py`).
- Added per-run `drmario_engine` pidfile tracking and best-effort cleanup on shutdown to reduce orphaned engine processes after crashes/forced worker termination (`envs/backends/cpp_engine_backend.py`, `training/run.py`).
- Normalized virus-clear rewards so the total per-episode virus-clear reward is constant across levels; updated reward config + docs.
- Updated `tools/plot_success_by_level.py` to default to confidence-lower-bound plots and skip the first 10 episodes; added a metric selector for alternate plots.
- Freed thread-local reachability BFS buffers for cpp-pool worker threads to prevent per-step memory leaks during parallel planning (`reach_native/drm_reach_full.c`, `game_engine/DrMarioPool.cpp`).
- Added SMDP-PPO checkpoint warm-start support via `train.init_checkpoint` (with optional optimizer/step restore) for extending runs with new curricula (`training/algo/ppo_smdp.py`).
- Fixed SMDP-PPO resume throughput stats to use steps since resume, and hardened checkpoint IO (atomic saves + clearer load errors) to avoid corrupted resumes (`training/algo/ppo_smdp.py`).
- Implemented an in-process batched C++ pool backend (`game_engine/libdrmario_pool`) that owns N engine instances, integrates the native reachability planner, and emits decision-time masks/obs plus step-time events/counters.
- Added a `cpp-pool` training backend (ctypes wrapper + lightweight vector env) and wired it into the real env factory (`envs/backends/drmario_pool.py`, `training/envs/drmario_pool_vec.py`, `training/envs/dr_mario_vec.py`).
- Added `python -m tools.build_drmario_pool` and a pytest smoke test for the pool backend (`tools/build_drmario_pool.py`, `tests/test_cpp_pool_smoke.py`).
- Updated the SMDP-PPO config to default to `backend: cpp-pool` (`training/configs/smdp_ppo.yaml`).
- Parallelized cpp-pool planner/step work across envs, made the native reachability helper thread-local, and added `DRMARIO_POOL_WORKERS`/`-pthread` support for tuning worker count (`game_engine/DrMarioPool.cpp`, `reach_native/drm_reach_full.c`, `game_engine/Makefile`).
- Added a candidate-scoring placement policy (packed feasible actions + explicit cost-to-lock feature) and wired it into SMDP-PPO (`models/policy/candidate_policy.py`, `models/policy/candidate_packing.py`, `training/algo/ppo_smdp.py`).
- Added a candidate-policy config + tests (`training/configs/smdp_ppo_candidate.yaml`, `tests/test_candidate_policy.py`).
- Added a cpp-engine integration smoke test for the candidate policy (skips if `game_engine/drmario_engine` is unavailable) (`tests/test_candidate_policy_cpp_engine_smoke.py`).
- Reconciled candidate-policy implementation with the existing docs/config/tests (restored missing modules + ensured interfaces match SMDP-PPO candidate mode); `pytest -q` passes.
- Suppressed Ctrl+C shutdown tracebacks in debug runs by catching `KeyboardInterrupt` during session/env teardown (`training/run.py`, `training/envs/dr_mario_vec.py`).
- Strengthened candidate-policy correctness + signal: deterministic packing tie-breaks, PPO update asserts repacked candidates contain the chosen macro action, and candidate-local patches now include color+virus planes (not just occupancy) (`models/policy/candidate_packing.py`, `training/algo/ppo_smdp.py`, `models/policy/candidate_policy.py`).
- Improved candidate-policy throughput: precomputed patch offsets to avoid per-forward allocations, prepacked candidates once per PPO update (no per-minibatch repacking), sanitized NaN costs, lowered default candidate `Kmax` to 128, and added targeted tests (`models/policy/candidate_policy.py`, `models/policy/candidate_packing.py`, `training/algo/ppo_smdp.py`, `training/configs/smdp_ppo_candidate.yaml`, `tests/test_candidate_policy.py`).
- Added a plotting utility to select a run log and plot `curriculum/rate_current` vs steps per curriculum level, breaking lines across gaps when a level is not active (`tools/plot_success_by_level.py`).
- Strengthened candidate scoring with spatial trunk context: gather CNN feature-map features (or column-token features) at the candidate’s two landing cells/columns and feed them to the per-candidate MLP (`models/policy/candidate_policy.py`, `tests/test_candidate_policy.py`).
- Pruned `runs/` to reclaim disk space (kept newest 3 runs under `runs/smdp_ppo_candidate/` and removed older run artifacts under `runs/smdp_ppo/` + `runs/ppo_example/`).
- Tweaked `ln_hop_back` curriculum defaults: skip immediate full hop-backs when a new probe stage is already above the k=1 pass target, hop back to the 3rd-highest mastered hop-back level (not always `start_level`), and add a configurable bailout (fraction of run `total_steps`) for stuck probe stages (`training/envs/curriculum.py`, `training/envs/dr_mario_vec.py`, `tests/test_curriculum_scheduler.py`).

## 2025-12-22 – Coding Agent (Codex CLI)

- Added line-by-line inline documentation to `training/configs/smdp_ppo_candidate.yaml`, cross-referencing the current SMDP-PPO + cpp-pool code paths and flagging unused/ignored knobs in the default experimental setup.
- Logged a new scrutiny item about unordered pill embeddings vs directed macro-action semantics (`notes/SCRUTINY.md`).

## 2025-11-22 – Coding Agent

- Implemented C++ game engine core logic (`game_engine/GameLogic.cpp`).
- Completed RNG, level generation, basic gravity/matching/clearing.
- Set up shared memory IPC between engine and Python.
- Created monitor tool for visual debugging (`game_engine/monitor.cpp`).

## 2025-01-04 – Coding Agent

- Completed SMDP-PPO placement policy implementation.
- Implemented 3 policy heads: dense, shift_score, factorized.
- Created `MaskedPlacementDist` for action masking.
- Added `DecisionRolloutBuffer` with SMDP discounting (Γ=γ^τ).
- Created training launcher (`training/launches/train_placement_smdp_ppo.py`).
- All 12 unit tests passing.
- Documented in `PLACEMENT_POLICY_IMPLEMENTATION.md` and `IMPLEMENTATION_COMPLETE.md`.

## 2025-12-16 – Coding Agent (Antigravity)

- Performed comprehensive codebase review.
- Created developer handoff report with prioritized findings.
- Identified critical gaps: no notes system, C++ engine missing DAS/wall kicks.
- Found 64 tests all passing.
- Implemented inter-session notes system (`notes/` directory).

## 2025-12-16 – Decruftification & Runner Refactor

- Deleted stub directories: io-bridge/, streaming/, sim-envpool/, retro/
- Deleted orphan files: patches, package.json, Screenshot
- Archived drmarioai/ Java bot to archive/ (reference only)
- Updated .gitignore: added cores/, checkpoints/, .venv-*/
- Updated docs/REFERENCES.md with drmarioai, Rich, Textual, WandB
- Created training/ui/tui.py: Rich-based TUI with sparklines (replaces Tkinter)
- Created training/utils/devices.py: unified MLX/PyTorch device resolution
- Created training/utils/wandb_logger.py: WandB integration with graceful fallback
- Enhanced training/run.py: added --ui tui|headless, --wandb, --wandb-project
- Updated pyproject.toml: added rich>=13.0, wandb, bumped version to 0.1.0
- Created training/ui/board_viewer.py: Rich-based board visualization with colored tiles
- Created training/ui/debug_viewer.py: interactive debug viewer with step controls

## 2025-12-16 – Critical Priority Tasks

- Created training/ui/event_handler.py: TUIEventHandler bridges EventBus → TUI
- Modified training/run.py: TUI integration with --ui tui flag
- Verified DAS physics already implemented in GameLogic.cpp (16-frame initial, 6-frame repeat)
- Verified wall kicks already implemented in GameLogic.cpp (kick-left on blocked rotation)
- Updated game_engine/AGENTS.md: corrected implementation status (was outdated from Nov 22)
- Updated notes/BACKLOG.md: marked critical priorities complete

## 2025-12-16 – C++ Engine Parity Testing

- Created tools/game_transcript.py: frame-by-frame recording format
  - JSON + MessagePack serialization
  - Delta encoding for board changes
  - Comparison utilities for divergence detection
  - Demo input parser (RLE → per-frame)
- Created tools/record_demo.py: C++ engine demo recorder
  - Uses --manual-step for synchronized stepping
  - Captures pill positions, board changes
- Ran parity test: initial board matches demo_field exactly
- Found divergence: C++ engine tops out at frame 292 (8 pills) vs NES ~5461 frames
- Root cause TBD: likely timing difference in input processing

## 2025-12-17 – NES Demo Parity Deep Dive

### Recording Layer Verification ✓
- Verified `demo_pills` array in C++ matches NES ROM exactly (45 bytes)
- Confirmed NES recorder captures board state correctly from RAM at `0x0400`
- Validated pills 1-3: positions AND colors match NES exactly (Y-Y, B-R, Y-B)
- Board state byte-for-byte identical after pill 3

### Fixes Applied
- **spawn_delay**: Added 35-frame delay matching NES throw animation
  - Files: `GameState.h` (new field), `GameLogic.cpp` (init/decrement), `engine_shm.py`
- **INPUT_OFFSET**: Changed from 158 to 124 (accounts for spawn_delay in input indexing)
  - File: `tools/record_demo.py`
- **pill_counter off-by-one**: Added second `generateNextPill()` in `init()`
  - Matches NES `level_init.asm` lines 125-126
  - File: `game_engine/GameLogic.cpp` lines 163-165

### Current State
- Pills 1-3: ✓ Full parity (positions, colors, board state)
- Pill 4: First divergence point
  - C++ lands at (4,6) as single tile; NES at (3,6-7) full pill
  - Board identical at spawn → behavioral difference in C++ engine
- Root cause: Cumulative timing drift (C++ 28 frames slower by pill 4)

### Handoff Notes
- Recording layer is rock-solid and trustworthy
- Divergence stems from C++ engine behavior, not recording
- See walkthrough.md for detailed analysis and next steps
- Probable causes: gravity counter timing, DAS timing, spawn_delay interaction

## 2025-12-18 – Coding Agent (Codex CLI)

- Reimplemented `game_engine/GameLogic.cpp` as a parity-first port of the NES frame loop (explicit NMI tick + `nextAction` / `pillPlacedStep` state machines).
- Corrected core ROM tables and counters for rules-exact timing: full NTSC `speedCounterTable`, full 128-byte `demo_pills`, 512-byte demo input stream semantics, and BCD counter behavior (viruses/pill counters).
- Updated demo tooling for ground-truth verification:
  - `tools/record_demo.py` now relies on engine-internal demo replay (no external input feeding) and matches NES recorder stop semantics.
  - `tools/game_transcript.py` comparison fixes + normalization.
- Added regression coverage: `tests/test_game_engine_demo.py::test_demo_trace_matches_nes_ground_truth` asserts full demo trace matches `data/nes_demo.json`.

## 2025-12-18 – Coding Agent (Codex CLI) – Engine Demo TUI

- Added an interactive Rich-based demo player for the C++ engine (`tools/engine_demo_tui.py`) with pause/step/speed/restart controls and live shared-memory state inspection.
- Fixed `training/ui/board_viewer.parse_board_bytes` to correctly accept raw `bytes`/`bytearray` board buffers.
- Improved demo TUI ergonomics + diagnostics: upcoming pill list, smoother FPS estimate, and an integrated benchmark suite (engine freerun vs manual-step vs TUI render costs).
- Fixed a pill-render “seam” artifact by filling pill tiles using background color in `training/ui/board_viewer.py`.
- Switched playback speed control from “seconds per frame” to an `x` multiplier target (e.g. `2.4x` NTSC), with region/base-FPS selection in `tools/engine_demo_tui.py`.
- Refreshed the tracked prebuilt engine artifacts (`game_engine/drmario_engine`, `game_engine/*.o`) to match the parity-correct C++ sources (fixes demo playback stalling/timeouts when running the shipped binary).

## 2025-12-18 – Coding Agent (Codex CLI) – Macro Placement Planner Rewrite

- Re-implemented the placement macro-action stack as a NES-accurate, spawn-latched SMDP wrapper:
  - Canonical 512-way `(o,row,col)` action space in `envs/retro/placement_space.py`.
  - Frame-accurate reachability in `envs/retro/fast_reach.py` (gravity + DAS + rotation quirks).
  - Spawn snapshot decoding + feasibility/cost masks + minimal-time controller reconstruction in `envs/retro/placement_planner.py`.
  - Gym wrapper `envs/retro/placement_env.py` returning `placements/*` masks and `placements/tau`.
- Kept `envs/retro/placement_wrapper.py` as a small compatibility shim for older scripts.
- Restored training/docs ergonomics:
  - `envs/retro/register_env.py` registers both `DrMarioPlacementEnv-v0` and legacy `DrMario-Placement-v0`.
  - Updated `docs/PLACEMENT_PLANNER.md`, `docs/PLACEMENT_POLICY.md`, and `QUICK_START_PLACEMENT_POLICY.md` to match the new wrapper.
- Added QuickNES update utility (`tools/update_quicknes_core.py`) and documented it in `docs/RETRO_CORE_NOTES.md`.
- Fixed regressions uncovered by unit tests:
  - Corrected τ=1 bootstrap semantics in `tests/test_placement_policy.py`.
  - Hardened `training/speedrun_experiment.py` episode finalization for missing/legacy `dones` tracking.

## 2025-12-18 – Coding Agent (Codex CLI) – New Runner: Real Env + Debug TUI

- Upgraded the unified runner (`training/run.py`) to support:
  - `--algo ppo_smdp` (uses `training/algo/ppo_smdp.py`).
  - `--ui debug` (Rich board visualization + pause/step + speed controls).
  - Convenience flags for real retro training: `--env-id`, `--core`, `--core-path`, `--rom-path`, `--backend`, `--level`, `--vectorization`.
- Implemented a real vector env factory in `training/envs/dr_mario_vec.py`:
  - Returns a Gymnasium VectorEnv for `DrMario*` env ids (with a wrapper that converts vector `infos` to a list-of-dicts).
  - Keeps `DummyVecEnv` for tests and non-retro configs.
- Added interactive playback control wrappers:
  - `training/envs/interactive.py` provides `PlaybackControl` + `RateLimitedVecEnv` (pause/single-step and target FPS using `placements/tau` when available).
  - `training/ui/runner_debug_tui.py` provides a terminal-based debug UI with board rendering from `raw_ram`.
- Added a unit test for the real env factory running on the mock backend: `tests/test_runner_real_env_factory.py`.
- Runner/config polish:
  - `training/run.py` now defaults `--algo/--engine` from the config file when not provided on the CLI (with `smdp_ppo` → `ppo_smdp` aliasing).
  - `training/configs/smdp_ppo.yaml` updated to `algo: ppo_smdp` and `env.id` (matches the unified runner + env factory).
  - Placement docs (`docs/PLACEMENT_POLICY.md`, `QUICK_START_PLACEMENT_POLICY.md`, `IMPLEMENTATION_COMPLETE.md`, `PLACEMENT_POLICY_IMPLEMENTATION.md`) now recommend `python -m training.run` over bespoke launch scripts.
- Fixed a reset-time state mismatch in `envs/retro/drmario_env.py`:
  - Rebuild `_state_cache` after the auto-start sequence so `reset()` returns observations consistent with the post-start `raw_ram` snapshot.
  - `viruses_remaining` now prefers the raw RAM counter during startup (avoids stale `_state_cache` during reset/start sequences).
- Fixed a `SyntaxError` in `training/ui/runner_debug_tui.py` (`f-string` quoting in the speed display).

## 2025-12-18 – Coding Agent (Codex CLI) – Runner Reset Fix + Native Reachability

- Fixed a training-time double-reset bug in `training/envs/dr_mario_vec.py` that prevented `DrMarioRetroEnv` from running the full 3-press auto-start sequence (led to empty board/viruses=0 on the placement env).
- Added a native reachability accelerator for the macro placement planner:
  - C BFS implementation: `reach_native/drm_reach_full.c`
  - Python wrapper + buffer management: `envs/retro/reach_native.py`
  - Build helper: `python -m tools.build_reach_native`
  - Bench harness: `python -m tools.bench_reachability`
- Integrated native backend into `envs/retro/placement_planner.py` (`reach_backend=auto|native|python`) while keeping `envs/retro/fast_reach.py` as the oracle.
- Surfaced the active planner backend in env `info` as `placements/reach_backend` and displayed it in the debug TUI stats panel.
- Updated docs: `docs/PLACEMENT_PLANNER.md`, `docs/PLACEMENT_POLICY.md`, `QUICK_START_PLACEMENT_POLICY.md`.

## 2025-12-18 – Coding Agent (Codex CLI) – SMDP-PPO Minibatch KL Bugfix

- Fixed a crash in `training/algo/ppo_smdp.py` where KL divergence was computed against the full-batch `log_probs_old` instead of the mini-batch slice (`mb_log_probs_old`), causing a 512-vs-128 tensor shape mismatch when `minibatch_size < decisions_per_update`.

## 2025-12-18 – Codex CLI – Auto-start Fix + RNG Randomization Toggle

- Fixed libretro auto-start after terminal episodes by using the correct default `start_presses=3` for backend resets (previous 1/2-press logic left the game in menus with viruses=0 after topout/clear).
- Made level alignment robust by reading the current level from RAM and tapping LEFT/RIGHT to reach the configured level (avoids wrap-around to level 20).
- Added per-env RNG randomization toggle:
  - New env attribute `rng_randomize` (used as the default for `reset(options.randomize_rng)` so Gymnasium vector autoresets still honor it).
  - `training.run` CLI: `--randomize-rng/--no-randomize-rng`
  - Debug TUI hotkey: `r` (shows `rng: on/off` in stats).

## 2025-12-18 – Codex CLI – Debug UI Responsiveness + Batched PPO Update

- Vectorized `SMDP-PPO` policy update in `training/algo/ppo_smdp.py` by computing masked log-probs/entropy for the full minibatch at once (removes per-sample Python loops that could stall the debug UI and reduce throughput).
- Fixed `MaskedPlacementDist` edge-case handling for batched masks (`models/policy/placement_dist.py`).
- Enhanced debug UI performance telemetry:
  - `emu_fps(step)` = frames/sec inside `env.step` only (planner + emu)
  - `emu_fps(total)` = frames/sec including training compute between env steps

## 2025-12-18 – Codex CLI – Placement Env: Skip No-Feasible Spawns

- Fixed `envs/retro/placement_env.py` to treat “decision points” with **zero feasible macro placements** (e.g., spawn-blocked top-out) as non-decision frames and keep stepping NOOP until the env transitions (lock/top-out/reset) instead of returning an empty mask that can cause an infinite invalid-action loop.
- Added regression coverage: `tests/test_placement_env_no_feasible_actions.py`.

## 2025-12-19 – Codex CLI – Reduced Bitplane Observations + Mask Injection

- Added two new state representations in `envs/specs/ram_to_state.py`:
  - `bitplane_reduced` (6ch): type-blind color planes + `virus_mask` + `pill_to_place` + `preview_pill`.
  - `bitplane_reduced_mask` (10ch): reduced + 4 feasibility-mask channels (`feasible_o0..feasible_o3`).
- Implemented feasibility-mask injection at true decision points in `envs/retro/placement_env.py` (fills reserved obs channels from `placements/feasible_mask`).
- Made the debug runner UI less noisy by hiding the channel index→name list by default (toggle with `p`) in `training/ui/runner_debug_tui.py`.
- Updated docs: `docs/STATE_OBS_AND_RAM_MAPPING.md`, `docs/RAM_TO_STATE.md`, `docs/PLACEMENT_POLICY.md`.
- Added tests for the new representations and mask injection:
  - `tests/test_bitplane_reduced_helpers.py`
  - `tests/test_feasible_mask_obs_injection.py`

## 2025-12-19 – Codex CLI – Bottle-Only Obs + Preview-Pill Vectors

- Added bottle-only state representations:
  - `bitplane_bottle` (4ch): bottle color planes + `virus_mask` (no falling/preview projection).
  - `bitplane_bottle_mask` (8ch): bottle-only + feasibility planes injected by placement env.
- Decoded falling/preview pill metadata directly from RAM in `envs/state_core.py` (observation-repr independent), and updated intent wrapper to decode falling coords from RAM.
- Updated the placement policy to condition on both **current** and **preview** pill colors as vectors (no longer requires `pill_to_place`/`preview_pill` planes), and extended the rollout buffer accordingly.

## 2025-12-19 – Codex CLI – Env Step Profiling Breakdown

- Added per-frame env timing keys (`perf/env_*_sec`) in `envs/retro/drmario_env.py` and aggregated them per macro decision in `envs/retro/placement_env.py`.
- Extended `training/envs/interactive.py` perf snapshot to report env breakdown ms/frame and `macro_other_ms/frame`.
- Updated `training/ui/runner_debug_tui.py` Perf panel to display the breakdown.
- Optimized state-mode stepping by avoiding redundant RAM refreshes, reusing the RAM snapshot for `info["raw_ram"]`, and fetching RGB frames lazily in `render()` unless `obs_mode=pixel`.
- Fixed a debug-TUI crash when `info["raw_ram"]` was present but `None` (`training/ui/board_viewer.py`), and added `ms/frame(total|accounted|unaccounted)` rows to reconcile `sps` vs per-component timings.
- Reduced per-frame reward overhead by gating adjacency/height computations on static-tile changes and vectorizing bottle-buffer scans (`envs/retro/drmario_env.py`).
- Reduced C++ engine backend step overhead by using a short spin-then-sleep polling loop (avoid guaranteed oversleep each frame) (`envs/backends/cpp_engine_backend.py`).
- Added optional encoder scaling via `encoder_blocks` (extra 64-channel residual blocks) and updated debug UI to show preview pill colors.

## 2025-12-18 – Codex CLI – Debug TUI: Perf Diagnostics (Inference/Planner)

- Added lightweight perf counters + timing breakdowns for interactive runs:
  - `DrMarioPlacementEnv` now emits planner timings (`perf/planner_build_sec`, `perf/planner_plan_sec`) and legacy planner-step keys (`placements/plan_calls`, `placements/plan_latency_ms_*`).
  - `RateLimitedVecEnv` accumulates inference/planner/update timing and exposes derived `ms/frame` and `ms/call` stats via `perf_snapshot()`.
  - `RunnerDebugTUI` displays inference/planner/update diagnostics alongside FPS.
- Added unit coverage: `tests/test_interactive_perf_counters.py`.

## 2025-12-18 – Codex CLI – Placement Env: Spawn-Latched Decisions (Fix Excess Replanning)

- Fixed `envs/retro/placement_env.py` to expose **exactly one macro decision per pill spawn** by gating decision-point detection on the ROM spawn counter (`pill_counter`, RAM `$0310`) in addition to `currentP_nextAction == nextAction_pillFalling`.
- Marked a spawn as “consumed” once we commit to a macro plan (or when `placements/options == 0`) so we don’t surface new decisions mid-fall (prevents “options ticking down” + hundreds of extra planner/inference calls per spawn).
- Adjusted planner-build timing emission so `perf/planner_build_sec` counts actual reachability builds (invalid-action retries reuse cached ctx).
- Added regression coverage: `tests/test_placement_env_spawn_latch.py`.

## 2025-12-18 – Codex CLI – Episode Stats + Live Return Metrics (Runner Debug UI)

- Added a vector-env wrapper episode-stat injector (`training/envs/dr_mario_vec.py`) that attaches:
  - `info["episode"] = {"r": return, "l": length_frames, "decisions": decisions}`
  - `info["drm"]` with lightweight end-of-episode summaries (e.g., `viruses_cleared`, `top_out`, `cleared`)
  This fixes `ret(last)`, `ret(mean100)`, and `len(last)` reporting for real envs.
- Extended the debug TUI stats panel (`training/ui/runner_debug_tui.py`) to show:
  - live per-episode return (`ret(curr)`), median of last 16 (`ret(med16)`), and current episode progress (`len(curr)`).
- Added unit coverage for episode stats injection: `tests/test_vec_env_episode_stats.py`.
- Documented the current base reward terms in `docs/REWARD_SHAPING.md`.

## 2025-12-18 – Codex CLI – Scripted Curriculum + WandB Wiring

- Wired `training.run --wandb/--wandb-project` into `DiagLogger` by ensuring `"wandb"` is added to `cfg.viz` when enabled.
- Implemented a scripted curriculum based on synthetic negative levels:
  - `DrMarioRetroEnv` interprets `level < 0` as a curriculum stage and patches the bottle RAM at reset time to reduce virus count (`-4..-1`), with `-4` using “any 4-match” (first clear event) as the success condition.
  - Added `training/envs/curriculum.py` (`CurriculumVecEnv`) to schedule levels based on rolling clear rate and optional rehearsal of lower levels.
  - Enabled the curriculum in `training/configs/smdp_ppo.yaml` and surfaced curriculum stats in `RunnerDebugTUI`.
- Updated docs to reflect implemented curriculum (`docs/PLACEMENT_POLICY.md`, `QUICK_START_PLACEMENT_POLICY.md`, `PLACEMENT_POLICY_IMPLEMENTATION.md`).

## 2025-12-18 – Codex CLI – Curriculum Clear Detection + Reward Breakdown Debugging

- Fixed curriculum `-4` clear detection by using the ROM’s clearing-tile markers (`CLEARED_TILE`/`FIELD_JUST_EMPTIED`) from bottle RAM (`envs/retro/drmario_env.py`) instead of relying on occupancy deltas (which can miss clears).
- Made curriculum stats explicitly “recent window” in the debug UI by surfacing `window_n/window_size` and ensuring terminal-step info reports the episode’s level (plus `next_env_level`) (`training/envs/curriculum.py`, `training/ui/runner_debug_tui.py`).
- Added reward breakdown aggregation for macro steps (`reward/*` totals and counts) and a new Reward column in the debug TUI to audit scoring live (`envs/retro/placement_env.py`, `training/envs/interactive.py`, `training/ui/runner_debug_tui.py`).
- Fixed the default reward config to apply a negative top-out penalty (`envs/specs/reward_config.json`).

## 2025-12-18 – Codex CLI – Fix `-4` False Clears + Debug UI Columns

- Fixed a false-positive in the curriculum level `-4` “any_clear” success detection: empty tiles are `0xFF` (high nibble `0xF0`) and must not be counted as `FIELD_JUST_EMPTIED` (`0xF0..0xF2`). Added a gameplay-mode guard and a regression test (`envs/retro/drmario_env.py`, `tests/test_clearing_tile_counter.py`).
- Corrected bitplane `clearing_mask` / `empty_mask` construction to distinguish `0xFF` empty from `0xF*` just-emptied tiles (`envs/specs/ram_to_state.py`).
- Restructured `RunnerDebugTUI` into 4 columns: board + perf + learning + reward, with wider side panels (`training/ui/runner_debug_tui.py`).

## 2025-12-18 – Codex CLI – Preview + Placement Verification + Curriculum Stages

- Made `preview_pill` consistently structured as a dict (`first_color`, `second_color`, `rotation`) and updated the Rich board renderer to show the next pill above the bottle with correct orientation (`envs/retro/drmario_env.py`, `training/ui/board_viewer.py`).
- Added placement verification metadata to macro steps by capturing the observed falling-pill pose at lock time and comparing it to the planner’s target pose (`placements/pose_ok`, `placements/target_pose`, `placements/lock_pose`) (`envs/retro/placement_env.py`).
- Expanded synthetic curriculum stages to include match-count levels `-10..-4` (1..7 matches) before the 1/2/3-virus stages (`-3..-1`) and updated configs to start at `-10` (`envs/retro/drmario_env.py`, `training/envs/curriculum.py`, `training/configs/smdp_ppo.yaml`).
- Made RNG randomization default-on in the standard training configs, and added spawn-level perf ratios (`infer/spawn`, `planner/spawn`) plus last terminal reason tracking to the debug UI (`training/configs/base.yaml`, `training/envs/interactive.py`, `training/ui/runner_debug_tui.py`).
- Surfaced placement verification status in the debug UI (`pose_ok`, `pose_err`) so mismatches are visible without digging through raw infos (`training/ui/runner_debug_tui.py`).
- Added coverage for the new negative-level mapping (`tests/test_synthetic_level_mapping.py`).

## 2025-12-19 – Codex CLI – Planner Parity + Canonical Clear Counting

- Fixed placement pose verification to capture the locked-pill pose from RAM state transitions (leaving `nextAction_pillFalling` / spawn counter advancing) instead of relying on `pill_bonus_adjusted` (which can be disabled by reward config). Added `placements/lock_reason` and surfaced `target_pose`/`lock_pose` in the debug UI.
- Corrected “down-only” soft drop parity: soft drop triggers when `frameCounter & 1 == 0` (not `== 1`). Updated:
  - Python oracle stepper (`envs/retro/fast_reach._step_state`)
  - Python packed BFS (`build_reachability` fast path)
  - Native BFS helper (`reach_native/drm_reach_full.c`, rebuild via `python -m tools.build_reach_native`)
  This restores frame-accurate agreement between native reachability scripts and the emulator (pose mismatches disappear).
- Replaced non-virus clear reward counting with a canonical bottle-buffer diff (`envs/specs/ram_to_state.count_tile_removals`) and cached bottle snapshots in `DrMarioRetroEnv` to avoid false positives from falling-pill overlays. Updated docs (`docs/REWARD_SHAPING.md`) and added unit coverage.

## 2025-12-19 – Codex CLI – Pose Mismatch Logging

- Added persistent pose mismatch counters (`placements/pose_mismatch_*`) and JSONL logging for rare planner/executor divergences, dumping snapshot + board + feasibility + plan + observed lock pose into `data/pose_mismatches.jsonl(.gz)` (override/disable via `DRMARIO_POSE_MISMATCH_LOG`). Optional per-frame trace capture is gated by `DRMARIO_POSE_MISMATCH_TRACE` (`envs/retro/placement_env.py`, `training/ui/runner_debug_tui.py`).

## 2025-12-19 – Codex CLI – Rotation Edge Semantics + Reward Config Safety

- Fixed reachability/planner rotation semantics to match the ROM: rotate uses `currentP_btnsPressed` (edge), so holding A/B across consecutive frames must not rotate repeatedly. Implemented by tracking a new per-frame `rot_hold` state across the Python stepper, Python packed BFS, and native BFS helper; rebuilt native dylib (`envs/retro/fast_reach.py`, `envs/retro/placement_planner.py`, `reach_native/drm_reach_full.c`, `envs/retro/reach_native.py`).
- Made reward-config failures non-silent and aligned `RewardConfig` dataclass defaults with `envs/specs/reward_config.json` so reward scale can’t unexpectedly jump to legacy “hundreds” defaults (`envs/retro/drmario_env.py`).

## 2025-12-19 – Codex CLI – Native Reachability Planner Performance

- Optimized the native reachability helper by (a) early-stopping once all in-bounds terminal poses are found, (b) pruning the early-stop target set via a timer-free geometric flood fill (to avoid max-depth blowups from sealed cavities), and (c) switching to a frontier-aggregated BFS that batches x positions per counter-state key (8-bit x masks). Added optional stats via `DRMARIO_REACH_STATS=1` and a replay test that validates native plans against the Python per-frame stepper (`reach_native/drm_reach_full.c`, `envs/retro/reach_native.py`, `tests/test_reach_native_smoke.py`).

## 2025-12-19 – Codex CLI – Virus Adjacency Shaping + Bitplane Policy Obs

- Added virus-specific adjacency shaping terms (`virus_adjacency_pair_bonus`, `virus_adjacency_triplet_bonus`) and surfaced the aggregate in the debug UI reward breakdown (`envs/retro/drmario_env.py`, `envs/specs/reward_config.json`, `envs/retro/placement_env.py`, `training/envs/interactive.py`, `training/ui/runner_debug_tui.py`).
- Set the placement SMDP-PPO config to use the `bitplane` state representation by default (type-blind color planes + virus mask) and improved preview decoding for bitplane states to infer rotation (`training/configs/smdp_ppo.yaml`, `envs/specs/ram_to_state.py`).
- Added unit coverage for virus adjacency shaping (`tests/test_virus_adjacency_reward.py`) and updated reward docs (`docs/REWARD_SHAPING.md`).

## 2025-12-19 – Codex CLI – Debug TUI: State Representation + Input Plane Names

- Added `ram_to_state.get_plane_names()` and a small test to keep plane-name lists consistent with channel counts (`envs/specs/ram_to_state.py`, `tests/test_state_plane_names.py`).
- Extended the debug TUI Perf panel to show `state_repr`, per-env observation shape, plane index→name map, next-pill colors, and mask/orientation conventions (`training/ui/runner_debug_tui.py`).

## 2025-12-19 – Codex CLI – C++ Engine ↔ Libretro Ghost Parity

- Added `cpp-engine` backend (shared memory + subprocess) with a synthetic 2 KB NES RAM view so the existing RAM→state pipeline works unchanged (`envs/backends/cpp_engine_backend.py`, `envs/backends/__init__.py`, `game_engine/engine_shm.py`, `game_engine/GameState.h`).
- Made libretro RNG seeding parity-robust by applying `rng_seed_bytes` at the `initData_level` boundary (mode==0x03), removed the engine’s menu-time RNG warmup hack, and added per-reset `frameCounter` low-byte seeding for exact soft-drop timing parity (`envs/retro/drmario_env.py`, `game_engine/GameLogic.cpp`).
- Added a ghosting parity harness that runs libretro and the C++ engine side-by-side and stops on first divergence (`tools/ghost_parity.py`).
- Hardened libretro auto-start: level selection clamps (no wrap-around) and added `start_sync_wait_frames` (`waitFrames` sync) so resets land in a stable post-virus-placement checkpoint across levels (`envs/retro/drmario_env.py`, `tools/ghost_parity.py`).
- Fixed demo recorder startup gating + transcript frame numbering so demo parity remains deterministic under the new frameCounter seeding (`tools/record_demo.py`, `tests/test_rng_randomization.py`), and updated docs (`docs/RETRO_CORE_NOTES.md`, `docs/CPP_SIM_NOTES.md`, `docs/DYNAMICS_SPEC.md`).

## 2025-12-19 – Codex CLI – Multi-Env C++ Backend: Design + P0 Backlog

- Wrote a multi-env scaling design doc (vectorization policy, scaling metrics, debug UI hotkeys, restart semantics) in `notes/DESIGN_MULTIENV.md`.
- Added hierarchical “Up Next / P0” backlog items for multi-env correctness/perf/UI (including a scaling benchmark harness) in `notes/BACKLOG.md`.

## 2025-12-20 – Codex CLI – C++ Batched Stepping for Multi-Env Throughput

- Fixed Gymnasium `AsyncVectorEnv` worker env creation by registering Dr. Mario env ids inside each env factory (subprocess-safe) and making registration idempotent to avoid “Overriding environment …” warnings (`training/envs/dr_mario_vec.py`, `envs/retro/register_env.py`).
- Added a shared-memory batched run protocol to the C++ engine (`run_request_id/run_ack_id`, run modes for fixed-frames and “until next decision”, plus cleared-tile counters) and surfaced it in the Python backend (`game_engine/GameState.h`, `game_engine/main.cpp`, `game_engine/GameLogic.cpp`, `game_engine/engine_shm.py`, `envs/backends/cpp_engine_backend.py`).
- Added `DrMarioRetroEnv.sync_after_backend_run()` and a cpp-engine fast path in `DrMarioPlacementEnv` that executes the planner controller script and wait-to-next-spawn via batched runs (gated by reward config + `DRMARIO_CPP_FAST`), materially increasing decisions/sec in the scaling benchmark while preserving core training reward terms (`envs/retro/drmario_env.py`, `envs/retro/placement_env.py`).
- Fixed an SMDP termination bug in the cpp-engine fast path: match-mode curriculum stages (0 viruses) must not end just because `viruses_remaining==0`; now termination respects `task_mode` and only ends after the configured match count or top-out (`envs/retro/placement_env.py`).

## 2025-12-20 – Codex CLI – Async Scaling Stability + Fast Reset Path

- Removed a multiprocessing spawn footgun: moved cpp-engine shared-memory ctypes bindings into an installable module (`envs/backends/cpp_engine_shm.py`) and updated the backend to import it, so `AsyncVectorEnv` workers no longer depend on `game_engine` being importable.
- Hardened cpp-engine reset behavior under high env counts: `DrMarioRetroEnv.reset()` now retries by restarting the backend (instead of silently falling back to mock dynamics) and records `_backend_last_error` for downstream wrappers.
- Eliminated per-frame stepping during autoresets: `DrMarioPlacementEnv.reset()` now uses `cpp-engine.run_until_decision()` + `sync_after_backend_run()` to reach the first decision point without calling `env.step()` in a loop, reducing timeouts and improving scaling.
- Reduced polling overhead in `CppEngineBackend` (both single-step and batched runs) with gentler backoff to avoid CPU thrash at high env counts.
- Added an `AsyncVectorEnv` regression test to exercise autoresets and ensure the placement env doesn’t lose `_state_cache` (`tests/test_async_vec_env_stability.py`).
- Made shutdown robust for long async runs: `_InfoListWrapper.close()` force-terminates `AsyncVectorEnv` if it has a pending call to avoid `close()` hanging on `step_wait` (`training/envs/dr_mario_vec.py`, `training/run.py`).

## 2025-12-20 – Codex CLI – Time/Spawn Budget Scaffolding (Curriculum)

- Fixed the TUI “Goal” label for synthetic match-count levels (now matches the env mapping `max(1, 16 + level)` for `-15..-4`) so negative levels don’t display as “clear -3 matches” (`training/ui/tui.py`).
- Added optional time-based task budgets (`task_max_frames`, `task_max_spawns`) plumbed via `CurriculumVecEnv.set_attr` and enforced inside `DrMarioPlacementEnv` with per-episode counters and `info` keys under `task/*` (`training/envs/curriculum.py`, `envs/retro/placement_env.py`).
- Added unit coverage for frame/spawn budget truncation and “clear over budget strips terminal bonus” semantics (`tests/test_task_budgets.py`).

## 2025-12-20 – Codex CLI – Confidence-Based Curriculum Windows + Mastery Time Budgets

- Switched curriculum advancement from fixed `window_episodes/min_episodes` gating to a sigma-based one-sided Wilson lower bound check (`p > target`), with window sizes derived from a “near-target” assumption and configurable `confidence_sigmas` (default 2σ). Added a separate perfect-streak window size helper for mastery gating (`training/envs/curriculum.py`).
- Added “time budget after mastery” plumbing: once a level hits a perfect-streak window long enough to certify mastery at `time_budget_mastery_sigmas/time_budget_mastery_target`, the curriculum begins setting a per-level `task_max_frames` that starts at mean clear time and tightens gradually with a MAD-capped drop. Exposed mean/MAD/budget via curriculum snapshots (`training/envs/curriculum.py`).
- Surfaced curriculum Wilson LB + time budget/mean/MAD in both the Rich TUI and debug TUI, and added `tools/report_curriculum.py --confidence-table` to print expected window sizes / requirements (`training/ui/tui.py`, `training/ui/runner_debug_tui.py`, `training/algo/ppo_smdp.py`, `tools/report_curriculum.py`).
- Hardened `DrMarioPlacementEnv` decision-context building to treat missing `_state_cache` as a backend error and return `truncated=True` (instead of raising), improving `AsyncVectorEnv` robustness under high env counts (`envs/retro/placement_env.py`).

## 2025-12-20 – Codex CLI – Stage-Local Hop-Back Stats + Persistent Best Times + Time Goal Demotion

- Made `ln_hop_back` stage tracking stage-local: when revisiting a level with a tighter threshold, success-window stats are now fresh (tracked per stage token/index), avoiding contamination from earlier easier passes (`training/envs/curriculum.py`).
- Added a persistent sqlite best-times DB (`data/best_times.sqlite3`, git-ignored) to track per-(level, rng_seed) best frames/spawns across runs, plus a small reporting script (`tools/report_best_times.py`).
- Extended time-goal logic: once base mastery is achieved, time budgets begin tightening against an increasing `1-exp(-k)` success target; if base-objective mastery drops, time goals are cleared and must be re-earned. Exposed `time_k/time_target` and spawn stats in curriculum info and UIs (`training/envs/curriculum.py`, `training/ui/tui.py`, `training/ui/runner_debug_tui.py`, `training/algo/ppo_smdp.py`).
- Removed a long-standing skipped test by replacing the optional `mlx.core` dependency with a deterministic fake-MLX module in tests, covering both `_mlx_set_row` code paths (`tests/test_discounting.py`).
- Made `--ui tui` shutdown on Ctrl+C cleanly (no traceback spam) by catching `KeyboardInterrupt` around training (`training/run.py`).
- Ignored best-times sqlite WAL/SHM sidecar files and local scratch directories (`notes/human_notes/`, `tests/user_testing/`) in `.gitignore`.

## 2025-12-20 – Codex CLI – Docs/Notes Pass

- Updated README + placement-policy docs to reflect fast `cpp-engine` multi-env training (`--vectorization async`) and new reporting tools (`tools/bench_multienv.py`, `tools/report_curriculum.py`, `tools/report_best_times.py`).
- Refreshed `notes/BACKLOG.md` to mark completed multi-env items and add next steps for best-times/time-goal iteration.

## 2025-12-21 – Codex CLI – Engine/Digital-Twin Interface Design Notes

- Added a design document specifying the engine/digital-twin/planner boundary and the decision-vs-telemetry channel split, grounded in the existing `libdrmario_pool` ABI (`notes/DESIGN_ENGINE_TWIN_PROTOCOL.md`).
- Updated `notes/MEMORY.md` with the architectural decision to keep board internals behind the decision-boundary ABI and co-locate planning with the timing model.
- Added a new scrutiny item for vision shadow-mode desync/drift risks and the need for explicit twin quality signals (`notes/SCRUTINY.md`).

## 2025-12-21 – Codex CLI – Fix VectorEnv Info Merge Crash (Task Budgets)

- Fixed `DrMarioPlacementEnv.reset()` to omit unset `task/max_frames` and `task/max_spawns` keys (instead of returning `None`), matching `step()` behavior and preventing Gymnasium VectorEnv dtype crashes. Added a SyncVectorEnv regression test (`envs/retro/placement_env.py`, `tests/test_task_budgets.py`).

## 2025-12-21 – Codex CLI – SMDP-PPO Aux Inputs (v1) + Speed Setting

- Added `smdp_ppo.aux_spec` (`none|v1`) and plumbed a 57-dim aux vector into the placement policy net + rollout buffer (speed/viruses/level/time/heights/clear-progress + a few cheap extras) (`training/algo/ppo_smdp.py`, `models/policy/placement_heads.py`, `training/rollout/decision_buffer.py`).
- Surfaced `speed_setting` (0/1/2) as a real env option across retro backends and `cpp-pool`, and emitted it in decision-time infos (`envs/retro/drmario_env.py`, `envs/retro/placement_env.py`, `training/envs/dr_mario_vec.py`, `training/envs/drmario_pool_vec.py`).
- Added `drm/viruses_initial` as a backend-agnostic info key to support a scalar “clearance progress” aux feature (`envs/retro/placement_env.py`, `training/envs/drmario_pool_vec.py`).
- Updated the default SMDP-PPO config to enable aux v1 and default to high game speed (`training/configs/smdp_ppo.yaml`).

## 2025-12-21 – Codex CLI – cpp-engine Async Timeout Recovery

- Prevented rare cpp-engine batched-run timeouts from crashing `AsyncVectorEnv`: placement fast-path now truncates the episode, records `placements/backend_error*`, and forces a backend restart on run-request failures (`envs/retro/placement_env.py`).
- Made `CppEngineBackend._run_request` use a progress-based watchdog + a more forgiving total timeout (with better diagnostics) to reduce false timeouts under heavy multi-env load (`envs/backends/cpp_engine_backend.py`).
- Added regression tests ensuring cpp-engine fast-path timeouts truncate instead of raising (`tests/test_cpp_engine_timeout_recovery.py`).

## 2025-12-22 – Coding Agent (Codex CLI) – Ordered Pill Pair Embedding

- Added `pill_embed_type` (`unordered` vs `ordered_onehot`/`ordered_pair`) and implemented an ordered 9-way pair embedding to preserve half identity for mixed-color pills in directed macro action spaces (`models/policy/placement_heads.py`, `models/policy/candidate_policy.py`, `training/algo/ppo_smdp.py`).
- Updated the candidate-policy training config to enable the ordered embedding (`training/configs/smdp_ppo_candidate.yaml`).
- Added unit tests for order sensitivity + selection wiring (`tests/test_placement_policy.py`, `tests/test_candidate_policy.py`).

## 2025-12-22 – Coding Agent (Codex CLI) – Compressed Artifacts + Checkpoint Scanner

- Defaulted run artifacts to gzip-compressed, streamable files: `metrics.jsonl.gz`, `env.txt.gz`, and `*.pt.gz` checkpoints; updated readers/tools to accept `.gz` (e.g., `tools/plot_success_by_level.py`, `tools/report_curriculum.py`, `training/utils/reproducibility.py`, `training/run.py`, `tests/test_adapters.py`).
- Added gzip-aware logging for pose mismatch and ghost-parity JSONL outputs and updated docs to match the new defaults (`envs/retro/placement_env.py`, `tools/ghost_parity.py`, `QUICK_START_PLACEMENT_POLICY.md`).
- Added a checkpoint validation tool to scan for corrupt checkpoint files and optionally delete them (`tools/check_checkpoints.py`).
- Added a live-updating curriculum plotter with an in-window chart selector that reuses the existing JSONL parser and avoids background threads (`tools/plot_success_live.py`).
- Fixed a startup ordering bug in the live plotter (status label initialized before trace callback).
- Tweaked live + static plot pickers to show `*.jsonl.gz` files in the file filter (`tools/plot_success_live.py`, `tools/plot_success_by_level.py`).
- Reworked the live picker flow to offer explicit file/dir buttons and centered the chooser to avoid off-screen dialogs (`tools/plot_success_live.py`).
- Made the live picker use the main Tk window (no hidden root) to avoid invisible/off-screen dialogs on macOS, and relaxed file filters so `.gz` isn’t hidden (`tools/plot_success_live.py`).
- Made `.jsonl.gz` readers tolerant of in-progress gzip streams so live plots work while training is still writing logs (`tools/plot_success_by_level.py`, `tools/report_curriculum.py`, `tools/plot_success_live.py`).
- Resized the live plot window after selection so it doesn’t inherit the small picker geometry (`tools/plot_success_live.py`).
