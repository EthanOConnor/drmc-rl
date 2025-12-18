# WORKLOG.md — drmc-rl

Chronological log of work done. Format: date, actor, brief summary.

---

## 2025-10-17 – Coding Agent (Codex CLI)

- Confirmed ROM revision (Dr. Mario Japan/USA rev0, CRC32 0xB1F7E3E9).
- Extracted and validated RAM map from disassembly.
- Implemented RAM→state mapping in `envs/specs/ram_to_state.py`.
- Added tests and CLI tooling (`tools/ram_planes_dump.py`).
- Documented RNG, virus placement, and state spec in `docs/`.

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
