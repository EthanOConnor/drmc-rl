# SCRUTINY.md — drmc-rl

Critical review and risk tracking. Capture concerns about correctness, performance, API contracts, and UX, plus how we'll validate or mitigate them.

---

## 2025-12-16 – Codebase Review Findings

### Critical Priority

**C1. C++ engine missing DAS physics**
- **Location**: `game_engine/GameLogic.cpp`, `game_engine/AGENTS.md` L55-58
- **Issue**: Current implementation moves pieces instantly on input. Real NES has Delayed Auto Shift with specific frame timings.
- **Impact**: Agent trained on C++ engine will have different timing assumptions than real hardware or emulator.
- **Mitigation**: Implement DAS per `fallingPill_checkXMove` in disassembly before using C++ engine for training.
- **Status (2025-12-18)**: Mitigated. DAS timing now follows the disassembly and is validated against the full demo trace.

**C2. C++ engine missing wall kicks**
- **Location**: `game_engine/GameLogic.cpp`, `game_engine/AGENTS.md` L59
- **Issue**: Rotation validation does not handle wall kicks (pushing piece left/right when blocked).
- **Impact**: Certain placements possible on NES are impossible in C++ engine, causing parity failure.
- **Mitigation**: Implement `pillRotateValidation` ($8E70) from disassembly.
- **Status (2025-12-18)**: Mitigated. Wall-kick behavior now follows the disassembly and is validated against the full demo trace.

**C3. No parity tests for C++ engine**
- **Location**: `game_engine/`, `tests/`
- **Issue**: No automated tests comparing C++ engine output vs NES demo data.
- **Impact**: Cannot verify correctness. May ship subtle bugs that affect training.
- **Mitigation**: Create parity test suite using demo mode input sequences and expected board states.
- **Status (2025-12-18)**: Mitigated. Added a regression test that asserts the full demo trace matches `data/nes_demo.json`.

---

### High Priority

**H1. speedrun_experiment.py is 5436 lines**
- **Location**: `training/speedrun_experiment.py`
- **Issue**: Single file with 170 functions. Hard to navigate, test, and maintain.
- **Impact**: Technical debt; slows onboarding and debugging.
- **Mitigation**: Refactor into modular components (runner, viewer, metrics, device utils). Tracked in BACKLOG.md.

**H2. Evaluator training incomplete**
- **Location**: `models/evaluator/train_qr.py`
- **Issue**: Skeleton only; QR-DQN distributional head not fully implemented.
- **Impact**: Cannot bootstrap RL with evaluator-based lookahead or dense reward shaping.
- **Mitigation**: Complete implementation and train on RAM-labeled corpus.

**H3. Seed registry empty**
- **Location**: `envs/retro/seeds/`
- **Issue**: No actual savestates captured for deterministic seed control.
- **Impact**: Cannot run reproducible evaluation sweeps or determinism tests.
- **Mitigation**: Capture 120 seeds per level with pill sequence and virus grid hashes.

**H4. No integration tests C++ ↔ Python**
- **Location**: `game_engine/engine_shm.py`, `tests/`
- **Issue**: Python shared memory interface exists but no tests verifying end-to-end behavior.
- **Impact**: Bugs in IPC could silently corrupt training data.
- **Mitigation**: Add integration tests that step the engine and verify state transitions.

---

### Medium Priority

**M1. Some documentation drift**
- **Location**: `docs/TASKS_NEXT.md`, various
- **Issue**: Some task items reference "next" work that was completed months ago.
- **Impact**: Confusing for new contributors.
- **Mitigation**: Sync docs with current implementation status. Low effort.

**M2. Type hints incomplete**
- **Location**: Various Python files
- **Issue**: Many internal functions use `Dict[str, Any]` or lack hints entirely.
- **Impact**: IDE support reduced; mypy cannot catch type errors.
- **Mitigation**: Gradual improvement; prioritize public APIs.

**M3. Reward config lacks schema validation**
- **Location**: `envs/retro/drmario_env.py` L305-336
- **Issue**: JSON reward configs are parsed without validation.
- **Impact**: Typos in config keys fail silently.
- **Mitigation**: Add JSON schema or pydantic validation.

---

### Low Priority / Notes

**L1. Multiple Python venv directories**
- **Observation**: `.venv`, `.venv-py313`, `.venv-py313t` all exist.
- **Impact**: None if gitignored, but could confuse contributors.
- **Note**: Document which venv to use in AGENTS.md.

**L2. Screenshot in repo root**
- **Location**: `Screenshot 2025-11-04 at 15.31.35.png` (401KB)
- **Issue**: Large binary file in repo root.
- **Impact**: Minor repo bloat.
- **Mitigation**: Move to `docs/` or remove if unused.

---

## Validation Plans

### C++ Engine Parity

1. Extract demo input sequences from `dr-mario-disassembly/data/drmario_data_demo_*.asm`.
2. Run same inputs through C++ engine.
3. Assert board state matches at each frame.
4. Automate as CI test.

---

## 2025-12-18 – Demo Parity Port: New Risks

**R1. Demo-input preroll is transcript-start dependent**
- **Concern**: `DEMO_INPUT_PREROLL_FRAMES = 7` is calibrated to the specific ground-truth capture start used for `data/nes_demo.json`.
- **Risk**: A new ground-truth capture (different recorder start frame) could require a different preroll, and parity would appear “broken” even if the engine is correct.
- **Mitigation**: Document the invariant (“align engine demo input pointer to transcript start”) and consider auto-calibrating by matching the first few non-zero input spans to the transcript.

**R2. Demo-end semantics are modeled as a mode flip**
- **Concern**: The ROM uses `flag_demo` / title-screen state to exit demo; the engine approximates this by switching `state.mode` from `MODE_DEMO` to `MODE_PLAYING` when the demo input stream ends.
- **Risk**: Downstream tooling might treat this as a gameplay mode transition rather than “demo ended”.
- **Mitigation**: Keep recorder semantics aligned (stop before appending the final frame) and consider adding an explicit `demo_active`/`demo_ended` flag in shared memory if needed by training/eval code.

**R3. Engine demo TUI relies on terminal raw mode**
- **Concern**: `tools/engine_demo_tui.py` uses `termios`/cbreak input to read single-key presses, which depends on running in a real TTY.
- **Risk**: In non-interactive environments (pipes, CI, some IDE consoles), the UI would otherwise appear “hung”.
- **Mitigation**: The TUI detects non-TTY stdin and runs in a non-interactive autoplay mode (exits on demo end / `--max-frames`).

**R4. Tracked engine binaries can drift from sources**
- **Concern**: The repo currently tracks `game_engine/drmario_engine` and `game_engine/*.o` alongside the C++ sources.
- **Risk**: If sources change without rebuilding (or rebuild artifacts aren’t updated in git), tools/tests that execute the shipped binary can appear to “hang” (e.g., manual-step timeouts / demo stalls) in ways that look like a logic regression.
- **Mitigation**: Keep build artifacts updated when changing engine sources (or, longer-term, stop tracking build outputs and build in CI / on demand). When debugging “engine step timed out”, first run `make -C game_engine clean && make -C game_engine`.

### Placement Policy Correctness

- Already covered by 12 unit tests in `tests/test_placement_policy.py`.
- All passing as of 2025-12-16.

### Overall Test Health

- **Current**: 64 tests, all passing.
- **Coverage gaps**: C++ engine, integration tests, slow evaluator training.

---

## 2025-12-18 – Macro Placement Planner: New Risks

**R5. Decision-point detection depends on ZP state + spawn counter**
- **Concern**: `DrMarioPlacementEnv` gates decisions on:
  - `currentP_nextAction == pillFalling`,
  - a non-empty falling mask, and
  - `pill_counter` (`$0310`) differing from the last consumed spawn.
- **Risk**: ROM revision differences, menu/ending edge cases, or 2P modes could violate these assumptions and cause timeouts or planning during non-controllable frames.
- **Mitigation**: Add emulator-driven regression traces (spawn/settle/ending) and consider additional guards (e.g., gameplay mode `$0046`, player count `$0727`).

**R6. Snapshot vs state-tensor address mismatch (ZP vs P1 RAM mirror)**
- **Concern**: The planner reads falling pill state from zero-page current-player addresses, while `ram_to_state` renders the falling mask from P1 RAM buffers.
- **Risk**: Synthetic tests or alternate backends that don’t keep both views consistent could produce contradictory observations vs masks.
- **Mitigation**: Document the invariant and add a helper to mirror ZP → P1 fields in test fixtures/backends when needed.

**R7. Planner performance and native/Python divergence risk**
- **Concern**: Frame-accurate reachability is inherently heavier than geometry-only BFS because it includes ROM counters (gravity, DAS, parity).
- **Risk**: The Python reference implementation is too slow for training; the native accelerator could drift from ROM semantics or the Python oracle over time.
- **Mitigation**:
  - Use the native accelerator (`reach_native/drm_reach_full.c` via `envs/retro/reach_native.py`) for training runs.
  - Keep `envs/retro/fast_reach.py` as the behavioural oracle and add small parity tests (start with “immediate lock” cases).
  - Add unit coverage that applies `reconstruct_actions()` outputs via `simulate_frame()` and asserts the locked terminal pose is actually reached (guards against drift between the packed BFS and the oracle stepper).
  - Provide a benchmark harness (`python -m tools.bench_reachability`) to catch performance regressions early.

**R13. Native reachability build + portability**
- **Concern**: The native planner is a locally-built shared library (clang toolchain, per-arch dylib/so).
- **Risk**: Missing builds silently fall back to Python (very slow), or platform-specific build issues block training.
- **Mitigation**: Document `python -m tools.build_reach_native` in quick-start docs and surface the active backend via `placements/reach_backend` in `info`/debug UI.

**R24. Native reachability frontier batching / early-stop correctness**
- **Concern**: The optimized native BFS uses frontier aggregation (batching x positions per counter-state key) and early-stop pruning (timer-free geometric reachable terminal set).
- **Risk**: A subtle under-approximation bug could silently drop feasible macro actions (empty/too-small feasible masks), or generate scripts that don’t execute correctly, leading to rare pose mismatches and unstable learning.
- **Mitigation**:
  - Keep `simulate_frame()` in `envs/retro/fast_reach.py` as the behavioural oracle and add replay-style unit tests (native plan → simulate_frame → verify lock pose).
  - Use `placements/pose_mismatch_*` JSONL logging to capture any remaining divergences post-hoc.
  - Enable `DRMARIO_REACH_STATS=1` when profiling/benchmarking to surface state/edge counts and ensure expected scaling.

**R8. Symmetry reduction for identical colors changes the effective action space**
- **Concern**: The macro env masks out H-/V- when the capsule colors match (orientation duplicates).
- **Risk**: Downstream code that assumes “always 4 orientations” could log misleading action stats if it doesn’t account for mask structure.
- **Mitigation**: Keep masking entirely within `placements/feasible_mask`/`legal_mask` (policy is already mask-aware) and surface option counts via `placements/options`.

**R25. Feasibility mask planes in observations can bias learning toward planner artifacts**
- **Concern**: `bitplane_reduced_mask` injects `placements/feasible_mask` into observation planes (`feasible_o0..feasible_o3`) at decision points.
- **Risk**:
  - Policies may overfit to idiosyncrasies of the planner/mask generator instead of learning a robust value function from board state alone.
  - Future training against a different backend (e.g. the C++ engine) must produce an identical feasibility mask to preserve policy behavior.
- **Mitigation**:
  - Keep action masking as the hard constraint (always apply feasibility mask to logits); treat mask planes as optional auxiliary input.
  - Run periodic ablations/evals on `bitplane_reduced` (no mask planes) to ensure the policy still learns sensible board features.
  - Maintain strict oracle parity tests for the planner so injected mask planes remain faithful to emulator dynamics.

**R26. Color encoding mismatch between RAM/board bytes and policy indices**
- **Concern**: Dr. Mario uses two natural encodings:
  - raw NES color bits (yellow=0, red=1, blue=2) in RAM/board bytes, and
  - canonical plane/policy indices (red=0, yellow=1, blue=2) used by `color_{r,y,b}` planes and policy embeddings.
- **Risk**: If a caller mixes these conventions (e.g., treats `preview_pill.first_color` as canonical when it is raw), the agent can receive inconsistent conditioning and learning can silently degrade.

---

## 2025-12-20 – Runner Logdir + Curriculum TUI

**R27. Default logdir now creates per-run subdirectories**
- **Concern**: `python -m training.run` now writes to `cfg.logdir/<run_id>` by default.
- **Risk**: Tools/scripts that assume a fixed path (e.g. `runs/smdp_ppo/metrics.jsonl`) will stop finding metrics/checkpoints automatically.
- **Mitigation**: Use `--logdir` to opt out; keep helper tools flexible (accept a run directory and pick the newest run when appropriate).

**R28. Event payload now includes nested curriculum snapshots**
- **Concern**: `update_end` events for `ppo_smdp` include a `curriculum` dict payload for UIs.
- **Risk**: Any downstream handler that assumes all payload values are numeric scalars could break.
- **Mitigation**: Keep logged scalar metrics unchanged; UI handlers must treat `curriculum` as optional and type-check it before use.
- **Mitigation**:
  - Treat UI/board-byte fields as “raw” and policy conditioning vectors as “canonical”; convert explicitly.
  - Prefer decoding policy-conditioning colors from source-of-truth RAM bytes (or from the RAM offsets spec) rather than from representation-dependent state tensors.
  - Add small unit tests around the conversions when adding new representations/backends.

**R14. Spawn-blocked “dead decisions” can produce empty feasible masks**
- **Concern**: Some spawns are immediately blocked (capsule locks offscreen/top-out before any actionable input window). The reachability planner correctly reports `placements/options == 0` for in-bounds macro placements.
- **Risk**: If the macro env returns `placements/needs_action == True` with an all-false feasible mask, policies can enter an infinite invalid-action loop (no emulator progress), freezing the debug UI and collapsing training metrics.
- **Mitigation**: `DrMarioPlacementEnv` now treats `options==0` decision points as non-decision frames and continues stepping NOOP until the env transitions (lock/top-out) or a later controllable spawn appears. Regression test: `tests/test_placement_env_no_feasible_actions.py`.

**R15. Spawn-latching depends on `pill_counter` semantics**
- **Concern**: The spawn-latched macro env relies on `pill_counter` (`$0310`) being a reliable “new pill appeared” signal and on it advancing *before* the first controllable `pillFalling` frame we want to treat as a decision.
- **Risk**: If a core/ROM revision reports `pill_counter` differently (e.g., increments earlier/later than expected, or is temporarily stale during reset/auto-start), the wrapper could (a) surface a decision late, shrinking feasibility masks, or (b) fail to surface a decision, hitting `max_wait_frames` timeouts.
- **Mitigation**: Keep decision detection conjunctive (`pillFalling` + falling mask + spawn counter change) and validate via recorded emulator traces; if needed, extend gating with additional “entered controllable state” transitions or gameplay-mode guards.

---

## 2025-12-18 – Unified Runner + Debug TUI: New Risks

**R9. Interactive debug UI uses raw terminal mode**
- **Concern**: `training/ui/runner_debug_tui.py` uses `termios`/cbreak mode and assumes stdin is a real TTY.
- **Risk**: In some environments (CI, IDE consoles, pipes) controls won’t work; a naive implementation can appear “hung”.
- **Mitigation**: The UI detects non-TTY stdin and disables controls; training remains stoppable via normal process interrupts.

**R10. Training runs in a background thread in debug mode**
- **Concern**: `training/run.py --ui debug` runs the training loop in a worker thread so the UI can remain responsive.
- **Risk**: Some libraries (certain GUI backends, GPU drivers, or non-thread-safe env backends) may behave unexpectedly in threads.
- **Mitigation**: Keep the default (`--ui headless` / `--ui tui`) in the main thread; treat `--ui debug` as a debugging-only mode. If needed, add a “main-thread training + step-gate polling” mode later.

**R11. `placements/tau` is only available for macro envs**
- **Concern**: Frame-accurate throttling depends on `placements/tau`.
- **Risk**: Controller envs will be throttled as “1 frame per step”, which is correct only when `frame_skip==1`.
- **Mitigation**: Document/assume `frame_skip==1` for debug playback; if we add configurable frame-skip, extend the wrapper to read that from info.

**R12. Libretro backend performance is dominated by per-frame video copies**
- **Concern**: `envs/backends/libretro_backend.py` copies the full 240×256 frame buffer in the libretro video callback on every `retro_run()` (even for state-mode runs that only need RAM).
- **Risk**: Reset/start sequences and training throughput can become effectively near-realtime (seconds per reset), which makes RL experiments impractically slow and can hide logic issues behind “it’s just slow”.
- **Mitigation**: Add an option for “RAM-only” stepping (skip video copies unless explicitly requested), and ensure reset/start sequences don’t pay video costs when `obs_mode=state` and rendering/video logging are disabled.

**R16. Episode length reporting depends on `placements/tau`**
- **Concern**: The vector-env wrapper injects `info["episode"]["l"]` as emulated frames using `placements/tau` when present.
- **Risk**: If an env forgets to emit `placements/tau` (or emits an incorrect value), episode lengths will be misreported and “time-to-clear” metrics can become misleading even if rewards/training are otherwise fine.
- **Mitigation**: Keep `placements/tau` required for macro envs (tests already cover presence for placement env steps) and add lightweight sanity checks (e.g., `tau >= 1`, `tau` not exploding) when debugging. Consider adding an integration test that asserts monotonic frame counters for a short recorded trace.

**R17. Scripted curriculum depends on RAM patching support**
- **Concern**: The curriculum uses `write_ram` to patch the bottle buffer (reduce virus count) at reset time.
- **Risk**: Backends without `write_ram` (or cores that don’t expose RAM writes) will silently ignore the patch, yielding unexpected virus counts and invalid curriculum statistics.
- **Mitigation**: Treat libretro as the required backend for curriculum runs; surface `synthetic_virus_target`/`curriculum_level` in `info` and the debug UI; consider adding a reset-time assertion (debug-only) that the derived virus count matches the target.

**R18. “Negative levels” can confuse logging and tooling**
- **Concern**: `env.level` is overloaded for curriculum staging, but `info["level"]` is often overwritten with `level_state` (0..20) from RAM.
- **Risk**: External tooling may read `level` and miss the curriculum stage; comparisons across runs can become ambiguous.
- **Mitigation**: Log and display `curriculum_level`/`curriculum/env_level` explicitly and document that `level_state` is the ROM level.

**R19. Curriculum wrapper assumes Gymnasium `autoreset_mode=NextStep`**
- **Concern**: `CurriculumVecEnv` sets per-env `level` immediately after observing a terminal step, relying on Gymnasium’s default `autoreset_mode=NextStep` to apply it on the next call to `step()`.
- **Risk**: If autoreset semantics change (or a custom vector env resets immediately), the next episode may start with the wrong curriculum level.
- **Mitigation**: Keep using Gymnasium’s default vector envs (which default to `NextStep`), and if this becomes configurable, explicitly set/validate the autoreset mode in the env factory.

**R20. Curriculum match-count success detection relies on ROM clear-animation tile codes**
- **Concern**: The synthetic curriculum stages `-15..-4` (0 viruses; terminate after N match events) detect matches by scanning the bottle RAM for the ROM’s clear-animation tile type codes (`CLEARED_TILE` / `FIELD_JUST_EMPTIED`) and counting *rising edges* where `tiles_clearing` crosses the ≥4 threshold.
- **Risk**: If a different ROM revision/core uses different marker codes or delays updating the bottle buffer, the terminal condition could misfire or lag.
- **Mitigation**: Require at least 4 clearing-marked tiles (`tiles_clearing >= 4`), explicitly exclude `FIELD_EMPTY == 0xFF` when matching `FIELD_JUST_EMPTIED` (empty tiles share the same high nibble `0xF0`), and gate the scan on `gameplay_active` to avoid menu/reset stale RAM. Keep the logic localized (single helper in `DrMarioRetroEnv`) so it’s easy to adapt per-ROM if needed.

**R21. Placement verification depends on a stable “lock boundary” signal**
- **Concern**: `DrMarioPlacementEnv` captures `placements/lock_pose` by tracking the last valid falling-pill pose while `currentP_nextAction == nextAction_pillFalling`, then freezing it when we leave that state (or when the spawn counter advances).
- **Risk**: If a ROM/core changes the `currentP_nextAction` semantics (or briefly toggles it during non-lock transitions), the captured pose could be stale or missing, which would mask planner/executor mismatches.
- **Mitigation**: Keep verification diagnostic-only. If this occurs in practice, extend capture to also use `lock_counter` transitions and/or bottle-buffer diffs (detect the two newly-written pill tiles) as alternate “lock boundary” detectors.

**R22. Pose mismatch logging overhead / log growth**
- **Concern**: Diagnosing rare mismatches benefits from rich dumps (feasible masks, costs, bottle grids, scripts).
- **Risk**: If mismatches become frequent (early development), JSONL logs can grow quickly and I/O may impact throughput; per-frame trace capture adds overhead if enabled.
- **Mitigation**: Log only on mismatch; keep trace capture behind `DRMARIO_POSE_MISMATCH_TRACE`; allow disabling/redirecting logs via `DRMARIO_POSE_MISMATCH_LOG` and capping with `DRMARIO_POSE_MISMATCH_LOG_MAX`.

**R23. Rotation edge semantics increases planner state space**
- **Concern**: Correct rotation modelling requires tracking whether A/B was held on the previous frame (`btnsPressed` edge semantics).
- **Risk**: The Python reachability backend’s BFS state space grows (~3×), making tests and any accidental Python-backend training noticeably slower.
- **Mitigation**: Prefer the native reachability backend for training; keep Python as the behavioural oracle + unit-test target. Consider reducing test `max_frames` where possible and/or adding a focused micro-benchmark to catch accidental Python-backend regressions.

**R25. Bitplane state representation mixes HUD preview into board planes**
- **Concern**: The `bitplane` representation encodes the next-pill preview (and the falling pill) by writing into the shared color planes, plus a `preview_mask`.
- **Risk**: If preview cells overlap with the falling pill spawn region (or if downstream code assumes the top rows are “pure bottle”), the observation can become multi-hot in color channels at a single cell and confuse policies/debugging.
- **Mitigation**: Keep preview decoding accurate (rotation inferred from `preview_mask`), and if this becomes an issue, introduce a dedicated policy observation wrapper that (a) strips preview/falling from the board planes and (b) passes preview pill colors separately as scalars.

**R26. RNG seeding boundary is ROM/core sensitive**
- **Concern**: Libretro parity seeding relies on observing the ROM mode transition to `initData_level` (`$0046 == 0x03`) during auto-start, then writing seed bytes to `$0017/$0018` immediately.
- **Risk**: Different ROM revisions/cores (or changes to auto-start timing) could skip the observable `0x03` boundary or move it earlier/later, causing the env to miss the intended seed point and fall back to menu-time seeding (or no seeding).
- **Mitigation**: Keep the logic localized in `DrMarioRetroEnv._run_start_sequence`, support a strict mode (`rng_seed_strict`) for parity harnesses, and validate regularly with `tools/ghost_parity.py` across multiple levels/seeds.

**R27. Auto-start sync uses `waitFrames` ($0051) heuristics**
- **Concern**: The “stable checkpoint” for ghost parity uses `waitFrames > 0` as the first post-virus-placement sync condition.
- **Risk**: If a future ROM/core changes when `$0051` is written, the sync could stop too early (mid init) or too late (past intro), breaking parity resets.
- **Mitigation**: Keep `start_sync_wait_frames` configurable (exact value vs first-nonzero) and cross-check with additional guards if needed (e.g., `mode==0x08` and board virus count stable).

---

## 2025-12-19 – State-Mode Perf Optimizations: Lazy RGB Frames

**R28. Lazy RGB frame refresh can surprise internal consumers**
- **Concern**: In state-mode, `DrMarioRetroEnv.step()` no longer refreshes `_last_frame` every frame; frames are marked dirty and refreshed in `render()` unless `obs_mode="pixel"`.
- **Risk**: Any tooling that reads `_last_frame` directly (without calling `render()`) may see stale frames.
- **Mitigation**: Treat `_last_frame` as internal; use `render()` (UI-driven, low Hz) or enable `obs_mode="pixel"` for frame-dependent tooling. Keep board/debug UIs RAM-derived by default.

---

## 2025-12-19 – Multi-Env C++ Backend: Scaling + Debug UI Risks

**R29. Process explosion with `AsyncVectorEnv` + per-env engine subprocess**
- **Concern**: Current `cpp-engine` backend uses a subprocess per env. `AsyncVectorEnv` also uses worker processes.
- **Risk**: Large `num_envs` can create “processes × processes” overhead (scheduler contention, file descriptor limits, slower-than-linear scaling, harder failure recovery).
- **Mitigation**: Benchmark scaling early; cap default `num_envs` conservatively; add smoke tests for clean shutdown/no SHM leaks. Long-term: consolidate to a multi-instance engine process or an in-process engine library backend (see backlog P0.4).

**R30. Cross-process payload dominates async scaling**
- **Concern**: `AsyncVectorEnv` must serialize observations and `info` across processes.
- **Risk**: Debug-heavy fields (`raw_ram`, large plane-name maps, verbose logs) can erase the speedup of parallel stepping even if the engine is fast.
- **Mitigation**: Provide explicit “training payload” vs “debug payload” modes (disable heavy fields by default in headless); keep observation tensors minimal (reduced bitplanes); measure IPC cost in a dedicated benchmark harness.

**R31. Env-count hotkeys can leak subprocesses or corrupt rollouts if not restart-safe**
- **Concern**: Debug TUI wants `[`/`]` to change `num_envs`.
- **Risk**: Attempting dynamic resizing in-place risks orphaned worker processes/SHM files or PPO buffer shape mismatches; even restart logic can leak if teardown isn’t robust.
- **Mitigation**: Make env-count changes “restart-only” and implement a controlled shutdown path (join training thread, close vec env, confirm engine subprocess exit, delete temp SHM files). Add an integration test that restarts N times and asserts no growth in live subprocesses.

**R32. Multi-env accounting errors can silently break learning and scaling metrics**
- **Concern**: Macro-step envs advance a variable number of frames (`placements/tau`).
- **Risk**: If we compute SPS from “steps” instead of frames, or mishandle per-env terminal boundaries, reported throughput and learning curves can be misleading (and PPO updates can become numerically degenerate).
- **Mitigation**: Treat `sum(tau_i)` as the canonical frame counter, log both frames/sec and decisions/sec, and add deterministic multi-env regression tests (num_envs=1 vs isolated num_envs>1 with fixed seeds).

**R33. Multi-board debug rendering can become the bottleneck**
- **Concern**: A grid view showing many env boards can require lots of terminal rendering work.
- **Risk**: Debug UI FPS may collapse and distort perceived simulator throughput.
- **Mitigation**: Keep render Hz capped (auto-downshift with env count), use compact mini-boards for summary view, avoid per-frame layout recomputation, and compute scaling metrics from training-thread counters (not from UI render loop timing).

**R34. Decision-buffer wrap could corrupt per-env GAE**
- **Concern**: SMDP-PPO now computes GAE per env using stored `env_id` sequences.
- **Risk**: If the ring buffer wraps within a rollout, per-env temporal order could be scrambled and advantages become incorrect.
- **Mitigation**: Keep buffer capacity > decisions_per_update + num_envs and clear after each update; add debug asserts if needed.

**R35. Debug UI exits can leave orphan engine subprocesses**
- **Concern**: The debug UI runs a separate training thread and the C++ backend spawns one subprocess per env.
- **Risk**: If the UI exits via an exception or Ctrl+C before cleanup, engine subprocesses can survive and accumulate.
- **Mitigation**: Ensure the debug runner always calls `session.stop()` in a `try/finally` block; continue to prefer `sync` vectorization in debug mode. Hard kills (SIGKILL) can still leak; use Activity Monitor or `pkill drmario_engine` as a last resort.

**R36. Batched stepping fast path only covers a subset of per-frame reward/info**
- **Concern**: The cpp-engine fast path in `DrMarioPlacementEnv` avoids per-frame `DrMarioRetroEnv.step()` calls and instead aggregates a macro reward from engine counters + adjacency-at-lock shaping.
- **Risk**: Features that depend on per-frame bookkeeping (e.g., `matches_completed` edge detection via `tiles_clearing`, height-penalty deltas, action penalties, pill-place bonuses, potential shaping) won’t match the slow path; enabling those knobs could silently change learning dynamics.
- **Mitigation**: Fast mode auto-disables when unsupported reward knobs are enabled or potential shaping is on, and can be forced off via `DRMARIO_CPP_FAST=0`. Extend support deliberately (with regression tests) before relaxing the guardrails.

**R37. Engine/Python SHM protocol mismatch can hang batched runs**
- **Concern**: Batched runs rely on matching `DrMarioState` layout and semantics between the C++ engine binary and Python `ctypes` struct.
- **Risk**: Running an old engine with a new Python driver (or vice versa) can cause run requests to never ack, leading to timeouts/hangs and confusing perf regressions.
- **Mitigation**: Keep `static_assert(sizeof(DrMarioState))` in C++ and `ctypes.sizeof` checks + demo parity tests in Python. Surface timeouts with engine stdout/stderr, and keep the per-frame manual-step path available as a fallback.

**R38. Reset-time per-frame stepping can timeout under AsyncVectorEnv load**
- **Concern**: `AsyncVectorEnv` autoresets call `env.reset()` in worker processes, which historically advanced to the first decision point by stepping per frame in Python.
- **Risk**: Under high env counts, per-frame lockstep stepping can time out (engine not scheduled within the per-step timeout) and/or burn CPU, causing instability and poor scaling.
- **Mitigation**: Use `cpp-engine.run_until_decision()` + `sync_after_backend_run()` during `DrMarioPlacementEnv.reset()` to fast-forward to the first decision without per-frame Python stepping.

**R39. Backend failures in state-mode can surface as missing `_state_cache`**
- **Concern**: The placement env relies on `DrMarioRetroEnv._state_cache` to build planner snapshots.
- **Risk**: If a backend reset/step fails and the env silently falls back to mock dynamics, `_state_cache` can be `None`, leading to confusing crashes (especially during `AsyncVectorEnv` autoreset on the next `step()` call).
- **Mitigation**: Record last backend error (`_backend_last_error`), retry cpp-engine resets by restarting the backend, and make `DrMarioPlacementEnv` return a truncated step (instead of raising) when state is unavailable so the vector env can recover via autoreset.

**R40. Multiprocessing spawn can break non-installed imports**
- **Concern**: On macOS, multiprocessing spawn workers may not inherit the repo root on `sys.path`.
- **Risk**: Imports from non-installed top-level packages (e.g., `game_engine.*`) can fail only in worker processes, causing flaky multi-env behavior that’s hard to reproduce in single-env runs.
- **Mitigation**: Keep cpp-engine driver code (SHM bindings) under installed packages (`envs.backends.*`) and add an async autoreset regression test to exercise spawn behavior.

**R41. `AsyncVectorEnv.close()` can hang if a worker dies mid-message**
- **Concern**: Gymnasium `AsyncVectorEnv` may enter a state where `_poll_pipe_envs()` reports ready but `pipe.recv()` blocks (e.g., partial IPC payload after a crash).
- **Risk**: Training/debug sessions can become hard to stop; Ctrl+C can produce confusing shutdown traces and leave orphaned processes.
- **Mitigation**: On close, force-terminate `AsyncVectorEnv` when it has a pending call, skipping `step_wait` and prioritizing teardown over graceful completion (`training/envs/dr_mario_vec.py`).

**R42. `ln_hop_back` pass thresholds can still drift toward “perfect or stall”**
- **Concern**: The hop-back thresholds use `1 - exp(-m·k)` (with `m=pass_ramp_exponent_multiplier`) and can still approach 1 as k increases.
- **Risk**: If `m` is too large (or k too high), later hop-back stages can become indistinguishable from “must be perfect” in finite windows and stall progress on noisy tasks.
- **Mitigation**: Cap k (`ln_hop_back_max_k`), lower `pass_ramp_exponent_multiplier`, and/or lower `confidence_sigmas`. Monitor `curriculum/stage_index`, `curriculum/success_threshold`, and `curriculum/confidence_lower_bound` in the TUI.

**R43. Time/spawn task budgets depend on wrapper counters (τ + spawn consumption)**
- **Concern**: Time-based curricula enforce budgets via `DrMarioPlacementEnv` counters (`task/frames_used`, `task/spawns_used`) rather than ROM-native timers.
- **Risk**: If τ accounting changes (e.g., reset-time fast-forward, pre-decision waits) or spawn consumption logic diverges across slow/fast paths, budgets could:
  - misclassify “success under budget” vs “over budget” (skewing curriculum signals), and/or
  - distort the time-goal terminal reward shaping applied on clear.
- **Mitigation**: Keep budgets based on actual frame deltas, increment spawn counts only when a spawn is explicitly consumed, and maintain parity across slow and cpp-engine fast paths. Cover with unit tests (`tests/test_task_budgets.py`) and log `task/*` in episode infos for spot-checking.

**R48. Time-goal terminal reward shaping changes reward sign/scale on clears**
- **Concern**: Clearing over-budget now yields a negative terminal component (bounded by the topout penalty), and clearing near the goal can yield a small terminal bonus.
- **Risk**: This can interact with existing time penalties/rewards and change learning dynamics (e.g., agents preferring “safe clear later” vs “riskier fast clears”), especially when budgets are tight or noisy.
- **Mitigation**: Monitor `task/budget_exceeded*`, `task/budget_delta`, and `task/budget_terminal_bonus_*` during training; tune budget schedules (`time_budget_*`) and ramp (`pass_ramp_exponent_multiplier`) if over-budget clears dominate or learning destabilizes.

**R49. EMA + pseudo-count Wilson LB is an approximation**
- **Concern**: The curriculum gate now uses an exponentially weighted moving average and computes a Wilson-style lower bound from fractional pseudo-counts.
- **Risk**: The resulting “LB” is not a strict frequentist confidence bound under a true i.i.d. Bernoulli model, and may be over/under-conservative depending on non-stationarity and correlation (common in on-policy RL).
- **Mitigation**: Treat the bound as a stable heuristic gate; tune `confidence_ema_half_life_episodes`, `confidence_min_effective_episodes`, and `min_stage_decisions` based on observed stall/flip behavior. Consider periodic fixed-seed evals as a higher-integrity certification check if needed.

**R50. AsyncVectorEnv info typing can crash on `None`**
- **Concern**: Gymnasium `AsyncVectorEnv` merges per-env infos into typed arrays.
- **Risk**: Returning `None` for a key that is numeric in other envs (or at other times) can crash training with `TypeError: int() argument must be ... not 'NoneType'`.
- **Mitigation**: Omit optional keys entirely when unset (instead of returning `None`) so the vector wrapper can use its presence-mask mechanism.

**R51. PID-file cleanup could terminate the wrong process if pids are reused**
- **Concern**: Run shutdown now attempts to kill leftover engine pids recorded in `$logdir/engine_pids`.
- **Risk**: If a pidfile persists across a crash and the OS later reuses that pid for a different process, naive cleanup could kill an unrelated process.
- **Mitigation**: Scope pidfiles to a run-local directory and verify `ps` command name contains `drmario_engine` before sending signals; keep cleanup best-effort and skip on validation failure.

**R44. Confidence-based rolling windows can become very large for strict targets**
- **Concern**: Curriculum advancement now uses a σ-level one-sided Wilson lower bound (`LB > target`) with window sizes derived from a “near-target” assumption.
- **Risk**: For targets close to 1 (especially `ln_hop_back` with large k), computed windows can become large, making stages slow to graduate and potentially amplifying non-i.i.d. effects (policy changes over time).
- **Mitigation**: Cap hop-back k (`ln_hop_back_max_k`), tune `confidence_sigmas`/targets, and monitor `curriculum/confidence_lower_bound` + `curriculum/window_*` in the TUI. Use `tools/report_curriculum.py --confidence-table` to sanity-check implied window sizes.

**R45. Stage-token mismatches can contaminate stage-local hop-back stats**
- **Concern**: `ln_hop_back` advancement uses stage-local rolling windows keyed by `stage_idx` and relies on a `stage_token` captured when an env’s episode starts.
- **Risk**: If `stage_token` is missing or incorrectly propagated (especially when multiple envs terminate in the same vec-step and the curriculum advances mid-loop), outcomes can be counted toward the wrong stage and cause premature advancement or stalls.
- **Mitigation**: `CurriculumVecEnv` stores `_env_stage_tokens` at assignment time and passes them into `note_episode`; `LnHopBackCurriculum.note_episode` refuses to advance the current stage when the token doesn’t match the active `stage_idx`.

**R46. Best-times sqlite DB can lock or slow under concurrent runs**
- **Concern**: The best-times tracker writes to a sqlite file (`data/best_times.sqlite3`) as episodes complete.
- **Risk**: Multiple training runs sharing one DB path can hit lock contention (`database is locked`), slow down training, or accumulate unbounded DB size over time.
- **Mitigation**: Use WAL mode + timeouts; keep a single writer in the main process (write only from the vec wrapper); treat writes as best-effort (swallow exceptions); allow per-run/per-machine paths via `DRMARIO_BEST_TIMES_DB`.

**R47. Clamping time budgets to “best observed” floors may reduce pressure to set new records**
- **Concern**: When applying dynamic budgets, the wrapper clamps `task_max_frames/task_max_spawns` to be ≥ best-known per-level floors from the DB.
- **Risk**: Once budgets reach the current best-known value, additional speed improvements become incidental rather than explicitly required by the curriculum.
- **Mitigation**: Consider making the clamp optional (or switching to a quantile floor like p10), and track/report top-K per-seed distributions to tune targets deliberately instead of binding to the absolute minimum.
