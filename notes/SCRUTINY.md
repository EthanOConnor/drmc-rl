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

**R5. Decision-point detection depends on a single ZP state (`currentP_nextAction`)**
- **Concern**: `DrMarioPlacementEnv` gates decisions on `currentP_nextAction == pillFalling` and a non-empty falling mask.
- **Risk**: ROM revision differences, menu/ending edge cases, or 2P modes could violate this heuristic and cause timeouts or planning during non-controllable frames.
- **Mitigation**: Add a small battery of emulator-driven regression traces (spawn/settle/ending) and consider additional guards (e.g., gameplay mode `$0046`, player count `$0727`).

**R6. Snapshot vs state-tensor address mismatch (ZP vs P1 RAM mirror)**
- **Concern**: The planner reads falling pill state from zero-page current-player addresses, while `ram_to_state` renders the falling mask from P1 RAM buffers.
- **Risk**: Synthetic tests or alternate backends that don’t keep both views consistent could produce contradictory observations vs masks.
- **Mitigation**: Document the invariant and add a helper to mirror ZP → P1 fields in test fixtures/backends when needed.

**R7. Planner performance is Python-bound**
- **Concern**: Frame-accurate reachability uses a bounded BFS over counter-augmented states per spawn.
- **Risk**: With large `num_envs`, planning latency could dominate wall-clock training throughput.
- **Mitigation**: Benchmark regularly and plan a native (C++/Numba) port once behaviour is fully locked down; keep the Python version as the reference oracle.

**R8. Symmetry reduction for identical colors changes the effective action space**
- **Concern**: The macro env masks out H-/V- when the capsule colors match (orientation duplicates).
- **Risk**: Downstream code that assumes “always 4 orientations” could log misleading action stats if it doesn’t account for mask structure.
- **Mitigation**: Keep masking entirely within `placements/feasible_mask`/`legal_mask` (policy is already mask-aware) and surface option counts via `placements/options`.

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
