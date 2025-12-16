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

**C2. C++ engine missing wall kicks**
- **Location**: `game_engine/GameLogic.cpp`, `game_engine/AGENTS.md` L59
- **Issue**: Rotation validation does not handle wall kicks (pushing piece left/right when blocked).
- **Impact**: Certain placements possible on NES are impossible in C++ engine, causing parity failure.
- **Mitigation**: Implement `pillRotateValidation` ($8E70) from disassembly.

**C3. No parity tests for C++ engine**
- **Location**: `game_engine/`, `tests/`
- **Issue**: No automated tests comparing C++ engine output vs NES demo data.
- **Impact**: Cannot verify correctness. May ship subtle bugs that affect training.
- **Mitigation**: Create parity test suite using demo mode input sequences and expected board states.

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

### Placement Policy Correctness

- Already covered by 12 unit tests in `tests/test_placement_policy.py`.
- All passing as of 2025-12-16.

### Overall Test Health

- **Current**: 64 tests, all passing.
- **Coverage gaps**: C++ engine, integration tests, slow evaluator training.
