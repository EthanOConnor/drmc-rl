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

