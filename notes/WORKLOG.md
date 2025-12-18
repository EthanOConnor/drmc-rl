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
