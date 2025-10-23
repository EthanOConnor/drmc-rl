# Design Overview

This document captures the current implementation status of the Dr. Mario RL stack along with the active roadmap. It supersedes the
original 2025 planning notes and reflects the code in this repository today.

## Environment (`envs/retro`)

### Core Gymnasium Env
- `DrMarioRetroEnv` wraps libretro (default), Stable-Retro, or a deterministic mock backend.
- Observation modes:
  - **Pixel** – 4× stacked 128×128 RGB frames (`float32` in `[0,1]`).
  - **State** – 4× stacked 16×8 tensors derived from NES RAM. Two representations are available:
    - `extended` (default, 16 channels: viruses/static/falling colors, orientation, gravity, lock, level, preview metadata).
    - `bitplane` (12 channels: color bitplanes + masks for viruses/locked/falling/preview/clearing + scalars).
  - `envs/specs/ram_to_state.py` controls the representation; update `STATE_IDX` helpers when changing layout.
- Action space: discrete 10 (NES button macros mapped in `envs/retro/action_adapters.py`).
- Determinism:
  - Savestates stored under `envs/retro/seeds/` with frame offsets and metadata in `registry.json`.
  - `reset(options={"frame_offset": N, "randomize_rng": True})` advances RNG deterministically or randomizes on demand.
- Reward shaping:
  - Configurable via `RewardConfig` dataclass or external JSON. Keys include pill placement bonuses, virus/clear rewards,
    adjacency heuristics, and top-out penalties. Config is hot-reloaded if the JSON file changes at runtime.
- Auto restart:
  - Menu macros (START/LEFT) run after top-outs or clears unless `auto_start=False`.
  - Parameters can be overridden via reset options or CLI flags in the demo script.
- Risk conditioning:
  - `risk_tau` scalar appended to observations when `include_risk_tau=True`.

### Placement & Intent Translators
- `placement_wrapper.py` exposes an action space over 16×8×orient placements. The wrapper:
  - Detects spawn events using RAM latches, accounts for identical-color capsules, and caches feasible plans per spawn.
  - Drives the emulator via `PlacementPlanner` paths (`placement_planner.py`) using adaptive retry windows for mis-timed inputs.
  - Provides `info` keys for masks, costs, and timing diagnostics (`placements/*`).
- `intent_wrapper.py` offers a lighter abstraction over “sticky” movement/rotation intents using RAM-based pill tracking.
- `state_viz.py` converts state tensors to RGB for the demo viewer and Sample Factory video dumps.

### Demo / Tooling
- `envs/retro/demo.py` exercises the environment with live visualization, frame capture, auto-throttling, and hotkeys. It is the
  canonical reference for backend configuration.
- `tools/dump_ram_and_state.py` captures RAM/state tensors for debugging parity issues.

## Specs & Reverse Engineering (`envs/specs`, `re/`)
- `ram_offsets.json` encodes the NES addresses used for state extraction, gravity/lock counters, and preview metadata.
- `STATE_OBS_AND_RAM_MAPPING.md` documents the mapping between RAM and observation tensors.
- Reverse-engineering notes, disassembly pointers, and RNG breakdowns live under `docs/` and `re/`.

## Training Stack (`training`, `models`)
- Sample Factory provides PPO training with risk-conditioned observations. Launcher: `training/run_sf.py`.
- Baseline configs:
  - `training/sf_configs/state_baseline.yaml` and `pixel_baseline.yaml` – mainline actor/learner setups.
  - `*_sf2.yaml` variants target the Sample Factory v2 CLI.
  - `state_eval_bootstrap.yaml` boots a distributional evaluator head for risk-aware action selection.
- Models (policy/evaluator) currently provide scaffolding; connect them when wiring custom heads into Sample Factory.

## Evaluation Harness (`eval/`)
- Scripts ingest saved rollouts to compute time-to-clear distributions, risk metrics (mean/CVaR), and visualization plots.
- Shared metrics checklist documented in `docs/METRICS.md`.

## Data & Curriculum
- Savestates and datasets live outside git (see `data/` placeholders). Capture reproducible seeds via `envs/retro/seeds` helpers.
- Curriculum focuses on level 0 single-player clears before expanding to higher levels or 2-player modes.

## Future Work / Roadmap
- Complete evaluator head integration and wire placement-aware action scoring into the policy loop.
- Expand PettingZoo wrappers for versus mode and align reward shaping with multiplayer rules.
- Implement the C++ `sim-envpool` backend for high-FPS parity testing.
- Continue reverse-engineering to solidify RNG models (virus placement, pill RNG, gravity tables).
- Prototype the row/column/orientation placement intent layer and event-driven inference cadence sketched in the 19 Oct 2025
  `docs/EOCLabbook.md` entry to shrink the action space and reduce per-episode inference counts.

## Stand-up Reference
- macOS/Linux stand-up steps live in `docs/ENV_STANDUP_MAC_LINUX.md`.
- RNG and placement research: `docs/RNG.md`, `docs/RNG_AND_PLACEMENT_NOTES.md`.
- Reward shaping: `docs/REWARD_SHAPING.md`.
- Legal considerations: `docs/LEGAL.md`.
