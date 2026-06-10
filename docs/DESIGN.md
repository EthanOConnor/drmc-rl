# Design Overview

This document describes the current `drmc-rl` architecture. It is intentionally
forward-facing: older emulator-first and EnvPool plans are retained only where
they still explain parity/debug code.

## Product Goal

Train a single-player Dr. Mario agent that chooses one macro placement per pill
spawn. The training problem is a placement SMDP, not a 60 Hz controller-policy
problem.

## Current Runtime Path

Default command:

```bash
python -m training.run --cfg training/configs/smdp_ppo.yaml --backend cpp-pool
```

Default config shape:

- `algo: ppo_smdp`
- `env.id: DrMarioPlacementEnv-v0`
- `env.backend: cpp-pool`
- `env.state_repr: bitplane_bottle_conn_mask`
- `curriculum.mode: ln_hop_back`

Data flow:

1. `game_engine/` builds `libdrmario_pool` from the `drmario-native` submodule.
2. `envs/backends/drmario_pool.py` loads the native pool with ctypes.
3. `training/envs/drmario_pool_vec.py` owns the current vector-env hot path.
4. `training/envs/dr_mario_vec.py` routes `DrMarioPlacementEnv-v0` +
   `backend=cpp-pool` directly to that hot path.
5. `training/algo/ppo_smdp.py` gathers one transition per placement decision and
   updates the policy with SMDP discounting.

`cpp-pool` does not go through the emulator backend registry and does not require
a ROM.

## Placement SMDP

Action space: a dense `4 x 16 x 8 = 512` placement grid.

- Action `a = (orientation, row, col)` selects the final locked pose.
- Feasibility comes from planner masks.
- `placements/cost_to_lock` or `placements/costs` records frames-to-lock.
- `placements/tau` records the SMDP duration through the macro step.

Current training observations are RAM/state-derived board tensors, especially
`bitplane_bottle_conn_mask`: bottle color planes, virus mask, locked-capsule
connection edges, and feasibility planes. Pill colors and preview data are
carried alongside the board.

## Policy Stack

Implemented policy modes:

- Dense, shift-score, and factorized 512-way placement heads.
- Candidate-scoring policy that packs only feasible placements and scores each
  candidate with explicit cost-to-lock features.

The default config is now candidate scoring in
`training/configs/smdp_ppo.yaml`. Use
`training/configs/smdp_ppo_heatmap.yaml` only for controlled heatmap baseline
comparisons, and `training/configs/smdp_ppo_candidate.yaml` as the verbose
annotated candidate experiment file.

## Curriculum

The current default curriculum is `ln_hop_back`.

- Synthetic negative levels cover match-count and low-virus tasks.
- Stage advancement uses EMA/Wilson-style confidence gates plus minimum decision
  budgets.
- Time budgets become active after mastery and are treated as soft goals.
- PPO rollouts stop at advancement boundaries so updates remain stage-pure.

## Backend Roles

- `cpp-pool`: default training backend.
- `cpp-engine`: older subprocess/shared-memory engine backend. Useful for
  compatibility and parity checks, not the default.
- `libretro`: emulator oracle/debug path with a legal ROM and NES core.
- `stable-retro`: legacy compatibility path.
- `mock`: dry-run and hermetic smoke behavior.

## Emulator Parity Boundary

Emulator-backed code remains important for:

- checking native-engine behavior against an oracle;
- recording and replaying traces;
- inspecting RAM and visual frames;
- validating ROM-specific timing assumptions.

It is not the onboarding or training default. See `docs/RETRO_CORE_NOTES.md` for
that lane.

## Current Non-Goals

These are not active default directions:

- EnvPool as the next simulator architecture;
- Stable-Retro-first setup;
- per-frame controller-policy training as the main learning setup;
- pixel-to-state as a prerequisite for training;
- 2-player/PettingZoo work as near-term scope.

If any of these become active again, promote them through `notes/BACKLOG.md` and
update this document at the same time.

## Design Records

Use `notes/MEMORY.md` for durable decisions, `notes/SCRUTINY.md` for risks, and
`notes/WORKLOG.md` for changes made. `docs/PROJECT_DEEP_DIVE_2026-05-07.md` is a
dated audit and handoff, not a replacement for current docs.
