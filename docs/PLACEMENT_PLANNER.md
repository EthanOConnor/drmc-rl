# Placement Planner: NES-Accurate Reachability + Macro-Action Execution (512-way)

This document describes the current placement planning stack for Dr. Mario.
The goal is *one decision per pill spawn*: the agent selects a macro placement,
and the environment executes the minimal-time per-frame controller script to
realise that landing under the retail NES rules.

## Overview

The macro action space is a dense 4×16×8 grid (512 actions):

- `action := (o, row, col)` where `(row, col)` is the landing cell of the **first**
  capsule half (the “1st color” in NES RAM), and `o` selects which adjacent cell
  contains the second half after the pill locks.
- The environment exposes **masks** and **costs** at each spawn:
  - `placements/legal_mask`: in-bounds actions (boundary actions masked out)
  - `placements/feasible_mask`: actions reachable under current board + counters
  - `placements/cost_to_lock`: native `cpp-pool` frames-to-lock (`0xFFFF` sentinel otherwise)
  - `placements/costs`: emulator-wrapper frames-to-lock (`inf` otherwise)
- Each `env.step(action)` advances the env until the **next decision point**,
  returning an SMDP duration `placements/tau` (frames elapsed).

## Action Space (Canonical)

Module: `drmc_rl/planning/placement_space.py`

Orientation convention (matches `drmc_rl/models/policy/placement_heads.py`):

- `o = 0 (H+)`: partner at `(row,   col+1)`
- `o = 1 (V+)`: partner at `(row+1, col)`
- `o = 2 (H-)`: partner at `(row,   col-1)`
- `o = 3 (V-)`: partner at `(row-1, col)`

The 4×16×8 grid intentionally includes boundary actions that point outside the
bottle; those are always invalid and must be masked (see `invalid_boundary_mask()`).

## Coordinate Model (NES Base-Cell Convention)

The NES stores the falling pill position as **the bottom-left cell of the pill’s
2×2 bounding box**, not “the first half’s cell”.

In planning code we work in top-origin rows (0=top, 15=bottom), but the ROM stores
`fallingPillY` as “row from bottom” (0=bottom). The planner converts between them.

Rotation codes are the ROM’s native values (0..3). Geometry depends on `rot & 1`,
but **color order depends on the full rotation code** (see ROM tables).

## Reachability Core (Frame-Accurate)

Module: `drmc_rl/planning/fast_reach.py`

This is the correctness-first reference implementation. It mirrors the ROM’s
per-frame falling-pill update order:

1) `fallingPill_checkYMove` (gravity / down-only soft drop; may lock immediately)
2) `fallingPill_checkXMove` (DAS timing via `hor_velocity`)
3) `fallingPill_checkRotate` (A/B rotation and `pillRotateValidation` quirks)

`build_reachability()` performs a bounded BFS over frame states that include the
relevant counters (`speed_counter`, `hor_velocity`, held direction, frame parity).
It records the earliest locked “terminal” nodes for each reachable `(x, y, rot)`
and parent pointers so we can reconstruct a minimal-time controller script.

## Native Acceleration (Recommended for Training)

Modules:
- Native BFS: `reach_native/drm_reach_full.c`
- Python wrapper: `drmc_rl/planning/native_reach.py`
- Build helper: `python -m tools.build_reach_native`

The Python reference BFS can be prohibitively slow when executed once per pill
spawn. The native helper implements the same per-frame step semantics and returns
for every reachable locked pose:
- minimal `tau` (frames-to-lock)
- a compact per-frame action script (indices into the stable 18-way frame action space)

Backend selection is handled by `PlacementPlanner(reach_backend=...)`:
- `auto` (default): use native if the shared library exists, else fall back to Python
- `native`: require native (raise if missing)
- `python`: force the reference implementation (useful as an oracle)

At runtime, the macro env surfaces the chosen backend as `placements/reach_backend`
in `info` (`native` or `python`).

## Current Training Path (`cpp-pool`)

For normal training, `DrMarioPlacementEnv-v0` plus `backend=cpp-pool` bypasses
Gymnasium vector wrappers and `DrMarioPlacementEnv` entirely. The active path is:

- `drmc_rl/training/envs/dr_mario_vec.py` dispatch
- `drmc_rl/training/envs/drmario_pool_vec.py` vector env
- `drmc_rl/envs/backends/drmario_pool.py` ctypes wrapper
- native `vendor/drmario_native/build/libdrmario_pool`

This path keeps the same canonical 512-way placement action contract and planner
semantics, but masks, costs, observations, and rewards come directly from the
native pool. Use the emulator wrapper path for parity/debugging.

## Planner (Spawn-Latched Masks + Plans)

Module: `drmc_rl/planning/planner.py`

Key responsibilities:

- Decode a `PillSnapshot` from RAM (base cell, rotation, counters, held buttons).
- Build a `BoardState` occupancy bitboard from the state planes (static + viruses).
- Compute `SpawnReachability` for the spawn:
  - `legal_mask`, `feasible_mask`, `costs` all shaped `(4, 16, 8)`
  - Python backend: `action_to_terminal_node` for reconstructing a plan to a specific action
- `plan_action(reach, action)` reconstructs the per-frame controller script:
  an array of holds (`left/right/down`) plus button taps (`NOOP`, `ROTATE_A/B`).

Note: placement costs are **frames-to-lock** from spawn; the SMDP step duration
`placements/tau` additionally includes any post-lock wait until the next pill is
controllable.

## Macro Environment Wrapper (SMDP)

Module: `drmc_rl/envs/libretro/placement_env.py`

`DrMarioPlacementEnv` wraps `DrMarioLibretroEnv` (state observations only) and:

- Detects decision points when:
  1) `currentP_nextAction == nextAction_pillFalling` (inputs are interpreted),
  2) a falling pill exists, and
  3) the pill spawn counter (`pill_counter`, RAM `$0310`) differs from the last
     *consumed* spawn id (spawn-latching: exactly one macro decision per pill).
- On reset and after each macro step, advances with `Action.NOOP` until a decision
  point, then emits masks/costs in `info`.
- If a “decision point” is reached but the planner reports **zero feasible
  in-bounds macro actions** (`placements/options == 0`), the wrapper treats it as
  a non-decision (spawn-blocked / immediate lock / top-out edge cases) and
  continues stepping NOOP until the env transitions or a later controllable spawn
  appears. These macro steps are flagged with `placements/no_feasible_actions`.
- On `step(action)`, executes the plan’s per-frame script, then advances until the
  next decision point and returns aggregated reward plus `placements/tau`.

Important `info` keys:

- `placements/legal_mask`, `placements/feasible_mask`, `placements/cost_to_lock` or `placements/costs`
- `placements/options` (count of feasible actions)
- `placements/spawn_id` (for caching logits “one inference per spawn”)
- `placements/tau` (SMDP duration in frames)
- `next_pill_colors` / `pill/colors` (color indices `[2]`)
