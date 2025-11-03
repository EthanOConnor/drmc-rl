# Placement Planner: Fast Reachability + Controller Synthesis

This document describes the current placement planning stack used by the Dr. Mario
placement action space. It summarizes the fast path (spawn-time feasibility and
costs), the single-target controller synthesis, and the available timing/diagnostic
instrumentation.

## Overview

We separate two concerns:

- Spawn-time options (microseconds): compute where the current pill can land and
  the earliest frame cost to each landing. We do this via a compact 512-state BFS
  over (x, y, orient) – no time-expansion, no button-combo branching.

- Final controller synthesis (milliseconds): for the selected action (one of the
  464 grid-directed edges), reconstruct the optimal route via parent pointers and
  synthesize a frame-by-frame controller script. We do this only for the chosen
  action to avoid Python overhead across all actions.

## Fast Reachability (reach512)

Module: `envs/retro/reach512.py`

- State space: 8×16×4 = 512 states (x∈[0..7], y∈[0..15], orient∈{H+,V+,H−,V−}).
- Transitions (unit cost): down/left/right if fits; rotate CW/CCW with a left
  kick for horizontal targets.
- Data:
  - `arrival[512]` (uint8) earliest frame to each state (0xFF for unreachable).
  - `parent[512]` and `code[512]` for optimal route reconstruction.
- APIs:
  - `feasibility_and_costs(cols, sx, sy, so)` → (legal_mask, feasible_mask, costs)
  - `run_reachability(cols, sx, sy, so)` → `Reach512`
  - `reconstruct_actions_to(reach, tx, ty, to)` → [(x, y, o, code)] forward route

This pass is microsecond‑class on modern CPUs and is used in `prepare_options()`.

## Controller Synthesis (fast single-target)

Module: `envs/retro/placement_planner.py`

- `plan_action_fast(board, capsule, action)`:
  1) Calls `reach512.run_reachability()` from the pill spawn.
  2) Reconstructs the route to the target anchor (x, y, orient) using parents.
  3) Synthesizes a controller sequence:
     - DOWN → `Action.DOWN_HOLD` (hold_down=True)
     - LEFT/RIGHT → `Action.NOOP` (hold_left/right=True)
     - ROT CW/CCW → `Action.ROTATE_A/B` (+hold_left=True if a left kick happened)

We only synthesize for the selected action. The exact counter-aware planner is
kept for diagnostics; the fast path never falls back to the heavy planner.

## Counter-Aware Planner (strict single-target)

Module: `envs/retro/fast_reach.py`

- Per-frame step order Y → X → Rotate, including lock buffer via
  `(grounded, lock_timer)` so slides/tucks are supported.
- Used only when explicitly invoked for diagnostics; not used in the fast path.

## Translator + Wrapper Behavior

Module: `envs/retro/placement_wrapper.py`

- The translator is constructed with `fast_options_only=True` by default.
  - `prepare_options(force=False)` uses `reach512.feasibility_and_costs`.
  - `get_plan(action)` uses `plan_action_fast` only; no fallback to the heavy
    planner for re-plans or mismatches.
- The wrapper only requests a new decision (NOOP step + needs_action flag) if
  no plan exists for the selected action.

## Instrumentation (zero overhead when off)

- CLI flag: `--placement-debug-log`
  - Enables `[timing] …` logs throughout the wrapper and launcher.
  - Pre-step markers: `pre_env_step main|exec|stall|no-plan` are printed right
    before calling `env.step`, so stalls inside `env.step` are bracketed.

- Environment variables (strict booleans: 1/true/yes/on):
  - `DRMARIO_TIMING`: backend boundary timing in `drmario_env`:
    `[backend] pre_step …` and `[backend] post_step dt_ms=…`.
  - `DRMARIO_DEBUG_INPUT`: per-frame input lines `[input] t=… action=… buttons=…`.

When logging is off, checks are gated and return immediately.

## Pitfalls Avoided

- Forced full re-plan mid-execution: previously, some re-plan paths called the
  heavy, counter-aware `plan_all`. These have been removed/guarded; re-plans run
  the fast reach512 path only.
- Per-action controller reconstruction at spawn: removed; only the selected
  action gets a controller sequence.

## Troubleshooting Long Gaps

- Use `--placement-debug-log` and `DRMARIO_TIMING=1`.
  - If there’s a long delay after `pre_get_plan action=X`, the slow path is inside
    `translator.get_plan`. The heavy fallback is disabled by default; a gap now
    indicates external issues.
  - If there’s a long delay between `[backend] pre_step` and `post_step`, the
    backend/emulator is stalling.
  - If neither `pre_env_step …` nor backend markers appear for a long time, the
    stall is above the main loop (e.g., UI/monitor). The loop gap logs in the
    launcher will print when the loop resumes.

## Status

- Fast reachability + spawn options: microsecond‑class (reach512).
- Single-target controller synthesis: milliseconds via reach512 parents.
- Strict counter-aware path retained for diagnostics; not used in fast path.

