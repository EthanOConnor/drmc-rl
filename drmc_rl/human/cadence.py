"""Replay-validated human execution cadence for placement scripts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from drmc_rl.planning.fast_reach import (
    FrameState,
    HoldDir,
    Rotation,
    frame_action_from_index,
    simulate_frame,
)

NEUTRAL_ACTION = 0
DOWN_ACTION = 3


def _locks_at(
    cols: np.ndarray,
    spawn: FrameState,
    script: Sequence[int],
    *,
    speed_threshold: int,
    target: tuple[int, int, int],
) -> bool:
    state = spawn
    for action in script:
        state = simulate_frame(cols, state, int(action), speed_threshold=speed_threshold)
        if state.locked:
            return (state.x, state.y, state.rot & 3) == target
    return False


def add_thinking_delay(
    cols: np.ndarray,
    spawn: FrameState,
    script: Sequence[int],
    *,
    speed_threshold: int,
    target: tuple[int, int, int],
    requested_frames: int,
) -> tuple[np.ndarray, int]:
    """Prepend as much requested human reaction time as remains executable.

    Human replay ``tau`` includes slack beyond the fastest planner path. A
    pre-action pause is both the dominant interpretable form of that slack and
    cheap to validate. We replay every candidate against the exact frame model;
    gravity therefore clamps the delay automatically when waiting longer would
    change the chosen placement.
    """

    base = np.asarray(script, dtype=np.uint8).reshape(-1)
    wanted = max(int(requested_frames), 0)
    if wanted == 0 or base.size == 0:
        return base.copy(), 0
    columns = np.asarray(cols, dtype=np.uint16).reshape(8)
    for delay in range(wanted, 0, -1):
        candidate = np.concatenate((np.zeros(delay, dtype=np.uint8), base))
        if _locks_at(
            columns,
            spawn,
            candidate,
            speed_threshold=int(speed_threshold),
            target=(int(target[0]), int(target[1]), int(target[2]) & 3),
        ):
            return candidate, delay
    return base.copy(), 0


def hold_soft_drop_suffix(
    cols: np.ndarray,
    spawn: FrameState,
    script: Sequence[int],
    *,
    speed_threshold: int,
    target: tuple[int, int, int],
) -> tuple[np.ndarray, bool]:
    """Hold Down after the final steering input when that remains exact.

    Reachability uses frame-level actions and may express the cartridge's
    every-other-frame soft-drop gate as Down taps. Humans hold the button. We
    preserve all reaction time and every lateral/rotation frame, replace only
    the terminal no-steering suffix, then replay the result before returning
    it. A route that descends before a late weave therefore remains untouched.
    """

    base = np.asarray(script, dtype=np.uint8).reshape(-1)
    first_action = next((index for index, action in enumerate(base) if action != 0), None)
    if first_action is None:
        return base.copy(), False
    last_steering = max(
        (
            index
            for index, action_index in enumerate(base)
            if (action := frame_action_from_index(int(action_index))).hold_dir
            is not HoldDir.NEUTRAL
            or action.rotation is not Rotation.NONE
        ),
        default=-1,
    )
    drop_start = max(first_action, last_steering + 1)
    candidate = base.copy()
    candidate[drop_start:] = DOWN_ACTION
    columns = np.asarray(cols, dtype=np.uint16).reshape(8)
    if _locks_at(
        columns,
        spawn,
        candidate,
        speed_threshold=int(speed_threshold),
        target=(int(target[0]), int(target[1]), int(target[2]) & 3),
    ):
        return candidate, True
    return base.copy(), False


__all__ = ["add_thinking_delay", "hold_soft_drop_suffix"]
