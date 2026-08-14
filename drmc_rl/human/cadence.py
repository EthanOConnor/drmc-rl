"""Replay-validated human execution cadence for placement scripts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from drmc_rl.planning.fast_reach import FrameState, simulate_frame

NEUTRAL_ACTION = 0


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


__all__ = ["add_thinking_delay"]
