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

# Equal player/game weighted medians from the verified August sample's
# training games, excluding player fold zero. These guide path shaping;
# they are not a certified hard execution envelope or absolute rating fit.
MOTOR_FIT = {
    "id": "covered-august-motor-medians-v1",
    "source_sha256": "61a506f2340eb196c163deb2c8d0608c0a7233c576168b7d0287d55db2b15848",
    "rating_centers": [1400.0, 1800.0, 2200.0],
    "reaction_frames": [15.0, 5.0, 4.0],
    "edge_interval_frames": [5.0, 2.0, 2.0],
    "players": [9, 40, 15],
    "scripts": [316, 23881, 11864],
    "speed": 2,
    "weighting": "equal players, equal games within player, equal sampled pills within game",
    "hard_profile_verified": False,
}


def motor_parameters(rating: float) -> tuple[int, int]:
    """Return fitted reaction and edge spacing, clamped to supported cohorts."""
    return tuple(int(round(np.interp(rating, MOTOR_FIT["rating_centers"], MOTOR_FIT[name])))
                 for name in ("reaction_frames", "edge_interval_frames"))


def _movement_script(
    cols: np.ndarray, spawn: FrameState, *, target: tuple[int, int, int],
    speed_threshold: int, total_frames: int, reaction_frames: int,
    edge_interval: int, order: int, das: bool,
) -> np.ndarray | None:
    """Steer with held buttons, then time the final drop against the NES clock.

    This is a bounded alternative witness, not a reachability oracle. Tucks
    that need a different route retain the exact planner witness.
    """
    state = spawn
    result = []
    previous = 0
    changed_at = -edge_interval
    for elapsed in range(512):
        dx = target[0] - state.x
        rotations = (state.rot - target[2]) & 3
        if elapsed >= reaction_frames and not dx and not rotations and elapsed - changed_at >= edge_interval:
            return _timed_drop(cols, state, result, speed_threshold=speed_threshold,
                               target=target, requested_frames=total_frames)
        moving = bool(dx)
        rotating = bool(rotations)
        if order == 0 and rotating:
            moving = False
        if order == 1 and moving:
            rotating = False
        direction = 1 if dx < 0 else 2 if dx > 0 else 0
        rotation = 2 if rotations == 3 else 1 if rotations else 0
        hold_dir = state.hold_dir.value
        rotation_hold = state.rot_hold.value
        if elapsed < reaction_frames:
            desired = 0
        elif dx or rotations:
            if not moving:
                direction = 0
            elif not das and hold_dir == direction:
                direction = 0
            if not rotating or rotation_hold == rotation:
                rotation = 0
            desired = direction * 6 + rotation
        else:
            desired = previous
        if desired != previous and elapsed - changed_at < edge_interval:
            desired = previous
        if desired != previous:
            changed_at = elapsed
        previous = desired
        result.append(desired)
        state = simulate_frame(cols, state, desired, speed_threshold=speed_threshold)
        if state.locked:
            return np.asarray(result, dtype=np.uint8) if (state.x, state.y, state.rot) == target else None
        if state.y > target[1]:
            return None
    return None


def _timed_drop(
    cols: np.ndarray, state: FrameState, prefix: Sequence[int], *,
    speed_threshold: int, target: tuple[int, int, int], requested_frames: int,
) -> np.ndarray | None:
    """Choose from a fixed family of wait-then-drop scripts at the final column.

    The family does not depend on requested pace. Selecting its nearest lock
    time is therefore monotone, including where natural gravity limits waiting.
    Keep an already-held rotation button down; it cannot generate another press.
    """
    wait_action = state.rot_hold.value
    drop_action = DOWN_ACTION + wait_action
    probe = state
    for _ in range(33):
        if probe.locked:
            break
        probe = simulate_frame(cols, probe, drop_action, speed_threshold=speed_threshold)
    if not probe.locked or (probe.x, probe.y, probe.rot) != target:
        return None
    best = None
    waiting = state
    limit = max(512, len(prefix) + 32)
    for wait in range(limit - len(prefix) + 1):
        remaining = target[1] - waiting.y + 1
        if waiting.locked:
            drop = 0
        elif speed_threshold == 0:
            drop = remaining
        elif waiting.frame_parity == 0:
            drop = 2 * remaining - 1
        elif waiting.speed_counter + 1 > speed_threshold:
            drop = 1 if remaining == 1 else 2 * (remaining - 1)
        else:
            drop = 2 * remaining
        tau = len(prefix) + wait + drop
        if tau <= limit:
            key = (abs(tau - requested_frames), tau, wait, drop)
            if best is None or key < best:
                best = key
        if waiting.locked:
            break
        waiting = simulate_frame(cols, waiting, wait_action, speed_threshold=speed_threshold)
    if best is None:
        return None
    _, _, wait, drop = best
    return np.concatenate((np.asarray(prefix, dtype=np.uint8),
                           np.full(wait, wait_action, dtype=np.uint8),
                           np.full(drop, drop_action, dtype=np.uint8)))


def _retime_planner_route(cols, spawn, base, *, speed_threshold, target, requested_frames):
    state = spawn
    final_steering = 0
    aligned = spawn
    for index, action in enumerate(base, 1):
        previous = state
        state = simulate_frame(cols, state, int(action), speed_threshold=speed_threshold)
        if (state.x, state.rot) != (previous.x, previous.rot):
            final_steering, aligned = index, state
        if state.locked:
            break
    if not state.locked or (state.x, state.y, state.rot) != target:
        raise ValueError("base planner route does not reach the requested target")
    return _timed_drop(cols, aligned, base[:final_steering], speed_threshold=speed_threshold,
                       target=target, requested_frames=requested_frames)


def shape_human_movement(
    cols: np.ndarray, spawn: FrameState, script: Sequence[int], *,
    speed_threshold: int, target: tuple[int, int, int], requested_frames: int,
    reaction_frames: int, edge_interval: int,
) -> tuple[np.ndarray, dict[str, int | bool]]:
    """Choose a replay-valid held-input route near the requested total duration.

    Placement and complete reachability stay authoritative. This alternative
    path generator improves common moves without claiming that its finite
    routes implement a complete constrained planner.
    """
    base = np.asarray(script, dtype=np.uint8).reshape(-1)
    wanted = max(len(base), int(requested_frames))
    columns = np.asarray(cols, dtype=np.uint16).reshape(8)
    interval = max(1, int(edge_interval))
    reaction = max(0, int(reaction_frames))
    candidates = []
    original = _retime_planner_route(columns, spawn, base, speed_threshold=int(speed_threshold),
                                    target=target, requested_frames=wanted)
    if original is not None:
        candidates.append((original, False))
    for das in (True, False):
        for order in (2, 0, 1):
            candidate = _movement_script(columns, spawn, target=target,
                speed_threshold=int(speed_threshold), total_frames=wanted,
                reaction_frames=reaction, edge_interval=interval, order=order, das=das)
            if candidate is not None:
                candidates.append((candidate, True))
    if not candidates:
        return base.copy(), {"shaped": False, "requested_frames": wanted, "edge_interval_frames": interval}
    ranked = []
    for candidate, human_route in candidates:
        masks = np.asarray([_action_mask(int(a)) for a in candidate], dtype=np.uint8)
        initial = (0, 2, 1)[spawn.hold_dir.value] | (0, 128, 64)[spawn.rot_hold.value]
        prior = np.concatenate(([initial], masks[:-1]))
        edges = sum(int(a ^ b).bit_count() for a, b in zip(masks, prior, strict=True))
        ranked.append((abs(len(candidate) - wanted), edges, len(candidate), not human_route, candidate))
    _, edges, _length, planner_route, best = min(ranked, key=lambda item: item[:4])
    replay = spawn
    for index, action in enumerate(best, 1):
        replay = simulate_frame(columns, replay, int(action), speed_threshold=int(speed_threshold))
        if replay.locked:
            if index != len(best) or (replay.x, replay.y, replay.rot) != target:
                raise RuntimeError("retimed movement failed exact lock-time replay")
            break
    if not replay.locked:
        raise RuntimeError("retimed movement did not lock")
    return best, {"shaped": not planner_route, "retimed": True,
                  "requested_frames": wanted, "edge_interval_frames": interval,
                  "edges": edges, "reaction_frames": reaction}


def _action_mask(action: int) -> int:
    return (0, 2, 1)[action // 6] | (4 if action % 6 >= 3 else 0) | (0, 128, 64)[action % 3]


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

    This is the fallback for routes unsupported by the held-input shaper.
    Replay rejects pauses that lose the selected placement; natural gravity
    can make actual added lock time much shorter than the prepended pause.
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


__all__ = ["MOTOR_FIT", "add_thinking_delay", "hold_soft_drop_suffix", "motor_parameters", "shape_human_movement"]
