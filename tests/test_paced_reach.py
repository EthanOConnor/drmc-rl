from collections import deque

import numpy as np
import pytest

from drmc_rl.execution.pace import PACES, resolve_pace
from drmc_rl.planning.fast_reach import FrameState, HoldDir, Rotation, simulate_frame
from drmc_rl.planning.native_reach import NativeReachabilityRunner


def spawn(y=0, parity=0):
    return FrameState(3, y, 0, 0, 0, HoldDir.NEUTRAL, parity, Rotation.NONE)


def reference_poses(columns, state, pace, threshold, horizon):
    """Independent exhaustive one-frame oracle, including redundant chords."""
    seen, poses = set(), set()
    queue = deque([(state, 0, -1000, -1000, 0)])
    while queue:
        state, previous, last_edge, last_motion, frame = queue.popleft()
        if frame >= horizon:
            continue
        key = (state, previous, max(0, pace.edge_interval - (frame - last_edge)),
               max(0, pace.motion_interval - (frame - last_motion)),
               max(0, pace.reaction_frames - frame))
        if key in seen:
            continue
        seen.add(key)
        for action in range(18):
            if frame < pace.reaction_frames and action != 0:
                continue
            if action != previous and frame - last_edge < pace.edge_interval:
                continue
            buttons = int(action // 6 != 0) + int(action % 6 >= 3) + int(action % 3 != 0)
            if buttons > pace.max_buttons:
                continue
            nxt = simulate_frame(columns, state, action, speed_threshold=threshold)
            moved = (nxt.x, nxt.rot) != (state.x, state.rot)
            if moved and frame - last_motion < pace.motion_interval:
                continue
            if nxt.locked:
                poses.add(nxt.x + 8 * nxt.y + 128 * nxt.rot)
            else:
                queue.append((nxt, action, frame if action != previous else last_edge,
                              frame if moved else last_motion, frame + 1))
    return poses


@pytest.mark.parametrize("pace", PACES, ids=lambda pace: pace.id)
@pytest.mark.parametrize("parity", [0, 1])
def test_every_tuck_script_obeys_motor_limits_and_replays(pace, parity):
    columns = np.zeros(8, dtype=np.uint16)
    columns[2] = 1 << 5
    initial = spawn(parity=parity)
    runner = NativeReachabilityRunner(max_frames=512)
    result = runner.bfs_full(columns, initial, speed_threshold=13, **pace.planner_args())
    assert np.any(result.costs_u16 != 65535)
    for pose in np.flatnonzero(result.costs_u16 != 65535):
        x, y, rotation = int(pose) & 7, (int(pose) >> 3) & 15, (int(pose) >> 7) & 3
        script = result.script_for_pose(x, y, rotation)
        audit = pace.validate(columns, initial, script, speed_threshold=13)
        assert (audit["x"], audit["y"], audit["rotation"]) == (x, y, rotation)
        assert len(script) == result.costs_u16[pose]


@pytest.mark.parametrize("pace", PACES, ids=lambda pace: pace.id)
def test_complete_feasibility_matches_independent_frame_oracle(pace):
    columns = np.zeros(8, dtype=np.uint16)
    columns[1] = (1 << 14) | (1 << 15)
    columns[6] = 1 << 15
    initial = spawn(y=13)
    # The native API requires a horizon that includes the reaction window,
    # even when gravity will lock the pill before that window elapses.
    horizon = max(40, pace.reaction_frames)
    runner = NativeReachabilityRunner(max_frames=horizon)
    actual = runner.bfs_full(columns, initial, speed_threshold=1, **pace.planner_args())
    expected = reference_poses(columns, initial, pace, 1, horizon)
    assert set(np.flatnonzero(actual.costs_u16 != 65535)) == expected


def test_slower_profiles_never_gain_mechanically_impossible_placements():
    columns = np.zeros(8, dtype=np.uint16)
    columns[2] = 1 << 5
    runner = NativeReachabilityRunner(max_frames=512)
    previous = set()
    for pace in PACES:
        result = runner.bfs_full(columns, spawn(), speed_threshold=3, **pace.planner_args())
        reachable = set(np.flatnonzero(result.costs_u16 != 65535))
        assert previous <= reachable
        previous = reachable


def test_legacy_paces_migrate_and_unknown_names_never_become_unrestricted():
    assert [resolve_pace(timing_scale=value).id for value in [0, 0.5, 1, 1.5]] == [
        "frame_perfect", "fast", "normal", "relaxed"]
    with pytest.raises(ValueError):
        resolve_pace("typo")
    with pytest.raises(ValueError):
        resolve_pace(timing_scale=float("nan"))


def test_relaxed_rejects_unrestricted_flash_tuck():
    columns = np.zeros(8, dtype=np.uint16)
    columns[2] = 1 << 5
    runner = NativeReachabilityRunner(max_frames=512)
    full = runner.bfs_full(columns, spawn(), speed_threshold=13)
    script = full.script_for_pose(2, 15, 1)
    assert script is not None
    with pytest.raises(ValueError, match="violates"):
        resolve_pace("relaxed").validate(columns, spawn(), script, speed_threshold=13)
