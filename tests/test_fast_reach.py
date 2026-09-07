import numpy as np
import pytest

from drmc_rl.planning.fast_reach import (
    FrameState,
    HoldDir,
    ReachabilityConfig,
    Rotation,
    build_reachability,
    compute_speed_threshold,
    frame_action_from_index,
    simulate_frame,
)
from drmc_rl.planning.placement_actions import GRID_HEIGHT, GRID_WIDTH


def _empty_columns() -> np.ndarray:
    return np.zeros(GRID_WIDTH, dtype=np.uint16)


@pytest.mark.parametrize("speed_ups,lock_frames", [(28, 80), (33, 64), (38, 48), (49, 16)])
def test_late_game_gravity_matches_retail_speed_transitions(speed_ups, lock_frames):
    # NTSC HI starts at speed-table index 31. At these late-game boundaries
    # the ROM waits 5/4/3/1 frames per row, including the blocked lock attempt.
    state = FrameState(x=3, y=0, rot=0, speed_counter=0, hor_velocity=0,
                       hold_dir=HoldDir.NEUTRAL, frame_parity=0)
    threshold = compute_speed_threshold(2, speed_ups)
    for elapsed in range(1, lock_frames + 1):
        state = simulate_frame(_empty_columns(), state, 0, speed_threshold=threshold)
        assert state.locked == (elapsed == lock_frames)
    assert (state.x, state.y, state.rot) == (3, 15, 0)


@pytest.mark.parametrize("x,rotation,direction", [(0, 0, HoldDir.LEFT), (6, 0, HoldDir.RIGHT), (7, 1, HoldDir.RIGHT)])
def test_bottle_boundary_does_not_charge_das_like_a_blocking_capsule(x, rotation, direction):
    state = FrameState(x=x, y=5, rot=rotation, speed_counter=0, hor_velocity=15,
                       hold_dir=direction, frame_parity=0)
    result = simulate_frame(_empty_columns(), state,
                            _action_index(direction, False, Rotation.NONE), speed_threshold=100)
    assert result.x == x
    assert result.hor_velocity == 10


def test_rotation_away_from_right_boundary_preserves_repeat_phase():
    state = FrameState(x=6, y=5, rot=0, speed_counter=0, hor_velocity=15,
                       hold_dir=HoldDir.RIGHT, frame_parity=0)
    for rotation in (Rotation.NONE, Rotation.CW, Rotation.CW):
        state = simulate_frame(_empty_columns(), state,
            _action_index(HoldDir.RIGHT, False, rotation), speed_threshold=100)
    assert (state.x, state.rot, state.hor_velocity) == (6, 3, 12)


def test_board_collision_still_charges_das_for_immediate_recovery():
    columns = _empty_columns()
    columns[7] = 1 << 5
    state = FrameState(x=5, y=5, rot=0, speed_counter=0, hor_velocity=0,
                       hold_dir=HoldDir.NEUTRAL, frame_parity=0)
    result = simulate_frame(columns, state,
        _action_index(HoldDir.RIGHT, False, Rotation.NONE), speed_threshold=100)
    assert result.x == 5
    assert result.hor_velocity == 15


def _action_index(hold_dir: HoldDir, hold_down: bool, rotation: Rotation) -> int:
    for idx in range(18):  # 3 hold dirs * 2 down states * 3 rotation states
        act = frame_action_from_index(idx)
        if act.hold_dir is hold_dir and act.hold_down == hold_down and act.rotation is rotation:
            return idx
    raise AssertionError(f"No action for {hold_dir}/{hold_down}/{rotation}")


def test_soft_drop_honours_frame_parity():
    cols = _empty_columns()
    down_action = _action_index(HoldDir.NEUTRAL, True, Rotation.NONE)
    state = FrameState(
        x=3,
        y=0,
        rot=0,
        speed_counter=0,
        hor_velocity=0,
        hold_dir=HoldDir.NEUTRAL,
        frame_parity=0,
        locked=False,
    )
    speed_threshold = 30  # large enough to avoid gravity triggering

    first = simulate_frame(cols, state, down_action, speed_threshold=speed_threshold)
    assert first.y == state.y + 1  # down-only soft drop triggers on parity==0
    assert first.speed_counter == 0
    assert not first.locked
    assert first.frame_parity == 1

    second = simulate_frame(cols, first, down_action, speed_threshold=speed_threshold)
    assert second.y == state.y + 1  # parity gate blocks soft drop on the next frame
    assert second.speed_counter == 1
    assert not second.locked


def test_horizontal_repeat_follows_nes_velocity():
    cols = _empty_columns()
    right_action = _action_index(HoldDir.RIGHT, False, Rotation.NONE)
    state = FrameState(
        x=2,
        y=5,
        rot=0,
        speed_counter=0,
        hor_velocity=0,
        hold_dir=HoldDir.NEUTRAL,
        frame_parity=0,
        locked=False,
    )
    speed_threshold = 100  # ensure gravity does not interfere

    # Initial press moves immediately and leaves velocity at 0
    state = simulate_frame(cols, state, right_action, speed_threshold=speed_threshold)
    assert state.x == 3
    assert state.hor_velocity == 0

    # Holding continues without movement until the velocity reaches the threshold.
    for _ in range(15):
        prev_x = state.x
        state = simulate_frame(cols, state, right_action, speed_threshold=speed_threshold)
        assert state.x == prev_x
    assert state.hor_velocity == 15

    # One more frame should trigger the repeat move and reload the velocity.
    prev_x = state.x
    state = simulate_frame(cols, state, right_action, speed_threshold=speed_threshold)
    assert state.x == prev_x + 1
    assert state.hor_velocity == 10


def test_rotation_with_left_kick_available():
    cols = _empty_columns()
    # Block the cell to the right of the spawn so the horizontal rotation needs the wall kick.
    cols[5] |= np.uint16(1 << 4)  # (row=4, col=5)
    spawn = FrameState(
        x=4,
        y=4,
        rot=1,  # vertical (geometry depends on rot&1)
        speed_counter=0,
        hor_velocity=0,
        hold_dir=HoldDir.NEUTRAL,
        frame_parity=0,
        locked=False,
    )
    reach = build_reachability(
        cols,
        spawn,
        speed_threshold=20,
        config=ReachabilityConfig(max_frames=4),
    )
    # Expect a reachable state with (x=3,y=4,orient=0) provided by the left kick.
    found = any(
        node.state.x == 3 and node.state.y == 4 and (node.state.rot & 1) == 0
        for node in reach.nodes
    )
    assert found, "Expected rotation with left kick to reach x-1"


def test_landing_locks_final_state():
    cols = _empty_columns()
    neutral = _action_index(HoldDir.NEUTRAL, False, Rotation.NONE)
    state = FrameState(
        x=1,
        y=0,
        rot=0,
        speed_counter=0,
        hor_velocity=0,
        hold_dir=HoldDir.NEUTRAL,
        frame_parity=0,
        locked=False,
    )
    speed_threshold = 1  # drop every other frame for a quick descent

    frames = 0
    while not state.locked and frames < 200:
        state = simulate_frame(cols, state, neutral, speed_threshold=speed_threshold)
        frames += 1
    assert state.locked
    assert state.y == GRID_HEIGHT - 1
    assert state.speed_counter == 0
