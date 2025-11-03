import numpy as np

from envs.retro.fast_reach import (
    FrameState,
    HoldDir,
    ReachabilityConfig,
    Rotation,
    build_reachability,
    frame_action_from_index,
    simulate_frame,
)
from envs.retro.placement_actions import GRID_HEIGHT, GRID_WIDTH


def _empty_columns() -> np.ndarray:
    return np.zeros(GRID_WIDTH, dtype=np.uint16)


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
        orient=0,
        speed_counter=0,
        hor_velocity=0,
        hold_dir=HoldDir.NEUTRAL,
        hold_down=False,
        frame_parity=0,
        locked=False,
    )
    speed_threshold = 30  # large enough to avoid gravity triggering

    first = simulate_frame(cols, state, down_action, speed_threshold=speed_threshold)
    assert first.y == state.y  # parity gate blocks the drop
    assert first.speed_counter == state.speed_counter + 1
    assert not first.locked
    assert first.frame_parity == 1

    second = simulate_frame(cols, first, down_action, speed_threshold=speed_threshold)
    assert second.y == state.y + 1  # drop occurs on the parity frame
    assert second.speed_counter == 0
    assert not second.locked


def test_horizontal_repeat_follows_nes_velocity():
    cols = _empty_columns()
    right_action = _action_index(HoldDir.RIGHT, False, Rotation.NONE)
    state = FrameState(
        x=2,
        y=5,
        orient=0,
        speed_counter=0,
        hor_velocity=0,
        hold_dir=HoldDir.NEUTRAL,
        hold_down=False,
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
        orient=1,  # vertical (down)
        speed_counter=0,
        hor_velocity=0,
        hold_dir=HoldDir.NEUTRAL,
        hold_down=False,
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
        node.state.x == 3 and node.state.y == 4 and node.state.orient == 0
        for node in reach.nodes
    )
    assert found, "Expected rotation with left kick to reach x-1"


def test_landing_locks_final_state():
    cols = _empty_columns()
    neutral = _action_index(HoldDir.NEUTRAL, False, Rotation.NONE)
    state = FrameState(
        x=1,
        y=0,
        orient=0,
        speed_counter=0,
        hor_velocity=0,
        hold_dir=HoldDir.NEUTRAL,
        hold_down=False,
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
