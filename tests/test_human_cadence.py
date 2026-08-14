import numpy as np

from drmc_rl.human.cadence import add_thinking_delay
from drmc_rl.planning.fast_reach import FrameState, HoldDir, Rotation, simulate_frame


def _spawn() -> FrameState:
    return FrameState(
        x=3,
        y=0,
        rot=0,
        speed_counter=0,
        hor_velocity=0,
        hold_dir=HoldDir.NEUTRAL,
        frame_parity=0,
        rot_hold=Rotation.NONE,
    )


def _minimal_left_script(cols: np.ndarray, threshold: int) -> list[int]:
    state = _spawn()
    script = [6]
    state = simulate_frame(cols, state, 6, speed_threshold=threshold)
    while not state.locked:
        script.append(0)
        state = simulate_frame(cols, state, 0, speed_threshold=threshold)
    assert (state.x, state.y, state.rot) == (2, 15, 0)
    return script


def test_cadence_adds_only_replay_valid_thinking_time():
    cols = np.zeros(8, dtype=np.uint16)
    threshold = 0x1F
    base = _minimal_left_script(cols, threshold)
    delayed, realized = add_thinking_delay(
        cols,
        _spawn(),
        base,
        speed_threshold=threshold,
        target=(2, 15, 0),
        requested_frames=20,
    )
    assert 0 < realized <= 20
    assert delayed[:realized].tolist() == [0] * realized

    state = _spawn()
    for action in delayed:
        state = simulate_frame(cols, state, int(action), speed_threshold=threshold)
        if state.locked:
            break
    assert (state.x, state.y, state.rot) == (2, 15, 0)


def test_zero_scale_path_preserves_minimal_script():
    cols = np.zeros(8, dtype=np.uint16)
    base = _minimal_left_script(cols, 0x1F)
    delayed, realized = add_thinking_delay(
        cols,
        _spawn(),
        base,
        speed_threshold=0x1F,
        target=(2, 15, 0),
        requested_frames=0,
    )
    assert realized == 0
    assert delayed.tolist() == base
