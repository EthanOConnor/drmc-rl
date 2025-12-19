import numpy as np
import pytest

from envs.retro import reach_native as reach_native_mod
from envs.retro.fast_reach import FrameState, HoldDir, Rotation, frame_action_from_index, simulate_frame
from envs.retro.placement_planner import BoardState, PillSnapshot, PlacementPlanner
from envs.retro.placement_space import GRID_WIDTH
from envs.retro.drmario_env import Action


def _blocked_under_spawn_board() -> BoardState:
    cols = np.zeros(GRID_WIDTH, dtype=np.uint16)
    # Spawn is horizontal at base (x=3,y=0). Occupy cells directly below to force
    # an immediate lock on the first gravity tick.
    cols[3] |= 1 << 1
    cols[4] |= 1 << 1
    return BoardState(columns=cols)


def _immediate_lock_snapshot() -> PillSnapshot:
    return PillSnapshot(
        base_row=0,
        base_col=3,
        rot=0,
        colors=(1, 2),
        speed_counter=0,
        speed_threshold=0,
        hor_velocity=0,
        frame_parity=0,
        hold_left=False,
        hold_right=False,
        hold_down=False,
        rot_hold=Rotation.NONE,
        speed_setting=0,
        speed_ups=0,
        spawn_id=1,
    )


def _frame_action_index(*, hold_left: bool, hold_right: bool, hold_down: bool, action: Action) -> int:
    hold_dir = HoldDir.NEUTRAL
    if hold_left and not hold_right:
        hold_dir = HoldDir.LEFT
    elif hold_right and not hold_left:
        hold_dir = HoldDir.RIGHT

    rotation = Rotation.NONE
    if action == Action.ROTATE_A:
        rotation = Rotation.CW
    elif action == Action.ROTATE_B:
        rotation = Rotation.CCW

    for idx in range(18):
        fa = frame_action_from_index(idx)
        if fa.hold_dir is hold_dir and fa.hold_down == bool(hold_down) and fa.rotation is rotation:
            return idx
    raise AssertionError("No frame action index for controller step")


def test_native_reachability_matches_python_on_immediate_lock():
    board = _blocked_under_spawn_board()
    snap = _immediate_lock_snapshot()

    py = PlacementPlanner(max_frames=4, reach_backend="python")
    py_reach = py.build_spawn_reachability(board, snap)
    assert int(py_reach.feasible_mask.sum()) == 1

    if not reach_native_mod.is_library_present():
        pytest.skip("native reachability library not built (python -m tools.build_reach_native)")

    native = PlacementPlanner(max_frames=4, reach_backend="native")
    native_reach = native.build_spawn_reachability(board, snap)

    assert int(native_reach.feasible_mask.sum()) == int(py_reach.feasible_mask.sum())
    assert np.array_equal(native_reach.feasible_mask, py_reach.feasible_mask)

    action = int(np.flatnonzero(py_reach.feasible_mask.reshape(-1))[0])
    py_plan = py.plan_action(py_reach, action)
    native_plan = native.plan_action(native_reach, action)
    assert py_plan is not None
    assert native_plan is not None
    assert py_plan.cost == native_plan.cost == 1


def test_native_plan_replays_to_reported_terminal_pose():
    if not reach_native_mod.is_library_present():
        pytest.skip("native reachability library not built (python -m tools.build_reach_native)")

    # Empty board: many placements should be feasible.
    board = BoardState(columns=np.zeros(GRID_WIDTH, dtype=np.uint16))
    snap = PillSnapshot(
        base_row=0,
        base_col=3,
        rot=0,
        colors=(1, 2),
        speed_counter=0,
        speed_threshold=0x45,
        hor_velocity=0,
        frame_parity=0,
        hold_left=False,
        hold_right=False,
        hold_down=False,
        rot_hold=Rotation.NONE,
        speed_setting=0,
        speed_ups=0,
        spawn_id=1,
    )

    planner = PlacementPlanner(max_frames=256, reach_backend="native")
    reach = planner.build_spawn_reachability(board, snap)
    feasible = np.flatnonzero(reach.feasible_mask.reshape(-1)).astype(int).tolist()
    assert feasible, "Expected at least one feasible macro placement on an empty board"

    # Replay a handful of feasible placements under the Python reference stepper.
    cols = board.columns.astype(np.uint16, copy=False)
    spawn = FrameState(
        x=int(snap.base_col),
        y=int(snap.base_row),
        rot=int(snap.rot) & 3,
        speed_counter=int(snap.speed_counter),
        hor_velocity=int(snap.hor_velocity),
        hold_dir=HoldDir.NEUTRAL,
        frame_parity=int(snap.frame_parity) & 1,
        rot_hold=snap.rot_hold,
        locked=False,
    )

    for action in feasible[:5]:
        plan = planner.plan_action(reach, int(action))
        assert plan is not None
        st = spawn
        for step in plan.controller:
            idx = _frame_action_index(
                hold_left=bool(step.hold_left),
                hold_right=bool(step.hold_right),
                hold_down=bool(step.hold_down),
                action=Action(int(step.action)),
            )
            st = simulate_frame(cols, st, idx, speed_threshold=int(snap.speed_threshold))
        assert st.locked is True
        assert (st.x, st.y, st.rot) == tuple(int(v) for v in plan.terminal_pose)
        assert len(plan.controller) == int(plan.cost)
