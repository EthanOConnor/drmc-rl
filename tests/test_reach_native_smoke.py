import numpy as np
import pytest

from envs.retro import reach_native as reach_native_mod
from envs.retro.placement_planner import BoardState, PillSnapshot, PlacementPlanner
from envs.retro.placement_space import GRID_WIDTH


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
        speed_setting=0,
        speed_ups=0,
        spawn_id=1,
    )


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

