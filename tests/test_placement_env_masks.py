import numpy as np

from envs.retro.fast_reach import Rotation
from envs.retro.placement_planner import BoardState, PillSnapshot, PlacementPlanner
from envs.retro.placement_space import GRID_HEIGHT, GRID_WIDTH, ORIENTATIONS, TOTAL_ACTIONS, invalid_boundary_mask


def test_invalid_boundary_mask_shape_and_count():
    mask = invalid_boundary_mask()
    assert mask.shape == (ORIENTATIONS, GRID_HEIGHT, GRID_WIDTH)
    # In-bounds directed adjacencies count = 464; boundary actions are masked out.
    assert int(mask.sum()) == 464


def test_planner_builds_nonempty_feasible_mask_on_empty_board():
    board = BoardState(columns=np.zeros(GRID_WIDTH, dtype=np.uint16))
    snapshot = PillSnapshot(
        base_row=0,
        base_col=3,
        rot=0,
        colors=(1, 2),
        speed_counter=0,
        speed_threshold=10,
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
    planner = PlacementPlanner(max_frames=512)
    reach = planner.build_spawn_reachability(board, snapshot)
    assert reach.legal_mask.shape == (ORIENTATIONS, GRID_HEIGHT, GRID_WIDTH)
    assert reach.feasible_mask.shape == (ORIENTATIONS, GRID_HEIGHT, GRID_WIDTH)
    assert reach.costs.shape == (ORIENTATIONS, GRID_HEIGHT, GRID_WIDTH)
    assert int(reach.feasible_mask.sum()) > 0


def test_planner_plan_action_returns_script_for_feasible_action():
    board = BoardState(columns=np.zeros(GRID_WIDTH, dtype=np.uint16))
    snapshot = PillSnapshot(
        base_row=0,
        base_col=3,
        rot=0,
        colors=(1, 2),
        speed_counter=0,
        speed_threshold=10,
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
    planner = PlacementPlanner(max_frames=512)
    reach = planner.build_spawn_reachability(board, snapshot)
    # Pick the first feasible macro action and ensure a controller script exists.
    flat = int(np.flatnonzero(reach.feasible_mask.reshape(-1))[0])
    plan = planner.plan_action(reach, flat)
    assert plan is not None
    assert plan.action == flat
    assert plan.cost == len(plan.controller)
    assert plan.cost >= 1
