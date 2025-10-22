import numpy as np

from envs.retro.placement_actions import action_from_cells, action_count
from envs.retro.placement_planner import BoardState, PillSnapshot, PlacementPlanner


def _board_from_rows(rows: list[str]) -> BoardState:
    columns = np.zeros(8, dtype=np.uint16)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                columns[c] |= 1 << r
    return BoardState(columns=columns)


def _empty_board() -> BoardState:
    return _board_from_rows(["........" for _ in range(16)])


def test_plan_simple_drop():
    board = _empty_board()
    capsule = PillSnapshot(
        row=0,
        col=3,
        orient=0,
        colors=(1, 0),
        gravity_counter=6,
        gravity_period=6,
        lock_counter=0,
        spawn_id=1,
    )
    planner = PlacementPlanner()
    action = action_from_cells((15, 3), (15, 4))
    plan = planner.plan_action(board, capsule, action)
    assert plan is not None
    assert plan.controller, "Expected non-empty controller sequence"
    assert plan.states[-1].locked


def test_legal_mask_has_bottom_placements():
    board = _empty_board()
    capsule = PillSnapshot(
        row=0,
        col=3,
        orient=0,
        colors=(1, 1),
        gravity_counter=6,
        gravity_period=6,
        lock_counter=0,
        spawn_id=2,
    )
    planner = PlacementPlanner()
    out = planner.plan_all(board, capsule)
    horizontal_action = action_from_cells((15, 3), (15, 4))
    vertical_action = action_from_cells((15, 3), (14, 3))
    assert out.legal_mask[horizontal_action]
    assert out.legal_mask[vertical_action]
    assert out.legal_mask.sum() > 0


def test_plan_all_returns_paths():
    rows = [
        "........",
        "........",
        "........",
        "........",
        "........",
        "........",
        "........",
        "........",
        "########",
        "########",
        "###..###",
        "###..###",
        "###..###",
        "###..###",
    ]
    board = _board_from_rows(rows)
    capsule = PillSnapshot(
        row=2,
        col=1,
        orient=0,
        colors=(1, 2),
        gravity_counter=8,
        gravity_period=8,
        lock_counter=0,
        spawn_id=4,
    )
    planner = PlacementPlanner()
    out = planner.plan_all(board, capsule)
    assert int(out.feasible_mask.sum()) >= 1
    assert out.costs.shape[0] == action_count()
    assert out.path_indices.shape[0] == action_count()
    assert out.plan_count == len(out.plans)
    for plan in out.plans:
        assert plan.cost >= 0
        assert 0 <= plan.path_index < out.plan_count
        assert np.isfinite(out.costs[plan.action])
        assert out.path_indices[plan.action] == plan.path_index
