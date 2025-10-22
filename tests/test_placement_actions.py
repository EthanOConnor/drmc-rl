from envs.retro.placement_actions import (
    GRID_HEIGHT,
    GRID_WIDTH,
    PLACEMENT_EDGES,
    action_count,
    action_from_cells,
    cells_for_action,
    opposite_actions,
)


def test_action_count_and_bounds():
    assert action_count() == len(PLACEMENT_EDGES) == GRID_HEIGHT * (GRID_WIDTH - 1) * 2 + (GRID_HEIGHT - 1) * GRID_WIDTH * 2


def test_action_round_trip():
    for edge in PLACEMENT_EDGES:
        idx = action_from_cells(edge.origin, edge.dest)
        assert idx == edge.index
        origin, dest, direction = cells_for_action(idx)
        assert origin == edge.origin
        assert dest == edge.dest
        assert direction == edge.direction


def test_opposite_pairs_cover_edges():
    seen = set()
    for left, right in opposite_actions():
        assert left != right
        seen.add(left)
        seen.add(right)
        edge_left = PLACEMENT_EDGES[left]
        edge_right = PLACEMENT_EDGES[right]
        assert edge_left.origin == edge_right.dest
        assert edge_left.dest == edge_right.origin
    assert len(seen) == len(PLACEMENT_EDGES)
