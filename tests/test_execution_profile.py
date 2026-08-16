import numpy as np

from drmc_rl.execution.profile import (
    BUTTON_A,
    BUTTON_LEFT,
    BUTTON_RIGHT,
    ExecutionProfile,
    pareto_frontier,
    script_metrics,
)


def test_script_metrics_capture_reaction_bursts_and_reversal() -> None:
    script = np.array([0, 0, BUTTON_LEFT, BUTTON_LEFT, 0, BUTTON_RIGHT, BUTTON_A, 0], dtype=np.uint8)
    metrics = script_metrics(script)
    assert metrics.reaction_frames == 2
    assert metrics.direction_reversals == 1
    assert metrics.rotation_presses == 1
    assert metrics.total_edges > 0


def test_profile_rejects_impossible_burst_and_chord() -> None:
    profile = ExecutionProfile(
        id="strict",
        description="strict",
        min_inter_edge_frames=1,
        max_edges_250ms=2,
        max_edges_1s=3,
        max_edges_10s=3,
        max_simultaneous_buttons=1,
    )
    result = profile.validate([BUTTON_LEFT | BUTTON_RIGHT, 0, BUTTON_A, 0])
    assert not result.valid
    assert "left_right_chord" in result.violations


def test_pareto_frontier_removes_slower_more_complex_script() -> None:
    fast = np.array([BUTTON_RIGHT, 0], dtype=np.uint8)
    slow = np.array([BUTTON_RIGHT, 0, 0, 0], dtype=np.uint8)
    frontier = pareto_frontier([fast, slow])
    assert len(frontier) == 1
    assert frontier[0][1].frames == 2
