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


def test_preheld_buttons_do_not_create_a_press_at_the_spawn_boundary() -> None:
    metrics = script_metrics([BUTTON_A, BUTTON_A, 0], initial_buttons=BUTTON_A)
    assert metrics.rotation_presses == 0
    assert metrics.rising_edges == 0
    assert metrics.falling_edges == 1
    assert metrics.total_edges == 1
    assert metrics.reaction_frames == 0
    assert script_metrics([BUTTON_RIGHT], initial_buttons=BUTTON_LEFT).direction_reversals == 1


def test_simultaneous_changes_are_chords_and_keep_the_full_burst_count() -> None:
    profile = ExecutionProfile(id="chords", description="chords allowed", min_inter_edge_frames=2)
    result = profile.validate([BUTTON_RIGHT | BUTTON_A, BUTTON_RIGHT | BUTTON_A, 0])
    assert result.valid
    assert result.metrics.min_inter_edge_frames == 2
    assert result.metrics.total_edges == 4
    assert result.metrics.max_simultaneous_buttons == 2
    assert result.metrics.peak_edges_250ms == 4
    assert not profile.validate([BUTTON_A, 0, BUTTON_A]).valid


def test_profile_validation_honors_the_initial_controller_state() -> None:
    profile = ExecutionProfile(id="held", description="held", max_edges_250ms=0)
    assert profile.validate([BUTTON_A] * 4, initial_buttons=BUTTON_A).valid
    assert not profile.validate([BUTTON_A] * 4).valid


def test_empty_execution_window_has_no_controller_activity() -> None:
    metrics = script_metrics([], initial_buttons=BUTTON_A)
    assert metrics.frames == metrics.total_edges == metrics.active_frames == 0
    assert ExecutionProfile.unrestricted().validate([]).valid


def test_pareto_frontier_removes_slower_more_complex_script() -> None:
    fast = np.array([BUTTON_RIGHT, 0], dtype=np.uint8)
    slow = np.array([BUTTON_RIGHT, 0, 0, 0], dtype=np.uint8)
    frontier = pareto_frontier([fast, slow])
    assert len(frontier) == 1
    assert frontier[0][1].frames == 2
