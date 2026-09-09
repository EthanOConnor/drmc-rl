import ctypes as C
import os

import numpy as np
import pytest

from drmc_rl.envs.backends.drmario_vs_pool import build_vs_reset_spec
from drmc_rl.envs.backends.vs_frames import FrameState, FrameVsPool
from drmc_rl.execution.pace import resolve_pace
from drmc_rl.game.pair_state import PairEventKind
from drmc_rl.human.anticipation import execution_for_action
from drmc_rl.human.backend import NoReachablePlacement, plan_candidates
from drmc_rl.human.motor_opportunity import MotorOpportunityLabeler, UNREACHABLE, action_cells
from drmc_rl.planning.native_reach import NativeReachabilityRunner


LIB = os.environ.get("DRMC_FRAME_LIBRARY")


def reset_sparse(pool, *, counter=9, speed_ups=0):
    board = np.full((2, 128), 255, np.uint8)
    board[0, 120:123] = 0xD0
    board[:, 127] = 0xD2
    spec = build_vs_reset_spec(level=(14, 14), speed_setting=(2, 2),
                              rng_override=True, rng_state=(81, 93))
    spec.checkpoint_enabled = 1
    for side in (0, 1):
        spec.checkpoint_board[side][:] = board[side]
        spec.checkpoint_falling_colors[side][:] = [1, 1]
        spec.checkpoint_preview_colors[side][:] = [0, 0]
        spec.checkpoint_pill_counter_total[side] = counter
        spec.checkpoint_speed_ups[side] = speed_ups
    pool._check(pool.lib.drm_vspool_frame_reset(
        pool.handle, None, C.byref(spec), pool.states, C.sizeof(FrameState)))
    while not pool.states[0].falling:
        pool.step()
    return pool.semantic(0, public_context=True)


def execute_to_spawn(pool, candidate, action, pace, delay):
    move = execution_for_action(candidate, action, pace, delay=delay)
    pool.step(count=delay)
    for bits in move["controller_frames"]:
        pool.step([bits, 0])
    assert not pool.states[0].falling
    while not pool.states[0].falling and not pool.states[0].terminal:
        pool.step()
    return move


@pytest.mark.parametrize("pace_name", ["sloth", "top_humans", "frame_perfect"])
def test_future_geometry_and_clears_match_two_actual_controller_placements(pace_name):
    planner = NativeReachabilityRunner()
    pace = resolve_pace(pace_name)
    delay = max(4, pace.reaction_frames)
    try:
        with FrameVsPool(lib_path=LIB) as pool, MotorOpportunityLabeler(planner, lib_path=LIB) as labeler:
            state = reset_sparse(pool)
            labels = labeler.label(state, pace)
            root = plan_candidates(planner, state, delay, pace)
            np.testing.assert_array_equal(labels.actions, np.flatnonzero(root[-1] != UNREACHABLE))
            assert (labels.actions >= 256).any()  # Both same-color color orders survive.
            # Exercise a sideways route that leaves useful access for the next
            # yellow pill, rather than only a straight soft drop.
            choices = [i for i, a in enumerate(labels.actions)
                       if a % 8 != 3 and (labels.next_clear_events[i] > 0).any()]
            assert choices
            slot = choices[len(choices) // 2]
            execute_to_spawn(pool, root, int(labels.actions[slot]), pace, delay)
            actual = pool.semantic(0, public_context=True)
            assert not pool.states[0].terminal
            np.testing.assert_array_equal(labels.after_fields[slot], np.asarray(pool.states[0].board))
            assert labels.next_speed_ups == actual["speed_ups"]
            assert actual["pill"] == state["preview"]
            parity = actual["falling"]["frame_parity"]
            future = plan_candidates(planner, actual, delay, pace)
            expected = future[-1].copy()
            expected[expected != UNREACHABLE] += delay
            np.testing.assert_array_equal(labels.next_costs[slot, parity], expected)
            possible = np.flatnonzero(expected != UNREACHABLE)
            clear = [a for a in possible if labels.next_clear_events[slot, parity, a]]
            chosen = int(clear[0] if clear else possible[0])
            before = int(pool.states[0].frame)
            execute_to_spawn(pool, future, chosen, pace, delay)
            events = [e for e in pool.public_state(0).recent_events
                      if e.side == 0 and e.kind == PairEventKind.CLEAR and e.frame_id > before]
            assert len(events) == labels.next_clear_events[slot, parity, chosen]
            assert sum(e.public_payload["viruses_cleared"] for e in events) == labels.next_viruses_cleared[slot, parity, chosen]
            assert labels.reachable_cells[slot, parity, list(action_cells(chosen))].max() <= expected[chosen]
    finally:
        planner.close()


def test_conditional_labels_ignore_opponent_and_report_missing_next_choices():
    planner = NativeReachabilityRunner()
    try:
        with FrameVsPool(lib_path=LIB) as pool, MotorOpportunityLabeler(planner, lib_path=LIB) as labeler:
            state = reset_sparse(pool, speed_ups=48)
            pace = resolve_pace("frame_perfect")
            labels = labeler.label(state, pace, compute_frames=0, validate_next_scripts=True)
            changed = {**state, "opponent_board_planes": np.zeros((8, 16, 8)), "opponent_pill": [2, 1]}
            other = labeler.label(changed, pace, compute_frames=0)
            for key, value in labels.arrays().items():
                np.testing.assert_array_equal(value, other.arrays()[key])
            assert labels.next_speed_ups == 49
            with pytest.raises(NoReachablePlacement):
                labeler.label(state, resolve_pace("sloth"))
            with pytest.raises(ValueError, match="BCD"):
                labeler.label({**state, "pill_counter_total": 0x1A}, pace)
    finally:
        planner.close()


def test_finishing_the_level_is_not_labeled_as_losing_next_pill_access():
    planner = NativeReachabilityRunner()
    try:
        with FrameVsPool(lib_path=LIB) as pool, MotorOpportunityLabeler(planner, lib_path=LIB) as labeler:
            state = reset_sparse(pool)
            state["board_planes"] = state["board_planes"].copy()
            state["board_planes"][:, 15, 7] = 0
            state["pill"] = [1, 1]
            labels = labeler.label(state, resolve_pace("normal"))
            won = labels.root_terminal == 1
            assert won.any()
            assert (labels.next_costs[won] == UNREACHABLE).all()
            assert labels.summary()["terminal_clear"] == int(won.sum())
            remaining = labels.root_terminal == 0
            missing = (labels.next_costs == UNREACHABLE).all(axis=(1, 2))
            assert labels.summary()["no_next_choice_both_parities"] == int((remaining & missing).sum())
    finally:
        planner.close()
