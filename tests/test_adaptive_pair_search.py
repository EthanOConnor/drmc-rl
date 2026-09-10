"""Independent complete matrices verify adaptive bounds and retained unknowns."""

from dataclasses import dataclass

import numpy as np
import pytest

from drmc_rl.game.pair_state import DecisionBoundary
from drmc_rl.search.adaptive_event import AdaptiveJointEventSearch
from drmc_rl.search.joint_event import ChanceOutcome, SearchConfig, WDL


@dataclass(frozen=True)
class State:
    phase: str = "both"
    value: float = 0.


class Matrix:
    def __init__(self, payoff):
        self.payoff = np.asarray(payoff, float)
        self.calls = []

    def key(self, state):
        return state

    def boundary(self, state):
        return dict(both=DecisionBoundary.BOTH, chance=DecisionBoundary.ADVANCE,
                    terminal=DecisionBoundary.TERMINAL)[state.phase]

    def terminal_value(self, state, root_side):
        if state.phase == "terminal":
            u = state.value * (1 - 2 * root_side)
            return WDL((1 + u) / 2, 0., (1 - u) / 2)

    def legal_actions(self, state, side):
        return list(range(self.payoff.shape[side]))

    def prior(self, state, side, actions):
        return [1. / (1 + i) for i in actions]

    def apply_actions(self, state, a, b):
        self.calls.append((a, b))
        return State("terminal", float(self.payoff[a, b]))

    def evaluate(self, state, side):
        raise AssertionError("the exact test game has no learned leaves")


def assert_valid_bounds(result, payoff):
    matrix = payoff[np.ix_(result.actions, result.opponent_actions)]
    assert np.all(matrix >= result.utility_lower - 1e-12)
    assert np.all(matrix <= result.utility_upper + 1e-12)
    p, q = result.policy_target, result.opponent_policy
    actual_gap = float((matrix @ q).max() - (p @ matrix).min())
    assert actual_gap <= result.gap_upper + 1e-12
    assert result.security_lower <= (p @ matrix).min() + 1e-12
    assert result.security_upper >= (matrix @ q).max() - 1e-12
    return actual_gap


@pytest.mark.parametrize("side", [0, 1])
def test_dominant_game_stops_early_with_full_inventory(side):
    matrix = np.zeros((10, 10))
    matrix[0, :] = .7
    matrix[1:, 0] = -.5
    model = Matrix(matrix)
    result = AdaptiveJointEventSearch(model, SearchConfig(depth_events=1, opponent_mode="mixed"),
        allocation_batch=1, response_gap=.0001).search(State(), root_side=side)
    assert result.certified and result.gap_upper <= .0001
    assert len(result.actions) == len(result.opponent_actions) == 10
    # Exploration order can differ by player perspective; neither requires
    # the full matrix, but no fixed percentage saving is an algorithm promise.
    assert result.evaluated.sum() < matrix.size
    assert len(model.calls) == result.allocated_joint_actions == result.evaluated.sum()
    assert len(set(model.calls)) == len(model.calls)
    assert_valid_bounds(result, matrix if side == 0 else -matrix.T)
    payload = result.to_dict()
    assert not payload["usable_for_quality_training"] and payload["candidate_truncation"] == 0
    assert any(v is None for row in payload["joint_wdl"] for v in row)
    assert np.all(result.utility_lower[~result.evaluated] == -1)
    assert np.all(result.utility_upper[~result.evaluated] == 1)


def test_mixed_game_bound_and_deterministic_budget_prefixes():
    matrix = np.array([[.8, -.4, .95], [-.2, .6, .95], [-.9, -.9, .95]])
    paths = []
    for budget in range(1, 10):
        model = Matrix(matrix)
        result = AdaptiveJointEventSearch(model, allocation_batch=1,
            max_joint_actions=budget, response_gap=.0001).search(State(), root_side=0)
        paths.append(model.calls)
        actual_gap = assert_valid_bounds(result, matrix)
        assert result.allocated_joint_actions <= budget
        if result.certified:
            assert actual_gap <= .0001
        else:
            with pytest.raises(ValueError, match="certificate"):
                result.select_action(np.random.default_rng(1))
    assert result.certified
    for a, b in zip(paths, paths[1:]):
        assert b[:len(a)] == a
    p = dict(zip(result.actions, result.policy_target))
    np.testing.assert_allclose([p[i] for i in range(3)], [.4, .6, 0], atol=1e-7)


def test_expiring_node_budget_does_not_turn_neutral_fallback_into_a_label():
    model = Matrix(np.array([[.8, -.4], [-.2, .6]]))
    result = AdaptiveJointEventSearch(model, SearchConfig(
        opponent_mode="mixed", max_nodes=1), allocation_batch=2).search(State(), root_side=0)
    assert not result.certified and result.stop_reason == "node_budget"
    assert result.discarded_joint_actions == 2 and result.evaluated.sum() == 0
    assert all(v is None for row in result.joint_wdl for v in row)


def test_adaptive_entries_include_all_correlated_forced_reveals():
    class Reveals(Matrix):
        def apply_actions(self, state, a, b):
            terminal = super().apply_actions(state, a, b)
            return State("chance", terminal.value)

        def chance_outcomes(self, state):
            return [ChanceOutcome(.25, State("terminal", state.value)),
                    ChanceOutcome(.75, State("terminal", 0))]

    matrix = np.array([[.8, -.4], [-.2, .6]])
    result = AdaptiveJointEventSearch(Reveals(matrix), SearchConfig(
        depth_events=1, chance_beam=1, opponent_mode="mixed"), allocation_batch=2).search(
            State(), root_side=0)
    assert result.certified and result.chance_outcomes == 2 * result.chance_nodes
    assert_valid_bounds(result, .25 * matrix)


def test_declared_evaluation_error_remains_in_known_cells():
    matrix = np.array([[.8, -.4], [-.2, .6]])
    result = AdaptiveJointEventSearch(Matrix(matrix), allocation_batch=1,
        evaluation_tolerance=.001, response_gap=.0001).search(State(), root_side=0)
    assert not result.certified and result.stop_reason == "response_tolerance"
    assert np.allclose((result.utility_upper - result.utility_lower)[result.evaluated], .002)
    rng = np.random.default_rng(71)
    for _ in range(20):
        assert_valid_bounds(result, matrix + rng.uniform(-.001, .001, matrix.shape))
