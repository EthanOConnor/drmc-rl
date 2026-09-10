from dataclasses import dataclass, replace

import numpy as np
import pytest

from drmc_rl.game.pair_state import DecisionBoundary
from drmc_rl.search.adaptive_event import AdaptiveDecisionResult, AdaptiveJointEventSearch
from drmc_rl.search.joint_event import ChanceOutcome, SearchConfig, WDL
from drmc_rl.search.queued_event import QueuedJointEventSearch
from tools.audit_adaptive_search import compare_adaptive_result


@dataclass(frozen=True)
class State:
    phase: str = "root"
    value: float = 0.


class Model:
    def __init__(self, side=0, values=(.8, -.2, .3), continuation=False):
        self.side, self.values, self.continuation = side, values, continuation
        self.root_calls, self.joint_calls = [], []

    def key(self, state):
        return state

    def boundary(self, state):
        return dict(root=DecisionBoundary.P1 if self.side == 0 else DecisionBoundary.P2,
                    chance=DecisionBoundary.ADVANCE, inner=DecisionBoundary.BOTH,
                    terminal=DecisionBoundary.TERMINAL)[state.phase]

    def terminal_value(self, state, side):
        if state.phase == "terminal":
            u = state.value if side == self.side else -state.value
            return WDL((1 + u) / 2, 0, (1 - u) / 2)

    def evaluate(self, state, side):
        raise AssertionError("test leaves are exact terminals")

    def legal_actions(self, state, side):
        if state.phase == "root":
            assert side == self.side, "inactive opponent must not be queried"
            return list(range(len(self.values)))
        return [0, 1]

    def prior(self, state, side, actions):
        if state.phase == "root":
            assert side == self.side
        return [1. / (1 + a) for a in actions]

    def apply_actions(self, state, a, b):
        if state.phase == "root":
            assert (b if self.side == 0 else a) is None
            action = a if self.side == 0 else b
            self.root_calls.append(action)
            return State("chance" if self.continuation else "terminal", self.values[action])
        self.joint_calls.append((state.value, a, b))
        own, opponent = (a, b) if self.side == 0 else (b, a)
        matrix = [[.1, -.1], [-.1, .1]]
        return State("terminal", state.value + matrix[own][opponent])

    def chance_outcomes(self, state):
        assert state.phase == "chance"
        return [ChanceOutcome(.25, State("inner", state.value + .05)),
                ChanceOutcome(.75, State("inner", state.value - .05))]


def check(result, exact):
    values = np.asarray(exact)[list(result.actions)]
    assert np.all(result.utility_lower <= values + 1e-12)
    assert np.all(result.utility_upper >= values - 1e-12)
    gap = values.max() - result.policy_target @ values
    assert gap <= result.gap_upper + 1e-12
    assert result.to_dict()["candidate_truncation"] == 0
    assert "opponent_actions" not in result.to_dict()


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("nested", [False, True])
def test_unilateral_certified_terminal_win_can_stop_with_unknown_alternatives(side, nested):
    model = Model(side, (1., .2, -.3))
    result = AdaptiveJointEventSearch(model, nested=nested, allocation_batch=1,
        response_gap=.0001).search(State(), root_side=side)
    assert isinstance(result, AdaptiveDecisionResult)
    assert result.certified and result.stop_reason == "regret_bound"
    assert model.root_calls == [0] and result.allocated_root_actions == 1
    assert result.total_allocated_joint_actions == 0
    assert result.select_action(np.random.default_rng(0)) == 0
    assert result.action_wdl[1:] == (None, None)
    assert result.to_dict()["root_boundary"] == ("p1" if side == 0 else "p2")
    check(result, model.values)


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("nested", [False, True])
def test_unilateral_full_pair_continuation_preserves_reveals_and_mixed_play(side, nested):
    model = Model(side, continuation=True)
    config = SearchConfig(depth_events=3, opponent_mode="mixed", chance_beam=1,
                          own_beam=1, opponent_beam=1)
    result = AdaptiveJointEventSearch(model, config, nested=nested, allocation_batch=2,
                                     response_gap=.0001).search(State(), root_side=side)
    assert result.certified and not result.node_budget_exhausted
    assert len(model.root_calls) == 3 and result.evaluated.all()
    assert result.chance_nodes == 3 and result.chance_outcomes == 6
    assert result.total_allocated_joint_actions == len(model.joint_calls) == 24
    check(result, np.asarray(model.values) - .025)
    if nested:
        assert result.action_wdl == (None,) * 3
    else:
        np.testing.assert_allclose([v.utility for v in result.action_wdl], [.775, -.225, .275])


@pytest.mark.parametrize("nested", [False, True])
def test_unexamined_candidate_cannot_be_hidden_by_root_allocation_budget(nested):
    model = Model(values=(.8, .9, 1.))
    result = AdaptiveJointEventSearch(model, nested=nested, max_root_actions=2,
        allocation_batch=2, response_gap=.0001).search(State(), root_side=0)
    assert not result.certified and result.stop_reason == "root_budget"
    assert result.allocated_root_actions == 2 and len(result.actions) == 3
    assert result.action_wdl[2] is None and result.utility_upper[2] == 1
    with pytest.raises(ValueError, match="certificate"):
        result.select_action(np.random.default_rng(0))
    check(result, model.values)


@pytest.mark.parametrize("nested", [False, True])
def test_descendant_joint_budget_stays_global_and_does_not_invent_values(nested):
    model = Model(continuation=True)
    result = AdaptiveJointEventSearch(model, SearchConfig(depth_events=3, opponent_mode="mixed"),
        nested=nested, allocation_batch=2, max_joint_actions=1, response_gap=.0001).search(State(), root_side=0)
    assert not result.certified and result.stop_reason == "joint_budget"
    assert result.total_allocated_joint_actions == len(model.joint_calls) == 1
    assert all(v is None for v in result.action_wdl)
    check(result, np.asarray(model.values) - .025)


@pytest.mark.parametrize("nested", [False, True])
def test_unilateral_node_budget_retains_unknown_values(nested):
    model = Model(continuation=True)
    result = AdaptiveJointEventSearch(model, SearchConfig(opponent_mode="mixed", max_nodes=1),
        nested=nested, allocation_batch=2).search(State(), root_side=0)
    assert not result.certified and result.stop_reason == "node_budget"
    assert result.allocated_root_actions <= 2
    assert all(v is None for v in result.action_wdl)


def test_wrong_acting_side_and_truncated_inventory_are_rejected():
    search = AdaptiveJointEventSearch(Model())
    with pytest.raises(ValueError, match="root player"):
        search.search(State(), root_side=1)
    with pytest.raises(ValueError, match="complete"):
        search.search(State(), root_side=0, root_actions=[0, 1])


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("nested", [False, True])
def test_independent_complete_reference_checks_reordered_inventory_and_rejects_false_bounds(side, nested):
    config = SearchConfig(depth_events=3, opponent_mode="mixed", policy_temperature=0.)
    complete = QueuedJointEventSearch(Model(side, continuation=True), config).search(State(), root_side=side)
    adaptive = AdaptiveJointEventSearch(Model(side, continuation=True), config,
        nested=nested, response_gap=.001).search(State(), root_side=side, root_actions=[2, 1, 0])
    comparison = compare_adaptive_result(complete, adaptive)
    assert comparison["bounds_hold"] and comparison["full_vector_regret"] < 1e-12
    assert comparison["total_root_actions"] == 3
    assert "total_joint_actions" not in comparison
    # The complete reference must use its precise WDL values, not the rounded
    # display utility array. A .01 falsification must fail independently.
    bad_lower = adaptive.utility_lower + .01
    assert not compare_adaptive_result(complete, replace(adaptive, utility_lower=bad_lower))["bounds_hold"]
    bad_policy = np.array([1., 0., 0.])
    assert not compare_adaptive_result(complete, replace(adaptive, policy_target=bad_policy))["bounds_hold"]
    with pytest.raises(RuntimeError, match="inventory"):
        compare_adaptive_result(complete, replace(adaptive, actions=(2, 1)))
    with pytest.raises(RuntimeError, match="reference"):
        compare_adaptive_result(replace(complete, budget_exhausted=True), adaptive)


@pytest.mark.parametrize("nested", [False, True])
def test_reused_search_changes_root_kind_without_stale_budget_or_perspective(nested):
    model = Model(continuation=True)
    search = AdaptiveJointEventSearch(model, SearchConfig(depth_events=3, opponent_mode="mixed"),
                                      nested=nested, response_gap=.001)
    first = search.search(State(), root_side=0)
    both = search.search(State("inner"), root_side=1)
    complete_both = QueuedJointEventSearch(model, search.config).search(State("inner"), root_side=1)
    assert compare_adaptive_result(complete_both, both)["bounds_hold"]
    last = search.search(State(), root_side=0)
    assert first.certified and both.certified and last.certified
    assert first.total_allocated_joint_actions == last.total_allocated_joint_actions == 24
    np.testing.assert_array_equal(first.utility_lower, last.utility_lower)
    assert both.to_dict()["schema"] == "drmc-adaptive-joint-search-v1"
