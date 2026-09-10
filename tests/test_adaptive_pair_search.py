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


class NestedMatrix(Matrix):
    """An independently solvable continuation matrix after every root pair."""
    def __init__(self, payoff, *, chance=False):
        super().__init__(payoff)
        self.inner = np.zeros((4, 5))
        self.inner[0, :] = .3
        self.inner[1:, 0] = -.6
        self.chance = chance
        self.batches = []

    def boundary(self, state):
        return dict(both=DecisionBoundary.BOTH, inner=DecisionBoundary.BOTH,
            chance=DecisionBoundary.ADVANCE, own=DecisionBoundary.P1,
            opponent=DecisionBoundary.P2, terminal=DecisionBoundary.TERMINAL)[state.phase]

    def legal_actions(self, state, side):
        if state.phase == 'both':
            return super().legal_actions(state, side)
        return list(range(self.inner.shape[side] if state.phase == 'inner' else 3))

    def apply_actions(self, state, a, b):
        self.calls.append((state.phase, a, b))
        if state.phase == 'both':
            return State('chance' if self.chance else 'inner', .6*float(self.payoff[a, b]))
        if state.phase == 'inner':
            return State('terminal', state.value+.3*float(self.inner[a, b]))
        delta = [-.05, 0, .05][a] if state.phase == 'own' else [-.08, 0, .08][b]
        return State('inner', state.value+delta)

    def chance_outcomes(self, state):
        return [ChanceOutcome(.25, State('own', state.value+.2)),
                ChanceOutcome(.75, State('opponent', state.value-.1))]

    def prepare_requests(self, requests):
        assert all(s.phase in ('both', 'inner', 'own', 'opponent') for s, _ in requests)
        self.batches.append(list(requests))


@pytest.mark.parametrize('side',[0,1])
@pytest.mark.parametrize('chance',[False,True])
def test_nested_bounds_cover_exact_mixed_chance_and_unilateral_values(side,chance):
    root=np.array([[.8,-.4,.95],[-.2,.6,.95],[-.9,-.9,.95]])
    model=NestedMatrix(root,chance=chance)
    config=SearchConfig(depth_events=4 if chance else 2,opponent_mode='mixed',
        own_beam=1,opponent_beam=1,chance_beam=1,max_nodes=20000)
    result=AdaptiveJointEventSearch(model,config,nested=True,allocation_batch=2,
        response_gap=.0001).search(State(),root_side=side)
    # The inner game has value .3. Public chance and intervening single-side
    # decisions contribute .25*(.2+.05)+.75*(-.1-.08) = -.0725.
    exact=.6*root+.09-(.0725 if chance else 0.)
    assert result.certified and result.nested
    assert_valid_bounds(result,exact if side==0 else -exact.T)
    assert all(v is None for row in result.joint_wdl for v in row)
    nested=[r for r in result.nested_certificates if not r['root']]
    assert nested and all(r['actions']*r['opponent_actions']==20 for r in nested)
    assert any(r['evaluated']<20 for r in nested)
    assert result.total_allocated_joint_actions==sum(phase in ('both','inner') for phase,_,_ in model.calls)
    assert result.total_allocated_joint_actions>result.allocated_joint_actions
    assert any(len(batch)>(2 if chance else 1) for batch in model.batches)
    if chance:
        assert result.chance_outcomes==2*result.chance_nodes
    payload=result.to_dict()
    assert not payload['usable_for_quality_training'] and payload['candidate_truncation']==0


@pytest.mark.parametrize('joint_budget',[1,5,25,100])
def test_nested_global_budget_keeps_partial_child_intervals_without_invented_wdl(joint_budget):
    root=np.array([[.8,-.4],[-.2,.6]])
    model=NestedMatrix(root)
    result=AdaptiveJointEventSearch(model,SearchConfig(depth_events=2,opponent_mode='mixed'),
        nested=True,allocation_batch=2,max_joint_actions=joint_budget,response_gap=.0001).search(
            State(),root_side=0)
    assert result.total_allocated_joint_actions<=joint_budget
    assert_valid_bounds(result,.6*root+.09)
    assert all(v is None for row in result.joint_wdl for v in row)
    if not result.certified:
        with pytest.raises(ValueError,match='certificate'):
            result.select_action(np.random.default_rng(1))


def test_nested_expired_chance_budget_retains_mass_and_never_evaluates_forced_states():
    root=np.array([[.8,-.4],[-.2,.6]])
    result=AdaptiveJointEventSearch(NestedMatrix(root,chance=True),SearchConfig(
        depth_events=4,opponent_mode='mixed',max_nodes=3,chance_beam=1),
        nested=True,allocation_batch=2).search(State(),root_side=0)
    assert not result.certified and result.node_budget_exhausted and result.nodes<=3
    assert_valid_bounds(result,.6*root+.09-.0725)
    assert result.chance_outcomes==2*result.chance_nodes


def test_nested_solver_failure_withholds_sampling_and_keeps_conservative_intervals(monkeypatch):
    model=NestedMatrix([[.8,-.4],[-.2,.6]])
    search=AdaptiveJointEventSearch(model,SearchConfig(depth_events=2,opponent_mode='mixed'),nested=True)
    original=search._solve_payoffs
    def fail(payoff):
        strategies=original(payoff)
        if search._matrix_games>=5:
            search._equilibrium_converged=False
            search._matrix_failures.append('injected bounded solver failure')
        return strategies
    monkeypatch.setattr(search,'_solve_payoffs',fail)
    result=search.search(State(),root_side=0)
    assert not result.certified and result.stop_reason=='solver_failure'
    assert result.matrix_failures
    assert_valid_bounds(result,.6*model.payoff+.09)


@pytest.mark.parametrize('side',[0,1])
def test_non_pure_interior_equilibrium_is_bounded_without_reporting_mixture_wdl(side):
    root=np.array([[.8,-.4],[-.2,.6]])
    model=NestedMatrix(root)
    model.inner=root.copy()  # p=(.4,.6), q=(.5,.5), value .2.
    result=AdaptiveJointEventSearch(model,SearchConfig(depth_events=2,opponent_mode='mixed'),
        nested=True,allocation_batch=1,response_gap=.0001).search(State(),root_side=side)
    exact=.6*root+.06
    assert result.certified
    assert_valid_bounds(result,exact if side==0 else -exact.T)
    assert all(v is None for row in result.joint_wdl for v in row)
    assert any(not c['root'] and c['certified'] for c in result.nested_certificates)
