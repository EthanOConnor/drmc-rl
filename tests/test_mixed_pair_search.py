"""Analytic simultaneous games and independent recursive/queued backups."""

import json
import sys
from dataclasses import dataclass

import numpy as np
import pytest

from drmc_rl.game.pair_state import DecisionBoundary
from drmc_rl.search.joint_event import ChanceOutcome, JointEventSearch, SearchConfig, WDL
from drmc_rl.search.queued_event import QueuedJointEventSearch


@dataclass(frozen=True)
class Position:
    phase: str = "both"
    utility: float = 0.0


class MatrixModel:
    # The 2x2 subgame has p=(.4,.6), q=(.5,.5), value=.2.
    # Row/column 2 are dominated but must still be evaluated.
    matrix = np.array([[.8, -.4, .95], [-.2, .6, .95], [-.9, -.9, .95]])

    def __init__(self):
        self.joint_actions = []
        self.prepared = []

    def key(self, state):
        return state

    def boundary(self, state):
        return {"both": DecisionBoundary.BOTH, "entry": DecisionBoundary.P1,
                "chance": DecisionBoundary.ADVANCE,
                "terminal": DecisionBoundary.TERMINAL}[state.phase]

    def legal_actions(self, state, side):
        return [0, 1, 2] if state.phase == "both" else [0]

    def prior(self, state, side, actions):
        return [1.0 if a == 2 else .0001 for a in actions]

    def apply_actions(self, state, action_p1, action_p2):
        if state.phase == "entry":
            return Position()
        self.joint_actions.append((action_p1, action_p2))
        return Position("terminal", float(self.matrix[action_p1, action_p2]))

    def terminal_value(self, state, root_side):
        if state.phase != "terminal":
            return None
        u = state.utility * (1 if root_side == 0 else -1)
        return WDL((1 + u) / 2, 0, (1 - u) / 2)

    def evaluate(self, state, root_side):
        raise AssertionError("analytic game needs no learned critic")

    def prepare_requests(self, requests):
        self.prepared.extend(requests)


@pytest.mark.parametrize("implementation", [JointEventSearch, QueuedJointEventSearch])
@pytest.mark.parametrize("root_side", [0, 1])
def test_complete_matrix_uses_mixture_and_preserves_perspective(implementation, root_side):
    model = MatrixModel()
    result = implementation(model, SearchConfig(
        depth_events=1, own_beam=1, opponent_beam=1, chance_beam=1,
        opponent_mode="mixed")).search(Position(), root_side=root_side)
    own = dict(zip(result.actions, result.policy_target, strict=True))
    other = dict(zip(result.opponent_actions, result.opponent_policy, strict=True))
    p, q = (own, other) if root_side == 0 else (other, own)
    np.testing.assert_allclose([p[i] for i in range(3)], [.4, .6, 0], atol=.01)
    np.testing.assert_allclose([q[i] for i in range(3)], [.5, .5, 0], atol=.01)
    assert result.root_value.utility == pytest.approx(.2 * (1 - 2 * root_side), abs=.01)
    assert set(model.joint_actions) == {(i, j) for i in range(3) for j in range(3)}
    assert result.equilibrium_converged and result.equilibrium_gap < .02
    assert result.matrix_games == 1 and result.matrix_solve_ms > 0
    assert result.action_selection == "sample_policy"
    assert not result.budget_exhausted
    with pytest.raises(ValueError, match="complete root"):
        implementation(model, SearchConfig(opponent_mode="mixed")).search(
            Position(), root_side=root_side, root_actions=[0, 1])


@pytest.mark.parametrize("implementation", [JointEventSearch, QueuedJointEventSearch])
def test_internal_simultaneous_node_uses_equilibrium(implementation):
    result = implementation(MatrixModel(), SearchConfig(
        depth_events=2, opponent_mode="mixed")).search(Position("entry"), root_side=0)
    assert result.root_value.utility == pytest.approx(.2, abs=.01)
    assert result.matrix_games == 1 and result.equilibrium_converged
    assert result.action_selection == "argmax"  # Root itself is a single-side choice.


@pytest.mark.parametrize("implementation", [JointEventSearch, QueuedJointEventSearch])
def test_mixed_keeps_complete_correlated_chance_support(implementation):
    class Reveals(MatrixModel):
        def apply_actions(self, state, action_p1, action_p2):
            terminal = super().apply_actions(state, action_p1, action_p2)
            return Position("chance", terminal.utility)

        def chance_outcomes(self, state):
            # Not independent one-ninth reveals. Chance outcomes have distinct
            # payoffs, so dropping and renormalizing either changes the game.
            return [ChanceOutcome(.25, Position("terminal", state.utility)),
                    ChanceOutcome(.75, Position("terminal", 0.0))]

    result = implementation(Reveals(), SearchConfig(
        depth_events=1, chance_beam=1, opponent_mode="mixed")).search(Position(), root_side=0)
    assert result.root_value.utility == pytest.approx(.05, abs=.01)
    assert result.chance_outcomes == 2 * result.chance_nodes
    assert result.equilibrium_converged


@pytest.mark.parametrize("implementation", [JointEventSearch, QueuedJointEventSearch])
def test_incomplete_or_unconverged_matrices_are_not_certified(implementation):
    model = MatrixModel()
    result = implementation(model, SearchConfig(
        max_nodes=1, opponent_mode="mixed")).search(Position(), root_side=0)
    assert result.budget_exhausted and not result.equilibrium_converged
    assert len(model.joint_actions) == 1
    assert result.matrix_solve_ms == 0
    result = implementation(MatrixModel(), SearchConfig(
        matrix_iterations=10, matrix_gap_tolerance=1e-10,
        matrix_solver="mirror_prox",
        opponent_mode="mixed")).search(Position(), root_side=0)
    assert not result.budget_exhausted and not result.equilibrium_converged
    assert result.equilibrium_gap > .02


def test_teacher_withholds_unconverged_targets(tmp_path, monkeypatch):
    from tools import joint_search_teacher

    source, output = tmp_path / "source.jsonl", tmp_path / "targets.jsonl"
    source.write_text('{"id":"analytic"}\n')
    monkeypatch.setattr(joint_search_teacher, "_adapter", lambda *args: (
        MatrixModel(), lambda payload: Position()))
    monkeypatch.setattr(sys, "argv", ["teacher", "--states", str(source), "--output",
        str(output), "--adapter", "test:model", "--opponent-mode", "mixed",
        "--matrix-solver", "mirror_prox",
        "--matrix-iterations", "10", "--matrix-gap-tolerance", "1e-10"])
    joint_search_teacher.main()
    row = json.loads(output.read_text())
    assert not row["usable_for_training"] and not row["equilibrium_converged"]
    assert row["unsearched_actions"] == []
    for field in ("policy_target", "opponent_policy", "utilities", "wdl", "best_action", "root_value"):
        assert row[field] is None


def test_common_critic_offset_changes_value_but_not_solver_dynamics():
    results = []
    for offset in (0, .8):
        model = MatrixModel()
        model.matrix = model.matrix * .1 + offset
        results.append(JointEventSearch(model, SearchConfig(
            opponent_mode="mixed")).search(Position(), root_side=0))
    a, b = results
    np.testing.assert_allclose(a.policy_target, b.policy_target, atol=1e-7)
    np.testing.assert_allclose(a.opponent_policy, b.opponent_policy, atol=1e-7)
    assert a.equilibrium_converged and b.equilibrium_converged
    assert b.root_value.utility - a.root_value.utility == pytest.approx(.8)


def test_execution_samples_reproducible_mixture_and_rejects_failed_solver(monkeypatch):
    from types import SimpleNamespace
    from scipy import optimize

    result = JointEventSearch(MatrixModel(), SearchConfig(opponent_mode="mixed")).search(
        Position(), root_side=0)
    left, right = np.random.default_rng(1923), np.random.default_rng(1923)
    choices = [result.select_action(left) for _ in range(100)]
    assert choices == [result.select_action(right) for _ in range(100)]
    assert set(choices) == {0, 1}  # A representative argmax alone is not execution.
    monkeypatch.setattr(optimize, "linprog", lambda *args, **kwargs: SimpleNamespace(
        success=False, message="time limit reached"))
    failed = JointEventSearch(MatrixModel(), SearchConfig(opponent_mode="mixed")).search(
        Position(), root_side=0)
    assert not failed.equilibrium_converged and failed.matrix_failures == ("time limit reached",)
    with pytest.raises(ValueError, match="unconverged"):
        failed.select_action(left)


def test_matrix_certificate_handles_nonunique_strategies_but_rejects_bad_ones():
    from dataclasses import replace
    from drmc_rl.search.matrix_check import compare_matrix_games

    result = JointEventSearch(MatrixModel(), SearchConfig(opponent_mode="mixed")).search(
        Position(), root_side=0)
    # Every strategy is optimal in a constant matrix. Vector identity would
    # reject two valid equilibria despite identical values and zero regret.
    left = replace(result, joint_utilities=np.full((3, 3), .7),
                   policy_target=np.array([1., 0, 0]), opponent_policy=(0., 1., 0.))
    right = replace(left, joint_utilities=np.full((3, 3), .700001),
                    policy_target=np.array([0., 0, 1.]), opponent_policy=(1., 0, 0.))
    report = compare_matrix_games(left, right)
    assert report["equivalent"] and report["own_policy_total_variation"] == 1
    assert report["bound_holds"] and report["rounding_gap_bound"] == pytest.approx(.000002)
    bad = replace(result, policy_target=np.eye(3)[0])
    assert not compare_matrix_games(result, bad)["equivalent"]
    with pytest.raises(ValueError, match="normalized"):
        compare_matrix_games(result, replace(result, opponent_policy=(float("nan"), 0, 1)))
