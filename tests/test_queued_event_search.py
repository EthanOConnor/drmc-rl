from dataclasses import dataclass

import numpy as np
import pytest

from drmc_rl.game.pair_state import DecisionBoundary
from drmc_rl.search.joint_event import ChanceOutcome, JointEventSearch, SearchConfig, WDL
from drmc_rl.search.queued_event import QueuedJointEventSearch


@dataclass(frozen=True)
class State:
    phase: str
    score: int = 0
    depth: int = 0


class Model:
    def __init__(self):
        self.batches = []
        self.evaluated = []

    def key(self, s):
        return s

    def boundary(self, s):
        return dict(
            both=DecisionBoundary.BOTH,
            own=DecisionBoundary.P1,
            opponent=DecisionBoundary.P2,
            chance=DecisionBoundary.ADVANCE,
            settle=DecisionBoundary.ADVANCE,
            terminal=DecisionBoundary.TERMINAL,
        )[s.phase]

    def legal_actions(self, s, side):
        return [0, 1, 2]

    def prior(self, s, side, actions):
        return np.asarray([3, 2, 1])[: len(actions)]

    def apply_actions(self, s, a, b):
        return State("settle", s.score + (a or 0) - (b or 0), s.depth + 1)

    def advance(self, s):
        return State("chance", s.score, s.depth)

    def chance_outcomes(self, s):
        if s.phase != "chance":
            return []
        phase = "terminal" if s.depth >= 3 else "opponent" if s.depth == 1 else "own"
        return [
            ChanceOutcome(0.7, State(phase, s.score, s.depth)),
            ChanceOutcome(0.3, State(phase, s.score - 1, s.depth)),
        ]

    def terminal_value(self, s, side):
        return (
            WDL.terminal(np.sign(s.score) * (1 if side == 0 else -1))
            if s.phase == "terminal"
            else None
        )

    def evaluate(self, s, side):
        assert s.phase in ("own", "opponent", "both")
        self.evaluated.append(s)
        win = 1 / (1 + np.exp(-s.score * (1 if side == 0 else -1)))
        return WDL(win, 0.0, 1 - win)

    def prepare_batch(self, states):
        assert all(s.phase in ("own", "opponent", "both") for s in states)
        self.batches.append(states)


@pytest.mark.parametrize("mode", ["expectation", "minimax"])
@pytest.mark.parametrize("depth", [1, 3, 7])
@pytest.mark.parametrize("root", [State("own"), State("both")])
def test_cooperative_frontier_matches_recursive_chance_and_opponent_backups(mode, depth, root):
    config = SearchConfig(depth_events=depth, own_beam=3, opponent_beam=3, opponent_mode=mode)
    reference = JointEventSearch(Model(), config).search(root, root_side=0)
    model = Model()
    actual = QueuedJointEventSearch(model, config, batch_size=8).search(root, root_side=0)
    assert actual.actions == reference.actions and actual.best_action == reference.best_action
    np.testing.assert_allclose(actual.utilities, reference.utilities, atol=1e-7)
    np.testing.assert_allclose(actual.policy_target, reference.policy_target, atol=1e-7)
    assert not actual.budget_exhausted
    assert any(len(batch) > 1 for batch in model.batches)


def test_queued_budget_exhaustion_is_explicit_and_never_evaluates_forced_events():
    result = QueuedJointEventSearch(Model(), SearchConfig(max_nodes=1)).search(
        State("both"), root_side=0
    )
    assert result.budget_exhausted
