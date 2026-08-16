from dataclasses import dataclass

import numpy as np

from drmc_rl.game.pair_state import DecisionBoundary
from drmc_rl.search.joint_event import ChanceOutcome, JointEventSearch, SearchConfig, WDL


@dataclass(frozen=True)
class State:
    phase: str
    score: int = 0


class Model:
    def key(self, state):
        return (state.phase, state.score)

    def boundary(self, state):
        return {
            "both": DecisionBoundary.BOTH,
            "p1": DecisionBoundary.P1,
            "p2": DecisionBoundary.P2,
            "chance": DecisionBoundary.ADVANCE,
            "terminal": DecisionBoundary.TERMINAL,
        }[state.phase]

    def legal_actions(self, state, side):
        return [0, 1]

    def prior(self, state, side, actions):
        return [0.7, 0.3]

    def apply_actions(self, state, action_p1, action_p2):
        if state.phase == "both":
            return State("chance", (action_p1 or 0) - (action_p2 or 0))
        if state.phase == "p1":
            return State("terminal", state.score + (action_p1 or 0))
        if state.phase == "p2":
            return State("terminal", state.score - (action_p2 or 0))
        raise AssertionError(state)

    def advance(self, state):
        return State("terminal", state.score)

    def chance_outcomes(self, state):
        return [
            ChanceOutcome(0.75, State("terminal", state.score)),
            ChanceOutcome(0.25, State("terminal", state.score - 1)),
        ]

    def terminal_value(self, state, root_side):
        if state.phase != "terminal":
            return None
        return WDL.terminal(np.sign(state.score if root_side == 0 else -state.score))

    def evaluate(self, state, root_side):
        return WDL(0.4, 0.2, 0.4)


def test_simultaneous_event_search_integrates_opponent_and_chance() -> None:
    search = JointEventSearch(Model(), SearchConfig(depth_events=3, own_beam=2, opponent_beam=2))
    result = search.search(State("both"), root_side=0)
    assert result.best_action == 1
    assert np.isclose(result.policy_target.sum(), 1.0)
    assert result.nodes > 0
