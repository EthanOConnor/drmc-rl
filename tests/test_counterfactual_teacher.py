from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drmc_rl.game.pair_state import DecisionBoundary
from drmc_rl.search.joint_event import ChanceOutcome, SearchConfig, WDL
from drmc_rl.teachers.counterfactual import CounterfactualTeacher


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
            "chance": DecisionBoundary.ADVANCE,
            "terminal": DecisionBoundary.TERMINAL,
        }[state.phase]

    def legal_actions(self, state, side):
        del state, side
        return [0, 1]

    def prior(self, state, side, actions):
        del state, side, actions
        return [0.7, 0.3]

    def apply_actions(self, state, action_p1, action_p2):
        return State("chance", state.score + (action_p1 or 0) - (action_p2 or 0))

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
        signed = state.score if root_side == 0 else -state.score
        return WDL.terminal(int(np.sign(signed)))

    def evaluate(self, state, root_side):
        del state, root_side
        return WDL(0.4, 0.2, 0.4)


def test_counterfactual_teacher_labels_all_actions() -> None:
    teacher = CounterfactualTeacher(Model(), config=SearchConfig(depth_events=3, own_beam=2))
    label = teacher.label(State("both"), root_side=0)
    assert len(label.candidates) == 2
    assert min(candidate.regret_win_logit for candidate in label.candidates) == 0
    assert label.best_action == 1
