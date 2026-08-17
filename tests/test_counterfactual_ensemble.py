from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from drmc_rl.game.pair_state import DecisionBoundary
from drmc_rl.search.joint_event import SearchConfig, WDL
from drmc_rl.teachers.counterfactual import CounterfactualTeacher, WeightedTeacherModels


@dataclass(frozen=True)
class State:
    action: int | None = None


class Model:
    def __init__(self, values: tuple[WDL, WDL]):
        self.values = values

    def key(self, state):
        return state.action

    def boundary(self, state):
        return DecisionBoundary.P1 if state.action is None else DecisionBoundary.TERMINAL

    def legal_actions(self, state, side):
        del state, side
        return (0, 1)

    def prior(self, state, side, actions):
        del state, side
        return [1.0] * len(actions)

    def apply_actions(self, state, action_p1, action_p2):
        del state, action_p2
        return State(int(action_p1))

    def advance(self, state):
        return state

    def chance_outcomes(self, state):
        del state
        return ()

    def terminal_value(self, state, root_side):
        del root_side
        return None if state.action is None else self.values[state.action]

    def evaluate(self, state, root_side):
        del state, root_side
        return WDL(0.4, 0.2, 0.4)


def test_weighted_teacher_exports_member_values_and_disagreement() -> None:
    first = Model((WDL(0.8, 0.1, 0.1), WDL(0.3, 0.2, 0.5)))
    second = Model((WDL(0.2, 0.2, 0.6), WDL(0.7, 0.1, 0.2)))
    ensemble = WeightedTeacherModels(
        models=(first, second), weights=(0.75, 0.25), ids=("main", "exploiter")
    )
    label = CounterfactualTeacher(
        ensemble, config=SearchConfig(depth_events=1, own_beam=2)
    ).label(State(), root_side=0)

    assert label.schema == "drmc-counterfactual-pair-label-v3"
    assert label.teacher_ids == ("main", "exploiter")
    assert label.teacher_weights == pytest.approx((0.75, 0.25))
    candidate = next(item for item in label.candidates if item.action == 0)
    assert candidate.win == pytest.approx(0.65)
    assert candidate.draw == pytest.approx(0.125)
    assert candidate.loss == pytest.approx(0.225)
    assert np.asarray(candidate.member_wdl) == pytest.approx(
        np.asarray(((0.8, 0.1, 0.1), (0.2, 0.2, 0.6)))
    )
    assert candidate.uncertainty is not None and candidate.uncertainty > 0
    assert candidate.uncertainty_js is not None and candidate.uncertainty_js > 0
    assert label.uncertainty_available is True


def test_weighted_teacher_rejects_invalid_weights() -> None:
    model = Model((WDL(1.0, 0.0, 0.0), WDL(0.0, 0.0, 1.0)))
    with pytest.raises(ValueError, match="positive mass"):
        WeightedTeacherModels((model, model), (0.0, 0.0))
