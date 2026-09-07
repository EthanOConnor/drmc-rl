from dataclasses import dataclass

import numpy as np
import pytest

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


def test_expired_depth_finishes_chance_before_evaluating_a_decision() -> None:
    class DecisionOnlyModel(Model):
        def evaluate(self, state, root_side):
            raise AssertionError("all forced branches terminate; no critic is needed")

    result = JointEventSearch(
        DecisionOnlyModel(), SearchConfig(depth_events=1, own_beam=2, opponent_beam=2)
    ).search(State("both"), root_side=0)
    assert result.best_action == 1
    assert result.chance_nodes == 3
    assert result.cache_hits == 1
    assert not result.budget_exhausted


def test_expired_depth_stops_before_the_next_player_action() -> None:
    class ForcedEvents(Model):
        def __init__(self):
            self.played = []
            self.evaluated = []

        def boundary(self, state):
            if state.phase == "settle":
                return DecisionBoundary.ADVANCE
            return super().boundary(state)

        def apply_actions(self, state, action_p1, action_p2):
            self.played.append(state.phase)
            assert state.phase == "p1"
            return State("settle", action_p1)

        def advance(self, state):
            return State("chance", state.score)

        def chance_outcomes(self, state):
            if state.phase == "settle":
                return []
            return [
                ChanceOutcome(.75, State("p2", state.score)),
                ChanceOutcome(.25, State("p2", state.score - 1)),
            ]

        def evaluate(self, state, root_side):
            assert state.phase == "p2"
            self.evaluated.append(state.score)
            return WDL(.5 + .2 * state.score, 0, .5 - .2 * state.score)

    model = ForcedEvents()
    result = JointEventSearch(model, SearchConfig(depth_events=1)).search(
        State("p1"), root_side=0
    )
    assert result.best_action == 1
    assert model.played == ["p1", "p1"]
    assert len(model.evaluated) == 4
    assert result.values[0].win == pytest.approx(.45)
    assert result.values[1].win == pytest.approx(.65)


def test_budget_exhaustion_never_falls_back_to_an_unsupported_critic() -> None:
    class NoFallback(Model):
        def evaluate(self, state, root_side):
            raise AssertionError("budget exhaustion cannot feed an unsupported state")

    result = JointEventSearch(NoFallback(), SearchConfig(depth_events=1, max_nodes=1)).search(
        State("both"), root_side=0
    )
    assert result.budget_exhausted
    assert np.isfinite(result.utilities).all()


def test_forced_advance_without_progress_fails_instead_of_looping() -> None:
    class Stuck(Model):
        def chance_outcomes(self, state):
            return []

        def advance(self, state):
            return state

    with pytest.raises(RuntimeError, match="advance made no progress"):
        JointEventSearch(Stuck(), SearchConfig(depth_events=1)).search(State("both"), root_side=0)


def test_observed_action_value_matches_full_root_search() -> None:
    search = JointEventSearch(Model(), SearchConfig(depth_events=2))
    full = search.search(State("both"), root_side=0)
    for action, value in zip(full.actions, full.values, strict=True):
        single = search.search(State("both"), root_side=0, root_actions=[action])
        assert single.actions == (action,)
        assert single.values == (value,)
        assert single.nodes < full.nodes
    for invalid in ([7], [0, 0]):
        with pytest.raises(ValueError, match="unique and legal"):
            search.search(State("both"), root_side=0, root_actions=invalid)
