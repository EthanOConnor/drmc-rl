"""Strict event-driven search over an asynchronous two-player pair model.

This module is intentionally backend-independent.  A native adapter supplies
restorable pair states and exact transitions; the search owns event ordering,
opponent integration, chance expectation, transposition caching, and root
policy improvement.  It replaces the conceptual model of alternating only the
learner's pills.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, Hashable, Protocol, Sequence, TypeVar, runtime_checkable

import numpy as np

from drmc_rl.game.pair_state import DecisionBoundary

StateT = TypeVar("StateT")


@dataclass(frozen=True, slots=True)
class WDL:
    win: float
    draw: float
    loss: float

    def __post_init__(self) -> None:
        values = np.asarray((self.win, self.draw, self.loss), dtype=np.float64)
        if not np.isfinite(values).all() or (values < -1e-9).any():
            raise ValueError(f"invalid W/D/L values: {values}")
        if abs(float(values.sum()) - 1.0) > 1e-5:
            raise ValueError(f"W/D/L must sum to one, got {values.sum()}")

    @property
    def utility(self) -> float:
        return float(self.win - self.loss)

    @property
    def expected_score(self) -> float:
        return float(self.win + 0.5 * self.draw)

    @classmethod
    def from_logits(cls, logits: Sequence[float]) -> "WDL":
        values = np.asarray(logits, dtype=np.float64).reshape(3)
        values -= values.max()
        probability = np.exp(np.clip(values, -60.0, 0.0))
        probability /= probability.sum()
        return cls(*map(float, probability))

    @classmethod
    def terminal(cls, outcome: int) -> "WDL":
        if outcome > 0:
            return cls(1.0, 0.0, 0.0)
        if outcome < 0:
            return cls(0.0, 0.0, 1.0)
        return cls(0.0, 1.0, 0.0)

    @classmethod
    def mixture(cls, weights: Sequence[float], values: Sequence["WDL"]) -> "WDL":
        if not values:
            raise ValueError("cannot mix an empty WDL sequence")
        probability = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(probability) != len(values) or (probability < 0).any():
            raise ValueError("mixture weights must match values and be non-negative")
        total = float(probability.sum())
        if total <= 0:
            raise ValueError("mixture weights must have positive mass")
        probability /= total
        matrix = np.asarray([(v.win, v.draw, v.loss) for v in values], dtype=np.float64)
        result = probability @ matrix
        result = np.maximum(result, 0.0)
        result /= result.sum()
        return cls(*map(float, result))


@dataclass(frozen=True, slots=True)
class ChanceOutcome(Generic[StateT]):
    probability: float
    state: StateT

    def __post_init__(self) -> None:
        if not math.isfinite(self.probability) or self.probability < 0:
            raise ValueError("chance probability must be finite and non-negative")


@runtime_checkable
class PairSearchModel(Protocol[StateT]):
    """Exact or approximate pair model consumed by :class:`JointEventSearch`."""

    def key(self, state: StateT) -> Hashable:
        """Return a stable transposition key for all value-relevant state."""

    def boundary(self, state: StateT) -> DecisionBoundary:
        """Return which side needs action, or whether to advance/terminate."""

    def legal_actions(self, state: StateT, side: int) -> Sequence[int]:
        """Return exact legal macro actions for one parked side."""

    def prior(self, state: StateT, side: int, actions: Sequence[int]) -> Sequence[float]:
        """Return action probabilities or unnormalized non-negative weights."""

    def apply_actions(
        self,
        state: StateT,
        action_p1: int | None,
        action_p2: int | None,
    ) -> StateT:
        """Apply supplied actions and advance causally to the next pair event."""

    def advance(self, state: StateT) -> StateT:
        """Advance a deterministic no-decision event to the next pair event."""

    def chance_outcomes(self, state: StateT) -> Sequence[ChanceOutcome[StateT]]:
        """Return newly revealed chance outcomes; empty means deterministic advance."""

    def terminal_value(self, state: StateT, root_side: int) -> WDL | None:
        """Return terminal W/D/L from the root side, or None if ongoing."""

    def evaluate(self, state: StateT, root_side: int) -> WDL:
        """Return a calibrated leaf estimate from public/belief state."""


@dataclass(frozen=True, slots=True)
class SearchConfig:
    depth_events: int = 4
    own_beam: int = 32
    opponent_beam: int = 12
    chance_beam: int = 16
    opponent_mode: str = "expectation"  # expectation|minimax
    policy_temperature: float = 0.25
    prior_floor: float = 1e-6
    max_nodes: int = 100000

    def __post_init__(self) -> None:
        if self.depth_events < 1:
            raise ValueError("depth_events must be positive")
        if min(self.own_beam, self.opponent_beam, self.chance_beam) < 1:
            raise ValueError("beam widths must be positive")
        if self.opponent_mode not in {"expectation", "minimax"}:
            raise ValueError("opponent_mode must be expectation or minimax")
        if self.policy_temperature < 0:
            raise ValueError("policy_temperature cannot be negative")
        if self.prior_floor <= 0:
            raise ValueError("prior_floor must be positive")
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be positive")


@dataclass(frozen=True, slots=True)
class SearchResult:
    actions: tuple[int, ...]
    values: tuple[WDL, ...]
    utilities: np.ndarray
    policy_target: np.ndarray
    best_action: int
    root_value: WDL
    nodes: int
    cache_hits: int
    depth_events: int
    budget_exhausted: bool


class JointEventSearch(Generic[StateT]):
    def __init__(self, model: PairSearchModel[StateT], config: SearchConfig | None = None):
        self.model = model
        self.config = config or SearchConfig()
        self._cache: dict[tuple[Hashable, int, int], WDL] = {}
        self._nodes = 0
        self._cache_hits = 0
        self._budget_exhausted = False

    def search(self, state: StateT, *, root_side: int) -> SearchResult:
        if root_side not in (0, 1):
            raise ValueError("root_side must be 0 or 1")
        self._cache.clear()
        self._nodes = 0
        self._cache_hits = 0
        self._budget_exhausted = False
        terminal = self.model.terminal_value(state, root_side)
        if terminal is not None:
            raise ValueError("cannot search from a terminal state")
        boundary = self.model.boundary(state)
        expected = DecisionBoundary.P1 if root_side == 0 else DecisionBoundary.P2
        if boundary not in {expected, DecisionBoundary.BOTH}:
            raise ValueError(f"root side {root_side} is not acting at boundary {boundary.value!r}")

        own_actions = self._ranked_actions(state, root_side, self.config.own_beam, maximize=True)
        if not own_actions:
            raise ValueError("root side has no legal actions")
        values: list[WDL] = []
        for action in own_actions:
            if boundary == DecisionBoundary.BOTH:
                value = self._simultaneous_given_own(
                    state, root_side, action, self.config.depth_events
                )
            else:
                p1 = action if root_side == 0 else None
                p2 = action if root_side == 1 else None
                child = self.model.apply_actions(state, p1, p2)
                value = self._value(child, self.config.depth_events - 1, root_side)
            values.append(value)
        utilities = np.asarray([value.utility for value in values], dtype=np.float64)
        policy = self._policy_target(utilities)
        best_index = int(np.argmax(utilities))
        root_value = WDL.mixture(policy, values)
        return SearchResult(
            actions=tuple(own_actions),
            values=tuple(values),
            utilities=utilities.astype(np.float32),
            policy_target=policy.astype(np.float32),
            best_action=int(own_actions[best_index]),
            root_value=root_value,
            nodes=self._nodes,
            cache_hits=self._cache_hits,
            depth_events=self.config.depth_events,
            budget_exhausted=self._budget_exhausted,
        )

    def _value(self, state: StateT, depth: int, root_side: int) -> WDL:
        self._nodes += 1
        if self._nodes > self.config.max_nodes:
            self._budget_exhausted = True
            return self.model.evaluate(state, root_side)
        terminal = self.model.terminal_value(state, root_side)
        if terminal is not None:
            return terminal
        if depth <= 0:
            return self.model.evaluate(state, root_side)
        cache_key = (self.model.key(state), int(depth), int(root_side))
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        boundary = self.model.boundary(state)
        if boundary == DecisionBoundary.ADVANCE:
            outcomes = tuple(self.model.chance_outcomes(state))
            if outcomes:
                value = self._chance_value(outcomes, depth, root_side)
            else:
                value = self._value(self.model.advance(state), depth - 1, root_side)
        elif boundary == DecisionBoundary.BOTH:
            value = self._simultaneous_value(state, depth, root_side)
        elif boundary in {DecisionBoundary.P1, DecisionBoundary.P2}:
            acting_side = 0 if boundary == DecisionBoundary.P1 else 1
            value = self._single_side_value(state, acting_side, depth, root_side)
        elif boundary == DecisionBoundary.TERMINAL:
            value = self.model.terminal_value(state, root_side) or self.model.evaluate(
                state, root_side
            )
        else:  # pragma: no cover - enum guard
            raise RuntimeError(f"unsupported decision boundary {boundary}")
        self._cache[cache_key] = value
        return value

    def _single_side_value(
        self, state: StateT, acting_side: int, depth: int, root_side: int
    ) -> WDL:
        is_root = acting_side == root_side
        beam = self.config.own_beam if is_root else self.config.opponent_beam
        actions = self._ranked_actions(state, acting_side, beam, maximize=is_root)
        if not actions:
            return self.model.evaluate(state, root_side)
        children: list[WDL] = []
        for action in actions:
            p1 = action if acting_side == 0 else None
            p2 = action if acting_side == 1 else None
            child = self.model.apply_actions(state, p1, p2)
            children.append(self._value(child, depth - 1, root_side))
        if is_root:
            return children[int(np.argmax([child.utility for child in children]))]
        return self._opponent_backup(state, acting_side, actions, children)

    def _simultaneous_value(self, state: StateT, depth: int, root_side: int) -> WDL:
        own_actions = self._ranked_actions(state, root_side, self.config.own_beam, maximize=True)
        if not own_actions:
            return self.model.evaluate(state, root_side)
        values = [
            self._simultaneous_given_own(state, root_side, action, depth) for action in own_actions
        ]
        return values[int(np.argmax([value.utility for value in values]))]

    def _simultaneous_given_own(
        self, state: StateT, root_side: int, own_action: int, depth: int
    ) -> WDL:
        opponent = 1 - root_side
        opp_actions = self._ranked_actions(
            state, opponent, self.config.opponent_beam, maximize=False
        )
        if not opp_actions:
            p1 = own_action if root_side == 0 else None
            p2 = own_action if root_side == 1 else None
            child = self.model.apply_actions(state, p1, p2)
            return self._value(child, depth - 1, root_side)
        values: list[WDL] = []
        for opp_action in opp_actions:
            p1 = own_action if root_side == 0 else opp_action
            p2 = own_action if root_side == 1 else opp_action
            child = self.model.apply_actions(state, p1, p2)
            values.append(self._value(child, depth - 1, root_side))
        return self._opponent_backup(state, opponent, opp_actions, values)

    def _opponent_backup(
        self,
        state: StateT,
        opponent_side: int,
        actions: Sequence[int],
        values: Sequence[WDL],
    ) -> WDL:
        if self.config.opponent_mode == "minimax":
            return values[int(np.argmin([value.utility for value in values]))]
        weights = self._prior_weights(state, opponent_side, actions)
        return WDL.mixture(weights, values)

    def _chance_value(
        self, outcomes: Sequence[ChanceOutcome[StateT]], depth: int, root_side: int
    ) -> WDL:
        ordered = sorted(outcomes, key=lambda item: item.probability, reverse=True)[
            : self.config.chance_beam
        ]
        if not ordered:
            return self.model.evaluate(outcomes[0].state, root_side)  # pragma: no cover
        weights = np.asarray([item.probability for item in ordered], dtype=np.float64)
        values = [self._value(item.state, depth - 1, root_side) for item in ordered]
        return WDL.mixture(weights, values)

    def _ranked_actions(self, state: StateT, side: int, beam: int, *, maximize: bool) -> list[int]:
        actions = [int(action) for action in self.model.legal_actions(state, side)]
        if not actions:
            return []
        weights = self._prior_weights(state, side, actions)
        # Priors determine expansion order/beam only. Root quality comes from
        # backed-up WDL values, so a weak prior cannot directly choose the move.
        order = np.argsort(-weights, kind="stable")
        return [actions[int(index)] for index in order[: min(int(beam), len(actions))]]

    def _prior_weights(self, state: StateT, side: int, actions: Sequence[int]) -> np.ndarray:
        raw = np.asarray(self.model.prior(state, side, actions), dtype=np.float64).reshape(-1)
        if raw.shape != (len(actions),):
            raise ValueError("model prior must match legal action count")
        if not np.isfinite(raw).all():
            raise ValueError("model prior contains non-finite values")
        if (raw < 0).any() or raw.sum() <= 0:
            # Treat arbitrary logits as logits if they are not probabilities.
            raw = raw - raw.max()
            raw = np.exp(np.clip(raw, -60.0, 0.0))
        raw = np.maximum(raw, self.config.prior_floor)
        return raw / raw.sum()

    def _policy_target(self, utilities: np.ndarray) -> np.ndarray:
        if self.config.policy_temperature == 0:
            target = np.zeros_like(utilities)
            target[int(np.argmax(utilities))] = 1.0
            return target
        centered = utilities - utilities.max()
        weights = np.exp(np.clip(centered / max(self.config.policy_temperature, 1e-8), -60.0, 0.0))
        return weights / weights.sum()


__all__ = [
    "ChanceOutcome",
    "JointEventSearch",
    "PairSearchModel",
    "SearchConfig",
    "SearchResult",
    "WDL",
]
