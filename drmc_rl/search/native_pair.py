"""Native strict-pair adapter used by counterfactual pilot tooling.

The included factory is deliberately named ``diagnostic_factory``: its leaf
value is a transparent public-state heuristic, not a calibrated competitive
teacher. It validates restore, causal branching, coverage, and release
mechanics before frozen checkpoint-mixture evaluation is admitted.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from drmc_rl.envs.backends.drmario_vs_pool import (
    VS_OUTCOME_DRAW,
    VS_OUTCOME_LOSS,
    VS_OUTCOME_WIN,
    DrMarioVsPoolRunner,
)
from drmc_rl.game.pair_state import DecisionBoundary, PrivilegedPairState
from drmc_rl.search.joint_event import ChanceOutcome, WDL


@dataclass(frozen=True, slots=True)
class NativePairSearchState:
    privileged: PrivilegedPairState
    legal_actions_by_side: tuple[tuple[int, ...], tuple[int, ...]]
    action_costs_by_side: tuple[tuple[int, ...], tuple[int, ...]]
    level: int = 20
    speed_setting: int = 2
    viruses_initial: tuple[int, int] = (84, 84)

    def __post_init__(self) -> None:
        for actions, costs in zip(
            self.legal_actions_by_side, self.action_costs_by_side, strict=True
        ):
            if len(actions) != len(costs):
                raise ValueError("native actions and costs must have matching lengths")
            if len(set(actions)) != len(actions) or any(
                not 0 <= action < 512 for action in actions
            ):
                raise ValueError("native legal actions must be unique indices in [0,511]")
            if any(cost <= 0 or cost >= 0xFFFF for cost in costs):
                raise ValueError("native legal action costs must be finite positive frames")
        if not 0 <= self.level <= 20 or self.speed_setting not in (0, 1, 2):
            raise ValueError("native search context has invalid level or speed")


class ContinuationEvaluator(Protocol):
    def prior(
        self, state: NativePairSearchState, side: int, actions: Sequence[int]
    ) -> Sequence[float]: ...

    def evaluate(self, state: NativePairSearchState, root_side: int) -> WDL: ...


class NativePairSearchModel:
    """Exact native transitions with a public-only diagnostic prior/value."""

    def __init__(
        self,
        runner: DrMarioVsPoolRunner,
        *,
        continuation: ContinuationEvaluator | None = None,
        reveal_chance: bool = False,
    ) -> None:
        if runner.num_pairs != 1:
            raise ValueError("native search adapter requires a dedicated one-pair runner")
        self.runner = runner
        self.continuation = continuation
        self.reveal_chance = bool(reveal_chance)

    def key(self, state: NativePairSearchState) -> str:
        return hashlib.sha256(state.privileged.engine_checkpoint).hexdigest()

    def boundary(self, state: NativePairSearchState) -> DecisionBoundary:
        return state.privileged.decision_boundary

    def legal_actions(self, state: NativePairSearchState, side: int) -> Sequence[int]:
        return state.legal_actions_by_side[int(side)]

    def prior(
        self, state: NativePairSearchState, side: int, actions: Sequence[int]
    ) -> Sequence[float]:
        if self.continuation is not None:
            return self.continuation.prior(state, side, actions)
        action_to_cost = dict(
            zip(
                state.legal_actions_by_side[int(side)],
                state.action_costs_by_side[int(side)],
                strict=True,
            )
        )
        costs = np.asarray([action_to_cost[int(action)] for action in actions], dtype=np.float64)
        costs -= costs.min()
        return np.exp(-costs / 24.0)

    def apply_actions(
        self,
        state: NativePairSearchState,
        action_p1: int | None,
        action_p2: int | None,
    ) -> NativePairSearchState:
        self.runner.restore(0, state.privileged.engine_checkpoint)
        actions = np.asarray(
            [
                -2 if action_p1 is None else int(action_p1),
                -2 if action_p2 is None else int(action_p2),
            ],
            dtype=np.int32,
        )
        if self.reveal_chance:
            self.runner.step_search(actions)
        else:
            self.runner.step_strict(actions)
        if np.any(self.runner.buffers.invalid_action >= 0):
            raise RuntimeError(f"native strict branch rejected actions {actions.tolist()}")
        return capture_native_state(
            self.runner,
            level=state.level,
            speed_setting=state.speed_setting,
            viruses_initial=state.viruses_initial,
        )

    def advance(self, state: NativePairSearchState) -> NativePairSearchState:
        return self.apply_actions(state, None, None)

    def chance_outcomes(
        self, state: NativePairSearchState
    ) -> Sequence[ChanceOutcome[NativePairSearchState]]:
        if not self.reveal_chance:
            return ()
        self.runner.restore(0, state.privileged.engine_checkpoint)
        reveal = self.runner.search_reveal_info(0)
        if reveal is None:
            return ()
        side, _reserve_index = reveal
        outcomes: list[ChanceOutcome[NativePairSearchState]] = []
        for left in range(3):
            for right in range(3):
                self.runner.restore(0, state.privileged.engine_checkpoint)
                self.runner.search_reveal(0, side, (left, right))
                outcomes.append(
                    ChanceOutcome(
                        1.0 / 9.0,
                        capture_native_state(
                            self.runner,
                            level=state.level,
                            speed_setting=state.speed_setting,
                            viruses_initial=state.viruses_initial,
                        ),
                    )
                )
        return tuple(outcomes)

    def terminal_value(self, state: NativePairSearchState, root_side: int) -> WDL | None:
        outcome = state.privileged.terminal_outcome[int(root_side)]
        if outcome == VS_OUTCOME_WIN:
            return WDL.terminal(1)
        if outcome == VS_OUTCOME_LOSS:
            return WDL.terminal(-1)
        if outcome == VS_OUTCOME_DRAW:
            return WDL.terminal(0)
        return None

    def evaluate(self, state: NativePairSearchState, root_side: int) -> WDL:
        if self.continuation is not None:
            return self.continuation.evaluate(state, root_side)
        public = state.privileged.public
        own = public.sides[int(root_side)]
        opponent = public.sides[1 - int(root_side)]
        own_viruses = float(own.viruses_remaining or 0)
        opponent_viruses = float(opponent.viruses_remaining or 0)
        own_height = _board_height(own.board)
        opponent_height = _board_height(opponent.board)
        pending = state.privileged.pending_attacks
        advantage = (
            0.10 * (opponent_viruses - own_viruses)
            + 0.18 * (opponent_height - own_height)
            + 0.10 * (pending[1 - int(root_side)] - pending[int(root_side)])
        )
        decisive_mass = 0.95
        win = decisive_mass / (1.0 + math.exp(-max(-20.0, min(20.0, advantage))))
        loss = decisive_mass - win
        return WDL(float(win), 1.0 - decisive_mass, float(loss))


def _board_height(board: bytes) -> int:
    for row in range(16):
        if any(tile != 0xFF for tile in board[row * 8 : (row + 1) * 8]):
            return 16 - row
    return 0


def capture_native_state(
    runner: DrMarioVsPoolRunner,
    *,
    viewer_side: int = 0,
    level: int = 20,
    speed_setting: int = 2,
    viruses_initial: tuple[int, int] | None = None,
) -> NativePairSearchState:
    from drmc_rl.game.pair_state import PublicPairState, VisibleSideState

    buffers = runner.buffers
    need = tuple(bool(item) for item in buffers.need_action[:2])
    outcome = tuple(int(item) for item in buffers.outcome[:2])
    if any(outcome):
        boundary = DecisionBoundary.TERMINAL
    elif need == (True, True):
        boundary = DecisionBoundary.BOTH
    elif need[0]:
        boundary = DecisionBoundary.P1
    elif need[1]:
        boundary = DecisionBoundary.P2
    else:
        boundary = DecisionBoundary.ADVANCE
    sides = tuple(
        VisibleSideState(
            board=bytes(buffers.board_bytes[side]),
            pill=tuple(int(value) for value in buffers.pill_colors[side]),
            preview=tuple(int(value) for value in buffers.preview_colors[side]),
            active=None,
            viruses_remaining=int(buffers.viruses_rem[side]),
            animation_phase="decision" if need[side] else "resolving",
        )
        for side in range(2)
    )
    public = PublicPairState(
        frame_id=int(max(buffers.side_frames[:2])),
        viewer_side=int(viewer_side),
        sides=sides,  # type: ignore[arg-type]
        decision_boundary=boundary,
        observable_clock_delta_frames=int(buffers.side_frames[0]) - int(buffers.side_frames[1]),
    )
    privileged = PrivilegedPairState(
        public=public,
        pair_clocks=tuple(int(item) for item in buffers.side_frames[:2]),
        need_action=need,  # type: ignore[arg-type]
        pending_attacks=tuple(int(item) for item in buffers.garbage_pending[:2]),
        native_phases=tuple("decision" if flag else "resolving" for flag in need),
        committed_actions=(None, None),
        engine_checkpoint=runner.snapshot(0),
        terminal_outcome=outcome,  # type: ignore[arg-type]
    )
    actions: list[tuple[int, ...]] = []
    costs: list[tuple[int, ...]] = []
    for side in range(2):
        legal = tuple(int(item) for item in np.flatnonzero(buffers.feasible_mask[side]))
        actions.append(legal)
        costs.append(tuple(int(buffers.cost_to_lock[side, action]) for action in legal))
    return NativePairSearchState(
        privileged=privileged,
        legal_actions_by_side=(actions[0], actions[1]),
        action_costs_by_side=(costs[0], costs[1]),
        level=int(level),
        speed_setting=int(speed_setting),
        viruses_initial=(
            tuple(int(item) for item in viruses_initial)
            if viruses_initial is not None
            else (min(84, 4 * (int(level) + 1)),) * 2
        ),
    )


def state_to_payload(state: NativePairSearchState) -> dict[str, Any]:
    return {
        "privileged": state.privileged.to_dict(),
        "legal_actions_by_side": [list(items) for items in state.legal_actions_by_side],
        "action_costs_by_side": [list(items) for items in state.action_costs_by_side],
        "level": state.level,
        "speed_setting": state.speed_setting,
        "viruses_initial": list(state.viruses_initial),
    }


def state_from_payload(payload: Mapping[str, Any]) -> NativePairSearchState:
    return NativePairSearchState(
        privileged=PrivilegedPairState.from_dict(dict(payload["privileged"])),
        legal_actions_by_side=tuple(
            tuple(int(item) for item in side) for side in payload["legal_actions_by_side"]
        ),  # type: ignore[arg-type]
        action_costs_by_side=tuple(
            tuple(int(item) for item in side) for side in payload["action_costs_by_side"]
        ),  # type: ignore[arg-type]
        level=int(payload.get("level", 20)),
        speed_setting=int(payload.get("speed_setting", payload.get("speed", 2))),
        viruses_initial=tuple(
            int(item)
            for item in payload.get(
                "viruses_initial",
                (min(84, 4 * (int(payload.get("level", 20)) + 1)),) * 2,
            )
        ),  # type: ignore[arg-type]
    )


def diagnostic_factory(args: Any):
    """CLI adapter for restore/coverage smoke pilots, never quality promotion."""

    runner = DrMarioVsPoolRunner(num_pairs=1)
    model = NativePairSearchModel(runner)

    def decode(payload: dict[str, Any]) -> NativePairSearchState:
        return state_from_payload(payload)

    return model, decode


__all__ = [
    "NativePairSearchModel",
    "NativePairSearchState",
    "capture_native_state",
    "diagnostic_factory",
    "state_from_payload",
    "state_to_payload",
]
