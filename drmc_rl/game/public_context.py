"""Versioned, allowlisted model inputs derived exclusively from public views.

The old zero-aux contract is unchanged. V3 preserves bottle bonds and exposes
both visible pills/previews, poses, phases, public events and own execution.
Missing observations have explicit masks; native restore bytes are never read.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from drmc_rl.game.pair_state import DecisionBoundary, PublicPairState
from drmc_rl.models.policy.event_belief import (
    EVENT_FEATURE_NAMES,
    pair_events_to_features,
)

PUBLIC_CONTEXT_SCHEMA = "public_pair_context_v3"
PHASES = (
    "unknown",
    "decision",
    "resolving",
    "falling",
    "clearing",
    "settling",
    "garbage",
    "spawn",
    "terminal",
)
SIDE_FEATURE_NAMES = (
    *(
        f"{piece}_{half}_{color}"
        for piece in ("pill", "preview")
        for half in ("a", "b")
        for color in ("r", "y", "b")
    ),
    "active_known",
    "column",
    "row_top",
    *(f"rotation_{i}" for i in range(4)),
    *(f"active_{half}_{color}" for half in ("a", "b") for color in ("r", "y", "b")),
    "controllable",
    "active_age",
    "viruses_known",
    "viruses_remaining",
    "state_age",
    *(f"phase_{phase}" for phase in PHASES),
)
SIDE_FEATURE_DIM = len(SIDE_FEATURE_NAMES)
HISTORY_LENGTH = 32
EXECUTION_FIELDS = (
    "reaction_frames",
    "edge_interval",
    "motion_interval",
    "max_buttons",
    "gravity_frames",
    "speed_ups",
    "decision_delay_frames",
    "compute_frames",
)
EXECUTION_SCALES = np.asarray((60.0, 12.0, 24.0, 3.0, 81.0, 49.0, 60.0, 60.0))
CONTEXT_FEATURE_NAMES = (
    *(f"{side}.{name}" for side in ("own", "opponent") for name in SIDE_FEATURE_NAMES),
    "own_decision",
    "opponent_decision",
    "terminal",
    "clock_delta_known",
    "clock_delta",
    "game_age",
    "execution_known",
    *EXECUTION_FIELDS,
    *(
        f"event_{i}.{name}"
        for i in range(HISTORY_LENGTH)
        for name in (*EVENT_FEATURE_NAMES, "known")
    ),
)
PUBLIC_CONTEXT_DIM = len(CONTEXT_FEATURE_NAMES)


@dataclass(frozen=True)
class PublicExecutionContext:
    reaction_frames: int
    edge_interval: int
    motion_interval: int
    max_buttons: int
    gravity_frames: int
    speed_ups: int
    decision_delay_frames: int
    compute_frames: int

    def __post_init__(self):
        if any(
            type(getattr(self, key)) is not int or getattr(self, key) < 0
            for key in EXECUTION_FIELDS
        ):
            raise ValueError("own execution fields require nonnegative integer frame counts")
        if (
            not 1 <= self.max_buttons <= 3
            or not 1 <= self.gravity_frames <= 81
            or self.speed_ups > 49
        ):
            raise ValueError("invalid own motor or gravity limits")


def _age(frames):
    return np.sign(frames) * np.log1p(abs(frames)) / np.log1p(600.0)


def _colors(colors):
    return np.eye(3, dtype=np.float32)[list(colors)].reshape(-1).tolist()


def _side_features(side):
    active = side.active
    features = _colors(side.pill) + _colors(side.preview)
    features += (
        [
            1.0,
            active.column / 7,
            active.row_top / 15,
            *np.eye(4)[active.rotation],
            *_colors(active.colors),
            float(active.controllable),
            _age(active.age_frames),
        ]
        if active
        else [0.0] * 15
    )
    features += [
        float(side.viruses_remaining is not None),
        (side.viruses_remaining or 0) / 84,
        _age(side.state_age_frames),
    ]
    phase = side.animation_phase if side.animation_phase in PHASES else "unknown"
    features += [float(phase == name) for name in PHASES]
    if len(features) != SIDE_FEATURE_DIM:
        raise AssertionError("public side feature layout drift")
    return features


def encode_public_context(
    public: PublicPairState, side: int, execution: PublicExecutionContext | None = None
):
    if type(public) is not PublicPairState or side not in (0, 1):
        raise TypeError("context requires a PublicPairState and a valid acting side")
    if execution is not None and (
        type(execution) is not PublicExecutionContext or side != public.viewer_side
    ):
        raise ValueError("execution context is known only for the public viewer")
    own_boundary = DecisionBoundary.P1 if side == 0 else DecisionBoundary.P2
    opp_boundary = DecisionBoundary.P2 if side == 0 else DecisionBoundary.P1
    features = _side_features(public.sides[side]) + _side_features(public.sides[1 - side])
    delta = public.observable_clock_delta_frames
    features += [
        float(public.decision_boundary in (own_boundary, DecisionBoundary.BOTH)),
        float(public.decision_boundary in (opp_boundary, DecisionBoundary.BOTH)),
        float(public.decision_boundary == DecisionBoundary.TERMINAL),
        float(delta is not None),
        _age((delta or 0) * (1 if side == 0 else -1)),
        _age(public.frame_id),
    ]
    features += [float(execution is not None)]
    features += (
        [
            getattr(execution, k) / scale
            for k, scale in zip(EXECUTION_FIELDS, EXECUTION_SCALES, strict=True)
        ]
        if execution
        else [0.0] * len(EXECUTION_FIELDS)
    )
    # Event side IDs are relative to the actor, so swapping physical ports does
    # not silently change the feature meaning.
    events = tuple(
        replace(e, side=None if e.side is None else int(e.side != side))
        for e in public.recent_events
    )
    history, mask = pair_events_to_features(
        events, current_frame=public.frame_id, max_events=HISTORY_LENGTH
    )
    features += np.concatenate((history, mask[:, None]), axis=1).reshape(-1).tolist()
    result = np.asarray(features, dtype=np.float32)
    if result.shape != (PUBLIC_CONTEXT_DIM,) or not np.isfinite(result).all():
        raise ValueError("invalid public context vector")
    return result


def context_from_info(info):
    if info.get("public_context_schema") != PUBLIC_CONTEXT_SCHEMA:
        raise ValueError("public context checkpoint requires its versioned input contract")
    public = info["public_pair_state"]
    return encode_public_context(
        public, int(info["public_acting_side"]), info.get("public_execution")
    )
