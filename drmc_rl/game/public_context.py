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
PROGRESS_CONTEXT_SCHEMA = "public_pair_progress_v1"
COUNTDOWN_CONTEXT_SCHEMA = "public_pair_progress_countdown_v1"
PUBLIC_CONTEXT_DIMS = {
    PUBLIC_CONTEXT_SCHEMA: PUBLIC_CONTEXT_DIM,
    PROGRESS_CONTEXT_SCHEMA: PUBLIC_CONTEXT_DIM + 2,
    COUNTDOWN_CONTEXT_SCHEMA: PUBLIC_CONTEXT_DIM + 4,
}


def progress_features(level, pill_counter_bcd, speed, speed_ups, *, countdown=False):
    """Own public level and spawn count; the existing game_age is the timer.

    Native and ROM counters are packed BCD, including the current falling pill.
    Countdown measures future spawns to an actual gravity-period decrease,
    skipping repeated speed-table entries. Zero plus a false mask means capped.
    """
    from drmc_rl.planning.fast_reach import compute_speed_threshold

    if (type(level) is not int or not 0 <= level <= 255
            or type(pill_counter_bcd) is not int or not 0 <= pill_counter_bcd <= 0x9999
            or type(speed) is not int or speed not in (0, 1, 2)
            or type(speed_ups) is not int or not 0 <= speed_ups <= 49):
        raise ValueError("invalid public progress counters")
    digits = [(pill_counter_bcd >> shift) & 15 for shift in (0, 4, 8, 12)]
    if any(d > 9 for d in digits):
        raise ValueError("pill counter must be packed BCD")
    count = sum(d * 10 ** i for i, d in enumerate(digits))
    result = [level / 20., np.log1p(count) / np.log1p(1000.)]
    if countdown:
        current = compute_speed_threshold(speed, speed_ups)
        next_change = next((i for i in range(speed_ups + 1, 50)
                            if compute_speed_threshold(speed, i) < current), None)
        remaining = (10 - count % 10 + 10 * (next_change - speed_ups - 1)
                     if next_change is not None else 0)
        if remaining and count + remaining > 9999:
            next_change, remaining = None, 0
        result += [float(next_change is not None), remaining / 100.]
    return np.asarray(result, dtype=np.float32)


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
    schema = info.get("public_context_schema")
    if schema not in PUBLIC_CONTEXT_DIMS:
        raise ValueError("public context checkpoint requires its versioned input contract")
    public = info["public_pair_state"]
    encoded = encode_public_context(
        public, int(info["public_acting_side"]), info.get("public_execution")
    )
    if schema != PUBLIC_CONTEXT_SCHEMA:
        progress = info["public_progress"]
        if int(info["public_acting_side"]) != public.viewer_side:
            raise ValueError("progress is known only for the public viewer")
        encoded = np.concatenate((encoded, progress_features(
            **progress, countdown=schema == COUNTDOWN_CONTEXT_SCHEMA)))
    return encoded
