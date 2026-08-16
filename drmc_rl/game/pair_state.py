"""Public and privileged state contracts for asynchronous two-player play.

The public contract is the only state accepted by deployed actors.  Privileged
state exists for centralized critics, counterfactual teachers, parity tools,
and joint-event search performed behind an explicit training boundary.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

PAIR_STATE_SCHEMA = "drmc-pair-state-v2"
BOARD_CELLS = 128
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "rng",
        "rng_state",
        "future_rng",
        "hidden_rng",
        "attack_size_internal",
        "pending_attack_internal",
        "garbage_pending",
        "pending_attack",
        "attack_size",
        "engine_checkpoint",
        "native_state",
        "committed_action_internal",
        "committed_action",
        "seed",
    }
)


def _board(value: bytes | bytearray | Sequence[int]) -> bytes:
    data = bytes(value)
    if len(data) != BOARD_CELLS:
        raise ValueError(f"board must contain {BOARD_CELLS} bytes, got {len(data)}")
    return data


def _colors(value: Sequence[int]) -> tuple[int, int]:
    colors = tuple(int(item) for item in value)
    if len(colors) != 2 or any(item not in (0, 1, 2) for item in colors):
        raise ValueError(f"pill colors must be two canonical indices in [0,2], got {colors}")
    return colors  # type: ignore[return-value]


class PairEventKind(str, Enum):
    OBSERVATION = "observation"
    SPAWN = "spawn"
    LOCK = "lock"
    CLEAR = "clear"
    VOLLEY = "volley"
    TOP_OUT = "top_out"
    STAGE_CLEAR = "stage_clear"
    TERMINAL = "terminal"


class DecisionBoundary(str, Enum):
    P1 = "p1"
    P2 = "p2"
    BOTH = "both"
    ADVANCE = "advance"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class FallingPillView:
    column: int
    row_top: int
    rotation: int
    colors: tuple[int, int]
    controllable: bool
    age_frames: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "colors", _colors(self.colors))
        if not 0 <= int(self.column) < 8:
            raise ValueError("falling pill column must be in [0,7]")
        if not -2 <= int(self.row_top) < 16:
            raise ValueError("falling pill top-origin row must be in [-2,15]")
        if not 0 <= int(self.rotation) < 4:
            raise ValueError("falling pill rotation must be in [0,3]")
        if int(self.age_frames) < 0:
            raise ValueError("falling pill age must be non-negative")


@dataclass(frozen=True, slots=True)
class VisibleSideState:
    board: bytes
    pill: tuple[int, int]
    preview: tuple[int, int]
    active: FallingPillView | None
    viruses_remaining: int | None = None
    animation_phase: str = ""
    state_age_frames: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "board", _board(self.board))
        object.__setattr__(self, "pill", _colors(self.pill))
        object.__setattr__(self, "preview", _colors(self.preview))
        if self.viruses_remaining is not None and not 0 <= int(self.viruses_remaining) <= 84:
            raise ValueError("viruses_remaining must be in [0,84]")
        if int(self.state_age_frames) < 0:
            raise ValueError("state_age_frames must be non-negative")


@dataclass(frozen=True, slots=True)
class PairEvent:
    kind: PairEventKind
    frame_id: int
    side: int | None
    public_payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.frame_id) < 0:
            raise ValueError("event frame_id must be non-negative")
        if self.side not in (None, 0, 1):
            raise ValueError("event side must be None, 0, or 1")
        audit_public_mapping(self.public_payload)


@dataclass(frozen=True, slots=True)
class PublicPairState:
    """Fair-play observation consumed by all deployable policies."""

    frame_id: int
    viewer_side: int
    sides: tuple[VisibleSideState, VisibleSideState]
    decision_boundary: DecisionBoundary
    recent_events: tuple[PairEvent, ...] = ()
    observable_clock_delta_frames: int | None = None
    own_controller_state: Mapping[str, int | float | bool] = field(default_factory=dict)
    schema: str = PAIR_STATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PAIR_STATE_SCHEMA:
            raise ValueError(f"unsupported public pair-state schema {self.schema!r}")
        if int(self.frame_id) < 0:
            raise ValueError("frame_id must be non-negative")
        if self.viewer_side not in (0, 1):
            raise ValueError("viewer_side must be 0 or 1")
        if len(self.sides) != 2:
            raise ValueError("public pair state requires exactly two sides")
        if self.observable_clock_delta_frames is not None and not -100000 <= int(
            self.observable_clock_delta_frames
        ) <= 100000:
            raise ValueError("observable clock delta is implausible")
        audit_public_mapping(self.own_controller_state)
        if len(self.recent_events) > 128:
            raise ValueError("public event history must be bounded")

    @property
    def own(self) -> VisibleSideState:
        return self.sides[self.viewer_side]

    @property
    def opponent(self) -> VisibleSideState:
        return self.sides[1 - self.viewer_side]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "frame_id": int(self.frame_id),
            "viewer_side": int(self.viewer_side),
            "decision_boundary": self.decision_boundary.value,
            "observable_clock_delta_frames": self.observable_clock_delta_frames,
            "own_controller_state": dict(self.own_controller_state),
            "sides": [_visible_side_to_dict(side) for side in self.sides],
            "recent_events": [
                {
                    "kind": event.kind.value,
                    "frame_id": event.frame_id,
                    "side": event.side,
                    "public_payload": dict(event.public_payload),
                }
                for event in self.recent_events
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PublicPairState":
        audit_public_mapping(value)
        sides_raw = value.get("sides")
        if not isinstance(sides_raw, Sequence) or isinstance(sides_raw, (str, bytes, bytearray)):
            raise ValueError("sides must be a two-entry sequence")
        events_raw = value.get("recent_events", ())
        if not isinstance(events_raw, Sequence) or isinstance(events_raw, (str, bytes, bytearray)):
            raise ValueError("recent_events must be a sequence")
        return cls(
            schema=str(value.get("schema", PAIR_STATE_SCHEMA)),
            frame_id=int(value["frame_id"]),
            viewer_side=int(value["viewer_side"]),
            decision_boundary=DecisionBoundary(str(value["decision_boundary"])),
            observable_clock_delta_frames=(
                None
                if value.get("observable_clock_delta_frames") is None
                else int(value["observable_clock_delta_frames"])
            ),
            own_controller_state=dict(value.get("own_controller_state", {})),
            sides=tuple(_visible_side_from_dict(item) for item in sides_raw),  # type: ignore[arg-type]
            recent_events=tuple(
                PairEvent(
                    kind=PairEventKind(str(item["kind"])),
                    frame_id=int(item["frame_id"]),
                    side=None if item.get("side") is None else int(item["side"]),
                    public_payload=dict(item.get("public_payload", {})),
                )
                for item in events_raw
            ),
        )

    def stable_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PrivilegedPairState:
    """Training-only state; never pass this object to a deployed actor."""

    public: PublicPairState
    pair_clocks: tuple[int, int]
    need_action: tuple[bool, bool]
    pending_attacks: tuple[int, int]
    native_phases: tuple[str, str]
    committed_actions: tuple[int | None, int | None]
    engine_checkpoint: bytes
    terminal_outcome: tuple[int, int] = (0, 0)
    schema: str = "drmc-privileged-pair-state-v2"

    def __post_init__(self) -> None:
        if len(self.pair_clocks) != 2 or min(self.pair_clocks) < 0:
            raise ValueError("pair_clocks must be two non-negative values")
        if len(self.need_action) != 2:
            raise ValueError("need_action must contain two flags")
        if len(self.pending_attacks) != 2 or min(self.pending_attacks) < 0:
            raise ValueError("pending_attacks must be two non-negative values")
        if len(self.native_phases) != 2 or len(self.committed_actions) != 2:
            raise ValueError("privileged state requires two side phases/actions")
        if len(self.terminal_outcome) != 2:
            raise ValueError("terminal_outcome must contain both sides")
        if not self.engine_checkpoint:
            raise ValueError("privileged state requires a restorable engine checkpoint")

    @property
    def decision_boundary(self) -> DecisionBoundary:
        if any(self.terminal_outcome):
            return DecisionBoundary.TERMINAL
        if self.need_action == (True, True):
            return DecisionBoundary.BOTH
        if self.need_action[0]:
            return DecisionBoundary.P1
        if self.need_action[1]:
            return DecisionBoundary.P2
        return DecisionBoundary.ADVANCE

    def public_view(self) -> PublicPairState:
        return self.public


def _visible_side_to_dict(side: VisibleSideState) -> dict[str, Any]:
    return {
        "board_b64": base64.b64encode(side.board).decode("ascii"),
        "pill": list(side.pill),
        "preview": list(side.preview),
        "active": None if side.active is None else asdict(side.active),
        "viruses_remaining": side.viruses_remaining,
        "animation_phase": side.animation_phase,
        "state_age_frames": side.state_age_frames,
    }


def _visible_side_from_dict(value: Mapping[str, Any]) -> VisibleSideState:
    active_raw = value.get("active")
    active = None
    if active_raw is not None:
        active_map = dict(active_raw)
        active = FallingPillView(
            column=int(active_map["column"]),
            row_top=int(active_map["row_top"]),
            rotation=int(active_map["rotation"]),
            colors=_colors(active_map["colors"]),
            controllable=bool(active_map["controllable"]),
            age_frames=int(active_map.get("age_frames", 0)),
        )
    return VisibleSideState(
        board=base64.b64decode(str(value["board_b64"]), validate=True),
        pill=_colors(value["pill"]),
        preview=_colors(value["preview"]),
        active=active,
        viruses_remaining=(
            None if value.get("viruses_remaining") is None else int(value["viruses_remaining"])
        ),
        animation_phase=str(value.get("animation_phase", "")),
        state_age_frames=int(value.get("state_age_frames", 0)),
    )


def audit_public_mapping(value: Any, *, path: str = "state") -> None:
    """Reject hidden-information keys recursively before actor ingestion."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            forbidden_prefix = normalized.startswith(("rng_", "future_rng", "hidden_rng"))
            if normalized in FORBIDDEN_PUBLIC_KEYS or forbidden_prefix:
                raise ValueError(f"forbidden public field at {path}.{key}")
            audit_public_mapping(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            audit_public_mapping(item, path=f"{path}[{index}]")


__all__ = [
    "BOARD_CELLS",
    "DecisionBoundary",
    "FORBIDDEN_PUBLIC_KEYS",
    "FallingPillView",
    "PAIR_STATE_SCHEMA",
    "PairEvent",
    "PairEventKind",
    "PrivilegedPairState",
    "PublicPairState",
    "VisibleSideState",
    "audit_public_mapping",
]
