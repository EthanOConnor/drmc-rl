"""Typed protocol-v2 messages for unified match and trainer products."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

import numpy as np

from drmc_rl.game.pair_state import PublicPairState
from drmc_rl.human.decoder import ProductMode

PROTOCOL_SCHEMA = "drmc-human-backend-v2"


@dataclass(frozen=True, slots=True)
class ProductControls:
    target_rating: float | None = None
    style: tuple[float, ...] = ()
    cadence_scale: float = 1.0
    execution_profile: str | None = None

    def __post_init__(self) -> None:
        if self.target_rating is not None and not np.isfinite(float(self.target_rating)):
            raise ValueError("target_rating must be finite")
        if not np.isfinite(float(self.cadence_scale)) or float(self.cadence_scale) < 0:
            raise ValueError("cadence_scale must be finite and non-negative")
        if any(not np.isfinite(float(value)) for value in self.style):
            raise ValueError("style controls must be finite")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> "ProductControls":
        payload = dict(value or {})
        style_raw = payload.get("style", ())
        if isinstance(style_raw, (str, bytes, bytearray)):
            raise ValueError("style must be a numeric sequence")
        return cls(
            target_rating=(
                None if payload.get("target_rating") is None else float(payload["target_rating"])
            ),
            style=tuple(float(item) for item in style_raw),  # type: ignore[arg-type]
            cadence_scale=float(payload.get("cadence_scale", 1.0)),
            execution_profile=(
                None
                if payload.get("execution_profile") in (None, "")
                else str(payload["execution_profile"])
            ),
        )


@dataclass(frozen=True, slots=True)
class DecisionRequestV2:
    request_id: int
    frame_id: int
    deadline_ms: float
    product: ProductMode
    controls: ProductControls
    state: PublicPairState
    schema: str = PROTOCOL_SCHEMA
    type: str = "decide"

    def __post_init__(self) -> None:
        if self.schema != PROTOCOL_SCHEMA or self.type != "decide":
            raise ValueError("invalid decision request schema/type")
        if min(int(self.request_id), int(self.frame_id)) < 0:
            raise ValueError("request_id and frame_id must be non-negative")
        if not np.isfinite(float(self.deadline_ms)) or float(self.deadline_ms) <= 0:
            raise ValueError("deadline_ms must be positive")
        if self.product == ProductMode.HUMAN_RATE and not self.controls.execution_profile:
            raise ValueError("human_rate requires a named execution_profile")
        if self.product == ProductMode.TRAINER and self.controls.target_rating is None:
            raise ValueError("trainer requires target_rating")
        if self.product != ProductMode.TRAINER and self.controls.style:
            raise ValueError("style control is available only to trainer mode")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "DecisionRequestV2":
        if value.get("schema") != PROTOCOL_SCHEMA:
            raise ValueError("unsupported human backend schema")
        state_raw = value.get("state")
        if not isinstance(state_raw, Mapping):
            raise ValueError("decision request state must be a mapping")
        controls_raw = value.get("controls")
        if controls_raw is not None and not isinstance(controls_raw, Mapping):
            raise ValueError("controls must be a mapping")
        return cls(
            request_id=int(value["request_id"]),
            frame_id=int(value["frame_id"]),
            deadline_ms=float(value["deadline_ms"]),
            product=ProductMode(str(value["product"])),
            controls=ProductControls.from_mapping(controls_raw),
            state=PublicPairState.from_dict(state_raw),
            schema=str(value["schema"]),
            type=str(value.get("type", "decide")),
        )


@dataclass(frozen=True, slots=True)
class DecisionResponseV2:
    request_id: int
    frame_id: int
    product: ProductMode
    action: int
    script: tuple[int, ...]
    best_win_probability: float
    chosen_win_probability: float
    regret_win_logit: float
    execution_profile: str
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    artifact_identity: Mapping[str, object] = field(default_factory=dict)
    schema: str = PROTOCOL_SCHEMA
    type: str = "decision"

    def __post_init__(self) -> None:
        if min(int(self.request_id), int(self.frame_id), int(self.action)) < 0:
            raise ValueError("response identifiers/action must be non-negative")
        if any(not 0 <= int(mask) <= 255 for mask in self.script):
            raise ValueError("script must contain byte-sized controller masks")
        for name in ("best_win_probability", "chosen_win_probability"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if float(self.chosen_win_probability) > float(self.best_win_probability) + 1e-8:
            raise ValueError("chosen win probability cannot exceed declared best")
        if not np.isfinite(float(self.regret_win_logit)) or self.regret_win_logit < -1e-8:
            raise ValueError("regret_win_logit must be non-negative")
        if not self.execution_profile:
            raise ValueError("response must declare execution profile")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["product"] = self.product.value
        payload["script"] = list(self.script)
        return payload


@dataclass(frozen=True, slots=True)
class BackendCapabilitiesV2:
    products: tuple[ProductMode, ...]
    public_state_schema: str
    execution_profiles: tuple[str, ...]
    artifact_identity: Mapping[str, object]
    style_dimensions: int = 0
    rating_range: tuple[float, float] | None = None
    schema: str = PROTOCOL_SCHEMA
    type: str = "capabilities"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["products"] = [product.value for product in self.products]
        return payload


__all__ = [
    "BackendCapabilitiesV2",
    "DecisionRequestV2",
    "DecisionResponseV2",
    "PROTOCOL_SCHEMA",
    "ProductControls",
]
