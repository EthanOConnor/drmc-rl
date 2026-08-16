"""Compact recurrent belief over public semantic pair events.

The actor never receives opaque native memory.  This encoder consumes bounded
spawn/lock/clear/volley/terminal events from :class:`PublicPairState` and
produces a context vector that can condition G5 or a public-information search
model.  Feature extraction is deterministic and ignores unknown payload keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from drmc_rl.game.pair_state import PairEvent, PairEventKind

_EVENT_KINDS = tuple(PairEventKind)
_EVENT_INDEX = {kind: index for index, kind in enumerate(_EVENT_KINDS)}
EVENT_FEATURE_NAMES = (
    *(f"kind_{kind.value}" for kind in _EVENT_KINDS),
    "side_none",
    "side_p1",
    "side_p2",
    "age_frames_log",
    "garbage_size",
    "tiles_cleared",
    "viruses_cleared",
    "lock_row",
    "lock_column",
    "rotation",
    "terminal_outcome",
)
EVENT_FEATURE_DIM = len(EVENT_FEATURE_NAMES)


def _number(payload: Mapping[str, object], names: Sequence[str], default: float = 0.0) -> float:
    for name in names:
        value = payload.get(name)
        if value is None:
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(result):
            return result
    return float(default)


def event_feature(event: PairEvent, *, current_frame: int) -> np.ndarray:
    """Encode one public event without consulting hidden/native state."""

    if current_frame < event.frame_id:
        raise ValueError("current_frame cannot precede a public event")
    out = np.zeros(EVENT_FEATURE_DIM, dtype=np.float32)
    out[_EVENT_INDEX[event.kind]] = 1.0
    side_offset = len(_EVENT_KINDS)
    out[side_offset + (0 if event.side is None else 1 + int(event.side))] = 1.0
    cursor = side_offset + 3
    out[cursor] = np.log1p(current_frame - event.frame_id) / np.log1p(600.0)
    payload = event.public_payload
    out[cursor + 1] = np.clip(
        _number(payload, ("garbage_size", "size", "volley_size")) / 4.0, 0.0, 1.0
    )
    out[cursor + 2] = np.clip(
        _number(payload, ("tiles_cleared", "cleared_tiles")) / 32.0, 0.0, 1.0
    )
    out[cursor + 3] = np.clip(
        _number(payload, ("viruses_cleared", "virus_delta")) / 16.0, 0.0, 1.0
    )
    out[cursor + 4] = np.clip(
        _number(payload, ("row_top", "row"), -1.0) / 15.0, -1.0, 1.0
    )
    out[cursor + 5] = np.clip(
        _number(payload, ("column", "col"), -1.0) / 7.0, -1.0, 1.0
    )
    out[cursor + 6] = np.clip(_number(payload, ("rotation", "rot")) / 3.0, 0.0, 1.0)
    out[cursor + 7] = np.clip(
        _number(payload, ("outcome", "terminal_outcome")), -1.0, 1.0
    )
    return out


def pair_events_to_features(
    events: Sequence[PairEvent],
    *,
    current_frame: int,
    max_events: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return chronological ``[T,F]`` features and valid mask, right padded."""

    width = max(1, int(max_events))
    selected = tuple(events)[-width:]
    features = np.zeros((width, EVENT_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(width, dtype=np.bool_)
    for index, event in enumerate(selected):
        features[index] = event_feature(event, current_frame=int(current_frame))
        mask[index] = True
    return features, mask


@dataclass(slots=True)
class PublicEventHistory:
    """Bounded runtime event buffer with monotone frame validation."""

    max_events: int = 32
    events: tuple[PairEvent, ...] = ()

    def append(self, event: PairEvent) -> None:
        if self.events and event.frame_id < self.events[-1].frame_id:
            raise ValueError("public events must be appended in frame order")
        self.events = (*self.events, event)[-max(1, int(self.max_events)) :]

    def features(self, current_frame: int) -> tuple[np.ndarray, np.ndarray]:
        return pair_events_to_features(
            self.events,
            current_frame=int(current_frame),
            max_events=self.max_events,
        )


class PublicEventBeliefEncoder(nn.Module):
    """GRU encoder for chronological event features with empty-history support."""

    def __init__(self, d_model: int, *, hidden_dim: int | None = None, layers: int = 1) -> None:
        super().__init__()
        hidden = int(hidden_dim or d_model)
        self.input = nn.Sequential(
            nn.LayerNorm(EVENT_FEATURE_DIM),
            nn.Linear(EVENT_FEATURE_DIM, hidden),
            nn.SiLU(),
        )
        self.gru = nn.GRU(hidden, hidden, num_layers=int(layers), batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, int(d_model)))
        self.empty = nn.Parameter(torch.zeros(int(d_model)))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != EVENT_FEATURE_DIM:
            raise ValueError(f"expected event features [B,T,{EVENT_FEATURE_DIM}]")
        valid = mask.bool()
        if tuple(valid.shape) != tuple(features.shape[:2]):
            raise ValueError("event mask must match [B,T]")
        encoded = self.input(features)
        output, _hidden = self.gru(encoded)
        lengths = valid.sum(dim=1)
        safe_index = (lengths - 1).clamp_min(0)
        row = torch.arange(features.shape[0], device=features.device)
        context = output[row, safe_index]
        projected = self.output(context)
        return torch.where(
            lengths.unsqueeze(1) > 0,
            projected,
            self.empty.to(projected.dtype).unsqueeze(0),
        )


__all__ = [
    "EVENT_FEATURE_DIM",
    "EVENT_FEATURE_NAMES",
    "PublicEventBeliefEncoder",
    "PublicEventHistory",
    "event_feature",
    "pair_events_to_features",
]
