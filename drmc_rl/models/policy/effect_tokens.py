"""Cheap exact candidate-effect tokens for G5 representation bakeoffs.

The policy's root encoder should not have to rediscover every deterministic
lock/clear/cascade consequence.  These tokens summarize resolved native
candidate afterstates without encoding requested human rating.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

GRID_H = 16
GRID_W = 8
TILE_EMPTY = 0xFF
TILE_ZERO = 0x00
TILE_CLEARED = 0xB0
TILE_VIRUS = 0xD0
TILE_JUST_EMPTIED = 0xF0
TYPE_MASK = 0xF0

_BASE_NAMES = (
    "changed_fraction",
    "root_occupancy",
    "after_occupancy",
    "occupancy_delta",
    "root_virus_fraction",
    "after_virus_fraction",
    "virus_delta",
    "root_max_height",
    "after_max_height",
    "max_height_delta",
    "root_holes",
    "after_holes",
    "holes_delta",
    "root_top4_occupancy",
    "after_top4_occupancy",
    "top4_delta",
    "root_roughness",
    "after_roughness",
    "roughness_delta",
)
EFFECT_TOKEN_NAMES = (
    *_BASE_NAMES,
    *(f"column_height_delta_{column}" for column in range(GRID_W)),
    *(f"column_occupancy_delta_{column}" for column in range(GRID_W)),
    "terminal_clear",
    "terminal_topout",
    "viruses_cleared",
    "nonviruses_cleared",
    "clear_events",
    "attack",
    "uncertainty",
)
EFFECT_TOKEN_DIM = len(EFFECT_TOKEN_NAMES)


def _occupied(fields: torch.Tensor) -> torch.Tensor:
    tile = fields.to(torch.int64)
    type_hi = tile & TYPE_MASK
    pills = (type_hi >= 0x40) & (type_hi <= 0x80)
    virus = type_hi == TILE_VIRUS
    return pills | virus


def _virus(fields: torch.Tensor) -> torch.Tensor:
    return (fields.to(torch.int64) & TYPE_MASK) == TILE_VIRUS


def _column_metrics(occupied: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return heights [..,8], holes [..], and roughness [..]."""

    shape = occupied.shape[:-1]
    grid = occupied.reshape(*shape, GRID_H, GRID_W)
    any_column = grid.any(dim=-2)
    # argmax on booleans returns first occupied row from the top; empty columns
    # are overwritten with height zero.
    first = grid.to(torch.int64).argmax(dim=-2)
    heights = torch.where(any_column, GRID_H - first, torch.zeros_like(first)).to(torch.float32)
    rows = torch.arange(GRID_H, device=grid.device).view(*([1] * len(shape)), GRID_H, 1)
    first_expanded = first.unsqueeze(-2)
    below_top = rows >= first_expanded
    holes = ((~grid) & below_top & any_column.unsqueeze(-2)).sum(dim=(-2, -1)).to(torch.float32)
    roughness = torch.abs(heights[..., 1:] - heights[..., :-1]).mean(dim=-1)
    return heights, holes, roughness


def _aux(
    value: torch.Tensor | np.ndarray | None,
    *,
    shape: tuple[int, int],
    device: torch.device,
    scale: float = 1.0,
) -> torch.Tensor:
    if value is None:
        return torch.zeros(shape, dtype=torch.float32, device=device)
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tuple(tensor.shape) != shape:
        raise ValueError(f"auxiliary target has shape {tuple(tensor.shape)}, expected {shape}")
    return tensor / float(scale)


def build_effect_tokens(
    root_fields: torch.Tensor | np.ndarray,
    afterstate_fields: torch.Tensor | np.ndarray,
    candidate_mask: torch.Tensor | np.ndarray,
    *,
    terminal_reason: torch.Tensor | np.ndarray | None = None,
    viruses_cleared: torch.Tensor | np.ndarray | None = None,
    nonviruses_cleared: torch.Tensor | np.ndarray | None = None,
    clear_events: torch.Tensor | np.ndarray | None = None,
    attack: torch.Tensor | np.ndarray | None = None,
    uncertainty: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    """Return normalized exact-effect tokens with shape ``[B,K,D]``."""

    after = torch.as_tensor(afterstate_fields)
    root = torch.as_tensor(root_fields, device=after.device)
    mask = torch.as_tensor(candidate_mask, dtype=torch.bool, device=after.device)
    if after.ndim != 3 or after.shape[-1] != GRID_H * GRID_W:
        raise ValueError("afterstate_fields must have shape [B,K,128]")
    batch, candidates, _ = after.shape
    if tuple(root.shape) != (batch, GRID_H * GRID_W):
        raise ValueError(f"root_fields must have shape {(batch, 128)}")
    if tuple(mask.shape) != (batch, candidates):
        raise ValueError(f"candidate_mask must have shape {(batch, candidates)}")

    root_expanded = root.unsqueeze(1).expand(-1, candidates, -1)
    root_occ = _occupied(root_expanded)
    after_occ = _occupied(after)
    root_virus = _virus(root_expanded)
    after_virus = _virus(after)
    root_heights, root_holes, root_roughness = _column_metrics(root_occ)
    after_heights, after_holes, after_roughness = _column_metrics(after_occ)

    changed = (after.to(torch.int64) != root_expanded.to(torch.int64)).sum(dim=-1).float() / 128.0
    root_occ_fraction = root_occ.float().mean(dim=-1)
    after_occ_fraction = after_occ.float().mean(dim=-1)
    root_virus_fraction = root_virus.float().sum(dim=-1) / 84.0
    after_virus_fraction = after_virus.float().sum(dim=-1) / 84.0
    root_max = root_heights.max(dim=-1).values / GRID_H
    after_max = after_heights.max(dim=-1).values / GRID_H
    root_top4 = root_occ.reshape(batch, candidates, GRID_H, GRID_W)[..., :4, :].float().mean(dim=(-2, -1))
    after_top4 = after_occ.reshape(batch, candidates, GRID_H, GRID_W)[..., :4, :].float().mean(dim=(-2, -1))
    root_column_occupancy = root_occ.reshape(batch, candidates, GRID_H, GRID_W).sum(dim=-2).float() / GRID_H
    after_column_occupancy = after_occ.reshape(batch, candidates, GRID_H, GRID_W).sum(dim=-2).float() / GRID_H

    scalars = [
        changed,
        root_occ_fraction,
        after_occ_fraction,
        after_occ_fraction - root_occ_fraction,
        root_virus_fraction,
        after_virus_fraction,
        root_virus_fraction - after_virus_fraction,
        root_max,
        after_max,
        after_max - root_max,
        root_holes / 128.0,
        after_holes / 128.0,
        (after_holes - root_holes) / 128.0,
        root_top4,
        after_top4,
        after_top4 - root_top4,
        root_roughness / GRID_H,
        after_roughness / GRID_H,
        (after_roughness - root_roughness) / GRID_H,
    ]
    column_height_delta = (after_heights - root_heights) / GRID_H
    column_occupancy_delta = after_column_occupancy - root_column_occupancy
    shape = (batch, candidates)
    terminal = _aux(terminal_reason, shape=shape, device=after.device)
    tactical = torch.stack(
        (
            terminal.eq(1).float(),
            terminal.eq(2).float(),
            _aux(viruses_cleared, shape=shape, device=after.device, scale=84.0),
            _aux(nonviruses_cleared, shape=shape, device=after.device, scale=32.0),
            _aux(clear_events, shape=shape, device=after.device, scale=8.0),
            _aux(attack, shape=shape, device=after.device, scale=4.0),
            _aux(uncertainty, shape=shape, device=after.device),
        ),
        dim=-1,
    )
    tokens = torch.cat(
        (
            torch.stack(scalars, dim=-1),
            column_height_delta,
            column_occupancy_delta,
            tactical,
        ),
        dim=-1,
    ).to(torch.float32)
    if tokens.shape[-1] != EFFECT_TOKEN_DIM:
        raise RuntimeError(f"effect-token dimension drift: {tokens.shape[-1]} != {EFFECT_TOKEN_DIM}")
    return tokens.masked_fill(~mask.unsqueeze(-1), 0.0)


class EffectTokenProjection(nn.Module):
    """Projection block ready to add to a G5 candidate token stream."""

    def __init__(self, d_model: int, *, hidden: int | None = None) -> None:
        super().__init__()
        hidden_dim = int(hidden or max(EFFECT_TOKEN_DIM, d_model // 2))
        self.net = nn.Sequential(
            nn.LayerNorm(EFFECT_TOKEN_DIM),
            nn.Linear(EFFECT_TOKEN_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, int(d_model)),
        )
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-1] != EFFECT_TOKEN_DIM:
            raise ValueError(
                f"expected effect token dimension {EFFECT_TOKEN_DIM}, got {tokens.shape[-1]}"
            )
        return torch.sigmoid(self.gate) * self.net(tokens)


__all__ = [
    "EFFECT_TOKEN_DIM",
    "EFFECT_TOKEN_NAMES",
    "EffectTokenProjection",
    "build_effect_tokens",
]
