"""Opponent-aware policy over exact post-placement bottle states.

The V2 human model scores poses from the pre-placement board.  This model makes
the strategic object explicit: every candidate is represented by the bottle
that actually results after the pill locks, clears, and cascades.  Competitive
quality is deliberately independent of requested human rating.  Human style is
a separate head conditioned on rating/history, so strength control can operate
on measured action regret instead of perturbing placement logits.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


HUMAN_AFTERSTATE_SCHEMA = "drmc-human-afterstate-v3"
GRID_H = 16
GRID_W = 8


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _BottleResidual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.silu(self.norm1(x)))
        y = self.conv2(F.silu(self.norm2(y)))
        return x + y


class TileBottleEncoder(nn.Module):
    """Encode the native engine's 128 tile bytes without lossy re-tokenizing."""

    def __init__(self, d_model: int, *, blocks: int, tile_dim: int = 32) -> None:
        super().__init__()
        self.tile_dim = int(tile_dim)
        self.tiles = nn.Embedding(256, int(tile_dim))
        self.stem = nn.Conv2d(int(tile_dim) + 2, int(d_model), 3, padding=1)
        self.blocks = nn.Sequential(*[_BottleResidual(int(d_model)) for _ in range(blocks)])
        self.out = nn.Sequential(
            nn.LayerNorm(2 * int(d_model)),
            nn.Linear(2 * int(d_model), int(d_model)),
            nn.SiLU(),
        )
        rows = torch.linspace(0.0, 1.0, GRID_H).view(1, 1, GRID_H, 1)
        cols = torch.linspace(0.0, 1.0, GRID_W).view(1, 1, 1, GRID_W)
        self.register_buffer("rows", rows.expand(1, 1, GRID_H, GRID_W), persistent=False)
        self.register_buffer("cols", cols.expand(1, 1, GRID_H, GRID_W), persistent=False)

    def encode_map(self, fields: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        shape = fields.shape
        if shape[-2:] == (GRID_H, GRID_W):
            flat = fields.reshape(-1, GRID_H, GRID_W)
        elif shape[-1:] == (GRID_H * GRID_W,):
            flat = fields.reshape(-1, GRID_H, GRID_W)
        else:
            raise ValueError(f"expected fields ending in (128,) or (16,8), got {tuple(shape)}")
        embedded = self.tiles(flat.long().clamp_(0, 255)).permute(0, 3, 1, 2)
        n = int(embedded.shape[0])
        coords = torch.cat((self.rows, self.cols), dim=1).expand(n, -1, -1, -1)
        x = self.blocks(self.stem(torch.cat((embedded, coords.to(embedded.dtype)), dim=1)))
        outer = shape[:-1] if shape[-1] == 128 else shape[:-2]
        return x, tuple(outer)

    def pool_map(self, feature_map: torch.Tensor, outer: tuple[int, ...]) -> torch.Tensor:
        x = feature_map
        pooled = torch.cat((x.mean(dim=(-2, -1)), x.amax(dim=(-2, -1))), dim=-1)
        return self.out(pooled).reshape(*outer, -1)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        feature_map, outer = self.encode_map(fields)
        return self.pool_map(feature_map, outer)


class _CandidateBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ff_mult: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.SiLU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, key_padding_mask=padding, need_weights=False)
        x = x + y
        return x + self.ff(self.norm2(x))


class AfterstatePolicyNet(nn.Module):
    """Joint competitive-value and human-style model.

    ``competitive_score`` and all tactical heads never receive the human
    condition. ``human_logits`` receives both the competitive candidate token
    and the semantic rating/history condition. This boundary is intentional:
    rating changes error/style selection, not what the model believes is good.
    """

    def __init__(
        self,
        *,
        condition_dim: int,
        d_model: int = 256,
        bottle_blocks: int = 6,
        candidate_layers: int = 4,
        heads: int = 8,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        d_model = int(d_model)
        self.condition_dim = int(condition_dim)
        self.d_model = d_model
        self.bottle = TileBottleEncoder(d_model, blocks=int(bottle_blocks))
        self.orientation = nn.Embedding(4, d_model)
        self.row = nn.Embedding(GRID_H, d_model)
        self.column = nn.Embedding(GRID_W, d_model)
        self.color = nn.Embedding(3, d_model // 4)
        self.pill = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU())
        self.cost = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.delta = nn.Sequential(
            nn.Linear(d_model + 2 * self.bottle.tile_dim + 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.delta_count = nn.Sequential(nn.Linear(1, d_model), nn.SiLU())
        self.fuse = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU())
        self.candidates = nn.ModuleList(
            [_CandidateBlock(d_model, int(heads), int(ff_mult)) for _ in range(candidate_layers)]
        )
        self.competitive = nn.Linear(d_model, 1)
        self.outcome = nn.Linear(d_model, 1)
        self.clear = nn.Linear(d_model, 1)
        self.topout = nn.Linear(d_model, 1)
        self.virus_delta = nn.Linear(d_model, 1)
        self.attack = nn.Linear(d_model, 1)
        self.style = nn.Sequential(
            nn.Linear(d_model + self.condition_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

    def _pill_embedding(self, pill: torch.Tensor, preview: torch.Tensor) -> torch.Tensor:
        colors = torch.cat((pill, preview), dim=-1).long().clamp_(0, 2)
        return self.pill(self.color(colors).flatten(1))

    def forward(
        self,
        afterstate_fields: torch.Tensor,
        root_fields: torch.Tensor,
        opponent_fields: torch.Tensor,
        pill: torch.Tensor,
        preview: torch.Tensor,
        candidate_actions: torch.Tensor,
        candidate_costs: torch.Tensor,
        candidate_mask: torch.Tensor,
        condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if afterstate_fields.ndim != 3 or afterstate_fields.shape[-1] != 128:
            raise ValueError("afterstate_fields must have shape [B,K,128]")
        batch, candidates, _ = afterstate_fields.shape
        if root_fields.shape != (batch, 128):
            raise ValueError(f"root_fields must have shape {(batch, 128)}")
        if condition.shape != (batch, self.condition_dim):
            raise ValueError(
                f"condition must have shape {(batch, self.condition_dim)}, got {tuple(condition.shape)}"
            )
        valid = candidate_mask.bool()
        actions = candidate_actions.long().clamp_min_(0)
        orientation = torch.div(actions, 128, rounding_mode="floor").clamp_(0, 3)
        cell = actions.remainder(128)
        row = torch.div(cell, GRID_W, rounding_mode="floor")
        column = cell.remainder(GRID_W)

        root_map, root_outer = self.bottle.encode_map(root_fields)
        root_global = self.bottle.pool_map(root_map, root_outer)
        changed = valid.unsqueeze(-1) & (afterstate_fields != root_fields.unsqueeze(1))
        changed_index = changed.nonzero(as_tuple=False)
        flat_candidate = changed_index[:, 0] * candidates + changed_index[:, 1]
        changed_cell = changed_index[:, 2]
        root_tile = root_fields[changed_index[:, 0], changed_cell]
        after_tile = afterstate_fields[changed_index[:, 0], changed_index[:, 1], changed_cell]
        context_map = root_map.permute(0, 2, 3, 1).reshape(batch, 128, self.d_model)
        local = context_map[changed_index[:, 0], changed_cell]
        position = torch.stack(
            (
                torch.div(changed_cell, GRID_W, rounding_mode="floor").to(local.dtype)
                / (GRID_H - 1),
                changed_cell.remainder(GRID_W).to(local.dtype) / (GRID_W - 1),
            ),
            dim=-1,
        )
        delta_token = self.delta(
            torch.cat(
                (
                    local,
                    self.bottle.tiles(root_tile.long()),
                    self.bottle.tiles(after_tile.long()),
                    position,
                ),
                dim=-1,
            )
        )
        flat_size = batch * candidates
        delta_sum = torch.zeros(
            (flat_size, self.d_model), dtype=delta_token.dtype, device=delta_token.device
        ).index_add(0, flat_candidate, delta_token)
        delta_n = torch.zeros(
            (flat_size, 1), dtype=delta_token.dtype, device=delta_token.device
        ).index_add(
            0,
            flat_candidate,
            torch.ones((len(delta_token), 1), dtype=delta_token.dtype, device=delta_token.device),
        )
        delta_context = delta_sum / delta_n.clamp_min(1.0)
        delta_context = delta_context + self.delta_count(torch.log1p(delta_n))
        own = root_global.unsqueeze(1) + delta_context.reshape(batch, candidates, self.d_model)
        opponent = self.bottle(opponent_fields).unsqueeze(1)
        pill_context = self._pill_embedding(pill, preview).unsqueeze(1)
        pose = self.orientation(orientation) + self.row(row) + self.column(column)
        cost = self.cost((candidate_costs.to(own.dtype) / 120.0).clamp(0.0, 4.0).unsqueeze(-1))
        tokens = self.fuse(own + opponent + pill_context + pose + cost)
        padding = ~valid
        safe_padding = padding & ~padding.all(dim=1, keepdim=True)
        for block in self.candidates:
            tokens = block(tokens, safe_padding)

        condition_expanded = condition.to(tokens.dtype).unsqueeze(1).expand(-1, candidates, -1)
        human_logits = self.style(torch.cat((tokens, condition_expanded), dim=-1)).squeeze(-1)
        masked = lambda value: value.masked_fill(~valid, -1e9)
        return {
            "competitive_score": masked(self.competitive(tokens).squeeze(-1)),
            "human_logits": masked(human_logits),
            "outcome_logit": masked(self.outcome(tokens).squeeze(-1)),
            "clear_logit": masked(self.clear(tokens).squeeze(-1)),
            "topout_logit": masked(self.topout(tokens).squeeze(-1)),
            "virus_delta": self.virus_delta(tokens).squeeze(-1).masked_fill(~valid, 0.0),
            "attack": self.attack(tokens).squeeze(-1).masked_fill(~valid, 0.0),
        }


def afterstate_policy_config(*, capacity: str = "base") -> dict[str, Any]:
    sizes = {
        "small": (128, 3, 2, 4),
        "base": (256, 6, 4, 8),
        "large": (320, 8, 6, 8),
    }
    if capacity not in sizes:
        raise ValueError(f"unknown capacity {capacity!r}")
    d_model, bottle_blocks, candidate_layers, heads = sizes[capacity]
    return {
        "schema": HUMAN_AFTERSTATE_SCHEMA,
        "condition_dim": 40,
        "capacity": capacity,
        "d_model": d_model,
        "bottle_blocks": bottle_blocks,
        "candidate_layers": candidate_layers,
        "heads": heads,
        "ff_mult": 4,
    }


def build_afterstate_policy(
    cfg: Mapping[str, Any], *, condition_dim: int, device: str = "cpu"
) -> AfterstatePolicyNet:
    return AfterstatePolicyNet(
        condition_dim=int(condition_dim),
        d_model=int(cfg.get("d_model", 256)),
        bottle_blocks=int(cfg.get("bottle_blocks", 6)),
        candidate_layers=int(cfg.get("candidate_layers", 4)),
        heads=int(cfg.get("heads", 8)),
        ff_mult=int(cfg.get("ff_mult", 4)),
    ).to(device)


__all__ = [
    "AfterstatePolicyNet",
    "HUMAN_AFTERSTATE_SCHEMA",
    "TileBottleEncoder",
    "afterstate_policy_config",
    "build_afterstate_policy",
]
