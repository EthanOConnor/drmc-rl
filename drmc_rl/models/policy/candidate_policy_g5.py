"""Opponent-aware G5 candidate policy.

G4 compresses both bottles through one unconditioned trunk and scores each
placement mostly in isolation.  G5 keeps the placement/planner contract, but
adds the structural biases needed for competitive play: a shared encoder for
the two bottles, pill/context-conditioned residual processing, explicit
cross-bottle interaction, and full candidate-set attention.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from drmc_rl.models.policy.candidate_policy import GRID_H, GRID_W, ORIENTS, _decode_actions
from drmc_rl.models.policy.placement_heads import OrderedPairEmbedding, UnorderedPillEmbedding


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _FiLMResidual(nn.Module):
    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        groups = _groups(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = nn.Linear(cond_dim, 4 * channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale1, shift1, scale2, shift2 = self.film(cond).chunk(4, dim=-1)
        y = self.norm1(x)
        y = y * (1.0 + scale1[:, :, None, None]) + shift1[:, :, None, None]
        y = self.conv1(F.silu(y))
        y = self.norm2(y)
        y = y * (1.0 + scale2[:, :, None, None]) + shift2[:, :, None, None]
        return x + self.conv2(F.silu(y))


class _SharedBottleEncoder(nn.Module):
    def __init__(self, in_channels: int, d_model: int, blocks: int, cond_dim: int) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_channels + 2, d_model, 3, padding=1)
        self.blocks = nn.ModuleList(
            [_FiLMResidual(d_model, cond_dim) for _ in range(int(blocks))]
        )
        rows = torch.linspace(0.0, 1.0, GRID_H).view(1, 1, GRID_H, 1)
        cols = torch.linspace(0.0, 1.0, GRID_W).view(1, 1, 1, GRID_W)
        self.register_buffer("rows", rows.expand(1, 1, GRID_H, GRID_W), persistent=False)
        self.register_buffer("cols", cols.expand(1, 1, GRID_H, GRID_W), persistent=False)

    def forward(self, bottle: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        batch = int(bottle.shape[0])
        coords = torch.cat((self.rows, self.cols), dim=1).expand(batch, -1, -1, -1)
        x = self.stem(torch.cat((bottle, coords.to(bottle.dtype)), dim=1))
        for block in self.blocks:
            x = block(x, cond)
        return x


class _TokenBlock(nn.Module):
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

    def forward(
        self, x: torch.Tensor, padding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, key_padding_mask=padding, need_weights=False)
        x = x + y
        return x + self.ff(self.norm2(x))


class G5CandidatePlacementPolicyNet(nn.Module):
    """Large shared-bottle, opponent-interacting placement policy."""

    def __init__(
        self,
        *,
        in_channels: int,
        board_channels: int,
        encoder_blocks: int,
        d_model: int,
        pill_embed_dim: int,
        pill_embed_type: str = "ordered_pair",
        aux_dim: int = 0,
        num_colors: int = 3,
        pos_embed_dim: int = 32,
        cost_embed_dim: int = 32,
        cand_hidden_dim: int = 768,
        transformer_heads: int = 8,
        transformer_ff_mult: int = 4,
        cross_layers: int = 3,
        interaction_layers: int = 2,
        patch_kernel: int = 9,
        cost_norm_denom: float = 64.0,
        value_atoms: int = 51,
        conditioned_trunk: bool = True,
        opponent_features: bool = True,
    ) -> None:
        super().__init__()
        if board_channels != 16:
            raise ValueError("G5 requires 16 board channels (8 own + 8 opponent)")
        if d_model % transformer_heads:
            raise ValueError("candidate_d_model must be divisible by transformer_heads")
        self.in_channels = int(in_channels)
        self.board_channels = 16
        self.d_model = int(d_model)
        self.aux_dim = int(aux_dim)
        self.patch_kernel = int(patch_kernel)
        self.cost_norm_denom = float(cost_norm_denom)
        self.logit_scale = d_model**-0.5
        self.value_atoms = int(value_atoms)
        self.conditioned_trunk = bool(conditioned_trunk)
        self.opponent_features = bool(opponent_features)

        embed_cls = (
            OrderedPairEmbedding
            if pill_embed_type.strip().lower() in {"ordered_onehot", "ordered", "onehot", "ordered_pair"}
            else UnorderedPillEmbedding
        )
        embed_kwargs = {"num_colors": num_colors, "output_dim": pill_embed_dim}
        if embed_cls is UnorderedPillEmbedding:
            embed_kwargs["embedding_dim"] = 16
        self.pill_embedding = embed_cls(**embed_kwargs)
        self.preview_embedding = embed_cls(**embed_kwargs)
        cond_in = 2 * pill_embed_dim + self.aux_dim
        self.condition = nn.Sequential(
            nn.Linear(cond_in, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

        self.bottle = _SharedBottleEncoder(8, d_model, encoder_blocks, d_model)
        self.column_pos = nn.Parameter(torch.randn(1, GRID_W, d_model) * 0.02)
        self.side = nn.Parameter(torch.randn(1, 2, 1, d_model) * 0.02)
        self.interaction = nn.ModuleList(
            [_TokenBlock(d_model, transformer_heads, transformer_ff_mult) for _ in range(interaction_layers)]
        )
        self.global_fusion = nn.Sequential(
            nn.LayerNorm(5 * d_model),
            nn.Linear(5 * d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
        )

        self.row_embed = nn.Embedding(GRID_H, pos_embed_dim)
        self.col_embed = nn.Embedding(GRID_W, pos_embed_dim)
        self.orient_embed = nn.Embedding(ORIENTS, pos_embed_dim)
        self.cost_mlp = nn.Sequential(
            nn.Linear(1, cost_embed_dim), nn.SiLU(), nn.Linear(cost_embed_dim, cost_embed_dim)
        )
        patch_dim = 2 * 4 * patch_kernel * patch_kernel
        candidate_in = pos_embed_dim + cost_embed_dim + patch_dim + 4 * d_model
        self.candidate = nn.Sequential(
            nn.Linear(candidate_in, cand_hidden_dim),
            nn.SiLU(),
            nn.Linear(cand_hidden_dim, d_model),
        )
        self.candidate_blocks = nn.ModuleList(
            [_TokenBlock(d_model, transformer_heads, transformer_ff_mult) for _ in range(cross_layers)]
        )
        self.policy = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, value_atoms)
        )
        self.register_buffer("value_support", torch.linspace(-1.0, 1.0, value_atoms), persistent=False)
        self.register_buffer("_dr", torch.tensor([0, 1, 0, -1], dtype=torch.int64), persistent=False)
        self.register_buffer("_dc", torch.tensor([1, 0, -1, 0], dtype=torch.int64), persistent=False)

    @staticmethod
    def _gather_map(fmap: torch.Tensor, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        batch, dim, _height, width = fmap.shape
        index = row * width + col
        flat = fmap.reshape(batch, dim, -1)
        return flat.gather(2, index.unsqueeze(1).expand(-1, dim, -1)).transpose(1, 2)

    def _patch(self, planes: torch.Tensor, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        radius = self.patch_kernel // 2
        unfolded = F.unfold(planes, self.patch_kernel, padding=radius)
        index = row * GRID_W + col
        return unfolded.gather(2, index.unsqueeze(1).expand(-1, unfolded.shape[1], -1)).transpose(1, 2)

    def forward(
        self,
        obs: torch.Tensor,
        pill_colors: torch.Tensor,
        preview_pill_colors: torch.Tensor,
        cand_actions: torch.Tensor,
        cand_cost: torch.Tensor,
        cand_mask: torch.Tensor,
        *,
        aux: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if obs.ndim != 4 or obs.shape[-2:] != (GRID_H, GRID_W):
            raise ValueError(f"expected obs [B,C,16,8], got {tuple(obs.shape)}")
        if obs.shape[1] < 16:
            raise ValueError("G5 observation lacks opponent bottle planes")
        if self.aux_dim and aux is None:
            raise ValueError("aux is required when aux_dim > 0")
        pieces = [self.pill_embedding(pill_colors), self.preview_embedding(preview_pill_colors)]
        if self.aux_dim:
            pieces.append(aux.to(obs.dtype))
        cond = self.condition(torch.cat(pieces, dim=-1))

        bottle_cond = cond if self.conditioned_trunk else torch.zeros_like(cond)
        own = self.bottle(obs[:, :8], bottle_cond)
        opponent_obs = obs[:, 8:16] if self.opponent_features else torch.zeros_like(obs[:, 8:16])
        opponent = self.bottle(opponent_obs, bottle_cond)
        columns = torch.stack(
            (own.mean(dim=2).transpose(1, 2), opponent.mean(dim=2).transpose(1, 2)), dim=1
        )
        columns = columns + self.column_pos.unsqueeze(1) + self.side
        tokens = columns.flatten(1, 2)
        for block in self.interaction:
            tokens = block(tokens)
        interacted = tokens.reshape(obs.shape[0], 2, GRID_W, self.d_model)
        opponent_columns = interacted[:, 1]
        global_context = self.global_fusion(
            torch.cat(
                (
                    own.mean(dim=(2, 3)),
                    own.amax(dim=(2, 3)),
                    opponent.mean(dim=(2, 3)),
                    opponent.amax(dim=(2, 3)),
                    cond,
                ),
                dim=-1,
            )
        )

        actions = cand_actions.long().clamp_min(0)
        orient, row, col = _decode_actions(actions)
        orient = orient.clamp(0, ORIENTS - 1)
        row = row.clamp(0, GRID_H - 1)
        col = col.clamp(0, GRID_W - 1)
        valid = cand_mask.bool()
        row2 = torch.where(valid, row + self._dr[orient], row).clamp(0, GRID_H - 1)
        col2 = torch.where(valid, col + self._dc[orient], col).clamp(0, GRID_W - 1)
        pose = self.row_embed(row) + self.col_embed(col) + self.orient_embed(orient)
        cost = self.cost_mlp(
            (cand_cost.to(obs.dtype).clamp_min(0) / self.cost_norm_denom).clamp_max(4).unsqueeze(-1)
        )
        raw = obs[:, :4]
        patches = torch.cat((self._patch(raw, row, col), self._patch(raw, row2, col2)), dim=-1)
        own_local = torch.cat((self._gather_map(own, row, col), self._gather_map(own, row2, col2)), dim=-1)
        col_index = col.unsqueeze(-1).expand(-1, -1, self.d_model)
        col2_index = col2.unsqueeze(-1).expand(-1, -1, self.d_model)
        threat = torch.cat(
            (opponent_columns.gather(1, col_index), opponent_columns.gather(1, col2_index)), dim=-1
        )
        candidate = self.candidate(torch.cat((pose, cost, patches, own_local, threat), dim=-1))
        candidate = candidate + global_context.unsqueeze(1)
        padding = ~valid
        safe_padding = padding & ~padding.all(dim=1, keepdim=True)
        for block in self.candidate_blocks:
            candidate = block(candidate, safe_padding)

        query = self.policy(candidate)
        logits = (query * global_context.unsqueeze(1)).sum(dim=-1) * self.logit_scale
        logits = logits.masked_fill(~valid, -1e9)
        value_logits = self.value_head(global_context)
        value = (value_logits.softmax(dim=-1) * self.value_support.to(value_logits.dtype)).sum(
            dim=-1, keepdim=True
        )
        if return_aux:
            return logits, value, {"value_logits": value_logits}
        return logits, value

    def distributional_value_loss(
        self, value_logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        targets = targets.clamp(-1.0, 1.0)
        spacing = float(2.0 / (self.value_atoms - 1))
        position = (targets + 1.0) / spacing
        lower = position.floor().long().clamp(0, self.value_atoms - 1)
        upper = position.ceil().long().clamp(0, self.value_atoms - 1)
        upper_weight = position - lower.to(position.dtype)
        # Distributional projection is a loss calculation, so keep it in
        # FP32 under mixed precision just like log_softmax below.
        target_dist = torch.zeros_like(value_logits, dtype=torch.float32)
        target_dist.scatter_add_(1, lower.unsqueeze(1), (1.0 - upper_weight).unsqueeze(1))
        target_dist.scatter_add_(1, upper.unsqueeze(1), upper_weight.unsqueeze(1))
        return -(target_dist * F.log_softmax(value_logits.float(), dim=-1)).sum(dim=-1).mean()


__all__ = ["G5CandidatePlacementPolicyNet"]
