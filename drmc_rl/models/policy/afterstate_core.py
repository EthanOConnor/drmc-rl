"""Afterstate public-context core: score what each placement actually does.

G5 scores a pose from the pre-lock bottle plus 9x9 raw patches, so clears,
cascades and the garbage a move sends must be inferred. This core feeds every
candidate's exact settled own bottle (``drmc_rl.game.afterstate``) and its
exact consequence facts into the candidate token, as the V3 human model does,
while keeping G5's public-context contract: both bottles through one
FiLM-conditioned residual trunk, per-side public conditioning, column
interaction, candidate-set attention and a 51-atom value head.

The trunk is 256 channels x 6 dense blocks (the V3 width/depth) instead of
320 x 8. Each afterstate is encoded by a small CNN that also sees the root
trunk's features through a 1x1 projection, so candidate cost stays a small
fraction of the trunk at browser batch 1.

The call signature matches ``G5CandidatePlacementPolicyNet`` so arena,
controller trainer and live backend use it unchanged. When ``afterstate`` is
omitted the exact afterstates are computed from the public model inputs on
the host; ONNX export passes them explicitly (``forward_features``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drmc_rl.game.afterstate import FACT_DIM, afterstate_batch
from drmc_rl.models.policy.candidate_policy import GRID_H, GRID_W, ORIENTS, _decode_actions
from drmc_rl.models.policy.candidate_policy_g5 import _SharedBottleEncoder, _TokenBlock
from drmc_rl.models.policy.placement_heads import OrderedPairEmbedding

AFTERSTATE_CORE_SCHEMA = "drmc-afterstate-core-v1"


def _tile_plane_table() -> torch.Tensor:
    """Lookup from native tile byte to the eight semantic planes."""

    table = torch.zeros(256, 8)
    for tile in range(256):
        kind, color = tile & 0xF0, tile & 0x03
        visible = tile not in (0xFF, 0) and kind not in (0xB0, 0xF0)
        if not visible:
            continue
        table[tile, {1: 0, 0: 1, 2: 2}.get(color, 0)] = 1.0 if color < 3 else 0.0
        table[tile, 3] = float(kind == 0xD0)
        table[tile, 4] = float(kind == 0x50)
        table[tile, 5] = float(kind == 0x40)
        table[tile, 6] = float(kind == 0x70)
        table[tile, 7] = float(kind == 0x60)
    return table


class _AfterResidual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.silu(self.norm1(x)))
        return x + self.conv2(F.silu(self.norm2(y)))


class AfterstateCorePolicyNet(nn.Module):
    """Public-context candidate policy over exact post-lock afterstates."""

    def __init__(
        self,
        *,
        in_channels: int = 20,
        board_channels: int = 16,
        encoder_blocks: int = 6,
        d_model: int = 256,
        pill_embed_dim: int = 128,
        aux_dim: int = 733,
        public_context_schema: str = "public_pair_context_v3",
        pos_embed_dim: int = 32,
        cost_embed_dim: int = 32,
        cand_hidden_dim: int = 768,
        transformer_heads: int = 8,
        transformer_ff_mult: int = 4,
        cross_layers: int = 3,
        cross_ff_mult: int = 4,
        interaction_layers: int = 2,
        after_channels: int = 32,
        after_blocks: int = 2,
        root_projection: int = 16,
        fact_dim: int = FACT_DIM,
        cost_norm_denom: float = 64.0,
        value_atoms: int = 51,
        terminal_wdl: bool = True,
        candidate_wdl: bool = True,
    ) -> None:
        super().__init__()
        from drmc_rl.game.public_context import PUBLIC_CONTEXT_DIMS, SIDE_FEATURE_DIM

        if board_channels != 16:
            raise ValueError("the afterstate core requires both bottles (16 board channels)")
        if public_context_schema not in PUBLIC_CONTEXT_DIMS or aux_dim != PUBLIC_CONTEXT_DIMS[public_context_schema]:
            raise ValueError("public context schema and auxiliary width must match")
        if d_model % transformer_heads:
            raise ValueError("d_model must be divisible by transformer_heads")
        self.in_channels = int(in_channels)
        self.board_channels = 16
        self.d_model = d = int(d_model)
        self.aux_dim = int(aux_dim)
        self.public_context_schema = public_context_schema
        self.public_side_dim = SIDE_FEATURE_DIM
        self.fact_dim = int(fact_dim)
        self.cost_norm_denom = float(cost_norm_denom)
        self.logit_scale = d**-0.5
        self.value_atoms = int(value_atoms)
        # Interface parity with G5 for callers that inspect these switches.
        self.critic_context = "global"
        self.motor_auxiliary = None

        self.pill_embedding = OrderedPairEmbedding(3, pill_embed_dim)
        self.preview_embedding = OrderedPairEmbedding(3, pill_embed_dim)
        self.condition = nn.Sequential(
            nn.Linear(2 * pill_embed_dim + self.aux_dim, d), nn.SiLU(), nn.Linear(d, d)
        )
        self.side_condition = nn.Sequential(
            nn.Linear(2 * pill_embed_dim + SIDE_FEATURE_DIM, d), nn.SiLU(), nn.Linear(d, d)
        )
        self.bottle = _SharedBottleEncoder(8, d, encoder_blocks, d, "dense")
        self.column_pos = nn.Parameter(torch.randn(1, GRID_W, d) * 0.02)
        self.side = nn.Parameter(torch.randn(1, 2, 1, d) * 0.02)
        self.interaction = nn.ModuleList(
            [_TokenBlock(d, transformer_heads, transformer_ff_mult) for _ in range(interaction_layers)]
        )
        self.global_fusion = nn.Sequential(
            nn.LayerNorm(5 * d), nn.Linear(5 * d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d)
        )

        self.row_embed = nn.Embedding(GRID_H, pos_embed_dim)
        self.col_embed = nn.Embedding(GRID_W, pos_embed_dim)
        self.orient_embed = nn.Embedding(ORIENTS, pos_embed_dim)
        self.cost_mlp = nn.Sequential(
            nn.Linear(1, cost_embed_dim), nn.SiLU(), nn.Linear(cost_embed_dim, cost_embed_dim)
        )
        # Afterstate encoder: settled planes, changed-cell mask, coordinates
        # and a projection of the shared root trunk features.
        self.after_channels = int(after_channels)
        self.root_projection = nn.Conv2d(d, int(root_projection), 1)
        self.after_stem = nn.Conv2d(8 + 1 + 2 + int(root_projection), self.after_channels, 3, padding=1)
        self.after_blocks = nn.ModuleList([_AfterResidual(self.after_channels) for _ in range(after_blocks)])
        self.after_out = nn.Sequential(
            nn.LayerNorm(4 * self.after_channels), nn.Linear(4 * self.after_channels, d), nn.SiLU()
        )
        self.facts = nn.Sequential(nn.Linear(self.fact_dim, 64), nn.SiLU())
        candidate_in = pos_embed_dim + cost_embed_dim + 4 * d + d + 64
        self.candidate = nn.Sequential(
            nn.Linear(candidate_in, cand_hidden_dim), nn.SiLU(), nn.Linear(cand_hidden_dim, d)
        )
        self.candidate_blocks = nn.ModuleList(
            [_TokenBlock(d, transformer_heads, cross_ff_mult) for _ in range(cross_layers)]
        )
        self.policy = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.value_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, value_atoms)
        )
        self.state_wdl_head = nn.Linear(d, 3) if terminal_wdl else None
        self.candidate_wdl_head = nn.Linear(d, 3) if candidate_wdl else None

        self.register_buffer("tile_planes", _tile_plane_table(), persistent=False)
        self.register_buffer("value_support", torch.linspace(-1.0, 1.0, value_atoms), persistent=False)
        self.register_buffer("_dr", torch.tensor([0, 1, 0, -1], dtype=torch.int64), persistent=False)
        self.register_buffer("_dc", torch.tensor([1, 0, -1, 0], dtype=torch.int64), persistent=False)
        rows = torch.linspace(0.0, 1.0, GRID_H).view(1, 1, GRID_H, 1).expand(1, 1, GRID_H, GRID_W)
        cols = torch.linspace(0.0, 1.0, GRID_W).view(1, 1, 1, GRID_W).expand(1, 1, GRID_H, GRID_W)
        self.register_buffer("_coords", torch.cat((rows, cols), 1), persistent=False)

    # ------------------------------------------------------------------
    @staticmethod
    def exact_afterstates(obs, pill_colors, cand_actions, cand_mask):
        """Host-side exact afterstate tensors for one model call."""

        tiles, facts = afterstate_batch(
            obs[:, :8].detach().to("cpu", torch.float32).numpy(),
            pill_colors.detach().cpu().numpy(),
            cand_actions.detach().cpu().numpy(),
            cand_mask.detach().cpu().numpy().astype(bool),
        )
        device = obs.device
        return torch.from_numpy(tiles).to(device), torch.from_numpy(facts).to(device)

    @staticmethod
    def _gather_map(fmap: torch.Tensor, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        batch, dim, _h, width = fmap.shape
        index = row * width + col
        return fmap.reshape(batch, dim, -1).gather(2, index.unsqueeze(1).expand(-1, dim, -1)).transpose(1, 2)

    def _encode_afterstates(self, own, root_tiles, after_tiles, valid):
        batch, width = valid.shape
        planes = self.tile_planes.to(own.dtype)[after_tiles.long()]  # [B,K,128,8]
        planes = planes.permute(0, 1, 3, 2).reshape(batch, width, 8, GRID_H, GRID_W)
        changed = (after_tiles != root_tiles.unsqueeze(1)).to(own.dtype).reshape(batch, width, 1, GRID_H, GRID_W)
        root = self.root_projection(own)
        root = root.unsqueeze(1).expand(-1, width, -1, -1, -1)
        coords = self._coords.to(own.dtype).expand(batch, width, -1, -1, -1)
        x = torch.cat((planes, changed, coords, root), dim=2).flatten(0, 1)
        flat_valid = valid.reshape(-1)
        packed = not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export()
        if packed:
            index = flat_valid.nonzero(as_tuple=True)[0]
            x = x.index_select(0, index)
        y = self.after_stem(x)
        for block in self.after_blocks:
            y = block(y)
        change_weight = x[:, 8:9]
        n_changed = change_weight.sum(dim=(2, 3)).clamp_min(1.0)
        pooled = torch.cat(
            (
                y.mean(dim=(2, 3)),
                y.amax(dim=(2, 3)),
                (y * change_weight).sum(dim=(2, 3)) / n_changed,
                (y * change_weight - (1.0 - change_weight) * 1e4).amax(dim=(2, 3)).clamp_min(-1e3),
            ),
            dim=-1,
        )
        encoded = self.after_out(pooled)
        if packed:
            full = encoded.new_zeros((batch * width, encoded.shape[-1]))
            encoded = full.index_copy(0, index, encoded)
        return encoded.reshape(batch, width, -1)

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
        afterstate: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        motor_geometry=None,
        prepared_bottles=None,
    ):
        if prepared_bottles is not None:
            raise ValueError("the afterstate core does not accept prepared G5 bottles")
        if afterstate is None:
            with torch.no_grad():
                afterstate = self.exact_afterstates(obs, pill_colors, cand_actions, cand_mask)
        after_tiles, facts = afterstate
        return self.forward_features(
            obs, pill_colors, preview_pill_colors, cand_actions, cand_cost, cand_mask, aux,
            after_tiles, facts, return_aux=return_aux,
        )

    def forward_features(
        self,
        obs,
        pill_colors,
        preview_pill_colors,
        cand_actions,
        cand_cost,
        cand_mask,
        aux,
        after_tiles,
        facts,
        *,
        return_aux: bool = False,
    ):
        if obs.ndim != 4 or obs.shape[1] < 16 or obs.shape[-2:] != (GRID_H, GRID_W):
            raise ValueError(f"expected obs [B,>=16,16,8], got {tuple(obs.shape)}")
        if aux is None:
            raise ValueError("the public-context afterstate core requires aux")
        dtype = obs.dtype
        aux = aux.to(dtype)
        pill_e = self.pill_embedding(pill_colors)
        preview_e = self.preview_embedding(preview_pill_colors)
        cond = self.condition(torch.cat((pill_e, preview_e, aux), dim=-1))
        own_public = aux[:, : self.public_side_dim]
        opp_public = aux[:, self.public_side_dim : 2 * self.public_side_dim]
        opp_pill = opp_public[:, :6].reshape(-1, 2, 3).argmax(-1)
        opp_preview = opp_public[:, 6:12].reshape(-1, 2, 3).argmax(-1)
        side_cond = self.side_condition(
            torch.cat(
                (
                    torch.cat((pill_e, preview_e, own_public), -1),
                    torch.cat((self.pill_embedding(opp_pill), self.preview_embedding(opp_preview), opp_public), -1),
                ),
                dim=0,
            )
        )
        batch = obs.shape[0]
        maps = self.bottle(torch.cat((obs[:, :8], obs[:, 8:16]), dim=0), side_cond)
        own, opponent = maps[:batch], maps[batch:]
        columns = torch.stack(
            (own.mean(dim=2).transpose(1, 2), opponent.mean(dim=2).transpose(1, 2)), dim=1
        )
        columns = columns + self.column_pos.unsqueeze(1) + self.side
        tokens = columns.flatten(1, 2)
        for block in self.interaction:
            tokens = block(tokens)
        opponent_columns = tokens.reshape(batch, 2, GRID_W, self.d_model)[:, 1]
        global_context = self.global_fusion(
            torch.cat(
                (own.mean(dim=(2, 3)), own.amax(dim=(2, 3)),
                 opponent.mean(dim=(2, 3)), opponent.amax(dim=(2, 3)), cond),
                dim=-1,
            )
        )

        valid = cand_mask.bool()
        orient, row, col = _decode_actions(cand_actions.long().clamp_min(0))
        orient = orient.clamp(0, ORIENTS - 1)
        row = row.clamp(0, GRID_H - 1)
        col = col.clamp(0, GRID_W - 1)
        row2 = torch.where(valid, row + self._dr[orient], row).clamp(0, GRID_H - 1)
        col2 = torch.where(valid, col + self._dc[orient], col).clamp(0, GRID_W - 1)
        pose = self.row_embed(row) + self.col_embed(col) + self.orient_embed(orient)
        cost = self.cost_mlp(
            (cand_cost.to(dtype).clamp_min(0) / self.cost_norm_denom).clamp_max(4).unsqueeze(-1)
        )
        own_local = torch.cat((self._gather_map(own, row, col), self._gather_map(own, row2, col2)), dim=-1)
        col_index = col.unsqueeze(-1).expand(-1, -1, self.d_model)
        col2_index = col2.unsqueeze(-1).expand(-1, -1, self.d_model)
        threat = torch.cat(
            (opponent_columns.gather(1, col_index), opponent_columns.gather(1, col2_index)), dim=-1
        )
        root_tiles = self._root_tiles(obs[:, :8])
        after = self._encode_afterstates(own, root_tiles, after_tiles, valid)
        fact = self.facts(facts.to(dtype))
        candidate = self.candidate(torch.cat((pose, cost, own_local, threat, after, fact), dim=-1))
        candidate = candidate + global_context.unsqueeze(1)
        padding = ~valid
        safe_padding = padding & ~padding.all(dim=1, keepdim=True)
        for block in self.candidate_blocks:
            candidate = block(candidate, safe_padding)
        logits = (self.policy(candidate) * global_context.unsqueeze(1)).sum(dim=-1) * self.logit_scale
        logits = logits.masked_fill(~valid, -1e9)
        value_logits = self.value_head(global_context)
        value = (value_logits.softmax(dim=-1) * self.value_support.to(value_logits.dtype)).sum(dim=-1, keepdim=True)
        if return_aux:
            extra = {
                "value_logits": value_logits,
                "value_context": global_context,
                "candidate_context": candidate,
                "global_context": global_context,
            }
            if self.state_wdl_head is not None:
                extra["state_wdl_logits"] = self.state_wdl_head(global_context)
            if self.candidate_wdl_head is not None:
                extra["candidate_wdl_logits"] = self.candidate_wdl_head(candidate)
            return logits, value, extra
        return logits, value

    def _root_tiles(self, planes: torch.Tensor) -> torch.Tensor:
        """Differentiable-free inverse of the semantic planes inside the graph."""

        occupied = planes[:, :3].sum(dim=1) > 0.5
        canonical = planes[:, :3].argmax(dim=1)
        low = torch.where(canonical == 0, 1, torch.where(canonical == 1, 0, 2))
        high = torch.full_like(low, 0x80)
        for channel, tile in ((4, 0x50), (5, 0x40), (6, 0x70), (7, 0x60)):
            high = torch.where(planes[:, channel] > 0.5, torch.full_like(high, tile), high)
        high = torch.where(planes[:, 3] > 0.5, torch.full_like(high, 0xD0), high)
        tiles = torch.where(occupied, high + low, torch.full_like(low, 0xFF))
        return tiles.reshape(planes.shape[0], GRID_H * GRID_W).to(torch.uint8)

    def distributional_value_loss(self, value_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.clamp(-1.0, 1.0)
        spacing = float(2.0 / (self.value_atoms - 1))
        position = (targets + 1.0) / spacing
        lower = position.floor().long().clamp(0, self.value_atoms - 1)
        upper = position.ceil().long().clamp(0, self.value_atoms - 1)
        upper_weight = position - lower.to(position.dtype)
        target = torch.zeros_like(value_logits, dtype=torch.float32)
        target.scatter_add_(1, lower.unsqueeze(1), (1.0 - upper_weight).unsqueeze(1))
        target.scatter_add_(1, upper.unsqueeze(1), upper_weight.unsqueeze(1))
        return -(target * F.log_softmax(value_logits.float(), dim=-1)).sum(dim=-1).mean()


def afterstate_core_config(**overrides) -> dict:
    """``smdp_ppo`` section of a new afterstate-core checkpoint config."""

    sp = {
        "policy_type": "candidate",
        "candidate_architecture": "afterstate",
        "afterstate_schema": AFTERSTATE_CORE_SCHEMA,
        "aux_spec": "public_pair_context_v3",
        "candidate_board_channels": 16,
        "candidate_max_candidates": 512,
        "encoder_blocks": 6,
        "candidate_d_model": 256,
        "pill_embed_dim": 128,
        "pill_embed_type": "ordered_pair",
        "candidate_pos_embed_dim": 32,
        "candidate_cost_embed_dim": 32,
        "candidate_hidden_dim": 768,
        "candidate_transformer_heads": 8,
        "candidate_transformer_ff_mult": 4,
        "candidate_cross_layers": 3,
        "candidate_cross_ff_mult": 4,
        "candidate_interaction_layers": 2,
        "candidate_value_atoms": 51,
        "candidate_terminal_wdl": True,
        "candidate_wdl": True,
        "afterstate_channels": 32,
        "afterstate_blocks": 2,
        "afterstate_root_projection": 16,
    }
    sp.update(overrides)
    return {"smdp_ppo": sp, "env": {"public_observations": True}}


def build_afterstate_core(sp: dict, in_channels: int, aux_dim: int) -> AfterstateCorePolicyNet:
    def g(key, default):
        return sp.get(key, default)

    if g("afterstate_schema", AFTERSTATE_CORE_SCHEMA) != AFTERSTATE_CORE_SCHEMA:
        raise ValueError("unknown afterstate core schema")
    return AfterstateCorePolicyNet(
        in_channels=int(in_channels),
        board_channels=int(g("candidate_board_channels", 16)),
        encoder_blocks=int(g("encoder_blocks", 6)),
        d_model=int(g("candidate_d_model", 256)),
        pill_embed_dim=int(g("pill_embed_dim", 128)),
        aux_dim=int(aux_dim),
        public_context_schema=str(g("aux_spec", "public_pair_context_v3")),
        pos_embed_dim=int(g("candidate_pos_embed_dim", 32)),
        cost_embed_dim=int(g("candidate_cost_embed_dim", 32)),
        cand_hidden_dim=int(g("candidate_hidden_dim", 768)),
        transformer_heads=int(g("candidate_transformer_heads", 8)),
        transformer_ff_mult=int(g("candidate_transformer_ff_mult", 4)),
        cross_layers=int(g("candidate_cross_layers", 3)),
        cross_ff_mult=int(g("candidate_cross_ff_mult", 4)),
        interaction_layers=int(g("candidate_interaction_layers", 2)),
        after_channels=int(g("afterstate_channels", 32)),
        after_blocks=int(g("afterstate_blocks", 2)),
        root_projection=int(g("afterstate_root_projection", 16)),
        value_atoms=int(g("candidate_value_atoms", 51)),
        terminal_wdl=bool(g("candidate_terminal_wdl", True)),
        candidate_wdl=bool(g("candidate_wdl", True)),
    )


__all__ = [
    "AFTERSTATE_CORE_SCHEMA",
    "AfterstateCorePolicyNet",
    "afterstate_core_config",
    "build_afterstate_core",
]
