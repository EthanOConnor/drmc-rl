"""Candidate-scoring placement policy networks.

Unlike the heatmap heads (which output a fixed 4×16×8 grid of logits), these
policies score only a packed list of planner-feasible macro actions and treat
the planner's frames-to-lock as an explicit per-candidate feature.

Two board encoder variants are supported:
  - `cnn`: CoordConv-style CNN trunk (pooled global + per-cell feature gathering)
  - `col_transformer`: 8-column token encoder + tiny transformer trunk
    (pooled global + per-column gathering)

The network outputs:
  - `logits`: [B, Kmax] candidate-slot logits (padding masked to -1e9)
  - `value`:  [B, 1] state-value estimate
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

from models.policy.placement_heads import OrderedPairEmbedding, UnorderedPillEmbedding

GRID_H = 16
GRID_W = 8
ORIENTS = 4


def _decode_actions(
    actions: "torch.Tensor",
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    a = actions.long()
    o = torch.div(a, GRID_H * GRID_W, rounding_mode="floor")
    rem = a % (GRID_H * GRID_W)
    row = torch.div(rem, GRID_W, rounding_mode="floor")
    col = rem % GRID_W
    return o, row, col


def _coord_channels(
    B: int, H: int, W: int, *, device: "torch.device", dtype: "torch.dtype"
) -> "torch.Tensor":
    y = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
    x = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)
    return torch.cat([y, x], dim=1)


def _gather_patch(
    occ: "torch.Tensor",
    row: "torch.Tensor",
    col: "torch.Tensor",
    *,
    kernel: int,
    pad_value: float = 1.0,
) -> "torch.Tensor":
    """Return flattened occupancy patches around each cell.

    Args:
        occ: [B,H,W] float tensor in {0,1}.
        row/col: [B,K] int tensors (will be clamped to in-bounds).
        kernel: odd patch kernel size (e.g. 3 or 5).
    """

    k = int(kernel)
    if k <= 1:
        B = int(occ.shape[0])
        b_idx = torch.arange(B, device=occ.device).unsqueeze(-1)
        row_i = row.clamp(0, GRID_H - 1).long()
        col_i = col.clamp(0, GRID_W - 1).long()
        return occ[b_idx, row_i, col_i].unsqueeze(-1)

    if k % 2 != 1:
        raise ValueError(f"patch_kernel must be odd, got {k}")
    r = k // 2
    B, H, W = occ.shape
    Hp = H + 2 * r
    Wp = W + 2 * r

    occ_p = F.pad(occ, (r, r, r, r), mode="constant", value=float(pad_value))
    occ_flat = occ_p.reshape(B, Hp * Wp)

    row_i = row.clamp(0, GRID_H - 1).long() + r
    col_i = col.clamp(0, GRID_W - 1).long() + r
    base = row_i * Wp + col_i  # [B,K]

    offsets = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            offsets.append(dy * Wp + dx)
    offs = torch.tensor(offsets, device=occ.device, dtype=torch.int64)  # [P]

    idx = base.unsqueeze(-1) + offs.view(1, 1, -1)  # [B,K,P]
    patch = occ_flat.gather(1, idx.reshape(B, -1)).reshape(B, base.shape[1], -1)
    return patch


class _ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(x + y)


class _BoardCNN(nn.Module):
    def __init__(self, *, in_ch: int, d_model: int, blocks: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch + 2, d_model, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[_ResBlock(d_model) for _ in range(int(max(0, blocks)))])

    def forward(self, board: "torch.Tensor") -> "torch.Tensor":
        B, _, H, W = board.shape
        coords = _coord_channels(B, H, W, device=board.device, dtype=board.dtype)
        x = torch.cat([board, coords], dim=1)
        x = self.stem(x)
        x = self.blocks(x)
        return x


class _TxBlock(nn.Module):
    def __init__(self, *, d_model: int, nhead: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=float(dropout), batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = float(dropout)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        y = self.ln1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        if self.dropout > 0:
            y = F.dropout(y, p=self.dropout, training=self.training)
        x = x + y

        y = self.ff(self.ln2(x))
        if self.dropout > 0:
            y = F.dropout(y, p=self.dropout, training=self.training)
        return x + y


class _BoardColumnTransformer(nn.Module):
    def __init__(self, *, in_ch: int, d_model: int, layers: int, nhead: int, ff_mult: int) -> None:
        super().__init__()
        token_in = int(in_ch) * GRID_H
        self.token_proj = nn.Linear(token_in, d_model)
        self.col_pos = nn.Parameter(torch.zeros(GRID_W, d_model))
        self.blocks = nn.Sequential(
            *[
                _TxBlock(
                    d_model=d_model,
                    nhead=int(max(1, nhead)),
                    ff_dim=int(max(d_model, int(ff_mult) * d_model)),
                    dropout=0.0,
                )
                for _ in range(int(max(0, layers)))
            ]
        )
        self.ln_out = nn.LayerNorm(d_model)
        nn.init.normal_(self.col_pos, mean=0.0, std=0.02)

    def forward(self, board: "torch.Tensor") -> "torch.Tensor":
        B, C, H, W = board.shape
        if H != GRID_H or W != GRID_W:
            raise ValueError(f"Expected board [B,C,16,8], got {tuple(board.shape)!r}")
        tokens = board.permute(0, 3, 1, 2).reshape(B, W, C * H)  # [B,8,C*16]
        x = self.token_proj(tokens) + self.col_pos.unsqueeze(0)
        x = self.blocks(x)
        x = self.ln_out(x)
        return x


class CandidatePlacementPolicyNet(nn.Module):
    """Global board embedding + per-candidate scorer."""

    def __init__(
        self,
        *,
        in_channels: int,
        board_channels: int,
        board_encoder: str,
        encoder_blocks: int,
        d_model: int,
        pill_embed_dim: int,
        pill_embed_type: str = "unordered",
        aux_dim: int = 0,
        num_colors: int = 3,
        pos_embed_dim: int = 32,
        cost_embed_dim: int = 32,
        cand_hidden_dim: int = 256,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        transformer_ff_mult: int = 4,
        patch_kernel: int = 7,
        cost_norm_denom: float = 64.0,
        use_trunk_gather: bool = True,
    ) -> None:
        super().__init__()
        if torch is None:
            raise RuntimeError("PyTorch is required for CandidatePlacementPolicyNet")

        self.in_channels = int(in_channels)
        self.board_channels = int(max(1, int(board_channels)))
        self.d_model = int(d_model)
        self.aux_dim = int(max(0, int(aux_dim)))
        self.patch_kernel = int(patch_kernel)
        self.cost_norm_denom = float(cost_norm_denom)
        self.patch_planes = int(min(4, self.board_channels))
        self.use_trunk_gather = bool(use_trunk_gather)

        k = int(self.patch_kernel)
        if k <= 0 or k % 2 != 1:
            raise ValueError(f"patch_kernel must be odd and >= 1, got {k}")
        r = k // 2
        self._patch_r = int(r)
        self._patch_wp = int(GRID_W + 2 * r)
        offsets = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                offsets.append(dy * self._patch_wp + dx)
        self.register_buffer(
            "_patch_offs", torch.tensor(offsets, dtype=torch.int64), persistent=False
        )

        enc = str(board_encoder or "").strip().lower()
        if enc == "cnn":
            self.board_trunk = _BoardCNN(
                in_ch=self.board_channels, d_model=self.d_model, blocks=int(encoder_blocks)
            )
        elif enc in {"col_transformer", "column_transformer", "columns", "tx"}:
            self.board_trunk = _BoardColumnTransformer(
                in_ch=self.board_channels,
                d_model=self.d_model,
                layers=int(transformer_layers),
                nhead=int(transformer_heads),
                ff_mult=int(transformer_ff_mult),
            )
        else:
            raise ValueError(f"Unknown board_encoder: {board_encoder!r}")

        # Conditioning on current + preview pill.
        pill_embed_type_norm = str(pill_embed_type or "").strip().lower()
        if pill_embed_type_norm in {"unordered", "deepsets", "unordered_embed"}:
            self.pill_embedding = UnorderedPillEmbedding(
                num_colors=int(num_colors),
                embedding_dim=16,
                output_dim=int(pill_embed_dim),
            )
            self.preview_embedding = UnorderedPillEmbedding(
                num_colors=int(num_colors),
                embedding_dim=16,
                output_dim=int(pill_embed_dim),
            )
        elif pill_embed_type_norm in {"ordered_onehot", "ordered", "onehot", "ordered_pair"}:
            self.pill_embedding = OrderedPairEmbedding(
                num_colors=int(num_colors),
                output_dim=int(pill_embed_dim),
            )
            self.preview_embedding = OrderedPairEmbedding(
                num_colors=int(num_colors),
                output_dim=int(pill_embed_dim),
            )
        else:
            raise ValueError(f"Unknown pill_embed_type: {pill_embed_type!r}")
        self.pill_fusion = nn.Sequential(
            nn.Linear(int(pill_embed_dim) * 2, int(pill_embed_dim)),
            nn.ReLU(inplace=True),
        )

        self.aux_encoder: Optional[nn.Module]
        self.cond_fusion: Optional[nn.Module]
        if self.aux_dim > 0:
            self.aux_encoder = nn.Sequential(
                nn.Linear(self.aux_dim, int(pill_embed_dim)),
                nn.ReLU(inplace=True),
                nn.Linear(int(pill_embed_dim), int(pill_embed_dim)),
                nn.ReLU(inplace=True),
            )
            self.cond_fusion = nn.Sequential(
                nn.Linear(int(pill_embed_dim) * 2, int(pill_embed_dim)),
                nn.ReLU(inplace=True),
            )
        else:
            self.aux_encoder = None
            self.cond_fusion = None

        self.cond_to_model = nn.Linear(int(pill_embed_dim), self.d_model)
        self.ln_g = nn.LayerNorm(self.d_model)

        # Candidate feature encoders.
        self.row_embed = nn.Embedding(GRID_H, int(pos_embed_dim))
        self.col_embed = nn.Embedding(GRID_W, int(pos_embed_dim))
        self.orient_embed = nn.Embedding(ORIENTS, int(pos_embed_dim))

        self.cost_mlp = nn.Sequential(
            nn.Linear(1, int(cost_embed_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(cost_embed_dim), int(cost_embed_dim)),
            nn.ReLU(inplace=True),
        )

        k2 = int(self.patch_kernel) * int(self.patch_kernel)
        patch_dim = (
            2 * int(self.patch_planes) * k2
            if int(self.patch_kernel) > 1
            else 2 * int(self.patch_planes)
        )
        trunk_local_dim = 2 * int(self.d_model) if self.use_trunk_gather else 0
        cand_in_dim = int(pos_embed_dim) + int(cost_embed_dim) + int(patch_dim) + trunk_local_dim

        self.cand_mlp = nn.Sequential(
            nn.Linear(cand_in_dim, int(cand_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(cand_hidden_dim), self.d_model),
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, 1),
        )

        # Offsets for partner cell in (row, col) coordinates.
        self.register_buffer(
            "_dr", torch.tensor([0, 1, 0, -1], dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "_dc", torch.tensor([1, 0, -1, 0], dtype=torch.int64), persistent=False
        )

    def _gather_trunk_map(
        self, fmap: "torch.Tensor", row: "torch.Tensor", col: "torch.Tensor"
    ) -> "torch.Tensor":
        """Gather per-cell trunk features from a [B,D,H,W] feature map."""

        B, D, H, W = fmap.shape
        idx = row.long() * int(W) + col.long()  # [B,K]
        flat = fmap.reshape(B, D, int(H) * int(W))
        idx_exp = idx.unsqueeze(1).expand(B, D, idx.shape[1])
        out = flat.gather(2, idx_exp).permute(0, 2, 1)  # [B,K,D]
        return out

    def _gather_trunk_tokens(self, tokens: "torch.Tensor", col: "torch.Tensor") -> "torch.Tensor":
        """Gather per-column trunk features from [B,8,D] tokens."""

        B, W, D = tokens.shape
        idx = col.long().unsqueeze(-1).expand(B, col.shape[1], D)
        return tokens.gather(1, idx)

    def _gather_patch_planes(
        self, planes: "torch.Tensor", row: "torch.Tensor", col: "torch.Tensor"
    ) -> "torch.Tensor":
        """Return flattened multi-plane patches around each cell.

        Notes:
          - This method intentionally does *not* clamp row/col in-bounds. If any
            supposedly-valid candidate encodes an out-of-bounds cell, we prefer
            failing loudly over silently folding indices back onto the board.
          - Offsets are precomputed as a buffer to avoid per-forward tensor
            allocations (important on MPS/GPU).
        """

        k = int(self.patch_kernel)
        if k == 1:
            B, P, H, W = planes.shape
            idx = row.long() * int(W) + col.long()  # [B,K]
            flat = planes.reshape(B, P, H * W)
            idx_exp = idx.unsqueeze(1).expand(B, P, idx.shape[1])
            out = flat.gather(2, idx_exp).permute(0, 2, 1)  # [B,K,P]
            return out
        r = int(self._patch_r)
        Wp = int(self._patch_wp)
        B, P, H, _W = planes.shape
        planes_p = F.pad(planes, (r, r, r, r), mode="constant", value=0.0)
        flat = planes_p.reshape(B, P, (H + 2 * r) * Wp)
        return self._gather_patch_planes_from_flat(flat, row, col)

    def _gather_patch_planes_from_flat(
        self, flat: "torch.Tensor", row: "torch.Tensor", col: "torch.Tensor"
    ) -> "torch.Tensor":
        """Gather multi-plane patches from a pre-padded, flattened plane tensor.

        Args:
            flat: [B,P,(H+2r)*(W+2r)] where W=GRID_W and r=patch_kernel//2.
        """

        r = int(self._patch_r)
        Wp = int(self._patch_wp)
        B, P, _ = flat.shape

        row_i = row.long() + r
        col_i = col.long() + r
        base = row_i * Wp + col_i  # [B,K]

        idx = base.unsqueeze(-1) + self._patch_offs.view(1, 1, -1)  # [B,K,K2]
        idx_exp = idx.unsqueeze(1).expand(B, P, idx.shape[1], idx.shape[2]).reshape(B, P, -1)
        patch = flat.gather(2, idx_exp).reshape(B, P, idx.shape[1], -1)
        patch = patch.permute(0, 2, 1, 3).reshape(B, idx.shape[1], -1)
        return patch

    def forward(
        self,
        obs: "torch.Tensor",
        pill_colors: "torch.Tensor",
        preview_pill_colors: "torch.Tensor",
        cand_actions: "torch.Tensor",
        cand_cost: "torch.Tensor",
        cand_mask: "torch.Tensor",
        *,
        aux: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Return (logits, value) for the packed candidate list."""

        if obs.ndim != 4:
            raise ValueError(f"Expected obs [B,C,16,8], got {tuple(obs.shape)!r}")
        B, C, H, W = obs.shape
        if H != GRID_H or W != GRID_W:
            raise ValueError(f"Expected obs [B,C,16,8], got {tuple(obs.shape)!r}")
        if C < self.board_channels:
            raise ValueError(f"obs has {C} channels but board_channels={self.board_channels}")

        board = obs[:, : self.board_channels, :, :]

        # Global board embedding.
        trunk = self.board_trunk(board)
        if trunk.ndim == 4:
            fmap = trunk  # [B,D,H,W]
            g_board = fmap.mean(dim=(2, 3))
        elif trunk.ndim == 3:
            tokens = trunk  # [B,8,D]
            g_board = tokens.mean(dim=1)
        else:
            raise ValueError(f"Unexpected board trunk output shape: {tuple(trunk.shape)!r}")

        # Conditioning vector (current + preview + optional aux).
        p_curr = self.pill_embedding(pill_colors)
        p_prev = self.preview_embedding(preview_pill_colors)
        p = self.pill_fusion(torch.cat([p_curr, p_prev], dim=-1))
        cond = p
        if self.aux_dim > 0:
            if aux is None:
                raise ValueError("aux is required when aux_dim > 0")
            aux_embed = self.aux_encoder(aux)  # type: ignore[misc]
            cond = self.cond_fusion(torch.cat([p, aux_embed], dim=-1))  # type: ignore[misc]
        cond_m = self.cond_to_model(cond)  # [B,D]

        g = self.ln_g(g_board + cond_m)  # [B,D]

        # Candidate features.
        a = cand_actions.long().clamp(min=0)
        o, row, col = _decode_actions(a)
        o = o.clamp(0, ORIENTS - 1)
        row = row.clamp(0, GRID_H - 1)
        col = col.clamp(0, GRID_W - 1)

        row2 = row + self._dr[o]
        col2 = col + self._dc[o]
        valid = cand_mask.bool()
        row2 = torch.where(valid, row2, row)
        col2 = torch.where(valid, col2, col)

        pos = self.row_embed(row) + self.col_embed(col) + self.orient_embed(o)  # [B,K,pos_dim]

        denom = float(max(1e-6, self.cost_norm_denom))
        cost_norm = (cand_cost.to(obs.dtype).clamp(min=0.0) / denom).clamp(0.0, 4.0)
        cost_e = self.cost_mlp(cost_norm.unsqueeze(-1))  # [B,K,cost_dim]

        # Local context patches (cheap, high signal):
        # gather raw bottle bitplanes (colors + virus) around each landing cell.
        plane_ch = int(min(self.patch_planes, board.shape[1]))
        planes = board[:, :plane_ch, :, :]
        if int(self.patch_kernel) == 1:
            patch1 = self._gather_patch_planes(planes, row, col)
            patch2 = self._gather_patch_planes(planes, row2, col2)
        else:
            r = int(self._patch_r)
            Wp = int(self._patch_wp)
            planes_p = F.pad(planes, (r, r, r, r), mode="constant", value=0.0)
            flat = planes_p.reshape(B, plane_ch, (GRID_H + 2 * r) * Wp)
            patch1 = self._gather_patch_planes_from_flat(flat, row, col)
            patch2 = self._gather_patch_planes_from_flat(flat, row2, col2)
        patch = torch.cat([patch1, patch2], dim=-1)

        if self.use_trunk_gather:
            if trunk.ndim == 4:
                local1 = self._gather_trunk_map(fmap, row, col)
                local2 = self._gather_trunk_map(fmap, row2, col2)
            else:
                local1 = self._gather_trunk_tokens(tokens, col)
                local2 = self._gather_trunk_tokens(tokens, col2)
            cand_in = torch.cat([pos, cost_e, patch, local1, local2], dim=-1)
        else:
            cand_in = torch.cat([pos, cost_e, patch], dim=-1)
        e = self.cand_mlp(cand_in) + cond_m.unsqueeze(1)  # [B,K,D]

        logits = (e * g.unsqueeze(1)).sum(dim=-1) / float(np.sqrt(max(1, self.d_model)))
        logits = logits.masked_fill(~cand_mask.bool(), -1e9)

        value = self.value_head(g)
        return logits, value


__all__ = ["CandidatePlacementPolicyNet"]
