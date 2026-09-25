"""Full-size public core with an additive exact-afterstate branch (arm C).

The unchanged G5 public-context core (320x8 FiLM trunk on both bottles,
column interaction, candidate attention, 51-atom value head, 9x9 pre-lock
patches) gains one term in each candidate token: a small CNN over that
candidate's exact settled own bottle (``drmc_rl.game.afterstate``), the
changed-cell mask and a projection of the root trunk features, fused with the
16 exact consequence facts. The branch ends in a zero-initialized projection
that is added to the candidate embedding before candidate attention, so a
model built from a G5 checkpoint computes exactly that checkpoint's logits and
values until training moves the projection.

The call signature matches ``G5CandidatePlacementPolicyNet``. When
``afterstate`` is omitted, afterstates are computed from the public model
inputs on the host (shared row cache with the afterstate core); ONNX export
passes ``after_tiles [B,K,128]`` and ``facts [B,K,16]`` via
``forward_features``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from drmc_rl.game.afterstate import FACT_DIM
from drmc_rl.models.policy.afterstate_core import (
    AfterstateCorePolicyNet,
    _AfterResidual,
    _tile_plane_table,
)
from drmc_rl.models.policy.candidate_policy import GRID_H, GRID_W
from drmc_rl.models.policy.candidate_policy_g5 import G5CandidatePlacementPolicyNet

AFTERSTATE_FULL_SCHEMA = "drmc-afterstate-full-core-v1"
BRANCH_PREFIX = "afterstate."


class _AfterstateBranch(nn.Module):
    def __init__(self, trunk_channels: int, d_model: int, channels: int, blocks: int,
                 root_projection: int, fact_dim: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.root_projection = nn.Conv2d(trunk_channels, int(root_projection), 1)
        self.stem = nn.Conv2d(8 + 1 + 2 + int(root_projection), self.channels, 3, padding=1)
        self.blocks = nn.ModuleList([_AfterResidual(self.channels) for _ in range(int(blocks))])
        self.board_out = nn.Sequential(
            nn.LayerNorm(4 * self.channels), nn.Linear(4 * self.channels, d_model), nn.SiLU()
        )
        self.facts = nn.Sequential(nn.Linear(int(fact_dim), 64), nn.SiLU())
        self.fuse = nn.Sequential(nn.Linear(d_model + 64, d_model), nn.SiLU())
        self.out = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("tile_planes", _tile_plane_table(), persistent=False)
        rows = torch.linspace(0.0, 1.0, GRID_H).view(1, 1, GRID_H, 1).expand(1, 1, GRID_H, GRID_W)
        cols = torch.linspace(0.0, 1.0, GRID_W).view(1, 1, 1, GRID_W).expand(1, 1, GRID_H, GRID_W)
        self.register_buffer("coords", torch.cat((rows, cols), 1), persistent=False)

    def forward(self, own, root_planes, after_tiles, facts, valid):
        batch, width = valid.shape
        dtype = own.dtype
        root_tiles = AfterstateCorePolicyNet._root_tiles(None, root_planes)
        planes = self.tile_planes.to(dtype)[after_tiles.long()]  # [B,K,128,8]
        planes = planes.permute(0, 1, 3, 2).reshape(batch, width, 8, GRID_H, GRID_W)
        changed = (after_tiles != root_tiles.unsqueeze(1)).to(dtype).reshape(batch, width, 1, GRID_H, GRID_W)
        root = self.root_projection(own).unsqueeze(1).expand(-1, width, -1, -1, -1)
        coords = self.coords.to(dtype).expand(batch, width, -1, -1, -1)
        x = torch.cat((planes, changed, coords, root), dim=2).flatten(0, 1)
        packed = not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export()
        if packed:
            index = valid.reshape(-1).nonzero(as_tuple=True)[0]
            x = x.index_select(0, index)
        y = self.stem(x)
        for block in self.blocks:
            y = block(y)
        change = x[:, 8:9]
        n_changed = change.sum(dim=(2, 3)).clamp_min(1.0)
        pooled = torch.cat(
            (
                y.mean(dim=(2, 3)),
                y.amax(dim=(2, 3)),
                (y * change).sum(dim=(2, 3)) / n_changed,
                (y * change - (1.0 - change) * 1e4).amax(dim=(2, 3)).clamp_min(-1e3),
            ),
            dim=-1,
        )
        board = self.board_out(pooled)
        if packed:
            board = board.new_zeros((batch * width, board.shape[-1])).index_copy(0, index, board)
        board = board.reshape(batch, width, -1)
        fused = self.fuse(torch.cat((board, self.facts(facts.to(dtype))), dim=-1))
        return self.out(fused) * valid.unsqueeze(-1).to(dtype)


class AfterstateFullCorePolicyNet(G5CandidatePlacementPolicyNet):
    """G5 public core whose candidate tokens also see exact afterstates."""

    def __init__(self, *, after_channels: int = 32, after_blocks: int = 2, root_projection: int = 16,
                 fact_dim: int = FACT_DIM, **g5) -> None:
        super().__init__(**g5)
        if self.public_context_schema is None:
            raise ValueError("the full afterstate core is defined only on a public context schema")
        self.fact_dim = int(fact_dim)
        self.afterstate = _AfterstateBranch(self.d_model, self.d_model, after_channels, after_blocks,
                                            root_projection, fact_dim)

    exact_afterstates = staticmethod(AfterstateCorePolicyNet.exact_afterstates)

    def forward(self, obs, pill_colors, preview_pill_colors, cand_actions, cand_cost, cand_mask, *,
                aux: Optional[torch.Tensor] = None, return_aux: bool = False,
                afterstate: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                motor_geometry=None, prepared_bottles=None):
        if afterstate is None:
            with torch.no_grad():
                afterstate = self.exact_afterstates(obs, pill_colors, cand_actions, cand_mask)
        return super().forward(obs, pill_colors, preview_pill_colors, cand_actions, cand_cost, cand_mask,
                               aux=aux, return_aux=return_aux, motor_geometry=motor_geometry,
                               prepared_bottles=prepared_bottles, candidate_extra=afterstate)

    def forward_features(self, obs, pill_colors, preview_pill_colors, cand_actions, cand_cost, cand_mask,
                         aux, after_tiles, facts, *, return_aux: bool = False):
        return super().forward(obs, pill_colors, preview_pill_colors, cand_actions, cand_cost, cand_mask,
                               aux=aux, return_aux=return_aux, candidate_extra=(after_tiles, facts))

    def _candidate_residual(self, own, obs, valid, candidate_extra):
        after_tiles, facts = candidate_extra
        return self.afterstate(own, obs[:, :8], after_tiles, facts, valid)


def afterstate_full_config(champion_cfg: dict, **overrides) -> dict:
    """Checkpoint config: the champion's G5 config with the afterstate branch switched on."""

    from copy import deepcopy

    cfg = deepcopy(champion_cfg)
    sp = cfg.setdefault("smdp_ppo", {})
    if sp.get("candidate_architecture") != "g5":
        raise ValueError("arm C extends a G5 public core")
    sp.update(candidate_architecture="g5_afterstate", afterstate_schema=AFTERSTATE_FULL_SCHEMA,
              afterstate_channels=32, afterstate_blocks=2, afterstate_root_projection=16)
    sp.update(overrides)
    return cfg


def from_g5(g5: G5CandidatePlacementPolicyNet, sp: dict) -> AfterstateFullCorePolicyNet:
    """Build the full afterstate core and load every G5 tensor unchanged."""

    from tools.eval_policy import _build_net_from_cfg

    net, _, _ = _build_net_from_cfg({"smdp_ppo": sp}, g5.in_channels, "cpu")
    missing, unexpected = net.load_state_dict(g5.state_dict(), strict=False)
    if unexpected or any(not key.startswith(BRANCH_PREFIX) for key in missing):
        raise ValueError(f"G5 weights do not map onto the full afterstate core: {missing} {unexpected}")
    return net.eval()


__all__ = [
    "AFTERSTATE_FULL_SCHEMA",
    "AfterstateFullCorePolicyNet",
    "BRANCH_PREFIX",
    "afterstate_full_config",
    "from_g5",
]
