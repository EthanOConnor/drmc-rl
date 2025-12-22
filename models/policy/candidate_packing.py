"""Helpers for packing feasible placement actions into fixed-size candidate lists.

The placement macro environment exposes:
  - `placements/feasible_mask` : bool [4,16,8]
  - `placements/cost_to_lock`  : uint16 [4,16,8] (cpp-pool) OR
    `placements/costs`         : float [4,16,8] (retro wrapper)

This module converts those into a fixed-size list suitable for candidate-scoring
policies:
  - actions[k] : int macro action index in [0,512) (padding = -1)
  - mask[k]    : bool valid-entry mask (padding = False)
  - cost[k]    : float32 cost-to-lock in frames (padding = 0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PackedCandidates:
    """Packed feasible candidates for a single decision point."""

    actions: np.ndarray  # (Kmax,) int32; macro action indices; padding = -1
    mask: np.ndarray  # (Kmax,) bool; True for valid candidates
    cost: np.ndarray  # (Kmax,) float32; cost-to-lock frames; padding = 0
    count: int  # number of valid candidates (<= Kmax)


def pack_feasible_candidates(
    feasible_mask: np.ndarray,
    cost_to_lock: np.ndarray,
    *,
    max_candidates: int,
    sort_by_cost: bool = True,
) -> PackedCandidates:
    """Return a padded list of feasible macro actions for one env state."""

    kmax = int(max(1, int(max_candidates)))

    mask = np.asarray(feasible_mask, dtype=np.bool_)
    cost = np.asarray(cost_to_lock)
    if mask.shape != (4, 16, 8):
        raise ValueError(f"Expected feasible_mask shape (4,16,8), got {mask.shape!r}")
    if cost.shape != (4, 16, 8):
        raise ValueError(f"Expected cost_to_lock shape (4,16,8), got {cost.shape!r}")

    if cost.dtype == np.uint16:
        cost_f = cost.astype(np.float32)
        cost_f[cost_f >= np.float32(0xFFFE)] = np.inf
    else:
        cost_f = cost.astype(np.float32, copy=False)
        if np.issubdtype(cost_f.dtype, np.floating) and np.isnan(cost_f).any():
            cost_f = cost_f.copy()
            cost_f[np.isnan(cost_f)] = np.inf

    flat_mask = mask.reshape(-1)
    flat_cost = cost_f.reshape(-1)
    idx = np.flatnonzero(flat_mask)

    actions_out = np.full((kmax,), -1, dtype=np.int32)
    mask_out = np.zeros((kmax,), dtype=np.bool_)
    cost_out = np.zeros((kmax,), dtype=np.float32)

    if idx.size == 0:
        return PackedCandidates(actions=actions_out, mask=mask_out, cost=cost_out, count=0)

    if bool(sort_by_cost):
        costs = flat_cost[idx]
        # Deterministic ordering: primary = cost, secondary = macro action id.
        # This avoids instability when multiple candidates share identical costs.
        order = np.lexsort((idx.astype(np.int64, copy=False), costs))
        idx = idx[order]

    if idx.size > kmax:
        idx = idx[:kmax]

    k = int(idx.size)
    actions_out[:k] = idx.astype(np.int32, copy=False)
    mask_out[:k] = True
    cost_out[:k] = flat_cost[idx].astype(np.float32, copy=False)

    return PackedCandidates(actions=actions_out, mask=mask_out, cost=cost_out, count=k)


__all__ = ["PackedCandidates", "pack_feasible_candidates"]
