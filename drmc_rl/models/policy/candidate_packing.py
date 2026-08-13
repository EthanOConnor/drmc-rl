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

import numpy as np
import torch


@dataclass(frozen=True)
class PackedCandidates:
    """Packed feasible candidates for a single decision point."""

    actions: np.ndarray  # (Kmax,) int32; macro action indices; padding = -1
    mask: np.ndarray  # (Kmax,) bool; True for valid candidates
    cost: np.ndarray  # (Kmax,) float32; cost-to-lock frames; padding = 0
    count: int  # number of valid candidates (<= Kmax)


@dataclass(frozen=True)
class PackedCandidateBatch:
    """Packed feasible candidates for a batch of decision points."""

    actions: np.ndarray  # (B,Kmax) int32; macro action indices; padding = -1
    mask: np.ndarray  # (B,Kmax) bool; True for valid candidates
    cost: np.ndarray  # (B,Kmax) float32; cost-to-lock frames; padding = 0
    count: np.ndarray  # (B,) int32; number of valid candidates per row


@dataclass(frozen=True)
class PackedCandidateTensorBatch:
    """Fixed-size candidates that remain on the input tensor's device."""

    actions: torch.Tensor  # (B,Kmax) int32; padding = -1
    mask: torch.Tensor  # (B,Kmax) bool
    cost: torch.Tensor  # (B,Kmax) float32; padding = 0
    count: torch.Tensor  # (B,) int32


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


def pack_feasible_candidates_batch(
    feasible_mask: np.ndarray,
    cost_to_lock: np.ndarray,
    *,
    max_candidates: int,
    sort_by_cost: bool = True,
) -> PackedCandidateBatch:
    """Pack a ``(B,4,16,8)`` batch without a Python loop over environments.

    Stable sorting preserves the single-row helper's deterministic tie break:
    candidates with equal costs remain in ascending macro-action order.
    """

    kmax = int(max(1, int(max_candidates)))
    mask = np.asarray(feasible_mask, dtype=np.bool_)
    cost = np.asarray(cost_to_lock)
    if mask.ndim != 4 or mask.shape[1:] != (4, 16, 8):
        raise ValueError(f"Expected feasible_mask shape (B,4,16,8), got {mask.shape!r}")
    if cost.shape != mask.shape:
        raise ValueError(f"Expected cost_to_lock shape {mask.shape!r}, got {cost.shape!r}")

    batch = int(mask.shape[0])
    flat_mask = mask.reshape(batch, -1)
    if cost.dtype == np.uint16:
        flat_cost = cost.reshape(batch, -1).astype(np.float32)
        flat_cost[flat_cost >= np.float32(0xFFFE)] = np.inf
    else:
        flat_cost = cost.reshape(batch, -1).astype(np.float32, copy=False)
        if np.isnan(flat_cost).any():
            flat_cost = flat_cost.copy()
            flat_cost[np.isnan(flat_cost)] = np.inf

    if sort_by_cost:
        keys = np.where(flat_mask, flat_cost, np.inf)
    else:
        # False sorts before true; stable order retains ascending action ids.
        keys = ~flat_mask
    order = np.argsort(keys, axis=1, kind="stable")[:, : min(kmax, flat_mask.shape[1])]
    valid = np.take_along_axis(flat_mask, order, axis=1)
    selected_cost = np.take_along_axis(flat_cost, order, axis=1)

    actions = np.full((batch, kmax), -1, dtype=np.int32)
    packed_mask = np.zeros((batch, kmax), dtype=np.bool_)
    packed_cost = np.zeros((batch, kmax), dtype=np.float32)
    width = int(order.shape[1])
    actions[:, :width] = np.where(valid, order, -1).astype(np.int32, copy=False)
    packed_mask[:, :width] = valid
    packed_cost[:, :width] = np.where(valid, selected_cost, 0.0).astype(
        np.float32, copy=False
    )
    return PackedCandidateBatch(
        actions=actions,
        mask=packed_mask,
        cost=packed_cost,
        count=packed_mask.sum(axis=1, dtype=np.int32),
    )


def pack_feasible_candidates_tensor_batch(
    feasible_mask: torch.Tensor,
    cost_to_lock: torch.Tensor,
    *,
    max_candidates: int,
    sort_by_cost: bool = True,
) -> PackedCandidateTensorBatch:
    """Pack candidates directly on CPU or CUDA with NumPy-equivalent ordering."""

    kmax = int(max(1, int(max_candidates)))
    if feasible_mask.ndim != 4 or tuple(feasible_mask.shape[1:]) != (4, 16, 8):
        raise ValueError(
            f"Expected feasible_mask shape (B,4,16,8), got {tuple(feasible_mask.shape)!r}"
        )
    if cost_to_lock.shape != feasible_mask.shape:
        raise ValueError(
            f"Expected cost_to_lock shape {tuple(feasible_mask.shape)!r}, "
            f"got {tuple(cost_to_lock.shape)!r}"
        )

    batch = int(feasible_mask.shape[0])
    flat_mask = feasible_mask.reshape(batch, -1).bool()
    flat_cost = cost_to_lock.reshape(batch, -1).float()
    flat_cost = torch.nan_to_num(flat_cost, nan=float("inf"))
    # Native pools use uint16 0xffff/0xfffe as unreachable sentinels. Casting
    # before this comparison also works around limited uint16 CUDA operators.
    flat_cost = flat_cost.masked_fill(flat_cost >= 65534.0, float("inf"))

    if sort_by_cost:
        keys = flat_cost.masked_fill(~flat_mask, float("inf"))
    else:
        keys = (~flat_mask).to(torch.int8)
    width = min(kmax, int(flat_mask.shape[1]))
    # Input columns are ascending action ids, so stable sorting gives the
    # NumPy reference's (cost, action-id) lexicographic order for tied costs.
    order = torch.argsort(keys, dim=1, stable=True)[:, :width]
    valid = flat_mask.gather(1, order)
    selected_cost = flat_cost.gather(1, order)

    actions = torch.full(
        (batch, kmax), -1, dtype=torch.int32, device=feasible_mask.device
    )
    packed_mask = torch.zeros(
        (batch, kmax), dtype=torch.bool, device=feasible_mask.device
    )
    packed_cost = torch.zeros(
        (batch, kmax), dtype=torch.float32, device=feasible_mask.device
    )
    actions[:, :width] = torch.where(valid, order, -1).to(torch.int32)
    packed_mask[:, :width] = valid
    packed_cost[:, :width] = torch.where(valid, selected_cost, 0.0)
    return PackedCandidateTensorBatch(
        actions=actions,
        mask=packed_mask,
        cost=packed_cost,
        count=packed_mask.sum(dim=1, dtype=torch.int32),
    )


__all__ = [
    "PackedCandidateBatch",
    "PackedCandidateTensorBatch",
    "PackedCandidates",
    "pack_feasible_candidates",
    "pack_feasible_candidates_batch",
    "pack_feasible_candidates_tensor_batch",
]
