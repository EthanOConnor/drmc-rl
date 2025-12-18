"""Canonical 4×16×8 placement action space (512 actions).

This module defines the *macro-action* space used by the placement policy:

    action := (orientation o, row i, col j)

where (i, j) is the landing cell for the *first* capsule half (the "1st color"
in NES RAM), and orientation selects which *adjacent* cell contains the second
half after the pill locks.

Orientation convention (matches `models/policy/placement_heads.py`):

    o = 0  (H+): partner at (i,   j+1)
    o = 1  (V+): partner at (i+1, j)
    o = 2  (H-): partner at (i,   j-1)
    o = 3  (V-): partner at (i-1, j)

Notes
-----
The 512-cell grid includes boundary actions that point outside the 16×8 bottle.
Those actions are always invalid and should be masked out by the environment.
This is intentional: a dense [4,16,8] logit map is convenient for CNN policy
heads; legality is handled via masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np

GRID_HEIGHT: int = 16
GRID_WIDTH: int = 8
ORIENTATIONS: int = 4
TOTAL_ACTIONS: int = ORIENTATIONS * GRID_HEIGHT * GRID_WIDTH  # 512

# (dr, dc) partner offsets in (row, col) coordinates.
ORIENT_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (0, 1),   # H+
    (1, 0),   # V+
    (0, -1),  # H-
    (-1, 0),  # V-
)


def flatten(o: int, row: int, col: int) -> int:
    """Flatten (o,row,col) into [0, 512)."""

    return int(o) * (GRID_HEIGHT * GRID_WIDTH) + int(row) * GRID_WIDTH + int(col)


def unflatten(action: int) -> Tuple[int, int, int]:
    """Unflatten [0, 512) into (o,row,col)."""

    action_i = int(action)
    o = action_i // (GRID_HEIGHT * GRID_WIDTH)
    rem = action_i % (GRID_HEIGHT * GRID_WIDTH)
    row = rem // GRID_WIDTH
    col = rem % GRID_WIDTH
    return int(o), int(row), int(col)


def partner_cell(o: int, row: int, col: int) -> Tuple[int, int]:
    """Return (row2,col2) for the partner half of (o,row,col)."""

    dr, dc = ORIENT_OFFSETS[int(o) & 3]
    return int(row) + int(dr), int(col) + int(dc)


def cells_for_action(action: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return (origin, dest) cells for a flat action index."""

    o, row, col = unflatten(action)
    return (row, col), partner_cell(o, row, col)


def action_from_cells(origin: Tuple[int, int], dest: Tuple[int, int]) -> int:
    """Return the flat action index corresponding to an adjacent origin→dest."""

    r0, c0 = origin
    r1, c1 = dest
    dr = int(r1) - int(r0)
    dc = int(c1) - int(c0)
    try:
        o = ORIENT_OFFSETS.index((dr, dc))
    except ValueError as exc:  # not adjacent / unsupported direction
        raise ValueError(f"Cells {origin}->{dest} are not a directed adjacency") from exc
    return flatten(o, int(r0), int(c0))


def in_bounds(row: int, col: int) -> bool:
    return 0 <= int(row) < GRID_HEIGHT and 0 <= int(col) < GRID_WIDTH


def is_valid_action(o: int, row: int, col: int) -> bool:
    """Return True iff both capsule halves are inside the 16×8 bottle grid."""

    if not in_bounds(row, col):
        return False
    r2, c2 = partner_cell(o, row, col)
    return in_bounds(r2, c2)


def iter_all_actions() -> Iterator[int]:
    """Yield all 512 flat action indices."""

    return iter(range(TOTAL_ACTIONS))


def invalid_boundary_mask() -> np.ndarray:
    """Return a boolean [4,16,8] mask that is True only for in-bounds actions."""

    mask = np.zeros((ORIENTATIONS, GRID_HEIGHT, GRID_WIDTH), dtype=np.bool_)
    for action in iter_all_actions():
        o, r, c = unflatten(action)
        if is_valid_action(o, r, c):
            mask[o, r, c] = True
    return mask


@dataclass(frozen=True)
class PlacementCells:
    """Decoded placement cells for a macro action."""

    origin: Tuple[int, int]
    dest: Tuple[int, int]


__all__ = [
    "GRID_HEIGHT",
    "GRID_WIDTH",
    "ORIENTATIONS",
    "TOTAL_ACTIONS",
    "ORIENT_OFFSETS",
    "flatten",
    "unflatten",
    "partner_cell",
    "cells_for_action",
    "action_from_cells",
    "in_bounds",
    "is_valid_action",
    "iter_all_actions",
    "invalid_boundary_mask",
    "PlacementCells",
]

