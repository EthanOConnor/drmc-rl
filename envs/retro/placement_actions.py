"""Directed placement action catalogue for Dr. Mario capsules.

The placement action space follows the "half-0 anchor" convention described in
the design discussion.  Each action selects a directed adjacency ``u -> v``
within the 16x8 bottle grid where ``u`` is the landing cell for the first half
of the capsule (the RAM "left"/"bottom" half depending on orientation) and
``v`` is the neighbour that will contain the second half once the piece locks.

The total number of directed edges on the grid is constant: 464.  We enumerate
them once in a deterministic order so that the policy head can emit logits for
``[0, 464)`` and the planner/translator can share a consistent mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

from envs.specs.ram_to_state import STATE_HEIGHT, STATE_WIDTH

GRID_HEIGHT = STATE_HEIGHT
GRID_WIDTH = STATE_WIDTH

GridCoord = Tuple[int, int]


@dataclass(frozen=True)
class DirectedEdge:
    """Represents a directed adjacency ``u -> v`` inside the bottle grid."""

    index: int
    origin: GridCoord
    dest: GridCoord
    direction: str

    def as_tuple(self) -> Tuple[int, GridCoord, GridCoord, str]:
        return self.index, self.origin, self.dest, self.direction


def _enumerate_edges() -> List[DirectedEdge]:
    edges: List[DirectedEdge] = []
    idx = 0
    dir_labels = ("U", "R", "D", "L")
    deltas = ((-1, 0), (0, 1), (1, 0), (0, -1))
    for r in range(STATE_HEIGHT):
        for c in range(STATE_WIDTH):
            u = (r, c)
            for label, (dr, dc) in zip(dir_labels, deltas):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < STATE_HEIGHT and 0 <= cc < STATE_WIDTH:
                    edges.append(DirectedEdge(idx, u, (rr, cc), label))
                    idx += 1
    return edges


PLACEMENT_EDGES: Tuple[DirectedEdge, ...] = tuple(_enumerate_edges())

# Lookup table [origin_flat][dest_flat] -> action index (or -1 if not adjacent)
_edge_lookup = np.full((STATE_HEIGHT * STATE_WIDTH, STATE_HEIGHT * STATE_WIDTH), -1, dtype=np.int32)
_opposites: List[Tuple[int, int]] = []
_undirected_seen: Dict[Tuple[int, int], int] = {}
for edge in PLACEMENT_EDGES:
    u_flat = edge.origin[0] * STATE_WIDTH + edge.origin[1]
    v_flat = edge.dest[0] * STATE_WIDTH + edge.dest[1]
    _edge_lookup[u_flat, v_flat] = edge.index
    # Track undirected pairs by sorted cell tuple
    undirected_key = tuple(sorted((u_flat, v_flat)))
    other = _undirected_seen.get(undirected_key)
    if other is None:
        _undirected_seen[undirected_key] = edge.index
    else:
        _opposites.append((other, edge.index))


def iter_edges() -> Iterator[DirectedEdge]:
    """Yield the directed edges in canonical order."""

    return iter(PLACEMENT_EDGES)


def action_count() -> int:
    """Return the number of placement actions (constant 464)."""

    return len(PLACEMENT_EDGES)


def edge_from_index(index: int) -> DirectedEdge:
    """Return the directed edge metadata for ``index``."""

    if index < 0 or index >= len(PLACEMENT_EDGES):
        raise IndexError(f"Placement action index {index} is out of range")
    return PLACEMENT_EDGES[index]


def cells_for_action(index: int) -> Tuple[GridCoord, GridCoord, str]:
    """Return ``(origin, dest, direction)`` for ``index``."""

    edge = edge_from_index(index)
    return edge.origin, edge.dest, edge.direction


def action_from_cells(origin: GridCoord, dest: GridCoord) -> int:
    """Return the action index corresponding to ``origin -> dest``."""

    o_flat = origin[0] * STATE_WIDTH + origin[1]
    d_flat = dest[0] * STATE_WIDTH + dest[1]
    action = int(_edge_lookup[o_flat, d_flat])
    if action < 0:
        raise ValueError(f"Cells {origin}->{dest} are not adjacent")
    return action


def opposite_actions() -> Sequence[Tuple[int, int]]:
    """Return ordered pairs mapping undirected adjacencies to action indices."""

    return tuple(_opposites)


__all__ = [
    "DirectedEdge",
    "GridCoord",
    "PLACEMENT_EDGES",
    "GRID_HEIGHT",
    "GRID_WIDTH",
    "opposite_actions",
    "iter_edges",
    "action_count",
    "edge_from_index",
    "cells_for_action",
    "action_from_cells",
]
