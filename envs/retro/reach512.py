"""Ultra-fast 512-state reachability and cost via bitboard BFS.

This module computes earliest arrival times to every (x,y,o) state for a single
falling pill using a compact BFS over 512 states. It ignores button timing
intricacies and instead treats per-frame transitions as unit-cost moves:

- Down by 1 if the shape fits
- Left/Right by 1 if the shape fits
- Rotate CW/CCW if the shape fits (with a single-step left kick)

The result is suitable for spawn-time feasibility and cost estimation in
microseconds. For final controller synthesis of a selected action, use the
counter-aware single-target planner to produce an exact sequence.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from envs.retro.placement_actions import GRID_HEIGHT, GRID_WIDTH, PLACEMENT_EDGES

O_HPOS = 0
O_VPOS = 1
O_HNEG = 2
O_VNEG = 3


def drm_idx(x: int, y: int, o: int) -> int:
    return ((int(o) & 3) << 7) | (int(y) << 3) | int(x)


OFFSETS = (
    ((0, 0), (0, 1)),  # HPOS
    ((0, 0), (1, 0)),  # VPOS
    ((0, 0), (0, -1)),  # HNEG
    ((0, 0), (-1, 0)),  # VNEG
)


def _fits(cols_u16: np.ndarray, x: int, y: int, o: int) -> bool:
    if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
        return False
    for dr, dc in OFFSETS[o & 3]:
        xx = x + dc
        yy = y + dr
        if xx < 0 or xx >= GRID_WIDTH or yy < 0 or yy >= GRID_HEIGHT:
            return False
        if int(cols_u16[xx]) & (1 << yy):
            return False
    return True


def _rotate_with_kick(cols_u16: np.ndarray, x: int, y: int, o: int, cw: bool) -> Tuple[int, int, int] | None:
    o2 = (o - 1) & 3 if cw else (o + 1) & 3
    if _fits(cols_u16, x, y, o2):
        return x, y, o2
    # Simple left kick for horizontal target
    if (o2 & 1) == 0:
        if _fits(cols_u16, x - 1, y, o2):
            return x - 1, y, o2
    return None


@dataclass
class Reach512:
    arrival: np.ndarray  # uint8 shape (512,), 0xFF unreachable
    parent: np.ndarray   # int16 shape (512,), -1 for root/unset
    code: np.ndarray     # uint8 shape (512,), 0:root, 1:DOWN,2:LEFT,3:RIGHT,4:CW,5:CCW


def run_reachability(cols_u16: np.ndarray, spawn_x: int, spawn_y: int, spawn_o: int) -> Reach512:
    arrival = np.full(512, 0xFF, dtype=np.uint8)
    parent = np.full(512, -1, dtype=np.int16)
    code = np.zeros(512, dtype=np.uint8)
    if not _fits(cols_u16, spawn_x, spawn_y, spawn_o & 3):
        return Reach512(arrival=arrival, parent=parent, code=code)
    q: deque[Tuple[int, int, int]] = deque()
    start = (int(spawn_x), int(spawn_y), int(spawn_o) & 3)
    sidx = drm_idx(*start)
    arrival[sidx] = 0
    q.append(start)
    while q:
        x, y, o = q.popleft()
        idx0 = drm_idx(x, y, o)
        t = int(arrival[idx0])
        t2 = np.uint8(min(t + 1, 255))
        # Down
        if y + 1 < GRID_HEIGHT and _fits(cols_u16, x, y + 1, o):
            idx = drm_idx(x, y + 1, o)
            if arrival[idx] == 0xFF:
                arrival[idx] = t2
                parent[idx] = idx0
                code[idx] = 1
                q.append((x, y + 1, o))
        # Left
        if x - 1 >= 0 and _fits(cols_u16, x - 1, y, o):
            idx = drm_idx(x - 1, y, o)
            if arrival[idx] == 0xFF:
                arrival[idx] = t2
                parent[idx] = idx0
                code[idx] = 2
                q.append((x - 1, y, o))
        # Right
        if x + 1 < GRID_WIDTH and _fits(cols_u16, x + 1, y, o):
            idx = drm_idx(x + 1, y, o)
            if arrival[idx] == 0xFF:
                arrival[idx] = t2
                parent[idx] = idx0
                code[idx] = 3
                q.append((x + 1, y, o))
        # Rotate CW
        rot = _rotate_with_kick(cols_u16, x, y, o, cw=True)
        if rot is not None:
            xx, yy, oo = rot
            idx = drm_idx(xx, yy, oo)
            if arrival[idx] == 0xFF:
                arrival[idx] = t2
                parent[idx] = idx0
                code[idx] = 4
                q.append((xx, yy, oo))
        # Rotate CCW
        rot = _rotate_with_kick(cols_u16, x, y, o, cw=False)
        if rot is not None:
            xx, yy, oo = rot
            idx = drm_idx(xx, yy, oo)
            if arrival[idx] == 0xFF:
                arrival[idx] = t2
                parent[idx] = idx0
                code[idx] = 5
                q.append((xx, yy, oo))
    return Reach512(arrival=arrival, parent=parent, code=code)


def feasibility_and_costs(
    cols_u16: np.ndarray, spawn_x: int, spawn_y: int, spawn_o: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (legal_mask, feasible_mask, costs) for all directed edges.

    - legal_mask: landing geometry (fits at origin, not at origin+1)
    - feasible_mask: geometry AND reachable from spawn (arrival != 0xFF)
    - costs: earliest arrival time in frames (float32); inf if infeasible
    """
    reach = run_reachability(cols_u16, spawn_x, spawn_y, spawn_o)
    arrival = reach.arrival
    legal = np.zeros(len(PLACEMENT_EDGES), dtype=np.bool_)
    feasible = np.zeros_like(legal)
    costs = np.full(len(PLACEMENT_EDGES), np.inf, dtype=np.float32)
    for edge in PLACEMENT_EDGES:
        r0, c0 = edge.origin
        r1, c1 = edge.dest
        dr = r1 - r0
        dc = c1 - c0
        if dr == 0 and dc == 1:
            o = O_HPOS
        elif dr == 0 and dc == -1:
            o = O_HNEG
        elif dr == 1 and dc == 0:
            o = O_VPOS
        elif dr == -1 and dc == 0:
            o = O_VNEG
        else:
            continue
        if _fits(cols_u16, c0, r0, o) and not _fits(cols_u16, c0, r0 + 1, o):
            legal[edge.index] = True
            t = int(arrival[drm_idx(c0, r0, o)])
            if t != 0xFF:
                feasible[edge.index] = True
                costs[edge.index] = float(t)
    return legal, feasible, costs


def reconstruct_path_to(reach: Reach512, target_x: int, target_y: int, target_o: int) -> list[tuple[int, int, int]]:
    idx = drm_idx(int(target_x), int(target_y), int(target_o) & 3)
    if int(reach.arrival[idx]) == 0xFF:
        return []
    path = []
    while idx >= 0 and int(reach.arrival[idx]) != 0xFF:
        o = (idx >> 7) & 3
        y = (idx >> 3) & 0x0F
        x = idx & 0x07
        path.append((x, y, o))
        p = int(reach.parent[idx])
        if p < 0:
            break
        idx = p
    path.reverse()
    return path


def reconstruct_actions_to(
    reach: Reach512, target_x: int, target_y: int, target_o: int
) -> list[tuple[int, int, int, int]]:
    """Return forward-ordered list of (x,y,o,code) from spawn to target.

    code values:
      0=root, 1=DOWN, 2=LEFT, 3=RIGHT, 4=CW, 5=CCW
    """
    idx = drm_idx(int(target_x), int(target_y), int(target_o) & 3)
    if int(reach.arrival[idx]) == 0xFF:
        return []
    seq = []
    while idx >= 0 and int(reach.arrival[idx]) != 0xFF:
        o = (idx >> 7) & 3
        y = (idx >> 3) & 0x0F
        x = idx & 0x07
        seq.append((x, y, o, int(reach.code[idx])))
        p = int(reach.parent[idx])
        if p < 0:
            break
        idx = p
    seq.reverse()
    return seq


__all__ = [
    "Reach512",
    "drm_idx",
    "run_reachability",
    "feasibility_and_costs",
    "reconstruct_actions_to",
    "reconstruct_path_to",
    "O_HPOS",
    "O_VPOS",
    "O_HNEG",
    "O_VNEG",
]
