"""NES-accurate falling-pill reachability (frame-level, counter-aware).

This module is the *reference* implementation for planning capsule movement
using the same per-frame update order as the retail NTSC Dr. Mario ROM:

    action_pillFalling:
        fallingPill_checkYMove   (gravity / soft drop)
        fallingPill_checkXMove   (DAS horizontal movement)
        fallingPill_checkRotate  (A/B rotation + kick-left quirks)

We model only the falling-pill sub-system (no clearing, no garbage, etc.). It is
used to build spawn-latched macro actions: one decision per pill spawn.

Coordinate system
-----------------
The NES stores the falling pill position as the *bottom-left cell* of the pill's
2×2 bounding box:

  - `fallingPillX` is the leftmost column of the capsule.
  - `fallingPillY` is the bottom cell's row, counted from the bottom (0=bottom).

Internally, we convert Y to a top-origin row index (0=top, 15=bottom):

    y_top = (GRID_H - 1) - y_from_bottom

Rotation codes (size=2 pill) are exactly the NES values (0..3). Geometry depends
only on parity (`rot & 1`), while color order depends on the full value.
See `halfPill_posOffset_rotationAndSizeBased` in `drmario_data_game.asm`.

Important behavioural details mirrored here
-------------------------------------------
* Gravity triggers when `speedCounter > speedCounterTable[idx]` (cmp + bcs exit).
* Soft drop ("down") is only checked every other frame (frameCounter & 1 != 0),
  and only when DOWN is the *only* d-pad direction held.
* Lock is *immediate* when a drop attempt is blocked: the ROM calls
  `confirmPlacement` inside `fallingPill_checkYMove`.
* DAS timing:
    - first repeat after 16 frames held (`hor_accel_speed = 0x10`)
    - repeats every 6 frames thereafter (`hor_max_speed = 0x06`)
    - blocked lateral movement sets velocity to 0x0F (move ASAP when free)
* Rotation quirks (`pillRotateValidation`):
    - collision checks depend on `rot & 1` (horizontal vs vertical)
    - when rotating *to horizontal* and LEFT is held, an extra "double-left"
      validation is attempted
    - when rotating to horizontal and blocked, a kick-left is attempted
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from envs.retro.placement_space import GRID_HEIGHT, GRID_WIDTH

# ---------------------------------------------------------------------------
# NES constants and lookup tables (NTSC retail build)
# ---------------------------------------------------------------------------


FAST_DROP_MASK = 0x01  # frameCounter & 1 gates down-only soft drop

HOR_ACCEL_SPEED = 0x10  # 16 frames until first repeat
HOR_MAX_SPEED = 0x06    # 6 frames between repeats
HOR_RELOAD = HOR_ACCEL_SPEED - HOR_MAX_SPEED  # 0x0A
HOR_BLOCKED = HOR_ACCEL_SPEED - 1             # 0x0F

# baseSpeedSettingValue and speedCounterTable are pulled from
# `dr-mario-disassembly/data/drmario_data_game.asm`.
BASE_SPEED_SETTING_VALUE: Tuple[int, ...] = (0x0F, 0x19, 0x1F)

SPEED_COUNTER_TABLE: Tuple[int, ...] = (
    0x45, 0x43, 0x41, 0x3F, 0x3D, 0x3B, 0x39, 0x37,
    0x35, 0x33, 0x31, 0x2F, 0x2D, 0x2B, 0x29, 0x27,
    0x25, 0x23, 0x21, 0x1F, 0x1D, 0x1B, 0x19, 0x17,
    0x15, 0x13, 0x12, 0x11, 0x10, 0x0F, 0x0E, 0x0D,
    0x0C, 0x0B, 0x0A, 0x09, 0x09, 0x08, 0x08, 0x07,
    0x07, 0x06, 0x06, 0x05, 0x05, 0x05, 0x05, 0x05,
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x04,
    0x04, 0x04, 0x04, 0x03, 0x03, 0x03, 0x03, 0x03,
    0x02, 0x02, 0x02, 0x02, 0x02, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00,
)

SPEED_COUNTER_MAX = len(SPEED_COUNTER_TABLE) - 1


def compute_speed_threshold(speed_setting: int, speed_ups: int) -> int:
    """Return the ROM table value used in `fallingPill_checkYMove` (gravity)."""

    setting_idx = max(0, min(int(speed_setting), len(BASE_SPEED_SETTING_VALUE) - 1))
    base_index = int(BASE_SPEED_SETTING_VALUE[setting_idx])
    raw_index = base_index + max(0, int(speed_ups))
    table_index = max(0, min(raw_index, SPEED_COUNTER_MAX))
    return int(SPEED_COUNTER_TABLE[table_index])


# ---------------------------------------------------------------------------
# Action/state definitions
# ---------------------------------------------------------------------------


class HoldDir(Enum):
    NEUTRAL = 0
    LEFT = 1
    RIGHT = 2


class Rotation(Enum):
    NONE = 0
    CW = 1   # NES "A" button: rotation index decrements
    CCW = 2  # NES "B" button: rotation index increments


@dataclass(frozen=True)
class FrameAction:
    """Buttons held/pressed for one frame of planning."""

    hold_dir: HoldDir
    hold_down: bool
    rotation: Rotation


@dataclass(frozen=True)
class FrameState:
    """Falling pill state at the start of a frame (planner coordinates)."""

    x: int  # base column (bottom-left of 2×2 bounding box)
    y: int  # base row, 0=top, 15=bottom (bottom cell of vertical pills)
    rot: int  # NES rotation code (0..3)

    speed_counter: int  # currentP_speedCounter ($0092)
    hor_velocity: int  # currentP_horVelocity ($0093)

    hold_dir: HoldDir  # held L/R from previous frame (for edge detection)
    frame_parity: int  # frameCounter & 1 at start of this frame
    locked: bool = False


@dataclass(frozen=True)
class FrameNode:
    state: FrameState
    parent: int  # node index; -1 for root
    action_index: int  # index into _ACTION_SPACE; -1 for root
    depth: int  # frames elapsed from spawn (root depth = 0)


@dataclass(frozen=True)
class ReachabilityConfig:
    max_frames: int = 2048


@dataclass
class ReachabilityResult:
    nodes: List[FrameNode]
    terminal_nodes: Dict[Tuple[int, int, int], int]  # (x,y,rot) -> node index (locked state)

    def best_terminal(self, x: int, y: int, rot: int) -> Optional[int]:
        return self.terminal_nodes.get((int(x), int(y), int(rot) & 3))


# Enumerate per-frame action space in a stable order (important for tie-breaking).
_ACTION_SPACE: Tuple[FrameAction, ...] = tuple(
    FrameAction(hold_dir=hold_dir, hold_down=hold_down, rotation=rotation)
    for hold_dir in (HoldDir.NEUTRAL, HoldDir.LEFT, HoldDir.RIGHT)
    for hold_down in (False, True)
    for rotation in (Rotation.NONE, Rotation.CW, Rotation.CCW)
)


def frame_action_from_index(index: int) -> FrameAction:
    if index < 0 or index >= len(_ACTION_SPACE):
        raise IndexError(f"Frame action index {index} is out of range")
    return _ACTION_SPACE[int(index)]


# ---------------------------------------------------------------------------
# Geometry / collision helpers (bitboard columns)
# ---------------------------------------------------------------------------


def _cell_occupied(cols_u16: np.ndarray, x: int, y: int) -> bool:
    if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
        return True
    return bool(int(cols_u16[x]) & (1 << int(y)))


def _fits(cols_u16: np.ndarray, x: int, y: int, rot: int) -> bool:
    """Return True iff the capsule fits at base cell (x,y) with NES rotation code rot."""

    x_i = int(x)
    y_i = int(y)
    rot_i = int(rot) & 3

    # Base cell must be in bounds.
    if x_i < 0 or x_i >= GRID_WIDTH or y_i < 0 or y_i >= GRID_HEIGHT:
        return False

    if _cell_occupied(cols_u16, x_i, y_i):
        return False

    if (rot_i & 1) == 0:
        # Horizontal geometry: occupies base and base+(0,+1)
        if x_i + 1 >= GRID_WIDTH:
            return False
        if _cell_occupied(cols_u16, x_i + 1, y_i):
            return False
    else:
        # Vertical geometry: occupies base and base+(-1,0) (above). Allow y=-1 as offscreen.
        if y_i - 1 >= 0 and _cell_occupied(cols_u16, x_i, y_i - 1):
            return False

    return True


# ---------------------------------------------------------------------------
# Rotation validation (pillRotateValidation)
# ---------------------------------------------------------------------------


def _apply_rotation(
    cols_u16: np.ndarray,
    *,
    x: int,
    y: int,
    rot: int,
    rotation: Rotation,
    hold_left: bool,
) -> Tuple[int, int]:
    """Apply rotation and return (new_x, new_rot).

    Mirrors `fallingPill_checkRotate` + `pillRotateValidation`:
      - A press: rot = (rot - 1) & 3 (CW)
      - B press: rot = (rot + 1) & 3 (CCW)
      - If rotating to horizontal and LEFT is held, attempt an extra x-1 shift.
      - If blocked rotating to horizontal, attempt a kick-left (x-1).
    """

    if rotation is Rotation.NONE:
        return int(x), int(rot) & 3

    x0 = int(x)
    rot0 = int(rot) & 3

    if rotation is Rotation.CW:
        rot1 = (rot0 - 1) & 3
    else:
        rot1 = (rot0 + 1) & 3

    if (rot1 & 1) == 0:
        # Target is horizontal.
        if _fits(cols_u16, x0, int(y), rot1):
            # Rotation accepted in-place.
            if hold_left and _fits(cols_u16, x0 - 1, int(y), rot1):
                return x0 - 1, rot1
            return x0, rot1
        # Kick-left attempt.
        if _fits(cols_u16, x0 - 1, int(y), rot1):
            return x0 - 1, rot1
        # Reject.
        return x0, rot0

    # Target is vertical: only in-place validation.
    if _fits(cols_u16, x0, int(y), rot1):
        return x0, rot1
    return x0, rot0


# ---------------------------------------------------------------------------
# Frame stepping (Y -> X -> Rotate)
# ---------------------------------------------------------------------------


def _dir_flags(direction: HoldDir) -> Tuple[bool, bool]:
    if direction is HoldDir.LEFT:
        return True, False
    if direction is HoldDir.RIGHT:
        return False, True
    return False, False


def _pack_state(state: FrameState) -> int:
    """Pack a FrameState into a small int key for visited pruning."""

    # Layout (low bits first):
    #   x:3, y:4, rot:2, speed_counter:7, hor_velocity:4, hold_dir:2, parity:1
    x = int(state.x) & 0x7
    y = int(state.y) & 0xF
    rot = int(state.rot) & 0x3
    sc = int(state.speed_counter) & 0x7F
    hv = int(state.hor_velocity) & 0xF
    hd = int(state.hold_dir.value) & 0x3
    p = int(state.frame_parity) & 0x1
    return x | (y << 3) | (rot << 7) | (sc << 9) | (hv << 16) | (hd << 20) | (p << 22)


def _step_state(
    cols_u16: np.ndarray,
    state: FrameState,
    action: FrameAction,
    *,
    speed_threshold: int,
) -> FrameState:
    """Simulate one full frame (Y then X then Rotate)."""

    if state.locked:
        return state

    prev_left, prev_right = _dir_flags(state.hold_dir)
    hold_left, hold_right = _dir_flags(action.hold_dir)
    hold_down = bool(action.hold_down)

    press_left = hold_left and not prev_left
    press_right = hold_right and not prev_right
    press_lr = press_left or press_right

    x = int(state.x)
    y = int(state.y)
    rot = int(state.rot) & 3
    speed_counter = int(state.speed_counter) & 0xFF
    hor_velocity = int(state.hor_velocity) & 0xFF
    parity = int(state.frame_parity) & FAST_DROP_MASK

    # ---------------- Y stage (gravity / soft drop) ----------------
    down_only = hold_down and (action.hold_dir is HoldDir.NEUTRAL)
    drop_triggered = False

    if (parity != 0) and down_only:
        # Down-only soft drop (checked every other frame).
        drop_triggered = True
        speed_counter = 0
    else:
        # Gravity: speedCounter++ and drop when speedCounter > tableValue.
        speed_counter = min(speed_counter + 1, 0xFF)
        if speed_counter > int(speed_threshold):
            drop_triggered = True
            speed_counter = 0

    if drop_triggered:
        if _fits(cols_u16, x, y + 1, rot):
            y += 1
        else:
            # Immediate lock: confirmPlacement happens inside checkYMove.
            next_parity = (parity + 1) & FAST_DROP_MASK
            return FrameState(
                x=x,
                y=y,
                rot=rot,
                speed_counter=0,
                hor_velocity=hor_velocity,
                hold_dir=action.hold_dir,
                frame_parity=next_parity,
                locked=True,
            )

    # ---------------- X stage (DAS movement) ----------------
    allow_move = False
    if press_lr:
        hor_velocity = 0
        allow_move = True
    else:
        if action.hold_dir is not HoldDir.NEUTRAL:
            hor_velocity = min(hor_velocity + 1, 0xFF)
            if hor_velocity >= HOR_ACCEL_SPEED:
                hor_velocity = HOR_RELOAD
                allow_move = True

    if allow_move:
        # ROM order: right check then left check.
        if hold_right:
            if _fits(cols_u16, x + 1, y, rot):
                x += 1
            else:
                hor_velocity = HOR_BLOCKED
        if hold_left:
            if _fits(cols_u16, x - 1, y, rot):
                x -= 1
            else:
                hor_velocity = HOR_BLOCKED

    # ---------------- Rotate stage ----------------
    x, rot = _apply_rotation(
        cols_u16,
        x=x,
        y=y,
        rot=rot,
        rotation=action.rotation,
        hold_left=hold_left,
    )

    next_parity = (parity + 1) & FAST_DROP_MASK
    return FrameState(
        x=x,
        y=y,
        rot=rot,
        speed_counter=speed_counter,
        hor_velocity=hor_velocity,
        hold_dir=action.hold_dir,
        frame_parity=next_parity,
        locked=False,
    )


def simulate_frame(
    cols_u16: np.ndarray,
    state: FrameState,
    action_index: int,
    *,
    speed_threshold: int,
) -> FrameState:
    """Advance ``state`` by one frame using the indexed action."""

    return _step_state(cols_u16, state, frame_action_from_index(action_index), speed_threshold=speed_threshold)


# ---------------------------------------------------------------------------
# BFS driver
# ---------------------------------------------------------------------------


def build_reachability(
    cols_u16: np.ndarray,
    spawn_state: FrameState,
    *,
    speed_threshold: int,
    config: Optional[ReachabilityConfig] = None,
) -> ReachabilityResult:
    """Enumerate all locked (x,y,rot) placements reachable from the spawn state."""

    cfg = config or ReachabilityConfig()

    if spawn_state.locked:
        raise ValueError("Spawn state cannot be locked")
    if not _fits(cols_u16, spawn_state.x, spawn_state.y, spawn_state.rot):
        # Unspawnable: no reachable placements.
        nodes = [FrameNode(state=spawn_state, parent=-1, action_index=-1, depth=0)]
        return ReachabilityResult(nodes=nodes, terminal_nodes={})

    nodes: List[FrameNode] = [FrameNode(state=spawn_state, parent=-1, action_index=-1, depth=0)]
    q: deque[int] = deque([0])
    visited: Dict[int, int] = {_pack_state(spawn_state): 0}
    terminal_nodes: Dict[Tuple[int, int, int], int] = {}

    while q:
        idx = q.popleft()
        node = nodes[idx]
        if node.depth >= int(cfg.max_frames):
            continue
        if node.state.locked:
            continue

        for action_index in range(len(_ACTION_SPACE)):
            next_state = _step_state(cols_u16, node.state, _ACTION_SPACE[action_index], speed_threshold=speed_threshold)
            child_depth = node.depth + 1
            child = FrameNode(
                state=next_state,
                parent=idx,
                action_index=action_index,
                depth=child_depth,
            )
            child_idx = len(nodes)

            if next_state.locked:
                key = (int(next_state.x), int(next_state.y), int(next_state.rot) & 3)
                # BFS order guarantees first hit is minimal depth; keep first.
                if key not in terminal_nodes:
                    nodes.append(child)
                    terminal_nodes[key] = child_idx
                continue

            packed = _pack_state(next_state)
            if packed in visited:
                continue
            visited[packed] = child_idx
            nodes.append(child)
            q.append(child_idx)

    return ReachabilityResult(nodes=nodes, terminal_nodes=terminal_nodes)


# ---------------------------------------------------------------------------
# Path reconstruction helpers
# ---------------------------------------------------------------------------


def reconstruct_actions(result: ReachabilityResult, node_index: int) -> List[int]:
    """Return action indices (root→node) for `node_index` in the reachability tree."""

    actions: List[int] = []
    idx = int(node_index)
    while idx >= 0:
        node = result.nodes[idx]
        if node.parent < 0:
            break
        actions.append(int(node.action_index))
        idx = int(node.parent)
    actions.reverse()
    return actions


def iter_frame_states(result: ReachabilityResult, node_index: int) -> Iterator[FrameState]:
    """Yield FrameState sequence from root to `node_index` inclusive."""

    seq: List[FrameState] = []
    idx = int(node_index)
    while idx >= 0:
        node = result.nodes[idx]
        seq.append(node.state)
        idx = int(node.parent)
    for st in reversed(seq):
        yield st


__all__ = [
    "FrameAction",
    "FrameNode",
    "FrameState",
    "HoldDir",
    "ReachabilityConfig",
    "ReachabilityResult",
    "Rotation",
    "build_reachability",
    "compute_speed_threshold",
    "simulate_frame",
    "iter_frame_states",
    "reconstruct_actions",
    "frame_action_from_index",
]

