"""NES-accurate falling pill reachability with frame-level counters.

This module implements a counter-aware breadth-first search that mirrors the
per-frame update order used by the retail NES Dr. Mario engine.  Each node in
the search corresponds to the full capsule state for a single frame:

    * anchor position (x, y) expressed in planner coordinates (y downward)
    * orientation (0=horizontal right, 1=vertical down, 2=horizontal left, 3=vertical up)
    * gravity counter (`currentP_speedCounter`)
    * horizontal velocity (`currentP_horVelocity`)
    * latched d-pad holds (left/right/down)
    * frame counter parity (`frameCounter & fast_drop_speed`)

Actions enumerate the button state for a single frame:

    HoldLeft / HoldRight / Neutral
      × HoldDown or not
      × RotateCW / RotateCCW / None

Transitions reproduce the in-game order:

    1. Update button registers (held + pressed bits)
    2. Y stage (`fallingPill_checkYMove`)
    3. X stage (`fallingPill_checkXMove`)
    4. Rotation (`fallingPill_checkRotate`)

The search terminates once a drop fails (piece locks).  The resulting plans
contain the exact per-frame controller sequence required for the emulator to
replay the move without drift.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from envs.retro.placement_actions import GRID_HEIGHT, GRID_WIDTH

# ---------------------------------------------------------------------------
# NES tables and constants (sourced from drmario_data_game.asm /
# drmario_constants.asm for the NTSC retail build).
# ---------------------------------------------------------------------------


FAST_DROP_MASK = 0x01  # frameCounter & FAST_DROP_MASK → down polling cadence

HOR_ACCEL_SPEED = 0x10  # frames until first lateral move when holding
HOR_MAX_SPEED = 0x06    # frames between repeats after the first move
HOR_RELOAD = HOR_ACCEL_SPEED - HOR_MAX_SPEED  # post-move reload (0x0A)
HOR_BLOCKED = HOR_ACCEL_SPEED - 1             # value loaded on block (0x0F)

# Lock buffer frames (approximate). If a drop is blocked this frame, allow X/Rot
# to adjust for a few frames before locking. Can be initialized from RAM $0307.
LOCK_BUFFER_FRAMES = 2

# baseSpeedSettingValue (low/medium/high) and speedCounterTable pulled directly
# from drmario_data_game.asm ($A3A7 / $A7AF).  These values control gravity.
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
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00,
)

SPEED_COUNTER_MAX = len(SPEED_COUNTER_TABLE) - 1

# ---------------------------------------------------------------------------
# Orientation helpers (mirrors envs.retro.placement_planner.ORIENT_OFFSETS)
# ---------------------------------------------------------------------------


class Orientation(IntEnum):
    H_POS = 0  # horizontal, partner at x+1
    V_POS = 1  # vertical, partner at y+1
    H_NEG = 2  # horizontal, partner at x-1
    V_NEG = 3  # vertical, partner at y-1


ORIENT_OFFSETS: Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...] = (
    ((0, 0), (0, 1)),   # H_POS
    ((0, 0), (1, 0)),   # V_POS
    ((0, 0), (0, -1)),  # H_NEG
    ((0, 0), (-1, 0)),  # V_NEG
)


# ---------------------------------------------------------------------------
# Action/state definitions
# ---------------------------------------------------------------------------


class HoldDir(Enum):
    NEUTRAL = 0
    LEFT = 1
    RIGHT = 2


class Rotation(Enum):
    NONE = 0
    CW = 1   # A button (clockwise in-game because rotation index decrements)
    CCW = 2  # B button (counter-clockwise)


@dataclass(frozen=True)
class FrameAction:
    """Button state for one frame."""

    hold_dir: HoldDir
    hold_down: bool
    rotation: Rotation


@dataclass
class FrameState:
    """Full capsule state after a frame step."""

    x: int
    y: int
    orient: int
    speed_counter: int
    hor_velocity: int
    hold_dir: HoldDir
    hold_down: bool
    frame_parity: int
    grounded: bool = False
    lock_timer: int = 0
    locked: bool = False


@dataclass
class FrameNode:
    """Search node storing the resulting state and backpointer metadata."""

    state: FrameState
    parent: int  # index into `nodes`; -1 for the root
    action_index: int  # index into ACTIONS applied to reach this node; -1 for root
    depth: int  # frames elapsed from spawn (root depth = 0)


@dataclass
class ReachabilityConfig:
    """Planner configuration parameters relevant to the fast reach search."""

    max_frames: int = GRID_HEIGHT * 6  # generous bound for pathological slides


@dataclass
class ReachabilityResult:
    """Result of the counter-aware BFS."""

    nodes: List[FrameNode]
    terminal_nodes: Dict[Tuple[int, int, int], int]
    stats: Dict[str, int]

    def best_node_for(self, anchor: Tuple[int, int, int]) -> Optional[int]:
        return self.terminal_nodes.get(anchor)


# Enumerate frame actions: HoldDir × HoldDown × Rotation.
_ACTION_SPACE: Tuple[FrameAction, ...] = tuple(
    FrameAction(hold_dir=hold_dir, hold_down=hold_down, rotation=rotation)
    for hold_dir in (HoldDir.NEUTRAL, HoldDir.LEFT, HoldDir.RIGHT)
    for hold_down in (False, True)
    for rotation in (Rotation.NONE, Rotation.CW, Rotation.CCW)
)


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------


def _cell_blocked(cols_u16: np.ndarray, x: int, y: int) -> bool:
    if x < 0 or x >= GRID_WIDTH:
        return True
    # Allow y=-1 for vertical pills spawning at top (top half at row 0, bottom half above screen)
    if y < -1:
        return True
    if y >= GRID_HEIGHT:
        return True
    # Cells above the grid (y < 0) are never blocked by the board
    if y < 0:
        return False
    col_bits = int(cols_u16[x])
    return bool(col_bits & (1 << y))


def _fits(cols_u16: np.ndarray, x: int, y: int, orient: int) -> bool:
    offsets = ORIENT_OFFSETS[int(orient) & 3]
    for dr, dc in offsets:
        if _cell_blocked(cols_u16, x + dc, y + dr):
            return False
    return True


# ---------------------------------------------------------------------------
# Gravity helpers
# ---------------------------------------------------------------------------


def compute_speed_threshold(speed_setting: int, speed_ups: int) -> int:
    """Return gravity threshold (frames between drops) for the current spawn."""

    setting_idx = max(0, min(speed_setting, len(BASE_SPEED_SETTING_VALUE) - 1))
    base_index = BASE_SPEED_SETTING_VALUE[setting_idx]
    raw_index = base_index + max(0, speed_ups)
    table_index = max(0, min(raw_index, SPEED_COUNTER_MAX))
    return SPEED_COUNTER_TABLE[table_index]


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def _dir_flags(direction: HoldDir) -> Tuple[bool, bool]:
    if direction is HoldDir.LEFT:
        return True, False
    if direction is HoldDir.RIGHT:
        return False, True
    return False, False


def _state_key(state: FrameState) -> Tuple[int, ...]:
    """Key used for visited pruning (excludes immutable spawn data)."""

    return (
        int(state.x),
        int(state.y),
        int(state.orient) & 3,
        int(state.speed_counter),
        int(state.hor_velocity),
        int(state.hold_dir.value),
        int(state.hold_down),
        int(state.frame_parity & FAST_DROP_MASK),
        1 if state.grounded else 0,
        int(min(max(state.lock_timer, 0), LOCK_BUFFER_FRAMES)),
    )


def _apply_rotation(
    cols_u16: np.ndarray,
    y: int,
    x: int,
    orient: int,
    rotation: Rotation,
    hold_left: bool,
) -> Tuple[int, int, bool]:
    """Apply the rotation stage. Returns (new_x, new_orient, rotated)."""

    if rotation is Rotation.NONE:
        return x, orient, False

    if rotation is Rotation.CW:
        new_orient = (orient - 1) & 3
    else:  # CCW
        new_orient = (orient + 1) & 3

    orig_x = x
    rotated = False

    if new_orient & 1 == 0:
        # Horizontal target: try in-place first.
        if _fits(cols_u16, x, y, new_orient):
            rotated = True
            if hold_left:
                kicked_x = x - 1
                if _fits(cols_u16, kicked_x, y, new_orient):
                    x = kicked_x
        else:
            # Wall kick: shift left once, then validate.
            kicked_x = x - 1
            if _fits(cols_u16, kicked_x, y, new_orient):
                rotated = True
                x = kicked_x
    else:
        # Vertical target: single validation.
        if _fits(cols_u16, x, y, new_orient):
            rotated = True

    if not rotated:
        return orig_x, orient, False
    return x, new_orient, True


def _step_state(
    cols_u16: np.ndarray,
    state: FrameState,
    action: FrameAction,
    speed_threshold: int,
) -> FrameState:
    """Simulate one frame given the current state and action."""

    prev_left, prev_right = _dir_flags(state.hold_dir)
    hold_left, hold_right = _dir_flags(action.hold_dir)
    hold_down = bool(action.hold_down)

    press_left = hold_left and not prev_left
    press_right = hold_right and not prev_right
    press_any_lr = press_left or press_right

    frame_flag = state.frame_parity & FAST_DROP_MASK

    # --- Y stage (gravity / soft drop) ---
    down_only = hold_down and not hold_left and not hold_right
    y = state.y
    x = state.x
    orient = state.orient & 3
    speed_counter = state.speed_counter
    grounded = bool(state.grounded)
    lock_timer = int(state.lock_timer)
    locked = False

    if frame_flag != 0 and down_only:
        drop_triggered = True
        # Note: speed_counter will be reset to 0 below if drop succeeds
    else:
        trial_counter = min(state.speed_counter + 1, speed_threshold + 1)
        if trial_counter > speed_threshold:
            drop_triggered = True
            speed_counter = 0
        else:
            drop_triggered = False
            speed_counter = min(trial_counter, speed_threshold)

    if drop_triggered:
        if _fits(cols_u16, x, y + 1, orient):
            y += 1
            speed_counter = 0
            grounded = False
        else:
            # Grounded this frame; defer locking to allow slide/tuck via lock buffer.
            grounded = True
            speed_counter = 0

    # --- X stage (horizontal movement) ---
    hor_velocity = state.hor_velocity
    if not locked:
        allow_move = False
        if press_any_lr:
            hor_velocity = 0
            allow_move = True
        elif hold_left or hold_right:
            trial_velocity = state.hor_velocity + 1
            if trial_velocity >= HOR_ACCEL_SPEED:
                hor_velocity = HOR_RELOAD
                allow_move = True
            else:
                hor_velocity = min(trial_velocity, HOR_BLOCKED)
        # Attempt moves in NES order: right first, then left.
        if allow_move:
            if hold_right:
                candidate = x + 1
                if _fits(cols_u16, candidate, y, orient):
                    x = candidate
                    grounded = False
                    lock_timer = LOCK_BUFFER_FRAMES
                else:
                    hor_velocity = HOR_BLOCKED
            if hold_left:
                candidate = x - 1
                if _fits(cols_u16, candidate, y, orient):
                    x = candidate
                    grounded = False
                    lock_timer = LOCK_BUFFER_FRAMES
                else:
                    hor_velocity = HOR_BLOCKED

    # --- Rotation stage ---
    if not locked:
        x, orient, _ = _apply_rotation(
            cols_u16=cols_u16,
            y=y,
            x=x,
            orient=orient,
            rotation=action.rotation,
            hold_left=hold_left,
        )
        # If rotation changed orientation successfully, treat as adjustment.
        if orient != (state.orient & 3):
            grounded = False
            lock_timer = LOCK_BUFFER_FRAMES

    next_parity = (state.frame_parity + 1) & FAST_DROP_MASK

    # Post-frame lock resolution
    if grounded and not locked:
        if lock_timer <= 0:
            locked = True
        else:
            lock_timer = max(0, lock_timer - 1)

    return FrameState(
        x=int(x),
        y=int(y),
        orient=int(orient),
        speed_counter=int(speed_counter),
        hor_velocity=int(hor_velocity),
        hold_dir=action.hold_dir,
        hold_down=hold_down,
        frame_parity=int(next_parity),
        grounded=bool(grounded),
        lock_timer=int(lock_timer),
        locked=locked,
    )


def simulate_frame(
    cols_u16: np.ndarray,
    state: FrameState,
    action_index: int,
    *,
    speed_threshold: int,
) -> FrameState:
    """Advance ``state`` by one frame using the indexed action."""

    action = frame_action_from_index(action_index)
    return _step_state(cols_u16, state, action, speed_threshold)


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
    """Enumerate all frame-accurate placements reachable from the spawn state."""
    import time
    t0 = time.perf_counter()
    
    cfg = config or ReachabilityConfig()
    nodes: List[FrameNode] = [FrameNode(state=spawn_state, parent=-1, action_index=-1, depth=0)]
    q: deque[int] = deque([0])
    visited: Dict[Tuple[int, ...], int] = {_state_key(spawn_state): 0}
    terminal_nodes: Dict[Tuple[int, int, int], int] = {}
    expanded = 0
    enqueued = 1
    last_log_time = t0
    
    print(f"[{t0:.4f}] build_reachability started")

    while q:
        idx = q.popleft()
        node = nodes[idx]
        state = node.state
        if node.depth >= cfg.max_frames:
            continue
        if state.locked:
            continue
        expanded += 1

        current_time = time.perf_counter()
        if current_time - last_log_time > 0.5: # Log every 500ms
            print(f"[{current_time:.4f}] build_reachability progress: expanded={expanded}, enqueued={enqueued}, qsize={len(q)}, depth={node.depth}")
            last_log_time = current_time

        for action_index, action in enumerate(_ACTION_SPACE):
            next_state = _step_state(cols_u16, state, action, speed_threshold)
            child_depth = node.depth + 1
            child = FrameNode(
                state=next_state,
                parent=idx,
                action_index=action_index,
                depth=child_depth,
            )
            child_idx = len(nodes)
            nodes.append(child)
            if next_state.locked:
                anchor = (next_state.x, next_state.y, next_state.orient & 3)
                prev_idx = terminal_nodes.get(anchor)
                if prev_idx is None or nodes[prev_idx].depth > child_depth:
                    terminal_nodes[anchor] = child_idx
                continue
            key = _state_key(next_state)
            prev_depth = visited.get(key)
            if prev_depth is not None and prev_depth <= child_depth:
                continue
            visited[key] = child_depth
            q.append(child_idx)
            enqueued += 1
    
    t1 = time.perf_counter()
    print(f"[{t1:.4f}] build_reachability finished in {t1-t0:.4f}s. expanded={expanded}, enqueued={enqueued}, terminals={len(terminal_nodes)}")

    stats = {
        "expanded": expanded,
        "enqueued": enqueued,
        "terminals": len(terminal_nodes),
    }
    return ReachabilityResult(nodes=nodes, terminal_nodes=terminal_nodes, stats=stats)


# ---------------------------------------------------------------------------
# Path reconstruction helpers
# ---------------------------------------------------------------------------


def reconstruct_actions(result: ReachabilityResult, terminal_index: int) -> List[int]:
    """Return action indices for the path ending at `terminal_index` (root→leaf)."""

    actions: List[int] = []
    idx = terminal_index
    while idx >= 0:
        node = result.nodes[idx]
        if node.parent < 0:
            break
        actions.append(node.action_index)
        idx = node.parent
    actions.reverse()
    return actions


def iter_frame_states(result: ReachabilityResult, terminal_index: int) -> Iterator[FrameState]:
    """Yield the FrameState sequence from spawn to the terminal node (inclusive)."""

    sequence: List[FrameState] = []
    idx = terminal_index
    while idx >= 0:
        node = result.nodes[idx]
        sequence.append(node.state)
        idx = node.parent
    for state in reversed(sequence):
        yield state


def frame_action_from_index(index: int) -> FrameAction:
    """Return the :class:`FrameAction` encoded at ``index`` in the action space."""

    if index < 0 or index >= len(_ACTION_SPACE):
        raise IndexError(f"Frame action index {index} is out of range")
    return _ACTION_SPACE[index]


__all__ = [
    "FrameAction",
    "FrameNode",
    "FrameState",
    "HoldDir",
    "Orientation",
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
