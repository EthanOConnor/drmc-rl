"""Spawn-latched macro-placement planner (NES-accurate).

This planner bridges:
  1) The NES/C++ engine *falling pill* state (base cell + rotation + counters)
  2) The policy's macro action space: a dense [4,16,8] grid (512 actions)

At each pill spawn we compute a feasibility mask over the 512 macro actions and
can reconstruct a *minimal-time* per-frame controller script for any feasible
placement.

The physics core is in `envs.retro.fast_reach` and mirrors the ROM routines:
`fallingPill_checkYMove`, `fallingPill_checkXMove`, `fallingPill_checkRotate`.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

import envs.specs.ram_to_state as ram_specs
from envs.retro.drmario_env import Action
from envs.state_core import DrMarioState
from envs.retro.fast_reach import (
    FrameState,
    HoldDir,
    ReachabilityConfig,
    ReachabilityResult,
    Rotation,
    build_reachability,
    compute_speed_threshold,
    frame_action_from_index,
    reconstruct_actions,
)
from envs.retro.placement_space import (
    GRID_HEIGHT,
    GRID_WIDTH,
    ORIENT_OFFSETS,
    ORIENTATIONS,
    TOTAL_ACTIONS,
    flatten as flatten_action,
    in_bounds,
    is_valid_action,
    partner_cell,
    unflatten as unflatten_action,
)

# NES controller bit masks (currentP_btnsHeld layout)
BTN_RIGHT = 0x01
BTN_LEFT = 0x02
BTN_DOWN = 0x04

# Zero-page current-player addresses (drmario_ram_zp.asm)
ZP_FALLING_PILL_COLOR_1 = 0x0081
ZP_FALLING_PILL_COLOR_2 = 0x0082
ZP_FALLING_PILL_X = 0x0085
ZP_FALLING_PILL_Y = 0x0086
ZP_SPEED_COUNTER = 0x0092
ZP_SPEED_UPS = 0x008A
ZP_SPEED_SETTING = 0x008B
ZP_HOR_VELOCITY = 0x0093
ZP_FALLING_PILL_ROT = 0x00A5

# Common addresses (NMI bookkeeping, player inputs)
ZP_FRAME_COUNTER = 0x0043
P1_BUTTONS_HELD = 0x00F7


class PlannerError(RuntimeError):
    """Raised when the planner cannot interpret the provided snapshot."""


# ---------------------------------------------------------------------------
# Capsule geometry helpers (NES base-cell convention)
# ---------------------------------------------------------------------------


# Offsets for (first_half, second_half) relative to base cell (bottom-left of 2×2 box).
# These are `halfPill_posOffset_rotationAndSizeBased` from drmario_data_game.asm,
# expressed in (row, col) coordinates with row increasing downward.
ROT_OFFSETS: Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...] = (
    ((0, 0), (0, 1)),    # rot=0: first at base, second right
    ((0, 0), (-1, 0)),   # rot=1: first at base, second above
    ((0, 1), (0, 0)),    # rot=2: first right, second at base
    ((-1, 0), (0, 0)),   # rot=3: first above, second at base
)


def iter_cells(base_row: int, base_col: int, rot: int) -> Iterator[Tuple[int, int]]:
    """Yield the (row,col) cells occupied by the falling pill at (base_row, base_col, rot)."""

    offsets = ROT_OFFSETS[int(rot) & 3]
    for dr, dc in offsets:
        yield int(base_row) + int(dr), int(base_col) + int(dc)


def _fits_base(cols_u16: np.ndarray, base_x: int, base_y: int, rot: int) -> bool:
    """Return True iff the capsule fits at (base_x,base_y,rot) on `cols_u16`."""

    x = int(base_x)
    y = int(base_y)
    rot_i = int(rot) & 3
    if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
        return False
    if int(cols_u16[x]) & (1 << y):
        return False
    if (rot_i & 1) == 0:
        # Horizontal: base and base+(0,+1)
        if x + 1 >= GRID_WIDTH:
            return False
        if int(cols_u16[x + 1]) & (1 << y):
            return False
    else:
        # Vertical: base and base+(-1,0). Allow y=-1 (offscreen).
        if y - 1 >= 0 and (int(cols_u16[x]) & (1 << (y - 1))):
            return False
    return True


def _macro_action_from_base(x: int, y: int, rot: int) -> Optional[int]:
    """Map a locked base-cell pill pose to a 512-way macro action index.

    Returns None for poses that place either half outside the visible 16×8 grid
    (e.g. top-out cases where the capsule is partly above the bottle).
    """

    rot_i = int(rot) & 3
    (dr1, dc1), (dr2, dc2) = ROT_OFFSETS[rot_i]
    r1, c1 = int(y) + int(dr1), int(x) + int(dc1)
    r2, c2 = int(y) + int(dr2), int(x) + int(dc2)
    if not (in_bounds(r1, c1) and in_bounds(r2, c2)):
        return None
    dr = r2 - r1
    dc = c2 - c1
    try:
        o = ORIENT_OFFSETS.index((dr, dc))
    except ValueError:
        return None
    return flatten_action(int(o), int(r1), int(c1))


def _base_pose_for_macro_action(action: int) -> Optional[Tuple[int, int, int]]:
    """Return (base_x, base_y, rot) for a macro action, or None if out-of-bounds."""

    o, row, col = unflatten_action(int(action))
    if not is_valid_action(o, row, col):
        return None

    # Derivation: base cell is bottom-left of the pill's 2×2 bounding box.
    if o == 0:  # H+ (origin left, dest right) -> rot 0, base at origin
        return int(col), int(row), 0
    if o == 2:  # H- (origin right, dest left) -> rot 2, base at dest (left cell)
        return int(col) - 1, int(row), 2
    if o == 3:  # V- (origin bottom, dest above) -> rot 1, base at origin (bottom cell)
        return int(col), int(row), 1
    if o == 1:  # V+ (origin top, dest below) -> rot 3, base at dest (bottom cell)
        return int(col), int(row) + 1, 3
    return None


# ---------------------------------------------------------------------------
# RAM snapshot decoding
# ---------------------------------------------------------------------------


def _read_u8(ram_bytes: bytes, addr: int, *, default: int = 0) -> int:
    if addr < 0 or addr >= len(ram_bytes):
        return int(default) & 0xFF
    return int(ram_bytes[addr]) & 0xFF


@dataclass(frozen=True)
class PillSnapshot:
    """Decoded falling-pill snapshot (base cell + NES rotation code)."""

    base_row: int  # 0=top, 15=bottom
    base_col: int  # 0..7
    rot: int       # 0..3 (NES rotation)
    colors: Tuple[int, int]  # (first_color, second_color) in {0,1,2}

    speed_counter: int
    speed_threshold: int
    hor_velocity: int
    frame_parity: int

    hold_left: bool
    hold_right: bool
    hold_down: bool

    speed_setting: int
    speed_ups: int
    spawn_id: Optional[int] = None

    @classmethod
    def from_state(cls, state: DrMarioState, offsets: Dict[str, Dict]) -> "PillSnapshot":
        ram_bytes = state.ram.bytes

        raw_row_from_bottom = _read_u8(ram_bytes, ZP_FALLING_PILL_Y)
        base_row = (GRID_HEIGHT - 1) - int(raw_row_from_bottom)
        base_col = _read_u8(ram_bytes, ZP_FALLING_PILL_X)
        rot = _read_u8(ram_bytes, ZP_FALLING_PILL_ROT) & 0x03

        if not (0 <= base_row < GRID_HEIGHT and 0 <= base_col < GRID_WIDTH):
            raise PlannerError(f"Falling pill base cell out of range: row={base_row}, col={base_col}")

        # First/second half colors (NES: fallingPill1stColor/2ndColor). Naming in offsets may
        # say "left/right" but order is rotation-dependent.
        c1 = _read_u8(ram_bytes, ZP_FALLING_PILL_COLOR_1) & 0x03
        c2 = _read_u8(ram_bytes, ZP_FALLING_PILL_COLOR_2) & 0x03

        speed_setting = _read_u8(ram_bytes, ZP_SPEED_SETTING)
        speed_ups = _read_u8(ram_bytes, ZP_SPEED_UPS)
        speed_threshold = compute_speed_threshold(speed_setting, speed_ups)
        speed_counter = _read_u8(ram_bytes, ZP_SPEED_COUNTER)
        hor_velocity = _read_u8(ram_bytes, ZP_HOR_VELOCITY)

        frame_parity = _read_u8(ram_bytes, ZP_FRAME_COUNTER) & 0x01
        held = _read_u8(ram_bytes, P1_BUTTONS_HELD)
        hold_left = bool(held & BTN_LEFT)
        hold_right = bool(held & BTN_RIGHT)
        hold_down = bool(held & BTN_DOWN)

        spawn_id = state.ram_vals.pill_counter

        return cls(
            base_row=int(base_row),
            base_col=int(base_col),
            rot=int(rot),
            colors=(int(c1), int(c2)),
            speed_counter=int(speed_counter),
            speed_threshold=int(speed_threshold),
            hor_velocity=int(hor_velocity),
            frame_parity=int(frame_parity),
            hold_left=bool(hold_left),
            hold_right=bool(hold_right),
            hold_down=bool(hold_down),
            speed_setting=int(speed_setting),
            speed_ups=int(speed_ups),
            spawn_id=int(spawn_id) if spawn_id is not None else None,
        )


# ---------------------------------------------------------------------------
# Board representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoardState:
    """Static bottle occupancy as 8×uint16 column bitboards."""

    columns: np.ndarray  # (8,) uint16; bit y=1 => occupied at row y (0=top)

    @classmethod
    def from_planes(cls, planes: np.ndarray) -> "BoardState":
        static_mask = ram_specs.get_static_mask(planes)
        virus_mask = ram_specs.get_virus_mask(planes)
        occupancy = np.asarray(static_mask | virus_mask, dtype=bool)
        cols = np.zeros(GRID_WIDTH, dtype=np.uint16)
        for c in range(GRID_WIDTH):
            bits = 0
            col = occupancy[:, c]
            for r in range(GRID_HEIGHT):
                if bool(col[r]):
                    bits |= 1 << r
            cols[c] = np.uint16(bits)
        return cls(columns=cols)


# ---------------------------------------------------------------------------
# Planner API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControllerStep:
    action: Action
    hold_left: bool
    hold_right: bool
    hold_down: bool


@dataclass(frozen=True)
class PlanResult:
    """Minimal-time controller script for a macro placement action."""

    action: int  # flat [0,512)
    controller: List[ControllerStep]
    cost: int  # frames (tau)
    terminal_pose: Tuple[int, int, int]  # (base_x, base_y, rot)


@dataclass(frozen=True)
class SpawnReachability:
    """Cached per-spawn reachability result."""

    legal_mask: np.ndarray  # bool [4,16,8]
    feasible_mask: np.ndarray  # bool [4,16,8]
    costs: np.ndarray  # float32 [4,16,8], inf if infeasible
    # Exactly one backend is populated:
    #   - Python reference (`reach`, `action_to_terminal_node`)
    #   - Native accelerator (`native`)
    reach: Optional[ReachabilityResult] = None
    action_to_terminal_node: Optional[Dict[int, int]] = None  # macro action -> reach.nodes index (locked)
    native: Optional["NativeReachability"] = None


class PlacementPlanner:
    """Compute feasibility masks and reconstruct minimal-time scripts."""

    def __init__(self, *, max_frames: int = 2048, reach_backend: str = "auto") -> None:
        self._max_frames = int(max_frames)
        self._reach_backend = str(reach_backend or "auto").lower()
        self._native_runner = None
        self._warned_native_failure = False

    @staticmethod
    def _hold_dir(snapshot: PillSnapshot) -> HoldDir:
        if snapshot.hold_left and not snapshot.hold_right:
            return HoldDir.LEFT
        if snapshot.hold_right and not snapshot.hold_left:
            return HoldDir.RIGHT
        return HoldDir.NEUTRAL

    def _spawn_state(self, snapshot: PillSnapshot) -> FrameState:
        return FrameState(
            x=int(snapshot.base_col),
            y=int(snapshot.base_row),
            rot=int(snapshot.rot) & 3,
            speed_counter=int(snapshot.speed_counter),
            hor_velocity=int(snapshot.hor_velocity),
            hold_dir=self._hold_dir(snapshot),
            frame_parity=int(snapshot.frame_parity) & 1,
            locked=False,
        )

    def _legal_mask(self, board: BoardState) -> np.ndarray:
        legal = np.zeros((ORIENTATIONS, GRID_HEIGHT, GRID_WIDTH), dtype=np.bool_)
        cols = board.columns.astype(np.uint16, copy=False)
        for action in range(TOTAL_ACTIONS):
            pose = _base_pose_for_macro_action(action)
            if pose is None:
                continue
            bx, by, rot = pose
            if bx < 0 or bx >= GRID_WIDTH or by < 0 or by >= GRID_HEIGHT:
                continue
            # Fits at (bx,by) and cannot fit one row lower -> would lock here.
            if not _fits_base(cols, bx, by, rot):
                continue
            if _fits_base(cols, bx, by + 1, rot):
                continue
            o, r, c = unflatten_action(action)
            legal[o, r, c] = True
        return legal

    def _native_available(self) -> bool:
        if self._reach_backend == "python":
            return False
        try:
            from envs.retro import reach_native

            return reach_native.is_library_present()
        except Exception:
            return False

    def _get_native_runner(self) -> Optional["NativeReachabilityRunner"]:
        if self._reach_backend == "python":
            return None
        # Allow forcing Python from env without touching configs.
        if str(os.environ.get("DRMARIO_REACH_BACKEND", "")).lower() == "python":
            return None
        if self._native_runner is not None:
            return self._native_runner
        if self._reach_backend == "auto" and not self._native_available():
            return None
        try:
            from envs.retro.reach_native import NativeReachabilityRunner

            self._native_runner = NativeReachabilityRunner(max_frames=self._max_frames)
            return self._native_runner
        except Exception as exc:
            if self._reach_backend == "native":
                raise
            if not self._warned_native_failure:
                warnings.warn(
                    "Native reachability unavailable; falling back to Python reference "
                    f"planner ({exc}). To build the native helper run: "
                    "`python -m tools.build_reach_native`."
                )
                self._warned_native_failure = True
            self._native_runner = None
            return None

    def build_spawn_reachability(self, board: BoardState, snapshot: PillSnapshot) -> SpawnReachability:
        cols = board.columns.astype(np.uint16, copy=False)
        spawn = self._spawn_state(snapshot)
        legal_mask = self._legal_mask(board)
        feasible_mask = np.zeros_like(legal_mask)
        costs = np.full_like(legal_mask, np.inf, dtype=np.float32)

        native_runner = self._get_native_runner()
        if native_runner is not None:
            native = native_runner.bfs_full(cols, spawn, speed_threshold=int(snapshot.speed_threshold))

            for action in range(TOTAL_ACTIONS):
                pose = _base_pose_for_macro_action(action)
                if pose is None:
                    continue
                bx, by, rot = pose
                if not legal_mask.reshape(-1)[action]:
                    continue
                cost = native.cost_for_pose(bx, by, rot)
                if cost is None:
                    continue
                o, r, c = unflatten_action(action)
                feasible_mask[o, r, c] = True
                costs[o, r, c] = float(cost)

            return SpawnReachability(
                legal_mask=legal_mask,
                feasible_mask=feasible_mask,
                costs=costs,
                native=native,
            )

        reach = build_reachability(
            cols,
            spawn,
            speed_threshold=int(snapshot.speed_threshold),
            config=ReachabilityConfig(max_frames=self._max_frames),
        )
        action_to_terminal: Dict[int, int] = {}
        for (x, y, rot), node_idx in reach.terminal_nodes.items():
            macro = _macro_action_from_base(x, y, rot)
            if macro is None:
                continue
            o, r, c = unflatten_action(macro)
            feasible_mask[o, r, c] = True
            costs[o, r, c] = float(reach.nodes[node_idx].depth)
            action_to_terminal[int(macro)] = int(node_idx)

        return SpawnReachability(
            legal_mask=legal_mask,
            feasible_mask=feasible_mask,
            costs=costs,
            reach=reach,
            action_to_terminal_node=action_to_terminal,
        )

    @staticmethod
    def _controller_from_frame_action(frame_action_index: int) -> ControllerStep:
        act = frame_action_from_index(frame_action_index)
        action = Action.NOOP
        if act.rotation is Rotation.CW:
            action = Action.ROTATE_A
        elif act.rotation is Rotation.CCW:
            action = Action.ROTATE_B
        return ControllerStep(
            action=action,
            hold_left=(act.hold_dir is HoldDir.LEFT),
            hold_right=(act.hold_dir is HoldDir.RIGHT),
            hold_down=bool(act.hold_down),
        )

    def plan_action(self, spawn: SpawnReachability, action: int) -> Optional[PlanResult]:
        """Return a minimal-time controller script for `action`, or None if infeasible."""

        action_i = int(action)
        try:
            o, r, c = unflatten_action(action_i)
            if not bool(spawn.feasible_mask[o, r, c]):
                return None
        except Exception:
            return None

        if spawn.native is not None:
            pose = _base_pose_for_macro_action(action_i)
            if pose is None:
                return None
            bx, by, rot = pose
            script = spawn.native.script_for_pose(bx, by, rot)
            if script is None:
                return None
            controller = [self._controller_from_frame_action(int(a)) for a in script]
            return PlanResult(
                action=action_i,
                controller=controller,
                cost=len(controller),
                terminal_pose=(int(bx), int(by), int(rot) & 3),
            )

        reach = spawn.reach
        action_map = spawn.action_to_terminal_node
        if reach is None or action_map is None:
            return None
        node_idx = action_map.get(action_i)
        if node_idx is None:
            return None
        node = reach.nodes[int(node_idx)]
        if not node.state.locked:
            return None

        frame_actions = reconstruct_actions(reach, int(node_idx))
        controller = [self._controller_from_frame_action(a) for a in frame_actions]
        return PlanResult(
            action=action_i,
            controller=controller,
            cost=len(controller),
            terminal_pose=(int(node.state.x), int(node.state.y), int(node.state.rot) & 3),
        )


__all__ = [
    "PlannerError",
    "PillSnapshot",
    "BoardState",
    "ControllerStep",
    "PlanResult",
    "SpawnReachability",
    "PlacementPlanner",
    "iter_cells",
    "partner_cell",
]


# Optional native backend types (avoid importing at module import time).
try:  # pragma: no cover - optional acceleration
    from envs.retro.reach_native import NativeReachability, NativeReachabilityRunner
except Exception:  # pragma: no cover - optional acceleration
    NativeReachability = None  # type: ignore[assignment]
    NativeReachabilityRunner = None  # type: ignore[assignment]
