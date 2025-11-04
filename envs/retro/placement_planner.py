"""NES-accurate placement planner built on the counter-aware fast reach core."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    iter_frame_states,
    reconstruct_actions,
)
from envs.retro.placement_actions import (
    GRID_HEIGHT,
    GRID_WIDTH,
    PLACEMENT_EDGES,
    action_count,
    action_from_cells,
    opposite_actions,
)
from envs.retro.reach512 import (
    O_HNEG,
    O_HPOS,
    O_VNEG,
    O_VPOS,
    drm_idx,
    reconstruct_actions_to,
    run_reachability,
)

GridCoord = Tuple[int, int]

# ---------------------------------------------------------------------------
# Capsule geometry helpers (shared with translator visualisations)
# ---------------------------------------------------------------------------


ORIENT_OFFSETS: Tuple[Tuple[GridCoord, GridCoord], ...] = (
    ((0, 0), (0, 1)),   # 0: horizontal, anchor on left
    ((0, 0), (1, 0)),   # 1: vertical (down), anchor on top
    ((0, 0), (0, -1)),  # 2: horizontal, anchor on right
    ((0, 0), (-1, 0)),  # 3: vertical (up), anchor on bottom
)


def iter_cells(row: int, col: int, orient: int) -> Iterator[GridCoord]:
    offsets = ORIENT_OFFSETS[int(orient) & 3]
    for dr, dc in offsets:
        yield row + dr, col + dc


def _edge_orientation(edge) -> Optional[int]:
    r0, c0 = edge.origin
    r1, c1 = edge.dest
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0 and dc == 1:
        return O_HPOS
    if dr == 0 and dc == -1:
        return O_HNEG
    if dr == 1 and dc == 0:
        return O_VPOS
    if dr == -1 and dc == 0:
        return O_VNEG
    return None


# ---------------------------------------------------------------------------
# RAM snapshot decoding helpers
# ---------------------------------------------------------------------------


BTN_RIGHT = 0x01
BTN_LEFT = 0x02
BTN_DOWN = 0x04

ZP_SPEED_COUNTER = 0x0092
ZP_SPEED_UPS = 0x008A
ZP_SPEED_SETTING = 0x008B
ZP_HOR_VELOCITY = 0x0093


class PlannerError(RuntimeError):
    """Raised when the planner cannot interpret the provided snapshot."""


def _read_byte(ram_bytes: bytes, addr: int) -> int:
    if addr < 0 or addr >= len(ram_bytes):
        return 0
    return int(ram_bytes[addr])


def _read_optional_hex(offsets: Dict[str, Dict], category: str, key: str, default: Optional[int] = None) -> Optional[int]:
    try:
        raw = offsets.get(category, {}).get(key)
    except Exception:
        raw = None
    if raw is None:
        return default
    try:
        return int(raw, 16)
    except (TypeError, ValueError):
        return default


@dataclass
class PillSnapshot:
    """Decoded RAM snapshot describing the currently falling capsule."""

    row: int
    col: int
    orient: int
    colors: Tuple[int, int]
    gravity_counter: int
    gravity_period: int
    lock_counter: int
    hor_velocity: int
    hold_left: bool
    hold_right: bool
    hold_down: bool
    frame_parity: int
    speed_setting: int
    speed_ups: int
    spawn_id: Optional[int] = None

    @property
    def speed_counter(self) -> int:
        return self.gravity_counter

    @classmethod
    def from_ram_state(
        cls,
        state: DrMarioState,
        offsets: Dict[str, Dict],
    ) -> "PillSnapshot":
        ram_bytes = state.ram.bytes

        falling_offsets = offsets.get("falling_pill", {})
        if not falling_offsets:
            raise PlannerError("Falling pill offsets are unavailable")
        row_addr = falling_offsets.get("row_addr")
        col_addr = falling_offsets.get("col_addr")
        orient_addr = falling_offsets.get("orient_addr")
        if not (row_addr and col_addr and orient_addr):
            raise PlannerError("Incomplete falling pill RAM offsets")
        try:
            raw_row = int(ram_bytes[int(row_addr, 16)])
            raw_col = int(ram_bytes[int(col_addr, 16)])
            orient_raw = int(ram_bytes[int(orient_addr, 16)]) & 0x03
        except (ValueError, IndexError, TypeError) as exc:
            raise PlannerError(f"Failed to decode pill RAM: {exc}") from exc

        # Counters and lock buffer
        lock_counter = state.ram_vals.lock_counter or 0

        speed_counter = state.ram_vals.gravity_counter or 0
        speed_setting = state.ram_vals.speed_setting or 0
        speed_ups = state.ram_vals.speed_ups or 0
        speed_threshold = compute_speed_threshold(speed_setting, speed_ups)

        hor_velocity = state.ram_vals.hor_velocity or 0
        frame_counter_addr = _read_optional_hex(offsets, "timers", "frame_counter_addr", 0x0043) or 0x0043
        frame_parity = _read_byte(ram_bytes, frame_counter_addr) & 0x01
        btns_addr = _read_optional_hex(offsets, "inputs", "p1_buttons_held_addr", 0x00F7) or 0x00F7
        btns_held = _read_byte(ram_bytes, btns_addr)
        hold_left = bool(btns_held & BTN_LEFT)
        hold_right = bool(btns_held & BTN_RIGHT)
        hold_down = bool(btns_held & BTN_DOWN)

        color_left = 0
        color_right = 0
        left_addr = falling_offsets.get("left_color_addr")
        right_addr = falling_offsets.get("right_color_addr")
        if left_addr:
            try:
                color_left = int(ram_bytes[int(left_addr, 16)]) & 0x03
            except (ValueError, IndexError, TypeError):
                color_left = 0
        if right_addr:
            try:
                color_right = int(ram_bytes[int(right_addr, 16)]) & 0x03
            except (ValueError, IndexError, TypeError):
                color_right = 0

        spawn_id = state.ram_vals.pill_counter

        # Convert RAM coordinates: rows count from bottom, orientation labels swapped.
        row = (GRID_HEIGHT - 1) - raw_row
        col = raw_col
        orient = orient_raw
        if orient == 1:
            orient = 3
        elif orient == 3:
            orient = 1
        if orient == 2:
            col += 1
        elif orient == 1:
            row -= 1
        if not (0 <= col < GRID_WIDTH):
            raise PlannerError(f"Falling pill column out of range: {col}")
        if not (-2 <= row < GRID_HEIGHT + 2):
            raise PlannerError(f"Falling pill row out of range: {row}")

        return cls(
            row=int(row),
            col=int(col),
            orient=int(orient),
            colors=(int(color_left), int(color_right)),
            gravity_counter=int(speed_counter),
            gravity_period=int(speed_threshold),
            lock_counter=int(lock_counter),
            hor_velocity=int(hor_velocity),
            hold_left=hold_left,
            hold_right=hold_right,
            hold_down=hold_down,
            frame_parity=int(frame_parity),
            speed_setting=int(speed_setting) & 0xFF,
            speed_ups=int(speed_ups) & 0xFF,
            spawn_id=spawn_id,
        )


# ---------------------------------------------------------------------------
# Board representation (bitboard for static occupancy)
# ---------------------------------------------------------------------------


@dataclass
class BoardState:
    """Bitboard representation of the static bottle."""

    columns: np.ndarray  # shape (GRID_WIDTH,), dtype=np.uint16

    @classmethod
    def from_state(cls, state: np.ndarray) -> "BoardState":
        static_mask = ram_specs.get_static_mask(state)
        virus_mask = ram_specs.get_virus_mask(state)
        occupancy = np.asarray(static_mask | virus_mask, dtype=bool)
        col_bits = np.zeros(GRID_WIDTH, dtype=np.uint16)
        for col in range(GRID_WIDTH):
            mask = 0
            column = occupancy[:, col]
            for row in range(GRID_HEIGHT):
                if column[row]:
                    mask |= 1 << row
            col_bits[col] = mask
        return cls(columns=col_bits)

    def fits(self, row: int, col: int, orient: int) -> bool:
        for rr, cc in iter_cells(row, col, orient):
            if rr < 0 or cc < 0 or cc >= GRID_WIDTH:
                return False
            if rr >= GRID_HEIGHT:
                return False
            if self.columns[cc] & (1 << rr):
                return False
        return True

    def without_capsule(self, pill: PillSnapshot) -> "BoardState":
        cols = self.columns.copy()
        for rr, cc in iter_cells(int(pill.row), int(pill.col), int(pill.orient)):
            if 0 <= cc < GRID_WIDTH and 0 <= rr < GRID_HEIGHT:
                cols[cc] &= ~(1 << rr)
        return BoardState(columns=cols)


# ---------------------------------------------------------------------------
# Planner API types
# ---------------------------------------------------------------------------


@dataclass
class PlannerParams:
    max_search_frames: int = GRID_HEIGHT * 6


@dataclass
class ControllerStep:
    action: Action = Action.NOOP
    hold_left: bool = False
    hold_right: bool = False
    hold_down: bool = False


@dataclass
class CapsuleState:
    row: int
    col: int
    orient: int
    speed_counter: int
    speed_threshold: int
    hor_velocity: int
    frame_parity: int
    hold_left: bool
    hold_right: bool
    hold_down: bool
    locked: bool
    frames: int = 0


@dataclass
class PlanResult:
    action: int
    controller: List[ControllerStep]
    states: List[CapsuleState]
    cost: int
    path_index: int = -1
    bfs_spawn: Tuple[int, int, int] | None = None  # (sx, sy, so)


@dataclass
class PlannerOutput:
    legal_mask: np.ndarray
    feasible_mask: np.ndarray
    costs: np.ndarray
    path_indices: np.ndarray
    plans: Tuple[PlanResult, ...] = field(default_factory=tuple)
    stats: Dict[str, int] = field(default_factory=dict)

    @property
    def plan_count(self) -> int:
        return len(self.plans)


# ---------------------------------------------------------------------------
# Placement planner implementation
# ---------------------------------------------------------------------------


class PlacementPlanner:
    """Fast reachability planner over the placement action space."""

    def __init__(self, params: Optional[PlannerParams] = None, *, debug: bool = False) -> None:
        self.params = params or PlannerParams()
        self._debug = bool(debug)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_all(self, board: BoardState, capsule: PillSnapshot) -> PlannerOutput:
        cols = board.columns.astype(np.uint16)
        spawn_state = self._spawn_state_from_snapshot(capsule)
        reach = build_reachability(
            cols,
            spawn_state,
            speed_threshold=int(capsule.gravity_period),
            config=ReachabilityConfig(max_frames=int(self.params.max_search_frames)),
        )

        legal_mask = self._compute_legal(board)
        feasible_mask = np.zeros(action_count(), dtype=np.bool_)
        costs = np.full(action_count(), np.inf, dtype=np.float32)
        path_indices = np.full(action_count(), -1, dtype=np.int32)

        plans: List[PlanResult] = []
        for anchor, node_idx in reach.terminal_nodes.items():
            action_index = self._action_index_for_anchor(anchor)
            plan = self._build_plan(reach, node_idx, action_index, capsule)
            plan.path_index = len(plans)
            plans.append(plan)
            feasible_mask[action_index] = True
            path_indices[action_index] = plan.path_index
            costs[action_index] = float(plan.cost)

        return PlannerOutput(
            legal_mask=legal_mask,
            feasible_mask=feasible_mask,
            costs=costs,
            path_indices=path_indices,
            plans=tuple(plans),
            stats=dict(reach.stats),
        )

    def plan_action(
        self,
        board: BoardState,
        capsule: PillSnapshot,
        action: int,
    ) -> Optional[PlanResult]:
        if action < 0 or action >= action_count():
            return None
        cols = board.columns.astype(np.uint16)
        spawn_state = self._spawn_state_from_snapshot(capsule)
        reach = build_reachability(
            cols,
            spawn_state,
            speed_threshold=int(capsule.gravity_period),
            config=ReachabilityConfig(max_frames=int(self.params.max_search_frames)),
        )
        target_anchor = self._anchor_for_action(int(action))
        node_idx = reach.best_node_for(target_anchor)
        if node_idx is None:
            return None
        plan = self._build_plan(reach, node_idx, int(action), capsule)
        plan.path_index = 0
        return plan

    def plan_action_fast(self, board: BoardState, capsule: PillSnapshot, action: int) -> Optional[PlanResult]:
        if action < 0 or action >= action_count():
            return None
        cols = board.columns.astype(np.uint16)
        sx, sy, so = int(capsule.col), int(capsule.row), int(capsule.orient) & 3
        reach = run_reachability(cols, sx, sy, so)
        orient = _edge_orientation(PLACEMENT_EDGES[int(action)])
        if orient is None:
            return None
        r0, c0 = PLACEMENT_EDGES[int(action)].origin
        target_idx = drm_idx(int(c0), int(r0), orient)
        arrival_cost = int(reach.arrival[target_idx])
        if arrival_cost == 0xFF:
            return None
        route = reconstruct_actions_to(reach, int(c0), int(r0), orient)
        if not route:
            return None
        return self._plan_from_route_fast(capsule, action, route, arrival_cost)

    def _plan_from_route_fast(
        self,
        capsule: PillSnapshot,
        action: int,
        route: List[Tuple[int, int, int, int]],
        arrival_cost: int,
    ) -> Optional[PlanResult]:
        if not route:
            return None
        controller: List[ControllerStep] = []
        states: List[CapsuleState] = []

        start_x, start_y, start_o, _ = route[0]
        base = snapshot_to_capsule_state(capsule)
        base.row = int(start_y)
        base.col = int(start_x)
        base.orient = int(start_o) & 3
        base.hold_left = False
        base.hold_right = False
        base.hold_down = False
        base.frames = 0
        states.append(base)

        frame_counter = 0
        hold_left = False
        hold_right = False
        hold_down = False

        def append_state(row: int, col: int, orient: int, *, locked: bool = False) -> None:
            states.append(
                CapsuleState(
                    row=int(row),
                    col=int(col),
                    orient=int(orient) & 3,
                    speed_counter=0,
                    speed_threshold=int(capsule.gravity_period),
                    hor_velocity=0,
                    frame_parity=0,
                    hold_left=bool(hold_left),
                    hold_right=bool(hold_right),
                    hold_down=bool(hold_down),
                    locked=bool(locked),
                    frames=frame_counter,
                )
            )

        for idx in range(1, len(route)):
            x_prev, y_prev, o_prev, _ = route[idx - 1]
            x_next, y_next, o_next, code = route[idx]

            if code == 1:  # DOWN
                hold_down = True
                hold_left = False
                hold_right = False
                controller.append(ControllerStep(action=Action.DOWN, hold_down=True))
                frame_counter += 1
                append_state(y_next, x_next, o_next)
            elif code == 2:  # LEFT
                hold_down = False
                hold_left = True
                hold_right = False
                controller.append(ControllerStep(action=Action.LEFT, hold_left=True))
                frame_counter += 1
                append_state(y_next, x_next, o_next)
                hold_left = False
                controller.append(ControllerStep(action=Action.NOOP, hold_left=False))
                frame_counter += 1
                append_state(y_next, x_next, o_next)
            elif code == 3:  # RIGHT
                hold_down = False
                hold_right = True
                hold_left = False
                controller.append(ControllerStep(action=Action.RIGHT, hold_right=True))
                frame_counter += 1
                append_state(y_next, x_next, o_next)
                hold_right = False
                controller.append(ControllerStep(action=Action.NOOP, hold_right=False))
                frame_counter += 1
                append_state(y_next, x_next, o_next)
            else:  # Rotations
                hold_down = False
                delta_orient = (int(o_next) - int(o_prev)) & 3
                action_code = Action.NOOP
                if delta_orient == 1 or code == 4:
                    action_code = Action.ROTATE_A
                elif delta_orient == 3 or code == 5:
                    action_code = Action.ROTATE_B
                kick_left = x_next < x_prev
                hold_left = kick_left
                hold_right = False
                controller.append(ControllerStep(action=action_code, hold_left=kick_left))
                frame_counter += 1
                append_state(y_next, x_next, o_next)
                if kick_left:
                    hold_left = False
                    controller.append(ControllerStep(action=Action.NOOP, hold_left=False))
                    frame_counter += 1
                    append_state(y_next, x_next, o_next)

        if not controller:
            return None
        states[-1].locked = True
        cost_val = int(arrival_cost) if arrival_cost >= 0 else len(controller)
        plan = PlanResult(action=int(action), controller=controller, states=states, cost=cost_val, path_index=0, bfs_spawn=None)
        return plan

    def enumerate_fast_options(
        self,
        board: BoardState,
        capsule: PillSnapshot,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[PlanResult, ...], Dict[str, int]]:
        cols = board.columns.astype(np.uint16)
        reach = run_reachability(cols, int(capsule.col), int(capsule.row), int(capsule.orient) & 3)
        arrival = reach.arrival
        legal = np.zeros(action_count(), dtype=np.bool_)
        feasible = np.zeros(action_count(), dtype=np.bool_)
        costs = np.full(action_count(), np.inf, dtype=np.float32)
        path_indices = np.full(action_count(), -1, dtype=np.int32)
        plans: List[PlanResult] = []
        for edge in PLACEMENT_EDGES:
            idx = edge.index
            orient = _edge_orientation(edge)
            if orient is None:
                continue
            r0, c0 = edge.origin
            if not board.fits(r0, c0, orient):
                continue
            if board.fits(r0 + 1, c0, orient):
                continue
            legal[idx] = True
            target_idx = drm_idx(int(c0), int(r0), orient)
            arrival_cost = int(arrival[target_idx])
            if arrival_cost == 0xFF:
                continue
            route = reconstruct_actions_to(reach, int(c0), int(r0), orient)
            if not route:
                continue
            plan = self._plan_from_route_fast(capsule, idx, route, arrival_cost)
            if plan is None:
                continue
            plan.path_index = len(plans)
            plans.append(plan)
            feasible[idx] = True
            path_indices[idx] = plan.path_index
            costs[idx] = float(plan.cost)
        stats = {
            "expanded": int(np.count_nonzero(arrival != 0xFF)),
            "terminals": int(feasible.sum()),
        }
        return legal, feasible, costs, path_indices, tuple(plans), stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hold_dir_from_bools(hold_left: bool, hold_right: bool) -> HoldDir:
        if hold_left and not hold_right:
            return HoldDir.LEFT
        if hold_right and not hold_left:
            return HoldDir.RIGHT
        return HoldDir.NEUTRAL

    def _spawn_state_from_snapshot(self, snapshot: PillSnapshot) -> FrameState:
        hold_dir = self._hold_dir_from_bools(snapshot.hold_left, snapshot.hold_right)
        return FrameState(
            x=int(snapshot.col),
            y=int(snapshot.row),
            orient=int(snapshot.orient) & 3,
            speed_counter=int(snapshot.gravity_counter),
            hor_velocity=int(snapshot.hor_velocity),
            hold_dir=hold_dir,
            hold_down=bool(snapshot.hold_down),
            frame_parity=int(snapshot.frame_parity) & 0x01,
            grounded=False,
            lock_timer=max(0, int(snapshot.lock_counter)) or 2,
            locked=False,
        )

    def _capsule_state_from_frame(
        self,
        frame_state: FrameState,
        capsule: PillSnapshot,
        frame_index: int,
    ) -> CapsuleState:
        hold_left = (frame_state.hold_dir is HoldDir.LEFT)
        hold_right = (frame_state.hold_dir is HoldDir.RIGHT)
        return CapsuleState(
            row=int(frame_state.y),
            col=int(frame_state.x),
            orient=int(frame_state.orient),
            speed_counter=int(frame_state.speed_counter),
            speed_threshold=int(capsule.gravity_period),
            hor_velocity=int(frame_state.hor_velocity),
            frame_parity=int(frame_state.frame_parity) & 0x01,
            hold_left=hold_left,
            hold_right=hold_right,
            hold_down=bool(frame_state.hold_down),
            locked=bool(frame_state.locked),
            frames=int(frame_index),
        )

    def _controller_from_action(self, action_index: int) -> ControllerStep:
        frame_action = frame_action_from_index(action_index)
        ctrl = ControllerStep()
        if frame_action.rotation is Rotation.CW:
            ctrl.action = Action.ROTATE_A
        elif frame_action.rotation is Rotation.CCW:
            ctrl.action = Action.ROTATE_B
        else:
            ctrl.action = Action.NOOP
        ctrl.hold_left = frame_action.hold_dir is HoldDir.LEFT
        ctrl.hold_right = frame_action.hold_dir is HoldDir.RIGHT
        ctrl.hold_down = bool(frame_action.hold_down)
        return ctrl

    def _build_plan(
        self,
        reach: ReachabilityResult,
        terminal_index: int,
        action_index: int,
        capsule: PillSnapshot,
    ) -> PlanResult:
        action_sequence = reconstruct_actions(reach, terminal_index)
        frame_states = list(iter_frame_states(reach, terminal_index))

        states: List[CapsuleState] = []
        for idx, frame_state in enumerate(frame_states):
            states.append(self._capsule_state_from_frame(frame_state, capsule, idx))
        if states:
            states[-1].locked = True

        controller = [self._controller_from_action(idx) for idx in action_sequence]
        cost = len(controller)
        bfs_spawn = (
            int(frame_states[0].x) if frame_states else int(capsule.col),
            int(frame_states[0].y) if frame_states else int(capsule.row),
            int(frame_states[0].orient) if frame_states else int(capsule.orient),
        )
        return PlanResult(
            action=int(action_index),
            controller=controller,
            states=states,
            cost=int(cost),
            bfs_spawn=bfs_spawn,
        )

    def _anchor_for_action(self, action: int) -> Tuple[int, int, int]:
        edge = PLACEMENT_EDGES[int(action)]
        origin, dest = edge.origin, edge.dest
        dr = dest[0] - origin[0]
        dc = dest[1] - origin[1]
        if dr == 0 and dc == 1:
            orient = 0
        elif dr == 0 and dc == -1:
            orient = 2
        elif dr == 1 and dc == 0:
            orient = 1
        elif dr == -1 and dc == 0:
            orient = 3
        else:
            raise ValueError("Invalid directed edge")
        return int(origin[1]), int(origin[0]), int(orient)

    def _action_index_for_anchor(self, anchor: Tuple[int, int, int]) -> int:
        x, y, orient = anchor
        origin = (int(y), int(x))
        offsets = ORIENT_OFFSETS[int(orient) & 3]
        dest = (origin[0] + offsets[1][0], origin[1] + offsets[1][1])
        return action_from_cells(origin, dest)

    def _compute_legal(self, board: BoardState) -> np.ndarray:
        legal = np.zeros(action_count(), dtype=np.bool_)
        for edge in PLACEMENT_EDGES:
            orient = self._orientation_for_edge(edge.origin, edge.dest)
            row, col = edge.origin
            if board.fits(row, col, orient) and not board.fits(row + 1, col, orient):
                legal[edge.index] = True
        return legal

    @staticmethod
    def _orientation_for_edge(origin: GridCoord, dest: GridCoord) -> int:
        dr = dest[0] - origin[0]
        dc = dest[1] - origin[1]
        if dr == 0 and dc == 1:
            return 0
        if dr == 0 and dc == -1:
            return 2
        if dr == 1 and dc == 0:
            return 1
        if dr == -1 and dc == 0:
            return 3
        raise ValueError("Invalid directed edge")


# ---------------------------------------------------------------------------
# Snapshot helper (public)
# ---------------------------------------------------------------------------


def snapshot_to_capsule_state(snapshot: PillSnapshot) -> CapsuleState:
    """Helper to build an initial :class:`CapsuleState` from a RAM snapshot."""

    return CapsuleState(
        row=snapshot.row,
        col=snapshot.col,
        orient=int(snapshot.orient) & 3,
        speed_counter=int(snapshot.gravity_counter),
        speed_threshold=int(snapshot.gravity_period),
        hor_velocity=int(snapshot.hor_velocity),
        frame_parity=int(snapshot.frame_parity) & 0x01,
        hold_left=bool(snapshot.hold_left),
        hold_right=bool(snapshot.hold_right),
        hold_down=bool(snapshot.hold_down),
        locked=False,
        frames=0,
    )


__all__ = [
    "BoardState",
    "CapsuleState",
    "ControllerStep",
    "PillSnapshot",
    "PlannerError",
    "PlannerOutput",
    "PlannerParams",
    "PlanResult",
    "PlacementPlanner",
    "iter_cells",
    "snapshot_to_capsule_state",
]
