"""Frame-accurate placement planner for Dr. Mario capsules.

The planner bridges the low-level emulator controller and the policy network's
placement intent head.  It receives a RAM-derived state snapshot at pill spawn
and enumerates every geometry-legal landing pose, then searches for micro plans
that respect gravity timing, lock-buffer windows, and the NES input model.  For
all reachable placements the planner records a per-frame controller script that
can be executed closed-loop by :class:`envs.retro.placement_wrapper`
(``DrMarioPlacementEnv``).

Design highlights
-----------------

* **Bitboard board model** – The static board is encoded as eight 16-bit column
  bitmasks.  Collision queries and support checks reduce to a few integer
  operations which keeps the successor generator tight even in Python.
* **Time-expanded search** – Each planner node encodes
  ``(row, col, orient, gravity, lock_buffer, held_left, held_right, held_down)``
  along with whether the capsule is currently grounded.  Successors advance one
  frame applying taps/holds and automatic gravity.  The branching factor is
  bounded (< 10) so A* over the reachable state space typically expands a few
  thousand nodes (< 2 ms on modern CPUs).
* **Slide/tuck modelling** – Horizontal movement or rotation while grounded
  resets a configurable lock buffer.  As long as the buffer remains positive the
  planner allows additional micro-adjustments, enabling classic Dr. Mario slides
  and boosted tucks.  The rotation helper mirrors the console behaviour closely
  enough to hit the timing windows observed in the disassembly/Java AI.
* **Multi-goal planning** – ``plan_all`` runs a single multi-target search,
  returning per-placement controller schedules alongside ``legal`` and
  ``feasible`` masks of length 464 (the directed edges in the bottle grid).
* **Closed-loop ready** – ``PlanResult`` stores both the controller script and
  the state trace so the wrapper can verify emulator state after each action and
  trigger replanning on mismatches.
"""

from __future__ import annotations

import enum
import heapq
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

import envs.specs.ram_to_state as ram_specs
from envs.retro.drmario_env import Action
from envs.retro.placement_actions import (
    GRID_HEIGHT,
    GRID_WIDTH,
    PLACEMENT_EDGES,
    action_count,
    opposite_actions,
)

GridCoord = Tuple[int, int]

# ---------------------------------------------------------------------------
# Capsule geometry helpers
# ---------------------------------------------------------------------------


ORIENT_OFFSETS: Tuple[Tuple[GridCoord, GridCoord], ...] = (
    ((0, 0), (0, 1)),  # 0: horizontal, anchor on left
    ((0, 0), (-1, 0)),  # 1: vertical, anchor on bottom
    ((0, 0), (0, -1)),  # 2: horizontal, anchor on right
    ((0, 0), (1, 0)),  # 3: vertical, anchor on top
)

_ROTATION_KICKS: Dict[Tuple[int, int], Tuple[GridCoord, ...]] = {
    (0, 1): ((0, 0), (0, -1), (1, 0)),
    (1, 0): ((0, 0), (0, 1), (-1, 0)),
    (1, 2): ((0, 0), (1, 0), (0, 1)),
    (2, 1): ((0, 0), (-1, 0), (0, -1)),
    (2, 3): ((0, 0), (0, 1), (-1, 0)),
    (3, 2): ((0, 0), (0, -1), (1, 0)),
    (0, 3): ((0, 0), (0, 1), (-1, 0)),
    (3, 0): ((0, 0), (0, -1), (1, 0)),
    (1, 3): ((0, 0), (1, 0)),
    (3, 1): ((0, 0), (-1, 0)),
    (0, 2): ((0, 0),),
    (2, 0): ((0, 0),),
}


class Rotation(enum.Enum):
    NONE = 0
    CW = 1
    CCW = 2
    HALF = 3


def orientation_successor(orient: int, rotation: Rotation) -> int:
    if rotation is Rotation.NONE:
        return orient
    if rotation is Rotation.HALF:
        return (orient + 2) % 4
    if rotation is Rotation.CW:
        return (orient + 1) % 4
    if rotation is Rotation.CCW:
        return (orient - 1) % 4
    raise ValueError(rotation)


def iter_cells(row: int, col: int, orient: int) -> Iterator[GridCoord]:
    offsets = ORIENT_OFFSETS[orient]
    for dr, dc in offsets:
        yield row + dr, col + dc


# ---------------------------------------------------------------------------
# RAM snapshot decoding
# ---------------------------------------------------------------------------


class PlannerError(RuntimeError):
    """Raised when the planner cannot interpret the provided snapshot."""


@dataclass
class PillSnapshot:
    row: int
    col: int
    orient: int
    colors: Tuple[int, int]
    gravity_counter: int
    gravity_period: int
    lock_counter: int
    spawn_id: Optional[int] = None

    @classmethod
    def from_ram_state(
        cls,
        state: Optional[np.ndarray],
        ram_bytes: bytes,
        offsets: Dict[str, Dict],
    ) -> "PillSnapshot":
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
            orient = int(ram_bytes[int(orient_addr, 16)]) & 0x03
        except (ValueError, IndexError, TypeError) as exc:
            raise PlannerError(f"Failed to decode pill RAM: {exc}") from exc

        gravity_offsets = offsets.get("gravity_lock", {})
        gravity_counter = gravity_period = 1
        lock_counter = 0
        if gravity_offsets:
            g_addr = gravity_offsets.get("gravity_counter_addr")
            p_addr = gravity_offsets.get("lock_counter_addr")
            if g_addr:
                try:
                    gravity_counter = max(0, int(ram_bytes[int(g_addr, 16)]))
                    gravity_period = max(1, gravity_counter or 1)
                except (ValueError, IndexError, TypeError):
                    gravity_counter = gravity_period = 1
            if p_addr:
                try:
                    lock_counter = max(0, int(ram_bytes[int(p_addr, 16)]))
                except (ValueError, IndexError, TypeError):
                    lock_counter = 0

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

        spawn_addr = offsets.get("pill_counter", {}).get("addr")
        spawn_id = None
        if spawn_addr:
            try:
                spawn_id = int(ram_bytes[int(spawn_addr, 16)]) & 0xFF
            except (ValueError, IndexError, TypeError):
                spawn_id = None

        # RAM row counts from the bottom of the bottle (0 == bottom) and the
        # column register always references the *left* half when horizontal.
        # The planner, however, anchors orientation ``2`` (reversed horizontal)
        # on the right half and orientation ``3`` (reversed vertical) on the
        # upper half.  Adjust the decoded coordinates so that
        # :func:`iter_cells` produces positions that match the pixels inferred
        # via :func:`envs.specs.ram_to_state.get_falling_mask`.
        row = (GRID_HEIGHT - 1) - raw_row
        col = raw_col
        if orient == 2:
            col += 1
        elif orient == 3:
            row -= 1
        if not (0 <= col < GRID_WIDTH):
            raise PlannerError(f"Falling pill column out of range: {col}")
        if not (-2 <= row < GRID_HEIGHT + 2):
            raise PlannerError(f"Falling pill row out of range: {row}")

        colors = (color_left, color_right)
        return cls(
            row=int(row),
            col=int(col),
            orient=int(orient),
            colors=colors,
            gravity_counter=int(gravity_counter),
            gravity_period=int(gravity_period),
            lock_counter=int(lock_counter),
            spawn_id=spawn_id,
        )


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

    def occupied(self, row: int, col: int) -> bool:
        if row < 0 or col < 0 or col >= GRID_WIDTH:
            return True
        if row >= GRID_HEIGHT:
            return True
        return bool(self.columns[col] & (1 << row))

    def fits(self, row: int, col: int, orient: int) -> bool:
        for rr, cc in iter_cells(row, col, orient):
            if rr < 0 or cc < 0 or cc >= GRID_WIDTH:
                return False
            if rr >= GRID_HEIGHT:
                return False
            if self.columns[cc] & (1 << rr):
                return False
        return True

    def resting(self, row: int, col: int, orient: int) -> bool:
        if not self.fits(row, col, orient):
            return False
        for rr, cc in iter_cells(row, col, orient):
            below = rr + 1
            if below >= GRID_HEIGHT:
                continue
            if not (self.columns[cc] & (1 << below)):
                return False
        return True


# ---------------------------------------------------------------------------
# Planner configuration
# ---------------------------------------------------------------------------


@dataclass
class PlannerParams:
    lock_buffer_frames: int = 2
    soft_drop_gravity: int = 1
    max_search_frames: int = GRID_HEIGHT * 6
    allow_half_rotations: bool = True


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
    gravity: int
    gravity_period: int
    lock_buffer: int
    grounded: bool
    locked: bool
    hold_left: bool
    hold_right: bool
    hold_down: bool
    frames: int = 0

    def key(self) -> Tuple[int, int, int, int, int, int, bool, bool, bool]:
        gravity_norm = min(max(self.gravity, 0), 60)
        lock_norm = min(max(self.lock_buffer, 0), 4)
        return (
            self.row,
            self.col,
            self.orient,
            gravity_norm,
            lock_norm,
            1 if self.grounded else 0,
            self.hold_left,
            self.hold_right,
            self.hold_down,
        )


@dataclass
class PlanResult:
    action: int
    controller: List[ControllerStep]
    states: List[CapsuleState]
    cost: int
    path_index: int = -1


@dataclass
class PlannerOutput:
    legal_mask: np.ndarray
    feasible_mask: np.ndarray
    costs: np.ndarray
    path_indices: np.ndarray
    plans: Tuple[PlanResult, ...] = field(default_factory=tuple)

    @property
    def plan_count(self) -> int:
        return len(self.plans)


# ---------------------------------------------------------------------------
# Planner implementation
# ---------------------------------------------------------------------------


class PlacementPlanner:
    """Search for frame-perfect placement plans."""

    def __init__(self, params: Optional[PlannerParams] = None) -> None:
        self.params = params or PlannerParams()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_all(self, board: BoardState, capsule: PillSnapshot) -> PlannerOutput:
        legal_mask = self._compute_legal(board)
        feasible_mask = np.zeros_like(legal_mask)
        costs = np.full(action_count(), np.inf, dtype=np.float32)
        path_indices = np.full(action_count(), -1, dtype=np.int32)
        targets = {idx for idx, is_legal in enumerate(legal_mask) if is_legal}
        plan_map = self._multi_goal_search(board, capsule, targets)
        plans: List[PlanResult] = []
        for action_idx, plan in sorted(plan_map.items()):
            feasible_mask[action_idx] = True
            plan_idx = len(plans)
            plan.path_index = plan_idx
            plans.append(plan)
            costs[action_idx] = float(plan.cost)
            path_indices[action_idx] = plan_idx
        return PlannerOutput(
            legal_mask=legal_mask,
            feasible_mask=feasible_mask,
            costs=costs,
            path_indices=path_indices,
            plans=tuple(plans),
        )

    def plan_action(
        self, board: BoardState, capsule: PillSnapshot, action: int
    ) -> Optional[PlanResult]:
        legal_mask = self._compute_legal(board)
        if action < 0 or action >= legal_mask.size or not legal_mask[action]:
            return None
        plan_map = self._multi_goal_search(board, capsule, {int(action)})
        plan = plan_map.get(int(action))
        if plan is not None:
            plan.path_index = 0
        return plan

    # ------------------------------------------------------------------
    # Legal geometry enumeration
    # ------------------------------------------------------------------

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
        if dr == -1 and dc == 0:
            return 1
        if dr == 1 and dc == 0:
            return 3
        raise ValueError(f"Invalid directed edge {origin}->{dest}")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _multi_goal_search(self, board, capsule, targets):
        remaining = set(targets)
        if not remaining:
            return {}

        start = CapsuleState(
            row=capsule.row,
            col=capsule.col,
            orient=capsule.orient,
            gravity=max(0, capsule.gravity_counter),
            gravity_period=max(1, capsule.gravity_period),
            lock_buffer=max(0, capsule.lock_counter // 4),
            grounded=False,
            locked=False,
            hold_left=False,
            hold_right=False,
            hold_down=False,
            frames=0,
        )
        start_key = start.key()
        frontier = [(0, 0, start)]
        came_from = {}
        state_cache = {start_key: start}
        cost_so_far = {start_key: 0}

        # dominance by (col,row,orient)
        dom = {}

        counter = 0
        plans = {}

        while frontier and remaining:
            _, _, cur = heapq.heappop(frontier)
            ck = cur.key()
            gc = cost_so_far.get(ck, 0)
            if cur.frames > self.params.max_search_frames:
                continue

            # dominance prune
            dkey = (cur.col, cur.row, cur.orient)
            cand = (cur.gravity, cur.lock_buffer, int(cur.hold_down) * -1)
            best = dom.get(dkey)
            if best is not None and (
                best[0] >= cand[0] and best[1] >= cand[1] and best[2] >= cand[2]
            ):
                continue
            dom[dkey] = cand

            # Goal match via predicate (robust to anchor conventions)
            if cur.locked and remaining:
                matched = [a for a in remaining if self._state_matches_action(board, cur, a)]
                for a in matched:
                    plan = self._reconstruct_plan(came_from, state_cache, ck, a, gc)
                    plans[a] = plan
                    remaining.remove(a)
                if not remaining:
                    break

            # successors
            for nxt, ctrl in self._successors(board, cur):
                nk = nxt.key()
                g2 = gc + 1
                if cost_so_far.get(nk, 1 << 30) <= g2:
                    continue
                cost_so_far[nk] = g2
                f2 = g2 + self._heuristic(nxt)  # make this tick-aware!
                counter += 1
                heapq.heappush(frontier, (f2, counter, nxt))
                came_from[nk] = (ck, ctrl)
                state_cache[nk] = nxt

        return plans

    def _heuristic(self, state: CapsuleState) -> int:
        # Encourage descent; horizontal distance is cheap relative to falling.
        return (GRID_HEIGHT - state.row) + (2 if state.orient in (1, 3) else 0)

    def _decode_action(self, action: int) -> Tuple[int, int, int]:
        """Map a placement action index to (col, row, orient).

        The tuple corresponds to the anchor used by the planner's state
        representation such that ``iter_cells(row, col, orient)`` yields the two
        cells for the placement encoded by ``PLACEMENT_EDGES[action]``.

        - Horizontal right (orient 0): anchor is the left cell (origin).
        - Horizontal left  (orient 2): anchor is the right cell (origin).
        - Vertical up      (orient 1): anchor is the bottom cell (origin).
        - Vertical down    (orient 3): anchor is the top cell (origin).
        """
        edge = PLACEMENT_EDGES[int(action)]
        orient = self._orientation_for_edge(edge.origin, edge.dest)
        row, col = edge.origin  # origin is the anchor under our orientation convention
        return int(col), int(row), int(orient)

    def _reachability_envelope(self, capsule: PillSnapshot) -> Tuple[List[int], List[int]]:
        """Conservative horizontal reachability envelope by row.

        Returns two lists ``(L, R)`` of length ``GRID_HEIGHT`` such that a
        placement whose anchor locks at ``(row, col)`` passes this pre-prune if
        ``L[row] <= col <= R[row]``. This implementation is deliberately
        conservative and allows the full width for every row, which preserves
        correctness and keeps the search functional if a more precise envelope
        is not available. More precise envelopes can be implemented to tighten
        successors and goals for performance.
        """
        L = [0 for _ in range(GRID_HEIGHT)]
        R = [GRID_WIDTH - 1 for _ in range(GRID_HEIGHT)]
        return L, R

    def _state_matches_action(self, board: BoardState, state: CapsuleState, action: int) -> bool:
        if not state.locked:
            return False
        # Match by occupied cells only; ignore orientation numbering to avoid
        # false negatives due to convention drift. A valid placement must occupy
        # exactly the two cells of the directed edge and be resting on support.
        edge = PLACEMENT_EDGES[int(action)]
        target = {edge.origin, edge.dest}
        cells = {cell for cell in iter_cells(state.row, state.col, state.orient)}
        if not target.issubset(cells):
            return False
        if board.fits(state.row + 1, state.col, state.orient):
            return False
        return True

    def _successors(
        self, board: BoardState, state: CapsuleState
    ) -> Iterator[Tuple[CapsuleState, ControllerStep]]:
        holds = [
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (False, False, True),
            (True, False, True),
            (False, True, True),
        ]
        rotations = [Rotation.NONE, Rotation.CW, Rotation.CCW]
        if self.params.allow_half_rotations:
            rotations.append(Rotation.HALF)
        for hold_left, hold_right, hold_down in holds:
            for rotation in rotations:
                ctrl = ControllerStep(
                    action=self._rotation_to_action(rotation),
                    hold_left=hold_left,
                    hold_right=hold_right,
                    hold_down=hold_down,
                )
                next_state = self._apply_control(board, state, ctrl, rotation)
                if next_state is None:
                    continue
                yield next_state, ctrl

    def _rotation_to_action(self, rotation: Rotation) -> Action:
        if rotation is Rotation.CW:
            return Action.ROTATE_A
        if rotation is Rotation.CCW:
            return Action.ROTATE_B
        if rotation is Rotation.HALF:
            return Action.BOTH_ROT
        return Action.NOOP

    def _apply_control(
        self,
        board: BoardState,
        state: CapsuleState,
        ctrl: ControllerStep,
        rotation: Rotation,
    ) -> Optional[CapsuleState]:
        row, col, orient = state.row, state.col, state.orient
        gravity = max(state.gravity - 1, 0)
        gravity_period = max(state.gravity_period, 1)
        lock_buffer = max(state.lock_buffer - 1, 0)
        grounded = state.grounded
        locked = False

        # Rotation attempt (before horizontal movement)
        if rotation is not Rotation.NONE:
            orient_new = orientation_successor(orient, rotation)
            pivot = (row, col)
            kicks = _ROTATION_KICKS.get((orient, orient_new), ((0, 0),))
            rotated = False
            for kick in kicks:
                candidate_row = pivot[0] + kick[0]
                candidate_col = pivot[1] + kick[1]
                if board.fits(candidate_row, candidate_col, orient_new):
                    row, col, orient = candidate_row, candidate_col, orient_new
                    rotated = True
                    grounded = False
                    lock_buffer = self.params.lock_buffer_frames
                    break
            if not rotated:
                return None

        moved_horizontally = False
        if ctrl.hold_left and not ctrl.hold_right:
            candidate_col = col - 1
            if board.fits(row, candidate_col, orient):
                col = candidate_col
                moved_horizontally = True
        elif ctrl.hold_right and not ctrl.hold_left:
            candidate_col = col + 1
            if board.fits(row, candidate_col, orient):
                col = candidate_col
                moved_horizontally = True

        if moved_horizontally:
            grounded = False
            lock_buffer = self.params.lock_buffer_frames

        if ctrl.hold_down:
            gravity = min(gravity, self.params.soft_drop_gravity)

        # Gravity tick
        if gravity <= 0:
            if board.fits(row + 1, col, orient):
                row += 1
                gravity = gravity_period
                grounded = False
                lock_buffer = max(lock_buffer, 0)
            else:
                grounded = True
                gravity = gravity_period
                if lock_buffer <= 0:
                    lock_buffer = self.params.lock_buffer_frames
                elif rotation is Rotation.NONE and not moved_horizontally:
                    lock_buffer = max(lock_buffer - 1, 0)
        else:
            gravity = gravity
            if (
                grounded
                and lock_buffer == 0
                and not moved_horizontally
                and rotation is Rotation.NONE
            ):
                lock_buffer = 0

        if grounded and lock_buffer == 0 and not board.fits(row + 1, col, orient):
            locked = True

        next_state = CapsuleState(
            row=row,
            col=col,
            orient=orient,
            gravity=max(gravity, 0),
            gravity_period=gravity_period,
            lock_buffer=max(lock_buffer, 0),
            grounded=grounded,
            locked=locked,
            hold_left=ctrl.hold_left,
            hold_right=ctrl.hold_right,
            hold_down=ctrl.hold_down,
            frames=state.frames + 1,
        )
        return next_state

    def _reconstruct_plan(
        self,
        came_from: Dict[Tuple[int, ...], Tuple[Tuple[int, ...], ControllerStep]],
        states: Dict[Tuple[int, ...], CapsuleState],
        key: Tuple[int, ...],
        action: int,
        cost: int,
    ) -> PlanResult:
        controller: List[ControllerStep] = []
        trace: List[CapsuleState] = []
        cur_key = key
        while True:
            state = states[cur_key]
            trace.append(state)
            if cur_key not in came_from:
                break
            prev_key, ctrl = came_from[cur_key]
            controller.append(ctrl)
            cur_key = prev_key
        trace.reverse()
        controller.reverse()
        return PlanResult(action=action, controller=controller, states=trace, cost=cost)


def snapshot_to_capsule_state(snapshot: PillSnapshot) -> CapsuleState:
    """Helper to build an initial :class:`CapsuleState` from a RAM snapshot."""

    return CapsuleState(
        row=snapshot.row,
        col=snapshot.col,
        orient=snapshot.orient,
        gravity=max(0, snapshot.gravity_counter),
        gravity_period=max(1, snapshot.gravity_period),
        lock_buffer=max(0, snapshot.lock_counter // 4),
        grounded=False,
        locked=False,
        hold_left=False,
        hold_right=False,
        hold_down=False,
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
    "snapshot_to_capsule_state",
]
