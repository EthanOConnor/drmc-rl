"""High-level placement action wrapper for :class:`DrMarioRetroEnv`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from time import perf_counter

import envs.specs.ram_to_state as ram_specs
from envs.retro.drmario_env import Action
from envs.retro.placement_actions import (
    GRID_HEIGHT,
    GRID_WIDTH,
    PLACEMENT_EDGES,
    action_count,
    opposite_actions,
)
from envs.retro.placement_planner import (
    BoardState,
    CapsuleState,
    PillSnapshot,
    PlanResult,
    PlacementPlanner,
    PlannerError,
    iter_cells,
    snapshot_to_capsule_state,
)


@dataclass
class _ExecutionOutcome:
    last_obs: Any
    info: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    replan_required: bool


@dataclass
class PlannerDebugSnapshot:
    board: BoardState
    pill: Optional[PillSnapshot]
    legal_mask: np.ndarray
    feasible_mask: np.ndarray
    plans: Tuple[PlanResult, ...]
    selected_plan: Optional[PlanResult]
    selected_action: Optional[int]


class PlacementTranslator:
    """Bridges emulator RAM and the placement planner."""

    def __init__(self, env: gym.Env, planner: Optional[PlacementPlanner] = None) -> None:
        self.env = env
        self._planner = planner or PlacementPlanner()
        self._offsets = getattr(env.unwrapped, "_ram_offsets", {})
        self._legal_mask = np.zeros(action_count(), dtype=np.bool_)
        self._feasible_mask = np.zeros_like(self._legal_mask)
        self._costs = np.full(action_count(), np.inf, dtype=np.float32)
        self._path_indices = np.full(action_count(), -1, dtype=np.int32)
        self._paths: Tuple[PlanResult, ...] = tuple()
        self._current_snapshot: Optional[PillSnapshot] = None
        self._board: Optional[BoardState] = None
        self._last_spawn_id: Optional[int] = None
        self._identical_color_pairs: Tuple[int, ...] = tuple()
        self._last_plan_latency: float = 0.0
        self._last_plan_count: int = 0

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        state, ram_bytes = self._read_state()
        board = BoardState.from_state(state)
        pill = self._extract_pill(state, ram_bytes)
        self._board = board
        if pill is None:
            self._current_snapshot = None
            self._last_spawn_id = None
            self._legal_mask[:] = False
            self._feasible_mask[:] = False
            self._paths = tuple()
            self._costs.fill(np.inf)
            self._path_indices.fill(-1)
            self._last_plan_count = 0
            return
        self._current_snapshot = pill
        self._last_spawn_id = pill.spawn_id
        start = perf_counter()
        planner_out = self._planner.plan_all(board, pill)
        self._last_plan_latency = perf_counter() - start
        self._legal_mask = planner_out.legal_mask
        self._feasible_mask = planner_out.feasible_mask
        self._paths = planner_out.plans
        self._costs = planner_out.costs.copy()
        self._path_indices = planner_out.path_indices.copy()
        self._last_plan_count = planner_out.plan_count
        self._mask_identical_colors(pill)

    def info(self) -> Dict[str, Any]:
        info = {
            "placements/legal_mask": self._legal_mask.copy(),
            "placements/feasible_mask": self._feasible_mask.copy(),
            "placements/options": int(self._feasible_mask.sum()),
            "placements/costs": self._costs.copy(),
            "placements/path_indices": self._path_indices.copy(),
        }
        if self._last_spawn_id is not None:
            info["pill/spawn_id"] = int(self._last_spawn_id)
            info["placements/spawn_id"] = int(self._last_spawn_id)
        return info

    def get_plan(self, action: int) -> Optional[PlanResult]:
        idx = int(self._path_indices[int(action)])
        if idx < 0 or idx >= len(self._paths):
            return None
        return self._paths[idx]

    def capture_capsule_state(self) -> Optional[CapsuleState]:
        state, ram_bytes = self._read_state()
        self._board = BoardState.from_state(state)
        pill = self._extract_pill(state, ram_bytes)
        if pill is None:
            return None
        return snapshot_to_capsule_state(pill)

    def current_board(self) -> Optional[BoardState]:
        return self._board

    def current_pill(self) -> Optional[PillSnapshot]:
        return self._current_snapshot

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "plan_latency_ms": self._last_plan_latency * 1000.0,
            "plan_count": self._last_plan_count,
        }

    def debug_snapshot(
        self, selected_action: Optional[int] = None
    ) -> Optional[PlannerDebugSnapshot]:
        if self._board is None:
            return None
        board_copy = BoardState(columns=self._board.columns.copy())
        selected_plan: Optional[PlanResult] = None
        action_value: Optional[int] = None
        if selected_action is not None:
            plan_candidate = self.get_plan(int(selected_action))
            if plan_candidate is not None:
                selected_plan = plan_candidate
                action_value = int(selected_action)
        return PlannerDebugSnapshot(
            board=board_copy,
            pill=self._current_snapshot,
            legal_mask=self._legal_mask.copy(),
            feasible_mask=self._feasible_mask.copy(),
            plans=self._paths,
            selected_plan=selected_plan,
            selected_action=action_value,
        )

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def states_consistent(self, actual: Optional[CapsuleState], expected: CapsuleState) -> bool:
        if actual is None:
            return expected.locked
        return (
            actual.row == expected.row
            and actual.col == expected.col
            and actual.orient == expected.orient
        )

    def _mask_identical_colors(self, pill: PillSnapshot) -> None:
        if pill.colors[0] != pill.colors[1]:
            self._identical_color_pairs = tuple()
            return
        masked = []
        for left, right in opposite_actions():
            left_idx = int(self._path_indices[left])
            right_idx = int(self._path_indices[right])
            left_cost = float(self._costs[left])
            right_cost = float(self._costs[right])
            left_legal = bool(self._legal_mask[left])
            right_legal = bool(self._legal_mask[right])

            if not (left_legal or right_legal):
                continue

            # Prefer whichever direction currently has a feasible plan. When both
            # exist we keep the lower-cost option but remain deterministic by
            # favouring the canonical left->right orientation on ties.
            drop = None
            if left_idx >= 0 and right_idx >= 0:
                if left_cost <= right_cost:
                    drop = right
                else:
                    drop = left
            elif left_idx >= 0:
                drop = right
            elif right_idx >= 0:
                drop = left
            else:
                # No feasible plan yet; fall back to canonical ordering.
                drop = right

            if drop is None:
                continue
            self._feasible_mask[drop] = False
            self._legal_mask[drop] = False
            self._path_indices[drop] = -1
            self._costs[drop] = np.inf
            masked.append(drop)

        self._identical_color_pairs = tuple(masked)
        self._last_plan_count = int(np.count_nonzero(self._path_indices >= 0))

    def _read_state(self) -> Tuple[np.ndarray, bytes]:
        base = self.env.unwrapped
        ram_arr = base._read_ram_array(refresh=True)
        if ram_arr is None:
            ram_arr = np.zeros(0x800, dtype=np.uint8)
        ram_bytes = bytes(ram_arr)
        state = ram_specs.ram_to_state(ram_bytes, self._offsets)
        return state, ram_bytes

    def _extract_pill(self, state: np.ndarray, ram_bytes: bytes) -> Optional[PillSnapshot]:
        falling_mask = ram_specs.get_falling_mask(state)
        if falling_mask.sum() == 0:
            return None
        try:
            return PillSnapshot.from_ram_state(state, ram_bytes, self._offsets)
        except PlannerError:
            return None


class DrMarioPlacementEnv(gym.Wrapper):
    """Wrapper exposing the 464-way placement action space."""

    def __init__(self, env: gym.Env, *, planner: Optional[PlacementPlanner] = None) -> None:
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(action_count())
        self._translator = PlacementTranslator(env, planner)
        self._last_obs: Any = None
        self._last_info: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        if info is None:
            info = {}
        self._translator.refresh()
        obs, info, _, _, _ = self._await_next_pill(obs, info)
        self._last_obs = obs
        self._last_info = info
        return obs, info

    def step(self, action: int):
        total_reward = 0.0
        terminated = False
        truncated = False
        last_obs = self._last_obs
        last_info: Dict[str, Any] = {}
        replan_attempts = 0
        planner_calls = 0
        planner_latency_ms_total = 0.0
        planner_latency_ms_max = 0.0
        planner_plan_count_total = 0.0
        planner_plan_count_last = 0.0
        planner_latency_ms_last = 0.0

        def record_refresh_metrics() -> None:
            nonlocal planner_calls, planner_latency_ms_total, planner_latency_ms_max
            nonlocal planner_plan_count_total, planner_plan_count_last
            diagnostics = self._translator.diagnostics()
            latency = float(diagnostics.get("plan_latency_ms", 0.0))
            plan_count = float(diagnostics.get("plan_count", 0.0))
            planner_calls += 1
            planner_latency_ms_total += latency
            planner_latency_ms_max = max(planner_latency_ms_max, latency)
            planner_plan_count_total += plan_count
            planner_plan_count_last = plan_count
            planner_latency_ms_last = latency

        while True:
            plan = self._translator.get_plan(int(action))
            outcome = self._execute_plan(plan, record_refresh_metrics)
            total_reward += outcome.reward
            last_obs = outcome.last_obs
            last_info = outcome.info
            terminated = outcome.terminated
            truncated = outcome.truncated
            if terminated or truncated:
                break
            if not outcome.replan_required:
                break
            replan_attempts += 1
            if replan_attempts > 3:
                last_info.setdefault("placements/replan_fail", replan_attempts)
                break
            self._translator.refresh()
            record_refresh_metrics()

        self._last_obs = last_obs
        enriched_info = dict(last_info)
        planner_latency_ms_avg = (
            planner_latency_ms_total / planner_calls if planner_calls > 0 else 0.0
        )
        if planner_calls > 0:
            planner_plan_count_avg = planner_plan_count_total / planner_calls
        else:
            planner_plan_count_avg = planner_plan_count_last
        enriched_info["placements/replan_attempts"] = replan_attempts
        enriched_info["placements/plan_calls"] = planner_calls
        enriched_info["placements/plan_latency_ms_total"] = planner_latency_ms_total
        enriched_info["placements/plan_latency_ms_avg"] = planner_latency_ms_avg
        enriched_info["placements/plan_latency_ms_max"] = planner_latency_ms_max
        enriched_info["placements/plan_count_total"] = planner_plan_count_total
        enriched_info["placements/plan_count_avg"] = planner_plan_count_avg
        enriched_info["placements/plan_count_last"] = planner_plan_count_last
        enriched_info["placements/plan_latency_ms_last"] = planner_latency_ms_last
        self._last_info = enriched_info
        return last_obs, total_reward, terminated, truncated, enriched_info

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _execute_plan(
        self,
        plan: Optional[PlanResult],
        record_refresh: Optional[Callable[[], None]] = None,
    ) -> _ExecutionOutcome:
        base = self.env.unwrapped
        total_reward = 0.0
        terminated = False
        truncated = False
        last_obs = self._last_obs
        last_info: Dict[str, Any] = {}
        replan_required = False

        if plan is None:
            obs, reward, terminated, truncated, info = self.env.step(int(Action.NOOP))
            total_reward += float(reward)
            last_obs = obs
            last_info = info or {}
            if not (terminated or truncated):
                self._translator.refresh()
                if record_refresh is not None:
                    record_refresh()
                if self._translator.current_pill() is None:
                    last_obs, last_info, extra_reward, terminated, truncated = (
                        self._await_next_pill(last_obs, last_info, record_refresh)
                    )
                    total_reward += extra_reward
                else:
                    last_info = dict(last_info)
                    last_info.update(self._translator.info())
            base._hold_left = False
            base._hold_right = False
            base._hold_down = False
            return _ExecutionOutcome(
                last_obs, last_info, total_reward, terminated, truncated, False
            )

        states = plan.states
        max_retries = 2
        retry_budget = max_retries
        idx = 0
        while idx < len(plan.controller):
            ctrl = plan.controller[idx]
            base._hold_left = bool(ctrl.hold_left)
            base._hold_right = bool(ctrl.hold_right)
            base._hold_down = bool(ctrl.hold_down)
            obs, reward, terminated, truncated, info = self.env.step(int(ctrl.action))
            total_reward += float(reward)
            last_obs = obs
            last_info = info or {}
            if terminated or truncated:
                break
            expected_state = states[min(idx + 1, len(states) - 1)]
            actual_state = self._translator.capture_capsule_state()
            if not self._translator.states_consistent(actual_state, expected_state):
                previous_state = states[idx]
                if (
                    retry_budget > 0
                    and self._translator.states_consistent(actual_state, previous_state)
                ):
                    retry_budget -= 1
                    continue
                replan_required = True
                break
            retry_budget = max_retries
            idx += 1

        base._hold_left = False
        base._hold_right = False
        base._hold_down = False

        if not (terminated or truncated) and not replan_required:
            self._translator.refresh()
            if record_refresh is not None:
                record_refresh()
            last_obs, last_info, extra_reward, terminated, truncated = self._await_next_pill(
                last_obs, last_info, record_refresh
            )
            total_reward += extra_reward

        if replan_required:
            last_info = dict(last_info)
            last_info["placements/replan_triggered"] = 1

        return _ExecutionOutcome(last_obs, last_info, total_reward, terminated, truncated, replan_required)

    def _await_next_pill(
        self,
        last_obs: Any,
        last_info: Dict[str, Any],
        record_refresh: Optional[Callable[[], None]] = None,
    ) -> Tuple[Any, Dict[str, Any], float, bool, bool]:
        """Advance the emulator until a new pill snapshot becomes available."""

        total_reward = 0.0
        terminated = False
        truncated = False
        obs = last_obs
        info = dict(last_info) if last_info is not None else {}

        if self._translator.current_pill() is not None:
            info.update(self._translator.info())
            return obs, info, total_reward, terminated, truncated

        while True:
            obs, reward, terminated, truncated, step_info = self.env.step(int(Action.NOOP))
            total_reward += float(reward)
            info = step_info or {}
            if terminated or truncated:
                break
            self._translator.refresh()
            if record_refresh is not None:
                record_refresh()
            if self._translator.current_pill() is not None:
                info = dict(info)
                info.update(self._translator.info())
                break

        return obs, info, total_reward, terminated, truncated


def _board_occupancy(board: BoardState) -> np.ndarray:
    occupancy = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.bool_)
    for col in range(GRID_WIDTH):
        mask = int(board.columns[col])
        if mask == 0:
            continue
        for row in range(GRID_HEIGHT):
            if mask & (1 << row):
                occupancy[row, col] = True
    return occupancy


def _blend(base: np.ndarray, color: Tuple[float, float, float], alpha: float) -> np.ndarray:
    return (1.0 - alpha) * base + alpha * np.asarray(color, dtype=np.float32)


def _cells_from_state(state: CapsuleState) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    for row, col in iter_cells(state.row, state.col, state.orient):
        if 0 <= row < GRID_HEIGHT and 0 <= col < GRID_WIDTH:
            cells.append((row, col))
    return cells


def render_planner_debug_view(
    snapshot: PlannerDebugSnapshot,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    board_mask = _board_occupancy(snapshot.board)
    canvas = np.zeros((GRID_HEIGHT, GRID_WIDTH, 3), dtype=np.float32)
    canvas[:] = (24.0, 24.0, 32.0)
    canvas[board_mask] = (118.0, 118.0, 118.0)

    legal_cells = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int32)
    feasible_cells = np.zeros_like(legal_cells)
    for edge in PLACEMENT_EDGES:
        if edge.index >= snapshot.legal_mask.size:
            continue
        row, col = edge.origin
        if not (0 <= row < GRID_HEIGHT and 0 <= col < GRID_WIDTH):
            continue
        if snapshot.legal_mask[edge.index]:
            legal_cells[row, col] += 1
        if snapshot.feasible_mask[edge.index]:
            feasible_cells[row, col] += 1

    legal_mask = legal_cells > 0
    feasible_mask = feasible_cells > 0
    if legal_mask.any():
        canvas[legal_mask] = _blend(canvas[legal_mask], (70.0, 70.0, 200.0), 0.35)
    if feasible_mask.any():
        canvas[feasible_mask] = _blend(canvas[feasible_mask], (40.0, 160.0, 90.0), 0.55)

    plan_to_show: Optional[PlanResult] = snapshot.selected_plan
    plan_action: Optional[int] = snapshot.selected_action
    if plan_to_show is None and snapshot.plans:
        plan_to_show = min(snapshot.plans, key=lambda plan: plan.cost)
        plan_action = plan_to_show.action

    if plan_to_show is not None:
        path_cells = set()
        for state in plan_to_show.states:
            path_cells.update(_cells_from_state(state))
        final_cells = _cells_from_state(plan_to_show.states[-1]) if plan_to_show.states else []
        for row, col in path_cells:
            canvas[row, col] = _blend(canvas[row, col], (255.0, 160.0, 40.0), 0.65)
        for row, col in final_cells:
            canvas[row, col] = (240.0, 32.0, 64.0)

    pill_cells: List[Tuple[int, int]] = []
    if snapshot.pill is not None:
        pill_cells = _cells_from_state(snapshot_to_capsule_state(snapshot.pill))
        for row, col in pill_cells:
            canvas[row, col] = _blend(canvas[row, col], (80.0, 200.0, 255.0), 0.6)

    cell_height = 8
    cell_width = 16
    enlarged = np.repeat(np.repeat(canvas, cell_height, axis=0), cell_width, axis=1)
    image = np.clip(enlarged, 0.0, 255.0).astype(np.uint8)

    stats: Dict[str, Any] = {
        "planner_debug": {
            "legal_count": int(snapshot.legal_mask.sum()),
            "feasible_count": int(snapshot.feasible_mask.sum()),
            "selected_action": None if plan_action is None else int(plan_action),
            "plan_cost": None if plan_to_show is None else int(plan_to_show.cost),
            "plan_steps": None
            if plan_to_show is None
            else int(len(plan_to_show.controller)),
        }
    }
    if snapshot.pill is not None:
        stats["planner_debug"]["pill_row"] = int(snapshot.pill.row)
        stats["planner_debug"]["pill_col"] = int(snapshot.pill.col)
        stats["planner_debug"]["pill_orient"] = int(snapshot.pill.orient)
        stats["planner_debug"]["pill_spawn_id"] = (
            None if snapshot.pill.spawn_id is None else int(snapshot.pill.spawn_id)
        )

    return image, stats


__all__ = [
    "DrMarioPlacementEnv",
    "PlannerDebugSnapshot",
    "render_planner_debug_view",
]
