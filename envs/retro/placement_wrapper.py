"""High-level placement action wrapper for :class:`DrMarioRetroEnv`."""

from __future__ import annotations

import math
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
    ControllerStep,
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
    planner_resynced: bool = False


@dataclass
class PlannerDebugSnapshot:
    board: BoardState
    pill: Optional[PillSnapshot]
    legal_mask: np.ndarray
    feasible_mask: np.ndarray
    plans: Tuple[PlanResult, ...]
    selected_plan: Optional[PlanResult]
    selected_action: Optional[int]
    state: Optional[np.ndarray] = None


class _InputTimingCalibrator:
    """Adaptive retry budgets for input timing mismatches."""

    def __init__(self, *, smoothing: float = 0.25, max_frames: int = 6) -> None:
        self._smoothing = float(np.clip(smoothing, 0.0, 1.0)) if smoothing > 0 else 0.0
        self._max_frames = max(1, int(max_frames))
        # Start with conservative defaults matching the legacy retry budget of two
        # additional frames (three total attempts per command).
        self._horizontal_frames = 3.0
        self._down_frames = 3.0

    @staticmethod
    def _is_horizontal(ctrl: ControllerStep) -> bool:
        return bool(ctrl.hold_left) ^ bool(ctrl.hold_right)

    @staticmethod
    def _is_soft_drop(ctrl: ControllerStep) -> bool:
        return bool(ctrl.hold_down)

    def _ema(self, current: float, observed: float) -> float:
        if self._smoothing <= 0.0:
            return observed
        return (1.0 - self._smoothing) * current + self._smoothing * observed

    def retry_budget_for(self, ctrl: ControllerStep) -> int:
        frames = 1.0
        if self._is_horizontal(ctrl):
            frames = self._horizontal_frames
        elif self._is_soft_drop(ctrl):
            frames = self._down_frames
        budget = int(math.ceil(min(self._max_frames, max(1.0, frames))) - 1)
        return max(0, budget)

    def observe_success(self, ctrl: ControllerStep, frames_used: int) -> None:
        if frames_used <= 0:
            return
        observed = float(min(self._max_frames, max(1, frames_used)))
        if self._is_horizontal(ctrl):
            self._horizontal_frames = self._ema(self._horizontal_frames, observed)
        elif self._is_soft_drop(ctrl):
            self._down_frames = self._ema(self._down_frames, observed)

    def observe_failure(self, ctrl: ControllerStep) -> None:
        increment = 1.0
        if self._is_horizontal(ctrl):
            self._horizontal_frames = min(self._max_frames, self._horizontal_frames + increment)
        elif self._is_soft_drop(ctrl):
            self._down_frames = min(self._max_frames, self._down_frames + increment)

    def info(self) -> Dict[str, float]:
        return {
            "placements/timing/frames_horizontal": float(self._horizontal_frames),
            "placements/timing/frames_down": float(self._down_frames),
            "placements/timing/retry_horizontal": float(self.retry_budget_for(ControllerStep(hold_left=True))),
            "placements/timing/retry_down": float(self.retry_budget_for(ControllerStep(hold_down=True))),
        }


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
        self._last_state: Optional[np.ndarray] = None
        size_hex = self._offsets.get("falling_pill", {}).get("size_addr")
        try:
            self._falling_size_addr: Optional[int] = int(size_hex, 16) if size_hex else None
        except (TypeError, ValueError):
            self._falling_size_addr = None
        self._last_spawn_id: Optional[int] = None
        self._identical_color_pairs: Tuple[int, ...] = tuple()
        self._last_plan_latency_ms: float = 0.0
        self._last_plan_count: int = 0
        self._cached_spawn_marker: Optional[int] = None
        self._spawn_generation: int = -1
        self._options_prepared: bool = False
        self._timing = _InputTimingCalibrator()

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        try:
            state, ram_bytes = self._read_state()
        except Exception:
            return
        try:
            board = BoardState.from_state(state)
        except Exception:
            board = None
        if board is not None:
            self._board = board
        self._last_state = np.array(state, copy=True)
        pill = self._extract_pill(state, ram_bytes, require_mask=False)
        if pill is None:
            self._current_snapshot = None
            self._last_spawn_id = None
            self._clear_cached_options()
            return
        self._current_snapshot = pill
        try:
            self._last_spawn_id = int(pill.spawn_id) if pill.spawn_id is not None else None
        except Exception:
            self._last_spawn_id = None
        if not self._spawn_matches_cache(pill):
            self._spawn_generation += 1
            self._clear_cached_options()
        else:
            # Cheap refresh does not trigger planning; ensure diagnostics reflect that.
            self._last_plan_latency_ms = 0.0
            self._last_plan_count = 0

    def refresh_state_only(self) -> None:
        """Deprecated shim; :meth:`refresh` is already cheap."""

        self.refresh()

    def prepare_options(self, *, force: bool = False) -> None:
        """Ensure placement options are prepared for the current spawn."""

        pill = self._current_snapshot
        board = self._board
        if pill is None or board is None:
            self._clear_cached_options()
            return
        spawn_marker = self._spawn_marker_for(pill)
        if not force and self._options_prepared and self._cached_spawn_marker == spawn_marker:
            return
        start = perf_counter()
        planner_out = self._planner.plan_all(board, pill)
        self._last_plan_latency_ms = (perf_counter() - start) * 1000.0
        self._legal_mask = planner_out.legal_mask.copy()
        self._feasible_mask = planner_out.feasible_mask.copy()
        self._paths = planner_out.plans
        self._costs = planner_out.costs.copy()
        self._path_indices = planner_out.path_indices.copy()
        self._cached_spawn_marker = spawn_marker
        self._options_prepared = True
        self._last_plan_count = int(planner_out.plan_count)
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
        info.update(self._timing.info())
        return info

    def get_plan(self, action: int) -> Optional[PlanResult]:
        # Lazily prepare options if they have not been computed yet for this spawn.
        self.prepare_options()
        idx = int(self._path_indices[int(action)])
        if idx < 0 or idx >= len(self._paths):
            return None
        return self._paths[idx]

    def capture_capsule_state(self) -> Optional[CapsuleState]:
        ram_bytes = self._read_ram_bytes()
        if ram_bytes is None:
            return None
        if self._falling_size_addr is not None:
            if self._falling_size_addr >= len(ram_bytes):
                return None
            size_val = ram_bytes[self._falling_size_addr]
            if size_val < 2:
                return None
        pill = self._extract_pill(None, ram_bytes, require_mask=False)
        if pill is None:
            return None
        return snapshot_to_capsule_state(pill)

    def current_board(self) -> Optional[BoardState]:
        return self._board

    def current_pill(self) -> Optional[PillSnapshot]:
        return self._current_snapshot

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "plan_latency_ms": self._last_plan_latency_ms,
            "plan_count": self._last_plan_count,
        }

    def retry_budget(self, ctrl: ControllerStep) -> int:
        return self._timing.retry_budget_for(ctrl)

    def record_timing_success(self, ctrl: ControllerStep, frames_used: int) -> None:
        self._timing.observe_success(ctrl, frames_used)

    def record_timing_failure(self, ctrl: ControllerStep) -> None:
        self._timing.observe_failure(ctrl)

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
            state=None if self._last_state is None else np.array(self._last_state, copy=True),
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

    def _read_ram_bytes(self) -> Optional[bytes]:
        base = self.env.unwrapped
        ram_arr = base._read_ram_array(refresh=True)
        if ram_arr is None:
            return None
        return bytes(ram_arr)

    def _read_state(self) -> Tuple[np.ndarray, bytes]:
        ram_bytes = self._read_ram_bytes()
        if ram_bytes is None:
            ram_arr = np.zeros(0x800, dtype=np.uint8)
            ram_bytes = bytes(ram_arr)
        state = ram_specs.ram_to_state(ram_bytes, self._offsets)
        return state, ram_bytes

    def _extract_pill(
        self,
        state: Optional[np.ndarray],
        ram_bytes: bytes,
        *,
        require_mask: bool = True,
    ) -> Optional[PillSnapshot]:
        if require_mask and state is not None:
            falling_mask = ram_specs.get_falling_mask(state)
            if falling_mask.sum() == 0:
                return None
        try:
            return PillSnapshot.from_ram_state(state, ram_bytes, self._offsets)
        except PlannerError:
            return None

    def _spawn_marker_for(self, pill: PillSnapshot) -> int:
        if pill.spawn_id is not None:
            try:
                return int(pill.spawn_id)
            except Exception:
                pass
        return int(self._spawn_generation)

    def _spawn_matches_cache(self, pill: PillSnapshot) -> bool:
        if self._cached_spawn_marker is None:
            return False
        return self._cached_spawn_marker == self._spawn_marker_for(pill)

    def _clear_cached_options(self) -> None:
        self._legal_mask[:] = False
        self._feasible_mask[:] = False
        self._costs.fill(np.inf)
        self._path_indices.fill(-1)
        self._paths = tuple()
        self._identical_color_pairs = tuple()
        self._options_prepared = False
        self._cached_spawn_marker = None
        self._last_plan_latency_ms = 0.0
        self._last_plan_count = 0


class DrMarioPlacementEnv(gym.Wrapper):
    """Wrapper exposing the 464-way placement action space (spawn-latched)."""

    def __init__(self, env: gym.Env, *, planner: Optional[PlacementPlanner] = None) -> None:
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(action_count())
        self._translator = PlacementTranslator(env, planner)
        self._last_obs: Any = None
        self._last_info: Dict[str, Any] = {}
        self._active_plan: Optional[PlanResult] = None
        self._latched_action: Optional[int] = None
        self._latched_spawn_id: int = -1
        self._spawn_id: int = 0
        self._capsule_present: bool = False
        self._spawn_marker: Optional[int] = None
        self._step_callback: Optional[
            Callable[[Any, Dict[str, Any], Optional[int], float, bool, bool], None]
        ] = None

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        if info is None:
            info = {}
        self._translator.refresh()
        self._active_plan = None
        self._latched_action = None
        self._latched_spawn_id = -1
        self._capsule_present = False
        self._spawn_marker = None
        self._spawn_id = 0
        obs, info, _, _, _ = self._await_next_pill(obs, info)
        self._last_obs = obs
        self._last_info = info
        self._capsule_present = bool(self._translator.current_pill() is not None)
        # Our spawn_id is purely internal and monotonic; do not overwrite from info.
        return obs, info

    def step(self, action: int):
        # --- Always resync snapshot at step start (fixes “one inference per run”) ---
        self._translator.refresh()

        # Allow a deliberate override to clear the execution latch for the same spawn.
        if (
            self._active_plan is not None
            and self._latched_spawn_id == self._spawn_id
            and self._latched_action is not None
            and int(action) != self._latched_action
        ):
            self._clear_latch()

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
            nonlocal planner_plan_count_total, planner_plan_count_last, planner_latency_ms_last
            diagnostics = self._translator.diagnostics() or {}
            latency = float(diagnostics.get("plan_latency_ms", 0.0) or 0.0)
            plan_count = float(diagnostics.get("plan_count", 0.0) or 0.0)
            if plan_count == 0.0 and latency == 0.0:
                return
            planner_calls += 1
            planner_latency_ms_total += latency
            planner_latency_ms_max = max(planner_latency_ms_max, latency)
            planner_plan_count_total += plan_count
            planner_plan_count_last = plan_count
            planner_latency_ms_last = latency

        record_refresh_metrics()

        def _read_spawn_marker() -> Optional[int]:
            snap = self._translator.current_pill()
            if snap is None:
                return None
            info_t = self._translator.info() or {}
            marker = info_t.get("placements/spawn_id", getattr(snap, "spawn_id", None))
            try:
                return int(marker) if marker is not None else None
            except Exception:
                return None

        def _mark_needs_action(base: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            out = dict(base or {})
            out["placements/needs_action"] = True
            out["placements/spawn_id"] = int(self._spawn_id)
            out.setdefault("pill_changed", 0)
            return out

        if self._translator.current_pill() is None:
            obs, info, reward_delta, terminated, truncated = self._await_next_pill(
                self._last_obs, self._last_info, record_refresh_metrics
            )
            total_reward += reward_delta
            self._last_obs, self._last_info = obs, info
            self._clear_latch()
            return obs, total_reward, terminated, truncated, info

        def request_new_decision(base_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            info_payload = _mark_needs_action(base_info)
            info_payload["pill_changed"] = 0
            for key in (
                "placements/legal_mask",
                "placements/feasible_mask",
                "placements/options",
                "placements/costs",
                "placements/path_indices",
            ):
                info_payload.pop(key, None)
            self._translator.prepare_options(force=True)
            record_refresh_metrics()
            info_payload.update(self._translator.info() or {})
            return info_payload

        while True:
            if self._active_plan is None or self._latched_spawn_id != self._spawn_id:
                plan = self._translator.get_plan(int(action))
                if plan is None:
                    last_obs = self._last_obs
                    last_info = request_new_decision(self._last_info)
                    terminated = False
                    truncated = False
                    self._clear_latch()
                    break
                self._active_plan = plan
                self._latched_action = int(action)
                self._latched_spawn_id = int(self._spawn_id)
            else:
                plan = self._active_plan

            outcome = self._execute_plan(plan, record_refresh_metrics)
            total_reward += outcome.reward
            last_obs = outcome.last_obs
            last_info = outcome.info
            terminated = outcome.terminated
            truncated = outcome.truncated
            # After executing a (multi-frame) plan, RAM has advanced; resync translator snapshot.
            self._translator.refresh()
            record_refresh_metrics()

            pill_now = self._translator.current_pill() is not None

            if terminated or truncated:
                self._clear_latch()
                break
            if outcome.replan_required:
                last_info = dict(last_info)
                last_info["placements/feasible_fp"] = int(
                    last_info.get("placements/feasible_fp", 0)
                ) + 1
                failed_idx = self._latched_action if self._latched_action is not None else action
                last_info["placements/failed_action_idx"] = int(failed_idx)
                replan_attempts += 1
                last_info = request_new_decision(last_info)
                self._clear_latch()
                break

            # Detect next-spawn arrival during/after plan execution.
            marker_now = _read_spawn_marker() if pill_now else None
            new_spawn = (
                pill_now
                and marker_now is not None
                and self._spawn_marker is not None
                and marker_now != self._spawn_marker
            )
            if new_spawn:
                # Bump OUR counter; keep translator marker only for detection.
                self._spawn_marker = marker_now
                self._spawn_id += 1
                self._capsule_present = True
                self._translator.prepare_options(force=True)
                record_refresh_metrics()
                info_now = _mark_needs_action(self._translator.info())
                info_now["pill_changed"] = 1
                last_info = {**last_info, **info_now}
                self._clear_latch()
                break

            # Happy path: plan executed; we're done for this env.step().
            if pill_now and marker_now is not None and self._spawn_marker is None:
                self._spawn_marker = marker_now
            self._capsule_present = pill_now
            break

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
        # Do NOT force a decision just because a pill exists; only on spawn/replan.
        if "placements/needs_action" not in enriched_info:
            enriched_info["placements/needs_action"] = False
        enriched_info.setdefault("placements/spawn_id", int(self._spawn_id))
        return last_obs, total_reward, terminated, truncated, enriched_info

    # ------------------------------------------------------------------
    # Viewer callback plumbing
    # ------------------------------------------------------------------

    def set_step_callback(
        self,
        callback: Optional[Callable[[Any, Dict[str, Any], Optional[int], float, bool, bool], None]],
    ) -> None:
        self._step_callback = callback

    def _notify_step_callback(
        self,
        obs: Any,
        info: Optional[Dict[str, Any]],
        action: Optional[int],
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if self._step_callback is None:
            return
        payload = dict(info or {})
        try:
            self._step_callback(
                obs,
                payload,
                None if action is None else int(action),
                float(reward),
                bool(terminated),
                bool(truncated),
            )
        except Exception:
            pass

    def _clear_latch(self) -> None:
        self._active_plan = None
        self._latched_action = None
        self._latched_spawn_id = -1

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
        planner_resynced = False

        if plan is None:
            obs, reward, terminated, truncated, info = self.env.step(int(Action.NOOP))
            total_reward += float(reward)
            last_obs = obs
            last_info = dict(info or {})
            if not (terminated or truncated):
                self._translator.refresh()
                if record_refresh is not None:
                    record_refresh()
                if self._step_callback is not None:
                    self._translator.refresh_state_only()
                if self._translator.current_pill() is None:
                    last_obs, last_info, extra_reward, terminated, truncated = (
                        self._await_next_pill(last_obs, last_info, record_refresh)
                    )
                    total_reward += extra_reward
                else:
                    extra_info = self._translator.info() or {}
                    if extra_info:
                        last_info.update(extra_info)
                    last_info["placements/needs_action"] = False
                    last_info.setdefault("pill_changed", 0)
            self._notify_step_callback(
                last_obs,
                last_info,
                int(Action.NOOP),
                float(reward),
                terminated,
                truncated,
            )
            base._hold_left = False
            base._hold_right = False
            base._hold_down = False
            last_info.setdefault("pill_changed", 0)
            return _ExecutionOutcome(
                last_obs,
                last_info,
                total_reward,
                terminated,
                truncated,
                False,
                False,
            )

        states = plan.states
        idx = 0
        while idx < len(plan.controller):
            ctrl = plan.controller[idx]
            retry_budget = self._translator.retry_budget(ctrl)
            frames_for_step = 0
            previous_state = states[idx]
            expected_state = states[min(idx + 1, len(states) - 1)]
            while True:
                frames_for_step += 1
                base._hold_left = bool(ctrl.hold_left)
                base._hold_right = bool(ctrl.hold_right)
                base._hold_down = bool(ctrl.hold_down)
                obs, reward, terminated, truncated, info = self.env.step(int(ctrl.action))
                total_reward += float(reward)
                last_obs = obs
                last_info = info or {}
                info_for_cb = dict(last_info)
                if self._step_callback is not None:
                    self._translator.refresh_state_only()
                    extra_info = self._translator.info() or {}
                    if extra_info:
                        info_for_cb.update(extra_info)
                    self._notify_step_callback(
                        obs,
                        info_for_cb,
                        int(ctrl.action),
                        float(reward),
                        terminated,
                        truncated,
                    )
                if terminated or truncated:
                    break
                actual_state = self._translator.capture_capsule_state()
                if self._translator.states_consistent(actual_state, expected_state):
                    if frames_for_step > 0:
                        self._translator.record_timing_success(ctrl, frames_for_step)
                    break
                if (
                    retry_budget > 0
                    and self._translator.states_consistent(actual_state, previous_state)
                ):
                    retry_budget -= 1
                    continue
                self._translator.record_timing_failure(ctrl)
                replan_required = True
                break
            if terminated or truncated or replan_required:
                break
            idx += 1

        base._hold_left = False
        base._hold_right = False
        base._hold_down = False

        if not (terminated or truncated) and not replan_required:
            self._translator.refresh()
            if record_refresh is not None:
                record_refresh()
            # Wait until the current capsule locks before surfacing the next spawn.
            stall_checks = 0
            while self._translator.current_pill() is not None:
                obs, reward, terminated, truncated, info = self.env.step(int(Action.NOOP))
                total_reward += float(reward)
                last_obs = obs
                last_info = info or {}
                if self._step_callback is not None:
                    self._translator.refresh_state_only()
                    info_for_cb = dict(last_info)
                    info_for_cb.update(self._translator.info() or {})
                    self._notify_step_callback(
                        obs,
                        info_for_cb,
                        int(Action.NOOP),
                        float(reward),
                        terminated,
                        truncated,
                    )
                if terminated or truncated:
                    break
                self._translator.refresh()
                if record_refresh is not None:
                    record_refresh()
                stall_checks += 1
                if stall_checks > 180 and self._translator.current_pill() is not None:
                    replan_required = True
                    break

            if not (terminated or truncated) and not replan_required:
                last_obs, last_info, extra_reward, terminated, truncated = self._await_next_pill(
                    last_obs, last_info, record_refresh
                )
                total_reward += extra_reward

        if replan_required:
            last_info = dict(last_info)
            last_info["placements/replan_triggered"] = 1
            if not (terminated or truncated):
                self._translator.refresh()
                self._translator.prepare_options(force=True)
                planner_resynced = True
                if record_refresh is not None:
                    record_refresh()

        if "pill_changed" not in last_info:
            last_info = dict(last_info)
            last_info.setdefault("pill_changed", 0)

        return _ExecutionOutcome(
            last_obs,
            last_info,
            total_reward,
            terminated,
            truncated,
            replan_required,
            planner_resynced,
        )

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
            if not self._capsule_present:
                self._spawn_id += 1
            self._capsule_present = True
            self._spawn_marker = None
            self._translator.prepare_options(force=True)
            if record_refresh is not None:
                record_refresh()
            try:
                self._spawn_marker = int((self._translator.info() or {}).get(
                    "placements/spawn_id", getattr(self._translator.current_pill(), "spawn_id", None)
                ))
            except Exception:
                pass
            info.update(self._translator.info() or {})
            info["placements/spawn_id"] = int(self._spawn_id)
            info["placements/needs_action"] = True
            info.setdefault("pill_changed", 1)
            return obs, info, total_reward, terminated, truncated

        self._capsule_present = False
        while True:
            obs, reward, terminated, truncated, step_info = self.env.step(int(Action.NOOP))
            total_reward += float(reward)
            info = step_info or {}
            if terminated or truncated:
                self._notify_step_callback(
                    obs,
                    info,
                    int(Action.NOOP),
                    float(reward),
                    terminated,
                    truncated,
                )
                break
            self._translator.refresh()
            if record_refresh is not None:
                record_refresh()
            if self._step_callback is not None:
                info = dict(info)
                info.update(self._translator.info() or {})
                self._translator.refresh_state_only()
                self._notify_step_callback(
                    obs,
                    info,
                    int(Action.NOOP),
                    float(reward),
                    terminated,
                    truncated,
                )
            if self._translator.current_pill() is not None:
                info = dict(info)
                self._translator.prepare_options(force=True)
                if record_refresh is not None:
                    record_refresh()
                info.update(self._translator.info() or {})
                if not self._capsule_present:
                    self._spawn_id += 1
                self._capsule_present = True
                # Capture translator marker at the moment of spawn detection.
                self._spawn_marker = None
                try:
                    self._spawn_marker = int((self._translator.info() or {}).get(
                        "placements/spawn_id", getattr(self._translator.current_pill(), "spawn_id", None)
                    ))
                except Exception:
                    pass
                info["placements/spawn_id"] = int(self._spawn_id)
                info["placements/needs_action"] = True
                info.setdefault("pill_changed", 1)
                break

        info.setdefault("pill_changed", 0)
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


_BACKGROUND_COLOR = np.array((24.0, 24.0, 32.0), dtype=np.float32)
_OCCUPIED_FALLBACK_COLOR = np.array((118.0, 118.0, 118.0), dtype=np.float32)
_STATIC_RGB = np.array(((180.0, 0.0, 0.0), (200.0, 180.0, 0.0), (0.0, 80.0, 200.0)), dtype=np.float32)
_VIRUS_RGB = np.array(((220.0, 40.0, 40.0), (240.0, 220.0, 40.0), (40.0, 120.0, 240.0)), dtype=np.float32)
_FALLING_RGB = np.array(((255.0, 128.0, 128.0), (255.0, 255.0, 120.0), (120.0, 120.0, 255.0)), dtype=np.float32)
_FALLING_COLOR_LOOKUP = {
    0: _FALLING_RGB[1],  # yellow
    1: _FALLING_RGB[0],  # red
    2: _FALLING_RGB[2],  # blue
}
_LEGAL_ONLY_COLOR = (150.0, 235.0, 170.0)
_FEASIBLE_COLOR = (40.0, 160.0, 90.0)
_PATH_COLOR = (255.0, 160.0, 40.0)
_TARGET_COLOR = (250.0, 250.0, 250.0)
_CELL_SCALE = 6


def _board_color_grid(state: Optional[np.ndarray]) -> np.ndarray:
    colors = np.zeros((GRID_HEIGHT, GRID_WIDTH, 3), dtype=np.float32)
    if state is None:
        return colors
    try:
        static_planes = ram_specs.get_static_color_planes(state)
        virus_planes = ram_specs.get_virus_color_planes(state)
    except Exception:
        return colors
    for idx, rgb in enumerate(_STATIC_RGB):
        if idx >= static_planes.shape[0]:
            break
        mask = static_planes[idx] > 0.1
        if mask.any():
            colors[mask] = rgb
    for idx, rgb in enumerate(_VIRUS_RGB):
        if idx >= virus_planes.shape[0]:
            break
        mask = virus_planes[idx] > 0.1
        if mask.any():
            colors[mask] = rgb
    return colors


def _apply_falling_colors(canvas: np.ndarray, state: Optional[np.ndarray]) -> None:
    if state is None:
        return
    try:
        falling_planes = ram_specs.get_falling_color_planes(state)
    except Exception:
        return
    for idx, rgb in enumerate(_FALLING_RGB):
        if idx >= falling_planes.shape[0]:
            break
        mask = falling_planes[idx] > 0.1
        if mask.any():
            canvas[mask] = rgb


def render_planner_debug_view(
    snapshot: PlannerDebugSnapshot,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    board_mask = _board_occupancy(snapshot.board)
    canvas = np.zeros((GRID_HEIGHT, GRID_WIDTH, 3), dtype=np.float32)
    canvas[:] = _BACKGROUND_COLOR

    board_colors = _board_color_grid(snapshot.state)
    colored_mask = board_colors.sum(axis=-1) > 0.1
    if colored_mask.any():
        canvas[colored_mask] = board_colors[colored_mask]
    fallback_mask = np.logical_and(board_mask, np.logical_not(colored_mask))
    if fallback_mask.any():
        canvas[fallback_mask] = _OCCUPIED_FALLBACK_COLOR

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
    legal_only_mask = np.logical_and(legal_mask, np.logical_not(feasible_mask))
    if legal_only_mask.any():
        canvas[legal_only_mask] = _blend(canvas[legal_only_mask], _LEGAL_ONLY_COLOR, 0.5)
    if feasible_mask.any():
        canvas[feasible_mask] = _blend(canvas[feasible_mask], _FEASIBLE_COLOR, 0.55)

    plan_to_show: Optional[PlanResult] = snapshot.selected_plan
    plan_action: Optional[int] = snapshot.selected_action
    if plan_to_show is None and snapshot.plans:
        plan_to_show = min(snapshot.plans, key=lambda plan: plan.cost)
        plan_action = plan_to_show.action

    final_cells: List[Tuple[int, int]] = []
    if plan_to_show is not None:
        path_cells = set()
        for state in plan_to_show.states:
            path_cells.update(_cells_from_state(state))
        if plan_to_show.states:
            final_cells = _cells_from_state(plan_to_show.states[-1])
        for row, col in path_cells:
            canvas[row, col] = _blend(canvas[row, col], _PATH_COLOR, 0.65)

    _apply_falling_colors(canvas, snapshot.state)
    if snapshot.state is None and snapshot.pill is not None:
        pill_state = snapshot_to_capsule_state(snapshot.pill)
        pill_cells = _cells_from_state(pill_state)
        for index, (row, col) in enumerate(pill_cells):
            color_value = snapshot.pill.colors[min(index, len(snapshot.pill.colors) - 1)]
            tint = _FALLING_COLOR_LOOKUP.get(int(color_value))
            if tint is None:
                continue
            canvas[row, col] = tint

    if final_cells:
        for row, col in final_cells:
            canvas[row, col] = _TARGET_COLOR

    enlarged = np.repeat(np.repeat(canvas, _CELL_SCALE, axis=0), _CELL_SCALE, axis=1)
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
