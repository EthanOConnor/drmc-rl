"""High-level placement action wrapper for :class:`DrMarioRetroEnv`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from time import perf_counter

import envs.specs.ram_to_state as ram_specs
from envs.retro.drmario_env import Action
from envs.retro.placement_actions import action_count, opposite_actions
from envs.retro.placement_planner import (
    BoardState,
    CapsuleState,
    PillSnapshot,
    PlanResult,
    PlacementPlanner,
    PlannerError,
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
        return {
            "placements/legal_mask": self._legal_mask.copy(),
            "placements/feasible_mask": self._feasible_mask.copy(),
            "placements/options": int(self._feasible_mask.sum()),
            "placements/costs": self._costs.copy(),
            "placements/path_indices": self._path_indices.copy(),
        }

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
        info.update(self._translator.info())
        self._last_info = info
        return obs, info

    def step(self, action: int):
        total_reward = 0.0
        terminated = False
        truncated = False
        last_obs = self._last_obs
        last_info: Dict[str, Any] = {}
        replan_attempts = 0

        while True:
            plan = self._translator.get_plan(int(action))
            outcome = self._execute_plan(plan)
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

        self._last_obs = last_obs
        self._last_info = last_info
        return last_obs, total_reward, terminated, truncated, last_info

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _execute_plan(self, plan: Optional[PlanResult]) -> _ExecutionOutcome:
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
                last_info.update(self._translator.info())
            base._hold_left = False
            base._hold_right = False
            base._hold_down = False
            return _ExecutionOutcome(last_obs, last_info, total_reward, terminated, truncated, False)

        states = plan.states
        for idx, ctrl in enumerate(plan.controller):
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
                replan_required = True
                break

        base._hold_left = False
        base._hold_right = False
        base._hold_down = False

        if not (terminated or truncated) and not replan_required:
            self._translator.refresh()
            last_info.update(self._translator.info())

        if replan_required:
            last_info = dict(last_info)
            last_info["placements/replan_triggered"] = 1

        return _ExecutionOutcome(last_obs, last_info, total_reward, terminated, truncated, replan_required)


__all__ = ["DrMarioPlacementEnv"]
