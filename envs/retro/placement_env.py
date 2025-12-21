"""Gym wrapper exposing spawn-latched macro placement actions (512-way).

`DrMarioRetroEnv` is a per-frame controller environment. This wrapper turns it
into a semi-Markov decision process (SMDP) where each `step(action)` selects a
*macro placement* for the currently falling pill:

    action ∈ [0, 512)  <->  (o,row,col) in a 4×16×8 grid

and the environment executes the minimal-time controller script to realize that
placement (or rejects it if infeasible).

The wrapper returns control at the next decision point: when the next pill
enters `nextAction_pillFalling` (i.e., the player can control the new pill).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

import envs.specs.ram_to_state as ram_specs
from envs.retro.drmario_env import Action
from envs.state_core import DrMarioState
from envs.retro.placement_planner import (
    BoardState,
    ControllerStep,
    PillSnapshot,
    PlacementPlanner,
    PlannerError,
    SpawnReachability,
)
from envs.retro.placement_space import ORIENTATIONS, GRID_HEIGHT, GRID_WIDTH, TOTAL_ACTIONS


# Zero-page current-player address for `currentP_nextAction` (drmario_ram_zp.asm).
ZP_CURRENT_P_NEXT_ACTION = 0x0097
NEXT_ACTION_PILL_FALLING = 0  # jumpTable_nextAction index 0

# Falling pill pose registers (zero-page current-player mirrors).
# These are the canonical inputs to the reachability planner and update every frame
# while `currentP_nextAction == nextAction_pillFalling`.
ZP_FALLING_PILL_X = 0x0085  # column (0..7)
ZP_FALLING_PILL_Y = 0x0086  # row from bottom (0=bottom)
ZP_FALLING_PILL_ROT = 0x00A5  # rotation code (0..3)
ZP_SPEED_COUNTER = 0x0092
ZP_HOR_VELOCITY = 0x0093
ZP_FRAME_COUNTER = 0x0043


def _read_u8(ram_bytes: bytes, addr: int) -> int:
    if addr < 0 or addr >= len(ram_bytes):
        return 0
    return int(ram_bytes[addr]) & 0xFF


def _read_falling_pose(ram_bytes: bytes) -> Optional[Tuple[int, int, int]]:
    """Return the falling pill pose (base_col, base_row_top, rot), or None if invalid."""

    raw_row_from_bottom = _read_u8(ram_bytes, ZP_FALLING_PILL_Y)
    base_row = (GRID_HEIGHT - 1) - int(raw_row_from_bottom)
    base_col = _read_u8(ram_bytes, ZP_FALLING_PILL_X)
    rot = _read_u8(ram_bytes, ZP_FALLING_PILL_ROT) & 0x03
    if not (0 <= base_row < GRID_HEIGHT and 0 <= base_col < GRID_WIDTH):
        return None
    return int(base_col), int(base_row), int(rot)


@dataclass
class _DecisionContext:
    snapshot: PillSnapshot
    board: BoardState
    reach: SpawnReachability
    planner_build_sec: float = 0.0


@dataclass
class _RewardBreakdown:
    """Accumulate reward components across multiple underlying per-frame steps.

    The placement macro environment executes a per-frame controller script for a
    chosen macro action. For debugging and curriculum/metrics purposes, it's
    useful to also aggregate the underlying environment's *reward components*
    (counts + per-term contributions) across the entire macro step.
    """

    r_total: float = 0.0
    r_env: float = 0.0
    r_shape: float = 0.0

    # Counts
    delta_v: int = 0
    tiles_cleared_total: int = 0
    tiles_cleared_non_virus: int = 0
    tiles_clearing_max: int = 0
    action_events: int = 0
    pill_locks: int = 0

    # Reward terms (already scaled)
    virus_clear_reward: float = 0.0
    non_virus_bonus: float = 0.0
    adjacency_bonus: float = 0.0
    virus_adjacency_bonus: float = 0.0
    pill_bonus_adjusted: float = 0.0
    height_penalty_delta: float = 0.0
    action_penalty: float = 0.0
    terminal_bonus: float = 0.0
    topout_penalty: float = 0.0
    time_reward: float = 0.0

    @staticmethod
    def _as_int(value: Any) -> int:
        if value is None:
            return 0
        try:
            if isinstance(value, np.ndarray):
                value = value.item()
        except Exception:
            return 0
        try:
            return int(value)
        except Exception:
            return 0

    @staticmethod
    def _as_float(value: Any) -> float:
        if value is None:
            return 0.0
        try:
            if isinstance(value, np.ndarray):
                value = value.item()
        except Exception:
            return 0.0
        try:
            return float(value)
        except Exception:
            return 0.0

    def add_frame(self, *, reward: float, info: Dict[str, Any]) -> None:
        self.r_total += float(reward)
        self.r_env += self._as_float(info.get("r_env"))
        self.r_shape += self._as_float(info.get("r_shape"))

        dv = self._as_int(info.get("delta_v"))
        self.delta_v += int(max(0, dv))
        self.tiles_cleared_total += int(max(0, self._as_int(info.get("tiles_cleared_total"))))
        self.tiles_cleared_non_virus += int(max(0, self._as_int(info.get("tiles_cleared_non_virus"))))
        self.tiles_clearing_max = max(self.tiles_clearing_max, self._as_int(info.get("tiles_clearing")))
        self.action_events += int(max(0, self._as_int(info.get("action_events"))))

        pill_bonus = self._as_float(info.get("pill_bonus_adjusted"))
        self.pill_bonus_adjusted += pill_bonus
        if pill_bonus > 0.0:
            self.pill_locks += 1

        self.virus_clear_reward += self._as_float(info.get("virus_clear_reward"))
        self.non_virus_bonus += self._as_float(info.get("non_virus_bonus"))
        self.adjacency_bonus += self._as_float(info.get("adjacency_bonus"))
        self.virus_adjacency_bonus += self._as_float(info.get("virus_adjacency_bonus"))
        self.height_penalty_delta += self._as_float(info.get("height_penalty_delta"))
        self.action_penalty += self._as_float(info.get("action_penalty"))
        self.terminal_bonus += self._as_float(info.get("terminal_bonus_reward"))
        self.topout_penalty += self._as_float(info.get("topout_penalty_reward"))
        self.time_reward += self._as_float(info.get("time_reward"))


@dataclass
class _PerfBreakdown:
    """Accumulate per-frame timing info from the underlying per-frame env.

    The placement environment executes many underlying `env.step()` calls per
    macro decision. The base env (DrMarioRetroEnv) emits `perf/env_*_sec` keys
    per frame; we sum them here and surface the totals on the macro step info
    so UI-side perf accounting can attribute time correctly.
    """

    env_step_calls: int = 0

    env_step_sec: float = 0.0
    env_backend_step_sec: float = 0.0
    env_get_frame_sec: float = 0.0
    env_pixel_stack_sec: float = 0.0
    env_get_ram_sec: float = 0.0
    env_ram_bytes_sec: float = 0.0
    env_build_state_sec: float = 0.0
    env_observe_sec: float = 0.0
    env_reward_sec: float = 0.0
    env_info_sec: float = 0.0

    @staticmethod
    def _as_float(value: Any) -> float:
        if value is None:
            return 0.0
        try:
            if isinstance(value, np.ndarray):
                value = value.item()
        except Exception:
            return 0.0
        try:
            return float(value)
        except Exception:
            return 0.0

    def add_frame(self, *, info: Dict[str, Any]) -> None:
        self.env_step_calls += 1
        self.env_step_sec += self._as_float(info.get("perf/env_step_sec"))
        self.env_backend_step_sec += self._as_float(info.get("perf/env_backend_step_sec"))
        self.env_get_frame_sec += self._as_float(info.get("perf/env_get_frame_sec"))
        self.env_pixel_stack_sec += self._as_float(info.get("perf/env_pixel_stack_sec"))
        self.env_get_ram_sec += self._as_float(info.get("perf/env_get_ram_sec"))
        self.env_ram_bytes_sec += self._as_float(info.get("perf/env_ram_bytes_sec"))
        self.env_build_state_sec += self._as_float(info.get("perf/env_build_state_sec"))
        self.env_observe_sec += self._as_float(info.get("perf/env_observe_sec"))
        self.env_reward_sec += self._as_float(info.get("perf/env_reward_sec"))
        self.env_info_sec += self._as_float(info.get("perf/env_info_sec"))


class DrMarioPlacementEnv(gym.Wrapper):
    """Spawn-latched placement environment (512-way macro actions)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: gym.Env,
        *,
        planner: Optional[PlacementPlanner] = None,
        max_wait_frames: int = 6000,
        debug: bool = False,
    ) -> None:
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(TOTAL_ACTIONS)
        self._planner = planner or PlacementPlanner()
        self._max_wait_frames = int(max_wait_frames)
        self._debug = bool(debug)
        # Training throughput mode: when using the C++ engine backend, allow the
        # wrapper to fast-forward many frames internally without calling the
        # per-frame `env.step()` loop. Disable with DRMARIO_CPP_FAST=0.
        self._cpp_fast = self._env_flag("DRMARIO_CPP_FAST", default=True)
        self._ctx: Optional[_DecisionContext] = None
        self._pose_mismatch_count = 0
        self._pose_mismatch_log_path = self._resolve_pose_mismatch_log_path()
        self._pose_mismatch_trace = self._env_flag("DRMARIO_POSE_MISMATCH_TRACE", default=False)
        self._pose_mismatch_log_max = self._env_int("DRMARIO_POSE_MISMATCH_LOG_MAX", default=0)
        # The NES exposes a monotonically increasing spawn counter (`pill_counter`,
        # RAM $0310) that increments whenever a new pill appears.
        #
        # The macro env must be *spawn-latched*: one decision per pill spawn, not
        # one per falling frame. We therefore treat a pill as "consumed" once we
        # either (a) execute a macro plan for it, or (b) determine that there are
        # no feasible in-bounds macro actions (spawn-blocked / immediate top-out).
        #
        # While a pill is falling, `currentP_nextAction == nextAction_pillFalling`
        # stays true for many frames; `pill_counter` is what lets us distinguish
        # "still the same spawn" vs "new spawn".
        self._consumed_spawn_id: Optional[int] = None
        self._last_obs: Any = None
        self._last_info: Dict[str, Any] = {}

        # NES button bits (match C++ engine + libretro conventions).
        self._BTN_RIGHT = 0x01
        self._BTN_LEFT = 0x02
        self._BTN_DOWN = 0x04
        self._BTN_B = 0x40
        self._BTN_A = 0x80

        # C++ engine batched-run reason codes (see `game_engine/main.cpp`).
        self._RUN_REASON_DECISION = 2
        self._RUN_REASON_TERMINAL = 3
        self._RUN_REASON_TIMEOUT = 4

        # Optional task budgets (for time-based curricula).
        # These are wrapper-level so curricula can set them without depending on
        # backend-specific counters.
        self.task_max_frames: Optional[int] = None
        self.task_max_spawns: Optional[int] = None
        self._task_frames_used: int = 0
        self._task_spawns_used: int = 0
        self._viruses_initial: Optional[int] = None

    # ------------------------------------------------------------------ utils

    def _consume_spawn(self, spawn_id: Optional[int]) -> None:
        """Mark a spawn as consumed and increment the episode spawn counter.

        A spawn can be consumed either because the agent committed to a macro
        action for it, or because the wrapper detected a spawn with zero feasible
        actions and skipped it (spawn-blocked / immediate top-out).
        """

        if spawn_id is None:
            return
        try:
            sid = int(spawn_id)
        except Exception:
            return
        if self._consumed_spawn_id is None or int(self._consumed_spawn_id) != sid:
            self._task_spawns_used += 1
        self._consumed_spawn_id = sid

    def _finalize_step(
        self,
        obs: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
        *,
        tau_frames: int,
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Update task counters, enforce budgets, and attach task info."""

        tau_i = int(max(0, int(tau_frames)))
        self._task_frames_used += tau_i

        objective_met = bool(info.get("goal_achieved", False))

        max_frames_raw = getattr(self, "task_max_frames", None)
        max_spawns_raw = getattr(self, "task_max_spawns", None)
        try:
            max_frames = None if max_frames_raw is None else int(max_frames_raw)
        except Exception:
            max_frames = None
        try:
            max_spawns = None if max_spawns_raw is None else int(max_spawns_raw)
        except Exception:
            max_spawns = None

        frames_exceeded = bool(max_frames is not None and self._task_frames_used > int(max_frames))
        spawns_exceeded = bool(max_spawns is not None and self._task_spawns_used > int(max_spawns))
        budget_exceeded = bool(frames_exceeded or spawns_exceeded)
        within_budget = not budget_exceeded
        success = bool(objective_met and within_budget)

        if "goal_achieved" not in info:
            info["goal_achieved"] = False
        if "cleared" not in info:
            info["cleared"] = False

        # Soft task budgets:
        # - If the agent clears over-budget, we still allow completion but
        #   replace the terminal clear bonus with a smooth time-goal reward:
        #     * under budget  -> positive, asymptotes to clear bonus
        #     * over budget   -> negative, asymptotes to topout penalty
        # - If the agent exceeds budget mid-episode, we do *not* truncate;
        #   instead `task/success` will stay false unless they finish in-budget.
        #
        # This avoids throwing away trajectories that narrowly miss the goal,
        # while still strongly incentivizing time-efficient clears.
        if objective_met and (max_frames is not None or max_spawns is not None):
            # Prefer frame budgets (time) if available; fall back to spawn budgets.
            metric = "frames" if max_frames is not None else "spawns"
            goal = int(max_frames) if max_frames is not None else int(max_spawns or 0)
            used = int(self._task_frames_used) if metric == "frames" else int(self._task_spawns_used)
            delta = int(goal) - int(used)  # positive == under budget

            old_bonus = 0.0
            for key in ("terminal_bonus_reward", "reward/terminal_bonus"):
                try:
                    old_bonus = float(info.get(key, 0.0) or 0.0)
                except Exception:
                    old_bonus = 0.0
                if old_bonus != 0.0:
                    break

            clear_bonus = float(old_bonus)
            if clear_bonus == 0.0:
                try:
                    reward_cfg = getattr(self.env.unwrapped, "reward_cfg", None)
                    clear_bonus = float(getattr(reward_cfg, "terminal_clear_bonus", 0.0) or 0.0)
                except Exception:
                    clear_bonus = 0.0

            topout_penalty = -1.0
            try:
                reward_cfg = getattr(self.env.unwrapped, "reward_cfg", None)
                topout_penalty = float(getattr(reward_cfg, "topout_penalty", -1.0) or -1.0)
            except Exception:
                topout_penalty = -1.0
            if topout_penalty >= 0.0:
                topout_penalty = -abs(float(topout_penalty))

            # Use tanh(delta/scale) to map to (-1,1) with 0 at exactly-on-budget.
            # Scale is a fraction of the budget to keep behavior consistent across levels.
            scale = max(1.0, 0.5 * float(max(1, abs(int(goal)))))
            y = math.tanh(float(delta) / float(scale))
            new_bonus = float(y * clear_bonus) if y >= 0.0 else float(y * abs(topout_penalty))

            bonus_delta = float(new_bonus) - float(old_bonus)
            if bonus_delta != 0.0:
                reward = float(reward) + float(bonus_delta)
                for key in ("r_env", "r_total", "reward/r_env", "reward/r_total"):
                    if key in info:
                        try:
                            info[key] = float(info[key]) + float(bonus_delta)
                        except Exception:
                            pass

            info["task/budget_terminal_bonus_raw"] = float(old_bonus)
            info["task/budget_terminal_bonus_shaped"] = float(new_bonus)
            info["task/budget_metric"] = str(metric)
            info["task/budget_delta"] = int(delta)
            info["terminal_bonus_reward"] = float(new_bonus)
            if "reward/terminal_bonus" in info:
                try:
                    info["reward/terminal_bonus"] = float(new_bonus)
                except Exception:
                    info["reward/terminal_bonus"] = float(new_bonus)

            if not within_budget:
                info["goal_achieved"] = False
                info["cleared"] = False

        # No truncate-on-budget-exceeded: budgets are soft constraints.

        info["task/objective_met"] = bool(objective_met)
        info["task/success"] = bool(success)
        info["task/within_budget"] = bool(within_budget)
        info["task/budget_exceeded"] = bool(budget_exceeded)
        info["task/budget_exceeded_frames"] = bool(frames_exceeded)
        info["task/budget_exceeded_spawns"] = bool(spawns_exceeded)
        # NOTE: Gymnasium AsyncVectorEnv cannot merge infos when a key is
        # sometimes numeric and sometimes `None` (it builds a typed array from
        # the first seen value). Omit unset keys instead of returning `None`.
        if max_frames is not None:
            info["task/max_frames"] = int(max_frames)
        if max_spawns is not None:
            info["task/max_spawns"] = int(max_spawns)
        info["task/frames_used"] = int(self._task_frames_used)
        info["task/spawns_used"] = int(self._task_spawns_used)
        if self._viruses_initial is not None:
            info["drm/viruses_initial"] = int(self._viruses_initial)
        if budget_exceeded:
            reasons: list[str] = []
            if frames_exceeded:
                reasons.append("frames")
            if spawns_exceeded:
                reasons.append("spawns")
            if reasons:
                info["task/budget_reason"] = "+".join(reasons)

        return obs, float(reward), bool(terminated), bool(truncated), info

    @staticmethod
    def _env_flag(name: str, *, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return bool(default)
        return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}

    @staticmethod
    def _env_int(name: str, *, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return int(default)
        try:
            return int(str(raw).strip())
        except Exception:
            return int(default)

    @staticmethod
    def _resolve_pose_mismatch_log_path() -> Optional[Path]:
        """Return the JSONL path for pose mismatch logs, or None to disable."""

        raw = os.environ.get("DRMARIO_POSE_MISMATCH_LOG")
        if raw is None:
            # Default to a git-ignored location for diagnostics.
            return Path("data") / "pose_mismatches.jsonl"
        normalized = str(raw).strip()
        if normalized.lower() in {"", "0", "false", "no", "off", "none"}:
            return None
        try:
            return Path(normalized).expanduser()
        except Exception:
            return None

    def _attach_pose_mismatch_info(
        self,
        out_info: Dict[str, Any],
        *,
        mismatch_last: bool = False,
        mismatch_id: Optional[int] = None,
    ) -> None:
        out_info["placements/pose_mismatch_count"] = int(self._pose_mismatch_count)
        out_info["placements/pose_mismatch_last"] = bool(mismatch_last)
        if mismatch_id is not None:
            out_info["placements/pose_mismatch_id"] = int(mismatch_id)
        if self._pose_mismatch_log_path is not None:
            out_info["placements/pose_mismatch_log_path"] = str(self._pose_mismatch_log_path)

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            json.dump(payload, f)
            f.write("\n")

    def _maybe_log_pose_mismatch(
        self,
        *,
        mismatch_id: int,
        ctx: _DecisionContext,
        action: int,
        plan: Any,
        target_pose: Optional[Tuple[int, int, int]],
        lock_capture: Dict[str, Any],
        bottle_before: Optional[np.ndarray],
        bottle_after: Optional[np.ndarray],
        out_info: Dict[str, Any],
    ) -> None:
        if self._pose_mismatch_log_path is None:
            return
        if self._pose_mismatch_log_max > 0 and mismatch_id > self._pose_mismatch_log_max:
            return

        def _grid_hex(grid: np.ndarray) -> list[str]:
            rows: list[str] = []
            for r in range(int(grid.shape[0])):
                rows.append(" ".join(f"{int(v) & 0xFF:02X}" for v in grid[r, :]))
            return rows

        snap = ctx.snapshot
        reach = ctx.reach
        board_cols = ctx.board.columns.astype(np.uint16, copy=False)
        feasible_flat = np.asarray(reach.feasible_mask, dtype=np.bool_).reshape(-1)
        legal_flat = np.asarray(reach.legal_mask, dtype=np.bool_).reshape(-1)
        costs_flat = np.asarray(reach.costs, dtype=np.float32).reshape(-1)
        costs_u16 = np.full_like(costs_flat, 0xFFFF, dtype=np.uint16)
        finite = np.isfinite(costs_flat)
        costs_u16[finite] = costs_flat[finite].astype(np.uint16, copy=False)

        controller_steps: list[Dict[str, Any]] = []
        try:
            for i, step in enumerate(plan.controller):
                controller_steps.append(
                    {
                        "i": int(i),
                        "action": int(step.action),
                        "action_name": getattr(step.action, "name", str(step.action)),
                        "hold_left": bool(step.hold_left),
                        "hold_right": bool(step.hold_right),
                        "hold_down": bool(step.hold_down),
                    }
                )
        except Exception:
            controller_steps = []

        trace = lock_capture.get("trace")
        trace_out: Optional[list[list[int]]] = None
        if isinstance(trace, (list, tuple)):
            # Already materialized (should not happen).
            trace_out = [[int(v) for v in row] for row in trace]  # type: ignore[assignment]
        elif trace is not None:
            try:
                trace_out = [[int(v) for v in row] for row in list(trace)]
            except Exception:
                trace_out = None
        trace_payload: Optional[Dict[str, Any]] = None
        if trace_out is not None:
            trace_payload = {
                "schema": [
                    "t",
                    "next_action",
                    "falling_x",
                    "falling_y",
                    "falling_rot",
                    "speed_counter",
                    "hor_velocity",
                    "frame_parity",
                    "spawn_id",
                ],
                "tuples": trace_out,
            }

        payload: Dict[str, Any] = {
            "type": "pose_mismatch",
            "schema": 1,
            "when_unix": float(time.time()),
            "mismatch_id": int(mismatch_id),
            "spawn_id": int(snap.spawn_id) if snap.spawn_id is not None else None,
            "reach_backend": "native" if reach.native is not None else "python",
            "action": int(action),
            "action_unflattened": None,
            "snapshot": {
                "base_row": int(snap.base_row),
                "base_col": int(snap.base_col),
                "rot": int(snap.rot),
                "colors": [int(snap.colors[0]), int(snap.colors[1])],
                "speed_counter": int(snap.speed_counter),
                "speed_threshold": int(snap.speed_threshold),
                "hor_velocity": int(snap.hor_velocity),
                "frame_parity": int(snap.frame_parity),
                "hold_left": bool(snap.hold_left),
                "hold_right": bool(snap.hold_right),
                "hold_down": bool(snap.hold_down),
                "speed_setting": int(snap.speed_setting),
                "speed_ups": int(snap.speed_ups),
            },
            "board": {
                "columns_u16": [int(v) for v in board_cols.tolist()],
                "bottle_before_u8": bottle_before.astype(np.uint8).tolist() if bottle_before is not None else None,
                "bottle_before_hex": _grid_hex(bottle_before) if bottle_before is not None else None,
                "bottle_after_u8": bottle_after.astype(np.uint8).tolist() if bottle_after is not None else None,
                "bottle_after_hex": _grid_hex(bottle_after) if bottle_after is not None else None,
            },
            "reachability": {
                "shape": [int(v) for v in reach.feasible_mask.shape],
                "legal_count": int(legal_flat.sum()),
                "feasible_count": int(feasible_flat.sum()),
                "legal_indices": np.flatnonzero(legal_flat).astype(int).tolist(),
                "feasible_indices": np.flatnonzero(feasible_flat).astype(int).tolist(),
                "costs_u16_flat": [int(v) for v in costs_u16.tolist()],
                "costs_u16_flat_inf": 65535,
            },
            "plan": {
                "terminal_pose": (
                    [int(v) for v in plan.terminal_pose] if getattr(plan, "terminal_pose", None) is not None else None
                ),
                "cost": int(getattr(plan, "cost", 0) or 0),
                "len": int(len(getattr(plan, "controller", []) or [])),
                "controller": controller_steps,
            },
            "result": {
                "target_pose": [int(v) for v in target_pose] if target_pose is not None else None,
                "lock_pose": (
                    [int(v) for v in lock_capture.get("lock_pose")]
                    if isinstance(lock_capture.get("lock_pose"), (list, tuple))
                    else None
                ),
                "lock_t": int(lock_capture.get("lock_t")) if lock_capture.get("lock_t") is not None else None,
                "lock_reason": lock_capture.get("lock_reason"),
                "pose_dx": out_info.get("placements/pose_dx"),
                "pose_dy": out_info.get("placements/pose_dy"),
                "pose_drot": out_info.get("placements/pose_drot"),
                "tau": int(out_info.get("placements/tau", 0) or 0),
            },
            "trace": trace_payload,
            "env": {
                "t": int(getattr(self.env.unwrapped, "_t", 0) or 0),
                "level": out_info.get("level"),
                "viruses_remaining": out_info.get("viruses_remaining"),
                "task_mode": out_info.get("task_mode"),
                "curriculum_level": out_info.get("curriculum/current_level"),
            },
        }

        try:
            from envs.retro.placement_space import unflatten as unflatten_action

            o, r, c = unflatten_action(int(action))
            payload["action_unflattened"] = {"o": int(o), "row": int(r), "col": int(c)}
        except Exception:
            payload["action_unflattened"] = None

        try:
            self._append_jsonl(self._pose_mismatch_log_path, payload)
        except Exception:
            # Never fail the environment step on diagnostic I/O.
            return

    def _state_cache(self) -> DrMarioState:
        state = getattr(self.env.unwrapped, "_state_cache", None)
        if state is None:
            base = self.env.unwrapped
            backend = getattr(base, "backend_name", None)
            backend_active = getattr(base, "_using_backend", None)
            last_err = getattr(base, "_backend_last_error", None)
            raise RuntimeError(
                "Underlying env does not expose _state_cache (state mode required). "
                f"backend={backend!r} backend_active={backend_active!r} last_backend_error={last_err!r}"
            )
        return state

    def _cpp_fast_backend(self) -> Optional[Any]:
        if not bool(self._cpp_fast):
            return None
        base = self.env.unwrapped
        if str(getattr(base, "backend_name", "")).lower() != "cpp-engine":
            return None
        # Guardrail: fast mode only supports the common training reward config.
        # If other reward knobs are enabled, fall back to per-frame stepping.
        reward_cfg = getattr(base, "reward_cfg", None)
        try:
            if reward_cfg is not None:
                if float(getattr(reward_cfg, "pill_place_base", 0.0)) != 0.0:
                    return None
                if float(getattr(reward_cfg, "pill_place_growth", 0.0)) != 0.0:
                    return None
                if float(getattr(reward_cfg, "column_height_penalty", 0.0)) != 0.0:
                    return None
                if float(getattr(reward_cfg, "action_penalty_scale", 0.0)) != 0.0:
                    return None
                if bool(getattr(reward_cfg, "punish_high_placements", False)) and float(
                    getattr(reward_cfg, "placement_height_penalty_multiplier", 0.0)
                ) != 0.0:
                    return None
        except Exception:
            return None
        if bool(getattr(base, "_use_shaping", False)):
            return None
        backend = getattr(base, "_backend", None)
        if backend is None:
            return None
        if not callable(getattr(backend, "run_frames_mask", None)):
            return None
        if not callable(getattr(backend, "run_until_decision", None)):
            return None
        if not callable(getattr(base, "sync_after_backend_run", None)):
            return None
        return backend

    def _cpp_backend_for_fast_advance(self) -> Optional[Any]:
        """Return a cpp-engine backend suitable for reset-time fast-forwarding.

        Unlike `_cpp_fast_backend`, this does *not* gate on reward config
        compatibility. Reset-time advance does not aggregate per-frame rewards,
        so we can safely use the engine's internal runner just to reach a
        decision point.
        """

        if not bool(self._cpp_fast):
            return None
        base = self.env.unwrapped
        if str(getattr(base, "backend_name", "")).lower() != "cpp-engine":
            return None
        backend = getattr(base, "_backend", None)
        if backend is None:
            return None
        if not callable(getattr(backend, "run_until_decision", None)):
            return None
        if not callable(getattr(base, "sync_after_backend_run", None)):
            return None
        return backend

    def _controller_step_to_mask(self, step: "ControllerStep") -> int:
        mask = 0
        if step.hold_left:
            mask |= self._BTN_LEFT
        if step.hold_right:
            mask |= self._BTN_RIGHT
        if step.hold_down:
            mask |= self._BTN_DOWN
        if step.action == Action.ROTATE_A:
            mask |= self._BTN_A
        elif step.action == Action.ROTATE_B:
            mask |= self._BTN_B
        elif step.action == Action.BOTH_ROT:
            mask |= self._BTN_A | self._BTN_B
        return int(mask) & 0xFF

    def _compress_controller(self, controller: list["ControllerStep"]) -> list[tuple[int, int]]:
        """Run-length encode controller steps as (buttons_mask, frames) segments."""

        runs: list[tuple[int, int]] = []
        last_mask: Optional[int] = None
        last_count = 0
        for step in controller:
            m = self._controller_step_to_mask(step)
            if last_mask is None:
                last_mask = m
                last_count = 1
                continue
            if m == last_mask:
                last_count += 1
                continue
            runs.append((int(last_mask), int(last_count)))
            last_mask = m
            last_count = 1
        if last_mask is not None and last_count > 0:
            runs.append((int(last_mask), int(last_count)))
        return runs

    def _spawn_id(self, state: DrMarioState) -> Optional[int]:
        try:
            spawn_id = getattr(state.ram_vals, "pill_counter", None)
            return int(spawn_id) if spawn_id is not None else None
        except Exception:
            return None

    def _at_decision_point(self, state: DrMarioState) -> bool:
        # "Controllable pill" state (player inputs are interpreted) is indicated
        # by `currentP_nextAction == nextAction_pillFalling` and a present
        # falling pill mask.
        #
        # This is true for *many* consecutive frames while the capsule falls, so
        # we additionally gate on the spawn counter to ensure one macro decision
        # per spawn.
        nxt = _read_u8(state.ram.bytes, ZP_CURRENT_P_NEXT_ACTION)
        if nxt != NEXT_ACTION_PILL_FALLING:
            return False
        try:
            if not bool(state.calc.falling_mask.any()):
                return False
        except Exception:
            return False

        spawn_id = self._spawn_id(state)
        if spawn_id is None:
            # Fallback for test stubs / unknown RAM layouts: treat any
            # controllable falling-pill state as a decision.
            return True
        if self._consumed_spawn_id is None:
            return True
        return int(spawn_id) != int(self._consumed_spawn_id)

    def _clear_holds(self) -> None:
        base = self.env.unwrapped
        for attr in ("_hold_left", "_hold_right", "_hold_down"):
            if hasattr(base, attr):
                setattr(base, attr, False)

    def _set_holds(self, *, left: bool, right: bool, down: bool) -> None:
        base = self.env.unwrapped
        if hasattr(base, "_hold_left"):
            base._hold_left = bool(left)
        if hasattr(base, "_hold_right"):
            base._hold_right = bool(right)
        if hasattr(base, "_hold_down"):
            base._hold_down = bool(down)

    def _decision_info(self, ctx: _DecisionContext) -> Dict[str, Any]:
        snap = ctx.snapshot
        info: Dict[str, Any] = {}
        # Emit planner-build timing once per reachability computation so UI-side
        # perf counters reflect actual work (invalid-action loops reuse the same
        # cached ctx and should not count as additional build calls).
        if float(ctx.planner_build_sec) > 0.0:
            info["perf/planner_build_sec"] = float(ctx.planner_build_sec)
            ctx.planner_build_sec = 0.0
        info["placements/legal_mask"] = ctx.reach.legal_mask.copy()
        info["placements/feasible_mask"] = ctx.reach.feasible_mask.copy()
        info["placements/costs"] = ctx.reach.costs.copy()
        info["placements/options"] = int(ctx.reach.feasible_mask.sum())
        info["placements/reach_backend"] = "native" if ctx.reach.native is not None else "python"
        if snap.spawn_id is not None:
            info["placements/spawn_id"] = int(snap.spawn_id)
            info["pill/spawn_id"] = int(snap.spawn_id)
        info["pill/base_row"] = int(snap.base_row)
        info["pill/base_col"] = int(snap.base_col)
        info["pill/rot"] = int(snap.rot)
        info["pill/speed_setting"] = int(snap.speed_setting)
        info["speed_setting"] = int(snap.speed_setting)
        info["pill/speed_counter"] = int(snap.speed_counter)
        info["pill/speed_threshold"] = int(snap.speed_threshold)
        info["pill/hor_velocity"] = int(snap.hor_velocity)
        info["pill/frame_parity"] = int(snap.frame_parity)
        info["pill/colors"] = np.asarray(snap.colors, dtype=np.int64)
        # Historical key used by placement policies
        info["next_pill_colors"] = np.asarray(snap.colors, dtype=np.int64)
        return info

    @staticmethod
    def _maybe_inject_feasible_mask(obs: Any, feasible_mask: np.ndarray) -> Any:
        """Optionally write the feasibility mask into dedicated observation planes.

        This is only active for representations that reserve feasibility mask
        channels (e.g. `bitplane_reduced_mask`). The RAM→state mapper emits
        those channels as zeros; the placement wrapper fills them at the true
        decision point where reachability is known.
        """

        channels = getattr(ram_specs.STATE_IDX, "feasible_mask_channels", None)
        if not channels:
            return obs
        try:
            dst = tuple(int(c) for c in channels)
        except Exception:
            return obs
        if len(dst) != 4:
            return obs

        try:
            mask = np.asarray(feasible_mask, dtype=np.float32)
        except Exception:
            return obs
        if mask.shape != (4, GRID_HEIGHT, GRID_WIDTH):
            return obs

        def _inject(arr: np.ndarray) -> np.ndarray:
            if arr.ndim == 4:
                # (T,C,H,W) frame-stack: repeat mask across the stack for
                # simplicity and to avoid stale channels in older frames.
                out = np.array(arr, copy=True)
                for o, ch in enumerate(dst):
                    out[:, ch, :, :] = mask[o]
                return out
            if arr.ndim == 3:
                # (C,H,W) single frame.
                out = np.array(arr, copy=True)
                for o, ch in enumerate(dst):
                    out[ch, :, :] = mask[o]
                return out
            return arr

        if isinstance(obs, dict) and "obs" in obs:
            out = dict(obs)
            try:
                out["obs"] = _inject(np.asarray(out["obs"]))
            except Exception:
                return obs
            return out

        try:
            return _inject(np.asarray(obs))
        except Exception:
            return obs

    def _update_lock_capture(self, lock_capture: Dict[str, Any], info: Dict[str, Any]) -> None:
        """Best-effort capture of the locked pill pose for the current macro step.

        The earlier implementation keyed off `pill_bonus_adjusted` which is:
          - zero when `pill_place_base == 0` (common during shaping-focused runs)
          - associated with spawn-counter increments, not the actual lock frame

        We instead use RAM state transitions:
          - While `currentP_nextAction == nextAction_pillFalling`, track the last
            valid falling pose.
          - When we leave that state *or* the spawn counter advances, we freeze
            the last observed falling pose as the lock pose.
        """

        if lock_capture.get("lock_pose") is not None:
            return
        try:
            state = self._state_cache()
        except Exception:
            return

        trace = lock_capture.get("trace")
        if trace is not None:
            try:
                t_val = info.get("t", getattr(self.env.unwrapped, "_t", 0) or 0)
                nxt_val = _read_u8(state.ram.bytes, ZP_CURRENT_P_NEXT_ACTION)
                pose = _read_falling_pose(state.ram.bytes) if nxt_val == NEXT_ACTION_PILL_FALLING else None
                if pose is None:
                    x, y, rot = -1, -1, -1
                else:
                    x, y, rot = (int(pose[0]), int(pose[1]), int(pose[2]))
                speed_ctr = _read_u8(state.ram.bytes, ZP_SPEED_COUNTER)
                hor_vel = _read_u8(state.ram.bytes, ZP_HOR_VELOCITY)
                parity = _read_u8(state.ram.bytes, ZP_FRAME_COUNTER) & 0x01
                spawn_id = self._spawn_id(state)
                trace.append((int(t_val), int(nxt_val), int(x), int(y), int(rot), int(speed_ctr), int(hor_vel), int(parity), int(spawn_id or -1)))
            except Exception:
                pass

        expected_spawn = lock_capture.get("spawn_id")
        cur_spawn = self._spawn_id(state)

        # If the spawn counter advanced, the previous pill must have locked;
        # freeze the last pose observed for the expected spawn.
        if (
            expected_spawn is not None
            and cur_spawn is not None
            and int(cur_spawn) != int(expected_spawn)
            and lock_capture.get("last_falling_pose") is not None
        ):
            lock_capture["lock_pose"] = tuple(int(v) for v in lock_capture["last_falling_pose"])
            lock_t = lock_capture.get("last_falling_t")
            if lock_t is None:
                lock_t = info.get("t", getattr(self.env.unwrapped, "_t", 0) or 0)
            lock_capture["lock_t"] = int(lock_t)
            lock_capture["lock_reason"] = "spawn_advanced"
            return

        try:
            nxt = _read_u8(state.ram.bytes, ZP_CURRENT_P_NEXT_ACTION)
        except Exception:
            nxt = 0

        if nxt == NEXT_ACTION_PILL_FALLING:
            pose = _read_falling_pose(state.ram.bytes)
            if pose is not None:
                lock_capture["last_falling_pose"] = tuple(int(v) for v in pose)
                lock_capture["last_falling_t"] = int(info.get("t", getattr(self.env.unwrapped, "_t", 0) or 0))
            return

        # Leaving controllable falling state: treat as lock (or top-out / stage transition).
        if lock_capture.get("last_falling_pose") is not None:
            lock_capture["lock_pose"] = tuple(int(v) for v in lock_capture["last_falling_pose"])
            lock_t = lock_capture.get("last_falling_t")
            if lock_t is None:
                lock_t = info.get("t", getattr(self.env.unwrapped, "_t", 0) or 0)
            lock_capture["lock_t"] = int(lock_t)
            lock_capture["lock_reason"] = "left_pill_falling"

    @staticmethod
    def _reward_breakdown_info(breakdown: _RewardBreakdown, *, r_total: float) -> Dict[str, Any]:
        """Format aggregated reward components for the wrapper `info` dict."""

        return {
            "reward/r_total": float(r_total),
            "reward/r_env": float(breakdown.r_env),
            "reward/r_shape": float(breakdown.r_shape),
            "reward/delta_v": int(breakdown.delta_v),
            "reward/tiles_cleared_total": int(breakdown.tiles_cleared_total),
            "reward/tiles_cleared_non_virus": int(breakdown.tiles_cleared_non_virus),
            "reward/tiles_clearing_max": int(breakdown.tiles_clearing_max),
            "reward/action_events": int(breakdown.action_events),
            "reward/pill_locks": int(breakdown.pill_locks),
            "reward/virus_clear_reward": float(breakdown.virus_clear_reward),
            "reward/non_virus_bonus": float(breakdown.non_virus_bonus),
            "reward/adjacency_bonus": float(breakdown.adjacency_bonus),
            "reward/virus_adjacency_bonus": float(breakdown.virus_adjacency_bonus),
            "reward/pill_bonus_adjusted": float(breakdown.pill_bonus_adjusted),
            "reward/height_penalty_delta": float(breakdown.height_penalty_delta),
            "reward/action_penalty": float(breakdown.action_penalty),
            "reward/terminal_bonus": float(breakdown.terminal_bonus),
            "reward/topout_penalty": float(breakdown.topout_penalty),
            "reward/time_reward": float(breakdown.time_reward),
        }

    @staticmethod
    def _perf_breakdown_info(perf: _PerfBreakdown) -> Dict[str, Any]:
        """Format aggregated per-frame env timings for the wrapper `info` dict."""

        return {
            "perf/env_step_calls": int(perf.env_step_calls),
            "perf/env_step_sec": float(perf.env_step_sec),
            "perf/env_backend_step_sec": float(perf.env_backend_step_sec),
            "perf/env_get_frame_sec": float(perf.env_get_frame_sec),
            "perf/env_pixel_stack_sec": float(perf.env_pixel_stack_sec),
            "perf/env_get_ram_sec": float(perf.env_get_ram_sec),
            "perf/env_ram_bytes_sec": float(perf.env_ram_bytes_sec),
            "perf/env_build_state_sec": float(perf.env_build_state_sec),
            "perf/env_observe_sec": float(perf.env_observe_sec),
            "perf/env_reward_sec": float(perf.env_reward_sec),
            "perf/env_info_sec": float(perf.env_info_sec),
        }

    def _build_decision_context(self) -> _DecisionContext:
        state = self._state_cache()
        offsets = getattr(self.env.unwrapped, "_ram_offsets", {})
        snap = PillSnapshot.from_state(state, offsets)
        board = BoardState.from_planes(state.calc.planes)
        t0 = time.perf_counter()
        reach = self._planner.build_spawn_reachability(board, snap)
        planner_build_sec = float(time.perf_counter() - t0)
        # Symmetry reduction: identical colors => drop H-/V- duplicates.
        if int(snap.colors[0]) == int(snap.colors[1]):
            reach = SpawnReachability(
                legal_mask=reach.legal_mask.copy(),
                feasible_mask=reach.feasible_mask.copy(),
                costs=reach.costs.copy(),
                reach=reach.reach,
                action_to_terminal_node=(
                    dict(reach.action_to_terminal_node) if reach.action_to_terminal_node is not None else None
                ),
                native=reach.native,
            )
            reach.feasible_mask[2, :, :] = False
            reach.feasible_mask[3, :, :] = False
            reach.legal_mask[2, :, :] = False
            reach.legal_mask[3, :, :] = False
            reach.costs[2, :, :] = np.inf
            reach.costs[3, :, :] = np.inf
            if reach.action_to_terminal_node is not None:
                # Remove disabled actions from the terminal map.
                to_drop = []
                for a in reach.action_to_terminal_node:
                    o = a // (GRID_HEIGHT * GRID_WIDTH)
                    if o in (2, 3):
                        to_drop.append(a)
                for a in to_drop:
                    reach.action_to_terminal_node.pop(a, None)
        return _DecisionContext(snapshot=snap, board=board, reach=reach, planner_build_sec=planner_build_sec)

    def _advance_until_decision(
        self,
        obs: Any,
        info: Dict[str, Any],
        *,
        breakdown: Optional[_RewardBreakdown] = None,
        perf: Optional[_PerfBreakdown] = None,
        lock_capture: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any], float, bool, bool]:
        """Advance the underlying env until the next macro decision point."""

        total_reward = 0.0
        terminated = False
        truncated = False
        saw_no_feasible = False

        self._clear_holds()
        # Invalidate any cached decision context while we advance the underlying
        # per-frame env.
        self._ctx = None
        for _ in range(self._max_wait_frames):
            try:
                state = self._state_cache()
            except Exception as exc:
                out_info = dict(info or {})
                out_info["placements/needs_action"] = False
                out_info["placements/backend_error"] = str(exc)
                return obs, out_info, float(total_reward), False, True
            if self._at_decision_point(state):
                try:
                    ctx = self._build_decision_context()
                except Exception as exc:
                    out_info = dict(info or {})
                    out_info["placements/needs_action"] = False
                    out_info["placements/backend_error"] = str(exc)
                    return obs, out_info, float(total_reward), False, True
                # If there are *no feasible in-bounds placements* for this spawn,
                # there is no meaningful macro-action to request. This can happen
                # when the spawn is immediately blocked and the capsule will lock
                # offscreen (top-out) before the player gets a usable input window.
                #
                # In that situation we keep stepping NOOP until the underlying env
                # transitions (lock / top-out / reset) or a later spawn becomes
                # controllable.
                if int(ctx.reach.feasible_mask.sum()) > 0:
                    self._ctx = ctx
                    out_info = dict(info or {})
                    out_info.update(self._decision_info(ctx))
                    out_info["placements/needs_action"] = True
                    if saw_no_feasible:
                        out_info["placements/no_feasible_actions"] = True
                    obs_out = self._maybe_inject_feasible_mask(obs, ctx.reach.feasible_mask)
                    return obs_out, out_info, float(total_reward), terminated, truncated
                saw_no_feasible = True
                # Mark this spawn as consumed (no feasible actions) so we don't
                # treat subsequent falling frames as new decisions.
                if ctx.snapshot.spawn_id is not None:
                    self._consume_spawn(ctx.snapshot.spawn_id)

            obs, r, terminated, truncated, info = self.env.step(int(Action.NOOP))
            total_reward += float(r)
            if breakdown is not None and isinstance(info, dict):
                breakdown.add_frame(reward=float(r), info=info)
            if perf is not None and isinstance(info, dict):
                perf.add_frame(info=info)
            if lock_capture is not None and isinstance(info, dict):
                self._update_lock_capture(lock_capture, info)
            if terminated or truncated:
                break

        # Timed out or terminated.
        out_info = dict(info or {})
        out_info["placements/needs_action"] = False
        if saw_no_feasible:
            out_info["placements/no_feasible_actions"] = True
        if self._debug and not (terminated or truncated):
            out_info["placements/wait_timeout"] = True
        return obs, out_info, float(total_reward), terminated, truncated

    def _advance_until_decision_after_reset(
        self,
        obs: Any,
        info: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any], bool, bool]:
        """Reset-time version of `_advance_until_decision`.

        If the cpp-engine backend supports batched fast-forwarding, prefer it to
        avoid per-frame Python stepping during `AsyncVectorEnv` autoresets.
        """

        backend = self._cpp_backend_for_fast_advance()
        if backend is None:
            obs, out_info, _r, terminated, truncated = self._advance_until_decision(obs, info)
            return obs, out_info, bool(terminated), bool(truncated)

        base = self.env.unwrapped
        sync_fn = getattr(base, "sync_after_backend_run", None)
        if not callable(sync_fn):
            obs, out_info, _r, terminated, truncated = self._advance_until_decision(obs, info)
            return obs, out_info, bool(terminated), bool(truncated)

        terminated = False
        truncated = False
        saw_no_feasible = False
        last_spawn_id: Optional[int] = None

        self._clear_holds()
        self._ctx = None

        remaining = int(self._max_wait_frames)
        while remaining > 0:
            try:
                state = self._state_cache()
            except Exception as exc:
                out_info = dict(info or {})
                out_info["placements/needs_action"] = False
                out_info["placements/backend_error"] = str(exc)
                return obs, out_info, False, True

            if self._at_decision_point(state):
                try:
                    ctx = self._build_decision_context()
                except Exception as exc:
                    out_info = dict(info or {})
                    out_info["placements/needs_action"] = False
                    out_info["placements/backend_error"] = str(exc)
                    return obs, out_info, False, True
                if int(ctx.reach.feasible_mask.sum()) > 0:
                    self._ctx = ctx
                    out_info = dict(info or {})
                    out_info.update(self._decision_info(ctx))
                    out_info["placements/needs_action"] = True
                    if saw_no_feasible:
                        out_info["placements/no_feasible_actions"] = True
                    obs_out = self._maybe_inject_feasible_mask(obs, ctx.reach.feasible_mask)
                    return obs_out, out_info, bool(terminated), bool(truncated)
                saw_no_feasible = True
                if ctx.snapshot.spawn_id is not None:
                    self._consume_spawn(ctx.snapshot.spawn_id)
                    last_spawn_id = int(ctx.snapshot.spawn_id)

            res = backend.run_until_decision(last_spawn_id=last_spawn_id, max_frames=int(remaining))
            waited = int(res.get("frames", 0))
            remaining -= waited
            try:
                sync_fn(int(waited))
                obs = getattr(base, "last_obs", obs)
            except Exception as exc:
                out_info = dict(info or {})
                out_info["placements/needs_action"] = False
                out_info["placements/backend_error"] = str(exc)
                return obs, out_info, False, True

            reason = int(res.get("reason", 0))
            if reason == self._RUN_REASON_TERMINAL:
                terminated = True
                break
            if reason == self._RUN_REASON_TIMEOUT or remaining <= 0:
                truncated = True
                break
            # Otherwise, loop and validate feasibility at the candidate decision.

        out_info = dict(info or {})
        out_info["placements/needs_action"] = False
        if saw_no_feasible:
            out_info["placements/no_feasible_actions"] = True
        if self._debug and not (terminated or truncated):
            out_info["placements/wait_timeout"] = True
        return obs, out_info, bool(terminated), bool(truncated)

    # ------------------------------------------------------------------ gym api

    def reset(self, **kwargs):
        max_retries = 0
        try:
            max_retries = max(0, int(os.environ.get("DRMARIO_PLACEMENT_RESET_RETRIES", "2")))
        except Exception:
            max_retries = 2
        max_attempts = int(max_retries) + 1

        last_exc: Optional[BaseException] = None
        for attempt in range(max_attempts):
            obs, info = self.env.reset(**kwargs)
            self._last_obs = obs
            self._last_info = dict(info or {})
            self._ctx = None
            self._consumed_spawn_id = None
            self._task_frames_used = 0
            self._task_spawns_used = 0
            try:
                obs, info, terminated, truncated = self._advance_until_decision_after_reset(obs, self._last_info)
            except Exception as exc:
                last_exc = exc
                # Retry resets that fail due to backend flakiness under high env counts.
                # This is especially important for AsyncVectorEnv autoresets, where
                # a single reset failure would otherwise crash the whole run.
                if attempt + 1 < max_attempts:
                    try:
                        base = self.env.unwrapped
                        if callable(getattr(base, "backend_reset", None)):
                            base.backend_reset()
                    except Exception:
                        pass
                    continue
                raise

            self._last_obs = obs
            self._last_info = dict(info or {})
            self._attach_pose_mismatch_info(self._last_info)
            # Start task budgets at the first decision point returned to the agent.
            self._consumed_spawn_id = None
            self._task_frames_used = 0
            self._task_spawns_used = 0
            out_info = dict(self._last_info)
            # NOTE: Gymnasium AsyncVectorEnv cannot merge infos when a key is
            # sometimes numeric and sometimes `None` (it builds a typed array from
            # the first seen value). Omit unset keys instead of returning `None`.
            if self.task_max_frames is not None:
                out_info["task/max_frames"] = int(self.task_max_frames)
            if self.task_max_spawns is not None:
                out_info["task/max_spawns"] = int(self.task_max_spawns)
            out_info["task/frames_used"] = 0
            out_info["task/spawns_used"] = 0
            out_info["task/budget_exceeded"] = False
            out_info["task/within_budget"] = True
            out_info["task/objective_met"] = False
            out_info["task/success"] = False
            try:
                self._viruses_initial = int(out_info.get("viruses_remaining", 0) or 0)
            except Exception:
                self._viruses_initial = 0
            out_info["drm/viruses_initial"] = int(self._viruses_initial or 0)
            return obs, out_info

        raise RuntimeError("DrMarioPlacementEnv.reset failed unexpectedly") from last_exc

    def step(self, action: int):
        frames_start_any = int(getattr(self.env.unwrapped, "_t", 0) or 0)

        total_reward = 0.0
        breakdown = _RewardBreakdown()
        perf = _PerfBreakdown()
        terminated = False
        truncated = False
        # Debug/validation: capture the observed pose when the pill locks so we
        # can confirm the controller script achieved the intended placement.
        target_pose: Optional[Tuple[int, int, int]] = None  # (x,y,rot)
        lock_capture: Dict[str, Any] = {
            "lock_pose": None,
            "lock_t": None,
            "lock_reason": None,
            "spawn_id": None,
            "last_falling_pose": None,
            "last_falling_t": None,
        }
        if self._pose_mismatch_trace:
            from collections import deque

            lock_capture["trace"] = deque(maxlen=1024)
        bottle_before: Optional[np.ndarray] = None
        bottle_after: Optional[np.ndarray] = None
        plan_calls = 0
        plan_latency_sec_total = 0.0
        plan_latency_sec_max = 0.0
        plan_options = 0

        # Ensure we are at a decision point.
        try:
            state = self._state_cache()
        except Exception as exc:
            # If the underlying env reset failed (common failure mode under heavy
            # AsyncVectorEnv loads), return a truncated step so the vector env can
            # autoreset this worker on the next call instead of crashing.
            out_info = dict(self._last_info)
            out_info["placements/needs_action"] = False
            out_info["placements/backend_error"] = str(exc)
            self._attach_pose_mismatch_info(out_info)
            obs_out, r_out, term_out, trunc_out, info_out = self._finalize_step(
                self._last_obs, 0.0, False, True, out_info, tau_frames=0
            )
            return obs_out, r_out, term_out, trunc_out, info_out
        if not self._at_decision_point(state):
            try:
                obs, info, r_wait, terminated, truncated = self._advance_until_decision(
                    self._last_obs, self._last_info, breakdown=breakdown, perf=perf
                )
            except Exception as exc:
                out_info = dict(self._last_info)
                out_info["placements/needs_action"] = False
                out_info["placements/backend_error"] = str(exc)
                self._attach_pose_mismatch_info(out_info)
                obs_out, r_out, term_out, trunc_out, info_out = self._finalize_step(
                    self._last_obs, 0.0, False, True, out_info, tau_frames=0
                )
                return obs_out, r_out, term_out, trunc_out, info_out
            total_reward += float(r_wait)
            self._last_obs, self._last_info = obs, dict(info or {})
            if terminated or truncated:
                frames_end = int(getattr(self.env.unwrapped, "_t", frames_start_any) or frames_start_any)
                out_info = dict(self._last_info)
                out_info["placements/tau"] = max(1, frames_end - frames_start_any)
                out_info.update(self._reward_breakdown_info(breakdown, r_total=float(total_reward)))
                out_info.update(self._perf_breakdown_info(perf))
                self._attach_pose_mismatch_info(out_info)
                tau_raw = max(0, frames_end - frames_start_any)
                obs_out, r_out, term_out, trunc_out, info_out = self._finalize_step(
                    obs, float(total_reward), terminated, truncated, out_info, tau_frames=tau_raw
                )
                return obs_out, r_out, term_out, trunc_out, info_out

        # Start τ after we have reached a decision for this spawn.
        frames_start = int(getattr(self.env.unwrapped, "_t", frames_start_any) or frames_start_any)

        if self._ctx is None:
            try:
                self._ctx = self._build_decision_context()
            except Exception as exc:
                out_info = dict(self._last_info)
                out_info["placements/needs_action"] = False
                out_info["placements/backend_error"] = str(exc)
                self._attach_pose_mismatch_info(out_info)
                obs_out, r_out, term_out, trunc_out, info_out = self._finalize_step(
                    self._last_obs, 0.0, False, True, out_info, tau_frames=0
                )
                return obs_out, r_out, term_out, trunc_out, info_out

        ctx = self._ctx
        lock_capture["spawn_id"] = getattr(ctx.snapshot, "spawn_id", None)
        lock_capture["last_falling_pose"] = (int(ctx.snapshot.base_col), int(ctx.snapshot.base_row), int(ctx.snapshot.rot) & 0x03)
        lock_capture["last_falling_t"] = int(getattr(self.env.unwrapped, "_t", 0) or 0)
        # If the planner reports no feasible in-bounds macro placements, we
        # cannot accept any `action` and must advance the underlying env until
        # it either terminates (top-out) or reaches a later controllable spawn.
        plan_options = int(ctx.reach.feasible_mask.sum())
        if plan_options == 0:
            # Consume this spawn (no feasible macro actions) to avoid returning
            # repeated "decisions" while the offscreen pill locks/top-outs.
            if ctx.snapshot.spawn_id is not None:
                self._consume_spawn(ctx.snapshot.spawn_id)
            obs, info, r_wait, terminated, truncated = self._advance_until_decision(
                self._last_obs, self._last_info, breakdown=breakdown, perf=perf
            )
            total_reward += float(r_wait)
            self._last_obs, self._last_info = obs, dict(info or {})
            frames_end = int(getattr(self.env.unwrapped, "_t", frames_start) or frames_start)
            out_info = dict(self._last_info)
            out_info["placements/no_feasible_actions"] = True
            out_info["placements/plan_count_last"] = float(plan_options)
            out_info["placements/tau"] = max(1, frames_end - frames_start)
            out_info["placements/needs_action"] = bool(out_info.get("placements/needs_action", False))
            out_info.update(self._reward_breakdown_info(breakdown, r_total=float(total_reward)))
            out_info.update(self._perf_breakdown_info(perf))
            self._attach_pose_mismatch_info(out_info)
            tau_raw = max(0, frames_end - frames_start)
            obs_out, r_out, term_out, trunc_out, info_out = self._finalize_step(
                obs, float(total_reward), terminated, truncated, out_info, tau_frames=tau_raw
            )
            return obs_out, r_out, term_out, trunc_out, info_out
        # Refresh decision context if spawn id changed.
        try:
            snap_now = PillSnapshot.from_state(self._state_cache(), getattr(self.env.unwrapped, "_ram_offsets", {}))
            if ctx.snapshot.spawn_id is not None and snap_now.spawn_id is not None and snap_now.spawn_id != ctx.snapshot.spawn_id:
                ctx = self._build_decision_context()
                self._ctx = ctx
        except Exception:
            pass

        plan_t0 = time.perf_counter()
        plan = self._planner.plan_action(ctx.reach, int(action))
        plan_latency_sec_total = float(time.perf_counter() - plan_t0)
        plan_calls = 1
        plan_latency_sec_max = max(plan_latency_sec_max, plan_latency_sec_total)
        if plan is None:
            # Invalid choice; surface mask again and request a new decision.
            out_info = dict(self._last_info)
            out_info.update(self._decision_info(ctx))
            out_info["placements/needs_action"] = True
            out_info["placements/invalid_action"] = int(action)
            out_info["perf/planner_plan_sec"] = float(plan_latency_sec_total)
            out_info["placements/plan_calls"] = int(plan_calls)
            out_info["placements/plan_latency_ms_total"] = float(plan_latency_sec_total) * 1000.0
            out_info["placements/plan_latency_ms_max"] = float(plan_latency_sec_max) * 1000.0
            out_info["placements/plan_count_last"] = float(plan_options)
            frames_end = int(getattr(self.env.unwrapped, "_t", frames_start) or frames_start)
            out_info["placements/tau"] = max(1, frames_end - frames_start)
            out_info.update(self._reward_breakdown_info(breakdown, r_total=float(total_reward)))
            out_info.update(self._perf_breakdown_info(perf))
            self._attach_pose_mismatch_info(out_info)
            tau_raw = max(0, frames_end - frames_start)
            obs_out, r_out, term_out, trunc_out, info_out = self._finalize_step(
                self._last_obs, float(total_reward), False, False, out_info, tau_frames=tau_raw
            )
            return obs_out, r_out, term_out, trunc_out, info_out

        try:
            if plan.terminal_pose is not None:
                bx, by, rot = plan.terminal_pose
                target_pose = (int(bx), int(by), int(rot) & 0x03)
        except Exception:
            target_pose = None

        try:
            reader = getattr(self.env.unwrapped, "_read_bottle_grid_u8", None)
            if callable(reader):
                bottle_before = reader()
        except Exception:
            bottle_before = None

        # We are committing to a macro action for this spawn; mark it as
        # consumed so `_advance_until_decision` ignores subsequent falling frames
        # until the next pill spawns.
        if ctx.snapshot.spawn_id is not None:
            self._consume_spawn(ctx.snapshot.spawn_id)
        # Invalidate cached ctx while we execute the plan.
        self._ctx = None

        # Execute the per-frame script.
        fast_backend = self._cpp_fast_backend()
        base = self.env.unwrapped
        sync_fn = None
        prev_state = None
        if fast_backend is not None:
            sync_fn = getattr(base, "sync_after_backend_run", None)
            prev_state = getattr(base, "_state_cache", None)
            # If the base env is in a bad state (e.g., backend crash under heavy
            # AsyncVectorEnv load), fall back to per-frame stepping rather than
            # crashing the whole run.
            if not callable(sync_fn) or prev_state is None:
                fast_backend = None

        if fast_backend is not None:
            assert callable(sync_fn)
            assert prev_state is not None
            prev_frame = prev_state.stack4[-1]
            v_start = int(getattr(base, "_viruses_remaining", 0))

            # Phase 1: execute the placement controller script via batched engine runs.
            runs = self._compress_controller(plan.controller)
            plan_frames_executed = 0
            cleared_total = 0
            cleared_nonvirus = 0
            cleared_virus = 0
            for mask, frames in runs:
                if frames <= 0:
                    continue
                res = fast_backend.run_frames_mask(buttons_mask=int(mask), frames=int(frames))
                plan_frames_executed += int(res.get("frames", 0))
                cleared_total += int(res.get("tiles_cleared_total", 0))
                cleared_virus += int(res.get("tiles_cleared_virus", 0))
                cleared_nonvirus += int(res.get("tiles_cleared_nonvirus", 0))
                if int(res.get("reason", 0)) == self._RUN_REASON_TERMINAL:
                    terminated = True
                    break
            sync_fn(int(plan_frames_executed))

            # Adjacency shaping is based on newly placed static pills (lock boundary).
            try:
                lock_frame = getattr(base, "_state_cache", None).stack4[-1]
            except Exception:
                lock_frame = prev_frame
            reward_cfg = getattr(base, "reward_cfg", None)
            adjacency_bonus = 0.0
            virus_adjacency_bonus = 0.0
            if reward_cfg is not None:
                adjacency_enabled = (
                    float(getattr(reward_cfg, "adjacency_pair_bonus", 0.0)) != 0.0
                    or float(getattr(reward_cfg, "adjacency_triplet_bonus", 0.0)) != 0.0
                )
                virus_adj_enabled = (
                    float(getattr(reward_cfg, "virus_adjacency_pair_bonus", 0.0)) != 0.0
                    or float(getattr(reward_cfg, "virus_adjacency_triplet_bonus", 0.0)) != 0.0
                )
                if adjacency_enabled:
                    adjacency_bonus = float(getattr(base, "_compute_adjacency_bonus")(prev_frame, lock_frame))
                if virus_adj_enabled:
                    virus_adjacency_bonus = float(
                        getattr(base, "_compute_virus_adjacency_bonus")(prev_frame, lock_frame)
                    )

            # Phase 2: fast-forward until next decision point (new spawn controllable).
            wait_frames_total = 0
            saw_no_feasible = False
            last_spawn_id = ctx.snapshot.spawn_id
            obs = getattr(base, "last_obs", self._last_obs)
            info: Dict[str, Any] = dict(self._last_info)

            if not terminated:
                remaining = int(self._max_wait_frames)
                while remaining > 0:
                    res = fast_backend.run_until_decision(
                        last_spawn_id=(int(last_spawn_id) if last_spawn_id is not None else None),
                        max_frames=int(remaining),
                    )
                    waited = int(res.get("frames", 0))
                    wait_frames_total += waited
                    remaining -= waited
                    cleared_total += int(res.get("tiles_cleared_total", 0))
                    cleared_virus += int(res.get("tiles_cleared_virus", 0))
                    cleared_nonvirus += int(res.get("tiles_cleared_nonvirus", 0))
                    sync_fn(int(waited))
                    obs = getattr(base, "last_obs", obs)

                    reason = int(res.get("reason", 0))
                    if reason == self._RUN_REASON_TERMINAL:
                        terminated = True
                        break
                    if reason == self._RUN_REASON_TIMEOUT or remaining <= 0:
                        truncated = True
                        break

                    # We hit a candidate decision point; validate feasibility.
                    if reason == self._RUN_REASON_DECISION:
                        try:
                            ctx_next = self._build_decision_context()
                        except Exception as exc:
                            truncated = True
                            info = dict(info or {})
                            info["placements/needs_action"] = False
                            info["placements/backend_error"] = str(exc)
                            break
                        if int(ctx_next.reach.feasible_mask.sum()) > 0:
                            self._ctx = ctx_next
                            info = dict(info or {})
                            info.update(self._decision_info(ctx_next))
                            info["placements/needs_action"] = True
                            if saw_no_feasible:
                                info["placements/no_feasible_actions"] = True
                            obs = self._maybe_inject_feasible_mask(obs, ctx_next.reach.feasible_mask)
                            break
                        saw_no_feasible = True
                        if ctx_next.snapshot.spawn_id is not None:
                            self._consume_spawn(ctx_next.snapshot.spawn_id)
                            last_spawn_id = int(ctx_next.snapshot.spawn_id)
                        # Keep waiting for a later controllable spawn.
                        continue

                if self._debug and not (terminated or truncated):
                    info["placements/wait_timeout"] = True

            # Aggregate macro reward terms (approximates the per-frame base env reward).
            v_end = int(getattr(base, "_viruses_remaining", 0))
            delta_v = int(max(0, v_start - v_end))

            virus_clear_reward = 0.0
            non_virus_bonus = 0.0
            terminal_bonus_reward = 0.0
            topout_penalty_reward = 0.0
            time_reward = 0.0

            r_env = 0.0
            if reward_cfg is not None:
                virus_clear_reward = float(getattr(reward_cfg, "virus_clear_bonus", 0.0)) * float(delta_v)
                non_virus_bonus = float(getattr(reward_cfg, "non_virus_clear_bonus", 0.0)) * float(
                    max(0, int(cleared_nonvirus))
                )
                r_env += float(virus_clear_reward + non_virus_bonus + adjacency_bonus + virus_adjacency_bonus)

            can_fail, can_clear = (None, None)
            if callable(getattr(base, "_canonical_ram_outcome", None)):
                can_fail, can_clear = base._canonical_ram_outcome()

            topout = bool(can_fail is True)
            task_mode = str(getattr(base, "_task_goal_mode", "viruses") or "viruses")
            goal_achieved = False
            goal_reason: Optional[str] = None
            match_target = getattr(base, "_match_target", None)
            matches_completed = int(getattr(base, "_matches_completed", 0) or 0)
            match_event = bool(int(cleared_total) >= 4)
            if task_mode in {"matches", "any_clear"}:
                target = int(match_target or 1)
                if match_event:
                    matches_completed += 1
                try:
                    base._matches_completed = int(matches_completed)
                    base._prev_clearing_active = False
                except Exception:
                    pass
                if matches_completed >= target:
                    goal_achieved = True
                    goal_reason = f"match_{matches_completed}"
            else:
                # Canonical RAM success is only meaningful for the standard "viruses" objective.
                goal_achieved = bool((not topout) and (task_mode == "viruses") and (can_clear is True or v_end == 0))
                if goal_achieved:
                    goal_reason = "clear"

            if goal_achieved:
                terminated = True
                if reward_cfg is not None:
                    terminal_bonus_reward = float(getattr(reward_cfg, "terminal_clear_bonus", 0.0))
                    r_env += terminal_bonus_reward
                try:
                    base._last_terminal = str(goal_reason or "clear")
                    base._prev_terminal = str(goal_reason or "clear")
                except Exception:
                    pass
            if topout:
                terminated = True
                if reward_cfg is not None:
                    topout_penalty_reward = float(getattr(reward_cfg, "topout_penalty", 0.0))
                    r_env += topout_penalty_reward
                try:
                    base._last_terminal = "topout"
                    base._prev_terminal = "topout"
                except Exception:
                    pass

            if terminated and reward_cfg is not None:
                elapsed_seconds = float(getattr(base, "_elapsed_frames", 0)) / 60.0
                if topout:
                    time_reward = float(getattr(reward_cfg, "time_bonus_topout_per_60_frames", 0.0)) * elapsed_seconds
                elif v_end == 0:
                    time_reward = -float(getattr(reward_cfg, "time_penalty_clear_per_60_frames", 0.0)) * elapsed_seconds
                if time_reward != 0.0:
                    r_env += float(time_reward)

            if int(getattr(base, "_t", 0) or 0) > int(getattr(base, "_t_max", 0) or 0):
                truncated = True

            if terminated or truncated:
                try:
                    base._elapsed_frames = 0
                except Exception:
                    pass

            total_reward = float(r_env)
            breakdown.r_total = float(total_reward)
            breakdown.r_env = float(r_env)
            breakdown.r_shape = 0.0
            breakdown.delta_v = int(delta_v)
            breakdown.tiles_cleared_total = int(max(0, cleared_total))
            breakdown.tiles_cleared_non_virus = int(max(0, cleared_nonvirus))
            breakdown.virus_clear_reward = float(virus_clear_reward)
            breakdown.non_virus_bonus = float(non_virus_bonus)
            breakdown.adjacency_bonus = float(adjacency_bonus)
            breakdown.virus_adjacency_bonus = float(virus_adjacency_bonus)
            breakdown.terminal_bonus = float(terminal_bonus_reward)
            breakdown.topout_penalty = float(topout_penalty_reward)
            breakdown.time_reward = float(time_reward)
            perf.env_step_calls = 0

            tau = int(max(1, int(plan_frames_executed) + int(wait_frames_total)))
            out_info = dict(info or {})
            out_info["t"] = int(getattr(base, "_t", 0) or 0)
            out_info["viruses_remaining"] = int(getattr(base, "_viruses_remaining", 0))
            out_info["delta_v"] = int(delta_v)
            out_info["r_env"] = float(r_env)
            out_info["r_shape"] = 0.0
            out_info["r_total"] = float(total_reward)
            out_info["virus_clear_reward"] = float(virus_clear_reward)
            out_info["non_virus_bonus"] = float(non_virus_bonus)
            out_info["adjacency_bonus"] = float(adjacency_bonus)
            out_info["virus_adjacency_bonus"] = float(virus_adjacency_bonus)
            out_info["terminal_bonus_reward"] = float(terminal_bonus_reward)
            out_info["topout_penalty_reward"] = float(topout_penalty_reward)
            out_info["time_reward"] = float(time_reward)
            out_info["tiles_cleared_total"] = int(max(0, cleared_total))
            out_info["tiles_cleared_non_virus"] = int(max(0, cleared_nonvirus))
            out_info["match_event"] = bool(match_event)
            # Omit when not in match-mode to keep vectorized infos type-stable.
            if match_target is not None:
                out_info["match_target"] = int(match_target or 0)
            out_info["matches_completed"] = int(matches_completed)
            out_info["task_mode"] = str(task_mode)
            out_info["goal_achieved"] = bool(goal_achieved)
            out_info["cleared"] = bool(goal_achieved)
            out_info["topout"] = bool(topout)
            tiles_clearing = 0
            if callable(getattr(base, "_count_clearing_tiles", None)):
                try:
                    tiles_clearing = int(base._count_clearing_tiles())
                except Exception:
                    tiles_clearing = 0
            out_info["tiles_clearing"] = int(max(0, tiles_clearing))
            out_info["clearing_active"] = bool(int(out_info["tiles_clearing"]) >= 4)

            if bool(getattr(base, "emit_raw_ram", False)):
                try:
                    out_info["raw_ram"] = base._raw_ram_bytes()
                except Exception:
                    pass

            if callable(getattr(base, "_augment_info", None)):
                try:
                    out_info = dict(base._augment_info(out_info))
                except Exception:
                    pass

            out_info["placements/tau"] = int(tau)
            out_info["placements/needs_action"] = bool(out_info.get("placements/needs_action", False))
            out_info["perf/planner_plan_sec"] = float(plan_latency_sec_total)
            out_info["placements/plan_calls"] = int(plan_calls)
            out_info["placements/plan_latency_ms_total"] = float(plan_latency_sec_total) * 1000.0
            out_info["placements/plan_latency_ms_max"] = float(plan_latency_sec_max) * 1000.0
            out_info["placements/plan_count_last"] = float(plan_options)
            out_info["placements/last_action"] = int(action)
            if target_pose is not None:
                out_info["placements/target_pose"] = tuple(int(v) for v in target_pose)

            out_info.update(self._reward_breakdown_info(breakdown, r_total=float(total_reward)))
            out_info.update(self._perf_breakdown_info(perf))
            self._attach_pose_mismatch_info(out_info)

            self._last_obs = obs
            self._last_info = dict(out_info)
            tau_raw = max(0, int(plan_frames_executed) + int(wait_frames_total))
            obs_out, r_out, term_out, trunc_out, info_out = self._finalize_step(
                obs, float(total_reward), bool(terminated), bool(truncated), out_info, tau_frames=tau_raw
            )
            return obs_out, r_out, term_out, trunc_out, info_out

        obs = self._last_obs
        info: Dict[str, Any] = dict(self._last_info)
        for step in plan.controller:
            self._set_holds(left=step.hold_left, right=step.hold_right, down=step.hold_down)
            obs, r, terminated, truncated, info = self.env.step(int(step.action))
            total_reward += float(r)
            if isinstance(info, dict):
                breakdown.add_frame(reward=float(r), info=info)
                perf.add_frame(info=info)
                self._update_lock_capture(lock_capture, info)
            if terminated or truncated:
                break

        # After locking, clear holds and advance until the next decision.
        self._clear_holds()
        if not (terminated or truncated):
            obs, info, r_wait, terminated, truncated = self._advance_until_decision(
                obs, info, breakdown=breakdown, perf=perf, lock_capture=lock_capture
            )
            total_reward += float(r_wait)

        self._last_obs = obs
        self._last_info = dict(info or {})

        frames_end = int(getattr(self.env.unwrapped, "_t", frames_start) or frames_start)
        tau = max(1, frames_end - frames_start)
        out_info = dict(self._last_info)
        out_info["placements/tau"] = int(tau)
        out_info["placements/needs_action"] = bool(out_info.get("placements/needs_action", False))
        out_info["perf/planner_plan_sec"] = float(plan_latency_sec_total)
        out_info["placements/plan_calls"] = int(plan_calls)
        out_info["placements/plan_latency_ms_total"] = float(plan_latency_sec_total) * 1000.0
        out_info["placements/plan_latency_ms_max"] = float(plan_latency_sec_max) * 1000.0
        out_info["placements/plan_count_last"] = float(plan_options)
        out_info["placements/last_action"] = int(action)
        if target_pose is not None:
            out_info["placements/target_pose"] = tuple(int(v) for v in target_pose)
        lock_pose = lock_capture.get("lock_pose")
        lock_t = lock_capture.get("lock_t")
        lock_reason = lock_capture.get("lock_reason")
        if isinstance(lock_pose, (list, tuple)) and len(lock_pose) == 3:
            out_info["placements/lock_pose"] = tuple(int(v) for v in lock_pose)
        if lock_t is not None:
            try:
                out_info["placements/lock_t"] = int(lock_t)
            except Exception:
                pass
        if lock_reason:
            out_info["placements/lock_reason"] = str(lock_reason)
        mismatch_last = False
        mismatch_id: Optional[int] = None
        if target_pose is not None and isinstance(lock_pose, (list, tuple)) and len(lock_pose) == 3:
            lock_pose_t = tuple(int(v) for v in lock_pose)
            out_info["placements/pose_ok"] = bool(lock_pose_t == target_pose)
            if lock_pose_t != target_pose:
                out_info["placements/pose_dx"] = int(lock_pose_t[0] - target_pose[0])
                out_info["placements/pose_dy"] = int(lock_pose_t[1] - target_pose[1])
                out_info["placements/pose_drot"] = int(((lock_pose_t[2] - target_pose[2]) + 4) % 4)
                self._pose_mismatch_count += 1
                mismatch_last = True
                mismatch_id = int(self._pose_mismatch_count)
                try:
                    reader = getattr(self.env.unwrapped, "_read_bottle_grid_u8", None)
                    if callable(reader):
                        bottle_after = reader()
                except Exception:
                    bottle_after = None
                self._maybe_log_pose_mismatch(
                    mismatch_id=mismatch_id,
                    ctx=ctx,
                    action=int(action),
                    plan=plan,
                    target_pose=target_pose,
                    lock_capture=lock_capture,
                    bottle_before=bottle_before,
                    bottle_after=bottle_after,
                    out_info=out_info,
                )
        out_info.update(self._reward_breakdown_info(breakdown, r_total=float(total_reward)))
        out_info.update(self._perf_breakdown_info(perf))
        self._attach_pose_mismatch_info(out_info, mismatch_last=mismatch_last, mismatch_id=mismatch_id)
        tau_raw = max(0, frames_end - frames_start)
        obs_out, r_out, term_out, trunc_out, info_out = self._finalize_step(
            obs, float(total_reward), terminated, truncated, out_info, tau_frames=tau_raw
        )
        return obs_out, r_out, term_out, trunc_out, info_out


__all__ = ["DrMarioPlacementEnv"]


def make_placement_env(**kwargs: Any) -> gym.Env:
    """Gymnasium entry-point: construct a state-backed retro env and wrap it."""

    from envs.retro.drmario_env import DrMarioRetroEnv

    # Planner requires RAM/state access; enforce state observations.
    obs_mode = kwargs.get("obs_mode")
    if obs_mode is None or str(obs_mode).lower() != "state":
        kwargs["obs_mode"] = "state"

    planner = kwargs.pop("planner", None)
    max_wait_frames = int(kwargs.pop("max_wait_frames", 6000))
    debug = bool(kwargs.pop("debug", False))

    base = DrMarioRetroEnv(**kwargs)
    return DrMarioPlacementEnv(base, planner=planner, max_wait_frames=max_wait_frames, debug=debug)


__all__.append("make_placement_env")
