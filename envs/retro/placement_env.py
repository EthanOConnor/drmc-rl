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

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from envs.retro.drmario_env import Action
from envs.state_core import DrMarioState
from envs.retro.placement_planner import (
    BoardState,
    PillSnapshot,
    PlacementPlanner,
    PlannerError,
    SpawnReachability,
)
from envs.retro.placement_space import ORIENTATIONS, GRID_HEIGHT, GRID_WIDTH, TOTAL_ACTIONS


# Zero-page current-player address for `currentP_nextAction` (drmario_ram_zp.asm).
ZP_CURRENT_P_NEXT_ACTION = 0x0097
NEXT_ACTION_PILL_FALLING = 0  # jumpTable_nextAction index 0


def _read_u8(ram_bytes: bytes, addr: int) -> int:
    if addr < 0 or addr >= len(ram_bytes):
        return 0
    return int(ram_bytes[addr]) & 0xFF


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
        self.height_penalty_delta += self._as_float(info.get("height_penalty_delta"))
        self.action_penalty += self._as_float(info.get("action_penalty"))
        self.terminal_bonus += self._as_float(info.get("terminal_bonus_reward"))
        self.topout_penalty += self._as_float(info.get("topout_penalty_reward"))
        self.time_reward += self._as_float(info.get("time_reward"))


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
        self._ctx: Optional[_DecisionContext] = None
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

    # ------------------------------------------------------------------ utils

    def _state_cache(self) -> DrMarioState:
        state = getattr(self.env.unwrapped, "_state_cache", None)
        if state is None:
            raise RuntimeError("Underlying env does not expose _state_cache (state mode required)")
        return state

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
        info["pill/speed_counter"] = int(snap.speed_counter)
        info["pill/speed_threshold"] = int(snap.speed_threshold)
        info["pill/hor_velocity"] = int(snap.hor_velocity)
        info["pill/frame_parity"] = int(snap.frame_parity)
        info["pill/colors"] = np.asarray(snap.colors, dtype=np.int64)
        # Historical key used by placement policies
        info["next_pill_colors"] = np.asarray(snap.colors, dtype=np.int64)
        return info

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
            "reward/pill_bonus_adjusted": float(breakdown.pill_bonus_adjusted),
            "reward/height_penalty_delta": float(breakdown.height_penalty_delta),
            "reward/action_penalty": float(breakdown.action_penalty),
            "reward/terminal_bonus": float(breakdown.terminal_bonus),
            "reward/topout_penalty": float(breakdown.topout_penalty),
            "reward/time_reward": float(breakdown.time_reward),
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
            state = self._state_cache()
            if self._at_decision_point(state):
                ctx = self._build_decision_context()
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
                    return obs, out_info, float(total_reward), terminated, truncated
                saw_no_feasible = True
                # Mark this spawn as consumed (no feasible actions) so we don't
                # treat subsequent falling frames as new decisions.
                if ctx.snapshot.spawn_id is not None:
                    self._consumed_spawn_id = int(ctx.snapshot.spawn_id)

            obs, r, terminated, truncated, info = self.env.step(int(Action.NOOP))
            total_reward += float(r)
            if breakdown is not None and isinstance(info, dict):
                breakdown.add_frame(reward=float(r), info=info)
            if lock_capture is not None and isinstance(info, dict):
                if lock_capture.get("lock_pose") is None:
                    try:
                        pb = float(info.get("pill_bonus_adjusted", 0.0) or 0.0)
                    except Exception:
                        pb = 0.0
                    if pb > 0.0:
                        offsets = getattr(self.env.unwrapped, "_ram_offsets", {})
                        try:
                            snap_lock = PillSnapshot.from_state(self._state_cache(), offsets)
                            lock_capture["lock_pose"] = (
                                int(snap_lock.base_col),
                                int(snap_lock.base_row),
                                int(snap_lock.rot) & 0x03,
                            )
                        except Exception:
                            lock_capture["lock_pose"] = None
                        try:
                            lock_capture["lock_t"] = int(info.get("t", getattr(self.env.unwrapped, "_t", 0) or 0))
                        except Exception:
                            lock_capture["lock_t"] = None
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

    # ------------------------------------------------------------------ gym api

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._last_info = dict(info or {})
        self._ctx = None
        self._consumed_spawn_id = None
        obs, info, _, terminated, truncated = self._advance_until_decision(obs, self._last_info)
        self._last_obs = obs
        self._last_info = dict(info or {})
        if terminated or truncated:
            return obs, info
        return obs, info

    def step(self, action: int):
        frames_start_any = int(getattr(self.env.unwrapped, "_t", 0) or 0)

        total_reward = 0.0
        breakdown = _RewardBreakdown()
        terminated = False
        truncated = False
        # Debug/validation: capture the observed pose when the pill locks so we
        # can confirm the controller script achieved the intended placement.
        target_pose: Optional[Tuple[int, int, int]] = None  # (x,y,rot)
        lock_capture: Dict[str, Any] = {"lock_pose": None, "lock_t": None}
        plan_calls = 0
        plan_latency_sec_total = 0.0
        plan_latency_sec_max = 0.0
        plan_options = 0

        # Ensure we are at a decision point.
        state = self._state_cache()
        if not self._at_decision_point(state):
            obs, info, r_wait, terminated, truncated = self._advance_until_decision(
                self._last_obs, self._last_info, breakdown=breakdown
            )
            total_reward += float(r_wait)
            self._last_obs, self._last_info = obs, dict(info or {})
            if terminated or truncated:
                frames_end = int(getattr(self.env.unwrapped, "_t", frames_start_any) or frames_start_any)
                out_info = dict(self._last_info)
                out_info["placements/tau"] = max(1, frames_end - frames_start_any)
                out_info.update(self._reward_breakdown_info(breakdown, r_total=float(total_reward)))
                return obs, float(total_reward), terminated, truncated, out_info

        # Start τ after we have reached a decision for this spawn.
        frames_start = int(getattr(self.env.unwrapped, "_t", frames_start_any) or frames_start_any)

        if self._ctx is None:
            self._ctx = self._build_decision_context()

        ctx = self._ctx
        # If the planner reports no feasible in-bounds macro placements, we
        # cannot accept any `action` and must advance the underlying env until
        # it either terminates (top-out) or reaches a later controllable spawn.
        plan_options = int(ctx.reach.feasible_mask.sum())
        if plan_options == 0:
            # Consume this spawn (no feasible macro actions) to avoid returning
            # repeated "decisions" while the offscreen pill locks/top-outs.
            if ctx.snapshot.spawn_id is not None:
                self._consumed_spawn_id = int(ctx.snapshot.spawn_id)
            obs, info, r_wait, terminated, truncated = self._advance_until_decision(
                self._last_obs, self._last_info, breakdown=breakdown
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
            return obs, float(total_reward), terminated, truncated, out_info
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
            return self._last_obs, float(total_reward), False, False, out_info

        try:
            if plan.terminal_pose is not None:
                bx, by, rot = plan.terminal_pose
                target_pose = (int(bx), int(by), int(rot) & 0x03)
        except Exception:
            target_pose = None

        # We are committing to a macro action for this spawn; mark it as
        # consumed so `_advance_until_decision` ignores subsequent falling frames
        # until the next pill spawns.
        if ctx.snapshot.spawn_id is not None:
            self._consumed_spawn_id = int(ctx.snapshot.spawn_id)
        # Invalidate cached ctx while we execute the plan.
        self._ctx = None

        # Execute the per-frame script.
        obs = self._last_obs
        info: Dict[str, Any] = dict(self._last_info)
        offsets = getattr(self.env.unwrapped, "_ram_offsets", {})
        for step in plan.controller:
            self._set_holds(left=step.hold_left, right=step.hold_right, down=step.hold_down)
            obs, r, terminated, truncated, info = self.env.step(int(step.action))
            total_reward += float(r)
            if isinstance(info, dict):
                breakdown.add_frame(reward=float(r), info=info)
                if lock_capture.get("lock_pose") is None:
                    try:
                        pb = float(info.get("pill_bonus_adjusted", 0.0) or 0.0)
                    except Exception:
                        pb = 0.0
                    if pb > 0.0:
                        try:
                            snap_lock = PillSnapshot.from_state(self._state_cache(), offsets)
                            lock_capture["lock_pose"] = (
                                int(snap_lock.base_col),
                                int(snap_lock.base_row),
                                int(snap_lock.rot) & 0x03,
                            )
                        except Exception:
                            lock_capture["lock_pose"] = None
                        try:
                            lock_capture["lock_t"] = int(info.get("t", getattr(self.env.unwrapped, "_t", 0) or 0))
                        except Exception:
                            lock_capture["lock_t"] = None
            if terminated or truncated:
                break

        # After locking, clear holds and advance until the next decision.
        self._clear_holds()
        if not (terminated or truncated):
            obs, info, r_wait, terminated, truncated = self._advance_until_decision(
                obs, info, breakdown=breakdown, lock_capture=lock_capture
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
        if isinstance(lock_pose, (list, tuple)) and len(lock_pose) == 3:
            out_info["placements/lock_pose"] = tuple(int(v) for v in lock_pose)
        if lock_t is not None:
            try:
                out_info["placements/lock_t"] = int(lock_t)
            except Exception:
                pass
        if target_pose is not None and isinstance(lock_pose, (list, tuple)) and len(lock_pose) == 3:
            lock_pose_t = tuple(int(v) for v in lock_pose)
            out_info["placements/pose_ok"] = bool(lock_pose_t == target_pose)
            if lock_pose_t != target_pose:
                out_info["placements/pose_dx"] = int(lock_pose_t[0] - target_pose[0])
                out_info["placements/pose_dy"] = int(lock_pose_t[1] - target_pose[1])
                out_info["placements/pose_drot"] = int(((lock_pose_t[2] - target_pose[2]) + 4) % 4)
        out_info.update(self._reward_breakdown_info(breakdown, r_total=float(total_reward)))
        return obs, float(total_reward), terminated, truncated, out_info


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
