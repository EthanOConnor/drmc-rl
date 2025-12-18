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
        self._last_obs: Any = None
        self._last_info: Dict[str, Any] = {}

    # ------------------------------------------------------------------ utils

    def _state_cache(self) -> DrMarioState:
        state = getattr(self.env.unwrapped, "_state_cache", None)
        if state is None:
            raise RuntimeError("Underlying env does not expose _state_cache (state mode required)")
        return state

    def _at_decision_point(self, state: DrMarioState) -> bool:
        # Decision point == currentP_nextAction == pillFalling, and a falling pill exists.
        nxt = _read_u8(state.ram.bytes, ZP_CURRENT_P_NEXT_ACTION)
        if nxt != NEXT_ACTION_PILL_FALLING:
            return False
        try:
            return bool(state.calc.falling_mask.any())
        except Exception:
            return False

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
        info["placements/legal_mask"] = ctx.reach.legal_mask.copy()
        info["placements/feasible_mask"] = ctx.reach.feasible_mask.copy()
        info["placements/costs"] = ctx.reach.costs.copy()
        info["placements/options"] = int(ctx.reach.feasible_mask.sum())
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

    def _build_decision_context(self) -> _DecisionContext:
        state = self._state_cache()
        offsets = getattr(self.env.unwrapped, "_ram_offsets", {})
        snap = PillSnapshot.from_state(state, offsets)
        board = BoardState.from_planes(state.calc.planes)
        reach = self._planner.build_spawn_reachability(board, snap)
        # Symmetry reduction: identical colors => drop H-/V- duplicates.
        if int(snap.colors[0]) == int(snap.colors[1]):
            reach = SpawnReachability(
                legal_mask=reach.legal_mask.copy(),
                feasible_mask=reach.feasible_mask.copy(),
                costs=reach.costs.copy(),
                reach=reach.reach,
                action_to_terminal_node=dict(reach.action_to_terminal_node),
            )
            reach.feasible_mask[2, :, :] = False
            reach.feasible_mask[3, :, :] = False
            reach.legal_mask[2, :, :] = False
            reach.legal_mask[3, :, :] = False
            reach.costs[2, :, :] = np.inf
            reach.costs[3, :, :] = np.inf
            # Remove disabled actions from the terminal map.
            to_drop = []
            for a in reach.action_to_terminal_node:
                o = a // (GRID_HEIGHT * GRID_WIDTH)
                if o in (2, 3):
                    to_drop.append(a)
            for a in to_drop:
                reach.action_to_terminal_node.pop(a, None)
        return _DecisionContext(snapshot=snap, board=board, reach=reach)

    def _advance_until_decision(self, obs: Any, info: Dict[str, Any]) -> Tuple[Any, Dict[str, Any], float, bool, bool]:
        """Advance the underlying env until the next macro decision point."""

        total_reward = 0.0
        terminated = False
        truncated = False

        self._clear_holds()
        for _ in range(self._max_wait_frames):
            state = self._state_cache()
            if self._at_decision_point(state):
                self._ctx = self._build_decision_context()
                out_info = dict(info or {})
                out_info.update(self._decision_info(self._ctx))
                out_info["placements/needs_action"] = True
                return obs, out_info, float(total_reward), terminated, truncated

            obs, r, terminated, truncated, info = self.env.step(int(Action.NOOP))
            total_reward += float(r)
            if terminated or truncated:
                break

        # Timed out or terminated.
        out_info = dict(info or {})
        out_info["placements/needs_action"] = False
        if self._debug:
            out_info["placements/wait_timeout"] = True
        return obs, out_info, float(total_reward), terminated, truncated

    # ------------------------------------------------------------------ gym api

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._last_info = dict(info or {})
        self._ctx = None
        obs, info, _, terminated, truncated = self._advance_until_decision(obs, self._last_info)
        self._last_obs = obs
        self._last_info = dict(info or {})
        if terminated or truncated:
            return obs, info
        return obs, info

    def step(self, action: int):
        frames_start = int(getattr(self.env.unwrapped, "_t", 0) or 0)

        total_reward = 0.0
        terminated = False
        truncated = False

        # Ensure we are at a decision point.
        state = self._state_cache()
        if not self._at_decision_point(state):
            obs, info, r_wait, terminated, truncated = self._advance_until_decision(
                self._last_obs, self._last_info
            )
            total_reward += float(r_wait)
            self._last_obs, self._last_info = obs, dict(info or {})
            if terminated or truncated:
                frames_end = int(getattr(self.env.unwrapped, "_t", frames_start) or frames_start)
                out_info = dict(self._last_info)
                out_info["placements/tau"] = max(1, frames_end - frames_start)
                return obs, float(total_reward), terminated, truncated, out_info

        if self._ctx is None:
            self._ctx = self._build_decision_context()

        ctx = self._ctx
        # Refresh decision context if spawn id changed.
        try:
            snap_now = PillSnapshot.from_state(self._state_cache(), getattr(self.env.unwrapped, "_ram_offsets", {}))
            if ctx.snapshot.spawn_id is not None and snap_now.spawn_id is not None and snap_now.spawn_id != ctx.snapshot.spawn_id:
                ctx = self._build_decision_context()
                self._ctx = ctx
        except Exception:
            pass

        plan = self._planner.plan_action(ctx.reach, int(action))
        if plan is None:
            # Invalid choice; surface mask again and request a new decision.
            out_info = dict(self._last_info)
            out_info.update(self._decision_info(ctx))
            out_info["placements/needs_action"] = True
            out_info["placements/invalid_action"] = int(action)
            frames_end = int(getattr(self.env.unwrapped, "_t", frames_start) or frames_start)
            out_info["placements/tau"] = max(1, frames_end - frames_start)
            return self._last_obs, 0.0, False, False, out_info

        # Execute the per-frame script.
        obs = self._last_obs
        info: Dict[str, Any] = dict(self._last_info)
        for step in plan.controller:
            self._set_holds(left=step.hold_left, right=step.hold_right, down=step.hold_down)
            obs, r, terminated, truncated, info = self.env.step(int(step.action))
            total_reward += float(r)
            if terminated or truncated:
                break

        # After locking, clear holds and advance until the next decision.
        self._clear_holds()
        if not (terminated or truncated):
            obs, info, r_wait, terminated, truncated = self._advance_until_decision(obs, info)
            total_reward += float(r_wait)

        self._last_obs = obs
        self._last_info = dict(info or {})

        frames_end = int(getattr(self.env.unwrapped, "_t", frames_start) or frames_start)
        tau = max(1, frames_end - frames_start)
        out_info = dict(self._last_info)
        out_info["placements/tau"] = int(tau)
        out_info["placements/needs_action"] = bool(out_info.get("placements/needs_action", False))
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
