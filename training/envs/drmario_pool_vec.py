from __future__ import annotations

"""Batched, in-process C++ pool vector environment for placement-SMDP training.

This env bypasses Gymnasium's `VectorEnv` wrappers and instead talks directly to
`libdrmario_pool` via ctypes. It is designed to be *training-hot-path*:
  - one C call per macro decision batch,
  - numpy buffers are persistent (no per-step allocations),
  - Python does only reward/curriculum bookkeeping + policy interaction.

The API intentionally matches the lightweight vector env contract used by
`training/algo/*` and `training/envs/curriculum.py`:
  - reset() -> (obs[N,...], infos[N])
  - step(actions[N]) -> (obs, rewards, terminated, truncated, infos)
  - set_attr(name, values)
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from gymnasium import spaces

import envs.specs.ram_to_state as ram_specs
from envs.backends.drmario_pool import (
    GRID_H,
    GRID_W,
    MACRO_ACTIONS,
    DrMarioPoolRunner,
    build_reset_spec,
)


def _canonical_to_raw_color(canonical: int) -> int:
    """Map canonical color idx (0=R,1=Y,2=B) -> NES raw bits (Y=0,R=1,B=2)."""

    c = int(canonical) & 0x03
    if c == 0:
        return 1
    if c == 1:
        return 0
    if c == 2:
        return 2
    return 0


def _resolve_reward_config_path() -> Optional[Path]:
    env_override = os.environ.get("DRMARIO_REWARD_CONFIG")
    if env_override:
        return Path(env_override).expanduser()
    return (Path(__file__).resolve().parents[2] / "envs" / "specs" / "reward_config.json").resolve()


@dataclass(frozen=True)
class _RewardCfg:
    virus_clear_bonus: float
    non_virus_clear_bonus: float
    terminal_clear_bonus: float
    topout_penalty: float
    time_bonus_topout_per_60_frames: float
    time_penalty_clear_per_60_frames: float
    adjacency_pair_bonus: float
    adjacency_triplet_bonus: float
    virus_adjacency_pair_bonus: float
    virus_adjacency_triplet_bonus: float

    @classmethod
    def load(cls) -> Tuple["_RewardCfg", Optional[Path]]:
        path = _resolve_reward_config_path()
        if path is None:
            return cls.default(), None
        if not path.is_file():
            raise FileNotFoundError(f"Reward config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        def _get(name: str, default: float) -> float:
            meta = payload.get(name, default)
            if isinstance(meta, dict) and "value" in meta:
                meta = meta["value"]
            try:
                return float(meta)
            except Exception:
                return float(default)

        cfg = cls(
            virus_clear_bonus=_get("virus_clear_bonus", 0.1),
            non_virus_clear_bonus=_get("non_virus_clear_bonus", 0.01),
            terminal_clear_bonus=_get("terminal_clear_bonus", 0.5),
            topout_penalty=_get("topout_penalty", -1.0),
            time_bonus_topout_per_60_frames=_get("time_bonus_topout_per_60_frames", 0.0),
            time_penalty_clear_per_60_frames=_get("time_penalty_clear_per_60_frames", 0.001),
            adjacency_pair_bonus=_get("adjacency_pair_bonus", 0.002),
            adjacency_triplet_bonus=_get("adjacency_triplet_bonus", 0.006),
            virus_adjacency_pair_bonus=_get("virus_adjacency_pair_bonus", 0.009),
            virus_adjacency_triplet_bonus=_get("virus_adjacency_triplet_bonus", 0.025),
        )
        return cfg, path

    @classmethod
    def default(cls) -> "_RewardCfg":
        return cls(
            virus_clear_bonus=0.1,
            non_virus_clear_bonus=0.01,
            terminal_clear_bonus=0.5,
            topout_penalty=-1.0,
            time_bonus_topout_per_60_frames=0.0,
            time_penalty_clear_per_60_frames=0.001,
            adjacency_pair_bonus=0.002,
            adjacency_triplet_bonus=0.006,
            virus_adjacency_pair_bonus=0.009,
            virus_adjacency_triplet_bonus=0.025,
        )


class DrMarioPoolVecEnv:
    """Vector environment backed by the in-process native pool library."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        *,
        num_envs: int,
        state_repr: str,
        level: int = 0,
        speed_setting: int = 2,
        randomize_rng: bool = False,
        include_risk_tau: bool = False,
        risk_tau: float = 1.0,
        emit_board: bool = False,
        max_lock_frames: int = 2048,
        max_wait_frames: int = 6000,
        lib_path: Optional[str] = None,
    ) -> None:
        self.num_envs = int(max(1, int(num_envs)))
        self.include_risk_tau = bool(include_risk_tau)
        self.default_risk_tau = float(risk_tau)
        self.rng_randomize = bool(randomize_rng)
        self.emit_board = bool(emit_board)
        self.speed_setting = int(max(0, min(int(speed_setting), 2)))

        state_repr_norm = str(state_repr or "").strip().lower()
        if state_repr_norm in {"bitplane_bottle", "bitplane-bottle"}:
            obs_spec = 1  # DRM_POOL_OBS_BITPLANE_BOTTLE
            obs_channels = 4
            ram_specs.set_state_representation("bitplane_bottle")
        elif state_repr_norm in {"bitplane_bottle_mask", "bitplane-bottle-mask"}:
            obs_spec = 2  # DRM_POOL_OBS_BITPLANE_BOTTLE_MASK
            obs_channels = 8
            ram_specs.set_state_representation("bitplane_bottle_mask")
        else:
            raise ValueError(
                "cpp-pool backend only supports state_repr in "
                "{bitplane_bottle, bitplane_bottle_mask}."
            )

        self.single_action_space = spaces.Discrete(MACRO_ACTIONS)
        self.single_observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_channels, GRID_H, GRID_W), dtype=np.float32
        )
        self.observation_space = self.single_observation_space

        self._levels = np.full((self.num_envs,), int(level), dtype=np.int32)
        self._task_max_frames = np.array([None for _ in range(self.num_envs)], dtype=object)
        self._task_max_spawns = np.array([None for _ in range(self.num_envs)], dtype=object)

        self._rng = np.random.default_rng(None)

        reward_cfg, reward_path = _RewardCfg.load()
        self._reward_cfg = reward_cfg
        self._reward_cfg_path = reward_path

        self._runner = DrMarioPoolRunner(
            num_envs=self.num_envs,
            obs_spec=obs_spec,
            obs_channels=obs_channels,
            max_lock_frames=int(max_lock_frames),
            max_wait_frames=int(max_wait_frames),
            lib_path=lib_path,
            emit_board=bool(self.emit_board),
        )

        # Views (no copies) into runner-owned buffers.
        self._obs = self._runner.buffers.obs
        self._mask_1d = self._runner.buffers.feasible_mask
        self._mask = self._mask_1d.reshape(self.num_envs, 4, GRID_H, GRID_W)
        self._cost_1d = self._runner.buffers.cost_to_lock
        self._cost = self._cost_1d.reshape(self.num_envs, 4, GRID_H, GRID_W)

        # Persistent per-env infos (updated in-place).
        self._infos: List[Dict[str, Any]] = [dict() for _ in range(self.num_envs)]

        # Autoreset (Gymnasium VectorEnv NEXT_STEP semantics).
        self._pending_reset = np.zeros((self.num_envs,), dtype=bool)

        # Episode stats (for `info["episode"]` + `info["drm"]`).
        self._ep_return = np.zeros((self.num_envs,), dtype=np.float64)
        self._ep_frames = np.zeros((self.num_envs,), dtype=np.int64)
        self._ep_decisions = np.zeros((self.num_envs,), dtype=np.int64)
        self._ep_viruses_cleared = np.zeros((self.num_envs,), dtype=np.int64)
        self._viruses_prev = np.full((self.num_envs,), -1, dtype=np.int32)
        self._viruses_initial = np.zeros((self.num_envs,), dtype=np.int32)
        self._elapsed_frames = np.zeros((self.num_envs,), dtype=np.int64)

        # Task tracking.
        self._task_frames_used = np.zeros((self.num_envs,), dtype=np.int64)
        self._task_spawns_used = np.zeros((self.num_envs,), dtype=np.int64)
        self._matches_completed = np.zeros((self.num_envs,), dtype=np.int64)

        # Decision-time state (spawn id) for spawn counting.
        self._spawn_id_at_decision = np.zeros((self.num_envs,), dtype=np.int32)

    # ------------------------------------------------------------------ vector API
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        opts = dict(options or {})
        if "randomize_rng" not in opts:
            opts["randomize_rng"] = bool(self.rng_randomize)

        specs = self._build_reset_specs(reset_mask=None, options=opts)
        self._runner.reset(None, specs)

        self._pending_reset.fill(False)
        self._ep_return.fill(0.0)
        self._ep_frames.fill(0)
        self._ep_decisions.fill(0)
        self._ep_viruses_cleared.fill(0)
        self._task_frames_used.fill(0)
        self._task_spawns_used.fill(0)
        self._matches_completed.fill(0)
        self._elapsed_frames.fill(0)

        v_now = self._runner.buffers.viruses_rem.astype(np.int32, copy=False)
        self._viruses_prev = v_now.copy()
        self._viruses_initial = v_now.copy()

        self._spawn_id_at_decision = self._runner.buffers.spawn_id.astype(np.int32, copy=True)
        self._apply_symmetry_reduction_in_place()
        self._refresh_infos(step_tau_raw=np.zeros((self.num_envs,), dtype=np.uint32), step_kind="reset")
        return self._wrap_obs(self._obs, risk_tau=opts.get("risk_tau")), list(self._infos)

    def step(
        self, actions: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        acts = np.asarray(list(actions), dtype=np.int32).reshape(self.num_envs)

        reset_mask = self._pending_reset.astype(np.uint8, copy=True)
        reset_specs = None
        if bool(reset_mask.any()):
            reset_specs = self._build_reset_specs(reset_mask=reset_mask, options={"randomize_rng": self.rng_randomize})

        # Preserve pre-step decision spawn ids for spawn accounting.
        spawn_before = self._spawn_id_at_decision.copy()

        self._runner.step(acts, reset_mask if reset_specs is not None else None, reset_specs)

        tau_raw = self._runner.buffers.tau_frames.astype(np.uint32, copy=False)
        invalid = self._runner.buffers.invalid_action.astype(np.int32, copy=False)
        term_native = self._runner.buffers.terminated.astype(np.uint8, copy=False)
        trunc_native = self._runner.buffers.truncated.astype(np.uint8, copy=False)
        reason_native = self._runner.buffers.terminal_reason.astype(np.uint8, copy=False)
        match_events = self._runner.buffers.match_events.astype(np.uint16, copy=False)

        v_now = self._runner.buffers.viruses_rem.astype(np.int32, copy=False)

        # Update spawn id snapshot (post-step decision context).
        self._spawn_id_at_decision = self._runner.buffers.spawn_id.astype(np.int32, copy=True)

        # Rewards + done flags.
        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        terminated = np.zeros((self.num_envs,), dtype=bool)
        truncated = np.zeros((self.num_envs,), dtype=bool)

        delta_v_arr = np.zeros((self.num_envs,), dtype=np.int32)
        virus_clear_reward_arr = np.zeros((self.num_envs,), dtype=np.float32)
        nonvirus_bonus_arr = np.zeros((self.num_envs,), dtype=np.float32)
        adjacency_bonus_arr = np.zeros((self.num_envs,), dtype=np.float32)
        virus_adjacency_bonus_arr = np.zeros((self.num_envs,), dtype=np.float32)
        terminal_bonus_arr = np.zeros((self.num_envs,), dtype=np.float32)
        topout_penalty_arr = np.zeros((self.num_envs,), dtype=np.float32)
        time_reward_arr = np.zeros((self.num_envs,), dtype=np.float32)
        goal_achieved_arr = np.zeros((self.num_envs,), dtype=bool)
        topout_arr = np.zeros((self.num_envs,), dtype=bool)
        terminal_reason_arr: List[str] = ["" for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            is_reset = bool(reset_mask[i] != 0)

            tau_i_raw = int(tau_raw[i])
            tau_info = int(max(1, tau_i_raw))

            # Episode bookkeeping uses the public tau (>=1), matching `_InfoListWrapper`.
            self._ep_decisions[i] += 1
            self._ep_frames[i] += int(tau_info)
            self._elapsed_frames[i] += int(tau_i_raw)

            # Delta-v tracking (also used for episode viruses_cleared).
            v_prev = int(self._viruses_prev[i])
            v_i = int(v_now[i])
            if v_prev >= 0:
                self._ep_viruses_cleared[i] += max(0, v_prev - v_i)
            self._viruses_prev[i] = int(v_i)

            if is_reset:
                # NEXT_STEP autoreset: ignore action and return reset state.
                self._task_frames_used[i] = 0
                self._task_spawns_used[i] = 0
                self._matches_completed[i] = 0
                self._viruses_initial[i] = int(v_i)
                self._pending_reset[i] = False
                continue

            if int(invalid[i]) != -1:
                # Invalid action: no advance.
                continue

            # Accepted action: update task counters.
            self._task_frames_used[i] += int(tau_i_raw)
            sp_after = int(self._spawn_id_at_decision[i]) & 0xFF
            sp_before = int(spawn_before[i]) & 0xFF
            sp_diff = int((sp_after - sp_before) & 0xFF)
            sp_delta = sp_diff if sp_diff > 0 else 1
            self._task_spawns_used[i] += int(sp_delta)

            # Match events: count rising edges observed during the macro step.
            me = int(match_events[i])
            if me > 0:
                self._matches_completed[i] += int(me)

            # Native terminal conditions (clear/topout/timeout).
            topout = bool(int(term_native[i]) != 0 and int(reason_native[i]) == 2)
            cleared_native = bool(int(term_native[i]) != 0 and int(reason_native[i]) == 1)
            timeout_native = bool(int(trunc_native[i]) != 0 or int(reason_native[i]) == 3)

            task_mode, match_target, synthetic_target = self._task_spec_for_level(int(self._levels[i]))
            goal_achieved = False
            goal_reason: Optional[str] = None
            if not topout:
                if task_mode == "matches":
                    target = int(match_target or 1)
                    if int(self._matches_completed[i]) >= target:
                        goal_achieved = True
                        goal_reason = f"match_{int(self._matches_completed[i])}"
                else:
                    goal_achieved = bool(cleared_native or int(v_i) == 0)
                    if goal_achieved:
                        goal_reason = "clear"

            # Reward components.
            r_env = 0.0
            delta_v = int(max(0, v_prev - v_i)) if v_prev >= 0 else 0
            virus_clear_reward = 0.0
            virus_bonus = float(self._reward_cfg.virus_clear_bonus)
            if delta_v > 0 and virus_bonus != 0.0:
                v0 = int(self._viruses_initial[i])
                if v0 > 0:
                    virus_clear_reward = virus_bonus * float(delta_v) / float(v0)
            nonvirus_bonus = float(self._reward_cfg.non_virus_clear_bonus) * float(
                int(self._runner.buffers.tiles_cleared_nonvirus[i])
            )

            adj_pair = self._runner.buffers.adj_pair[i].astype(np.int32, copy=False)
            adj_triplet = self._runner.buffers.adj_triplet[i].astype(np.int32, copy=False)
            v_adj_pair = self._runner.buffers.virus_adj_pair[i].astype(np.int32, copy=False)
            v_adj_triplet = self._runner.buffers.virus_adj_triplet[i].astype(np.int32, copy=False)

            adjacency_bonus = (
                float(self._reward_cfg.adjacency_triplet_bonus) * float(int(adj_triplet.sum()))
                + float(self._reward_cfg.adjacency_pair_bonus) * float(int(adj_pair.sum()))
            )
            virus_adjacency_bonus = (
                float(self._reward_cfg.virus_adjacency_triplet_bonus) * float(int(v_adj_triplet.sum()))
                + float(self._reward_cfg.virus_adjacency_pair_bonus) * float(int(v_adj_pair.sum()))
            )

            terminal_bonus_reward = 0.0
            topout_penalty_reward = 0.0
            if goal_achieved and not topout:
                terminal_bonus_reward = float(self._reward_cfg.terminal_clear_bonus)
            if topout:
                topout_penalty_reward = float(self._reward_cfg.topout_penalty)
            time_reward = 0.0
            if goal_achieved or topout:
                elapsed_seconds = float(max(0, int(self._elapsed_frames[i]))) / 60.0
                if topout:
                    time_reward = float(self._reward_cfg.time_bonus_topout_per_60_frames) * elapsed_seconds
                elif task_mode == "viruses" and int(v_i) == 0:
                    time_reward = -float(self._reward_cfg.time_penalty_clear_per_60_frames) * elapsed_seconds

            r_env = float(
                virus_clear_reward
                + nonvirus_bonus
                + adjacency_bonus
                + virus_adjacency_bonus
                + terminal_bonus_reward
                + topout_penalty_reward
                + time_reward
            )

            # Populate done flags. (Budgets are applied later as soft constraints.)
            done = bool(topout or goal_achieved)
            if timeout_native and not done:
                truncated[i] = True
                done = True
            terminated[i] = bool(done and not truncated[i])

            rewards[i] = float(r_env)
            delta_v_arr[i] = int(delta_v)
            virus_clear_reward_arr[i] = float(virus_clear_reward)
            nonvirus_bonus_arr[i] = float(nonvirus_bonus)
            adjacency_bonus_arr[i] = float(adjacency_bonus)
            virus_adjacency_bonus_arr[i] = float(virus_adjacency_bonus)
            terminal_bonus_arr[i] = float(terminal_bonus_reward)
            topout_penalty_arr[i] = float(topout_penalty_reward)
            time_reward_arr[i] = float(time_reward)
            goal_achieved_arr[i] = bool(goal_achieved)
            topout_arr[i] = bool(topout)
            if topout:
                terminal_reason_arr[i] = "topout"
            elif goal_reason:
                terminal_reason_arr[i] = str(goal_reason)
            elif timeout_native:
                terminal_reason_arr[i] = "timeout"

        # Apply symmetry reduction after each call (decision outputs).
        self._apply_symmetry_reduction_in_place()

        # Attach per-env info dicts.
        self._refresh_infos(step_tau_raw=tau_raw, step_kind="step")

        # Fill reward/status fields computed above.
        for i in range(self.num_envs):
            info_i = self._infos[i]
            info_i["delta_v"] = int(delta_v_arr[i])
            info_i["virus_clear_reward"] = float(virus_clear_reward_arr[i])
            info_i["non_virus_bonus"] = float(nonvirus_bonus_arr[i])
            info_i["adjacency_bonus"] = float(adjacency_bonus_arr[i])
            info_i["virus_adjacency_bonus"] = float(virus_adjacency_bonus_arr[i])
            info_i["terminal_bonus_reward"] = float(terminal_bonus_arr[i])
            info_i["topout_penalty_reward"] = float(topout_penalty_arr[i])
            info_i["time_reward"] = float(time_reward_arr[i])

            info_i["goal_achieved"] = bool(goal_achieved_arr[i])
            info_i["cleared"] = bool(goal_achieved_arr[i])
            info_i["topout"] = bool(topout_arr[i])
            info_i["terminal_reason"] = str(terminal_reason_arr[i] or "")

            r_env = float(rewards[i])
            info_i["r_env"] = r_env
            info_i["r_shape"] = 0.0
            info_i["r_total"] = r_env

            # `reward/*` mirrors (preferred by the debug UI wrappers).
            info_i["reward/r_env"] = r_env
            info_i["reward/r_shape"] = 0.0
            info_i["reward/r_total"] = r_env
            info_i["reward/delta_v"] = int(delta_v_arr[i])
            info_i["reward/virus_clear_reward"] = float(virus_clear_reward_arr[i])
            info_i["reward/non_virus_bonus"] = float(nonvirus_bonus_arr[i])
            info_i["reward/adjacency_bonus"] = float(adjacency_bonus_arr[i])
            info_i["reward/virus_adjacency_bonus"] = float(virus_adjacency_bonus_arr[i])
            info_i["reward/terminal_bonus"] = float(terminal_bonus_arr[i])
            info_i["reward/topout_penalty"] = float(topout_penalty_arr[i])
            info_i["reward/time_reward"] = float(time_reward_arr[i])

        # Apply soft budgets and shape terminal bonus if needed.
        self._apply_task_budgets_in_place(rewards, terminated, truncated, step_tau_raw=tau_raw)

        # Keep reward mirrors in sync after budget shaping.
        for i in range(self.num_envs):
            info_i = self._infos[i]
            try:
                info_i["reward/r_env"] = float(info_i.get("r_env", rewards[i]))
                info_i["reward/r_total"] = float(info_i.get("r_total", rewards[i]))
                info_i["reward/terminal_bonus"] = float(info_i.get("terminal_bonus_reward", 0.0))
            except Exception:
                pass

        # Episode return uses the final reward (post-budget-shaping).
        self._ep_return += rewards.astype(np.float64)

        for i in range(self.num_envs):
            try:
                self._infos[i]["placements/needs_action"] = not bool(terminated[i] or truncated[i])
            except Exception:
                pass

        # Finalize episode info + mark pending resets.
        for i in range(self.num_envs):
            if bool(terminated[i] or truncated[i]):
                info_i = self._infos[i]
                info_i["episode"] = {
                    "r": float(self._ep_return[i]),
                    "l": int(self._ep_frames[i]),
                    "decisions": int(self._ep_decisions[i]),
                }
                info_i["drm"] = {
                    "viruses_cleared": int(self._ep_viruses_cleared[i]),
                    "viruses_remaining": int(self._runner.buffers.viruses_rem[i]),
                    "viruses_initial": int(self._viruses_initial[i]),
                    "top_out": bool(info_i.get("topout", False)),
                    "cleared": bool(info_i.get("cleared", False)),
                    "level": int(self._levels[i]),
                    "speed_setting": int(self.speed_setting),
                }

                # Reset episode counters for the next episode (NEXT_STEP autoreset).
                self._ep_return[i] = 0.0
                self._ep_frames[i] = 0
                self._ep_decisions[i] = 0
                self._ep_viruses_cleared[i] = 0
                self._elapsed_frames[i] = 0
                self._task_frames_used[i] = 0
                self._task_spawns_used[i] = 0
                self._matches_completed[i] = 0
                self._pending_reset[i] = True

        obs_out = self._wrap_obs(self._obs, risk_tau=None)
        return obs_out, rewards.astype(np.float32), terminated, truncated, list(self._infos)

    def close(self) -> None:
        if hasattr(self._runner, "close"):
            self._runner.close()

    def render(self, *args: Any, **kwargs: Any) -> Optional[np.ndarray]:
        try:
            from envs.retro.state_viz import state_to_rgb
        except Exception:
            return None
        try:
            info0 = self._infos[0] if self._infos else {}
            frame = self._obs[0].astype(np.float32, copy=False)
            stack = frame[None, ...]
            return state_to_rgb(stack, info0)
        except Exception:
            return None

    def set_attr(self, name: str, values: Any, indices: Optional[Sequence[int]] = None) -> None:
        idxs = list(range(self.num_envs)) if indices is None else [int(i) for i in indices]
        if isinstance(values, (list, tuple, np.ndarray)) and not isinstance(values, (str, bytes, bytearray)):
            vals = list(values)
        else:
            vals = [values for _ in idxs]
        if len(vals) != len(idxs):
            raise ValueError(f"set_attr expected {len(idxs)} values, got {len(vals)}")

        if name == "level":
            for i, v in zip(idxs, vals):
                self._levels[int(i)] = int(v)
            return
        if name == "task_max_frames":
            for i, v in zip(idxs, vals):
                self._task_max_frames[int(i)] = None if v is None else int(v)
            return
        if name == "task_max_spawns":
            for i, v in zip(idxs, vals):
                self._task_max_spawns[int(i)] = None if v is None else int(v)
            return
        if name in {"rng_randomize", "randomize_rng"}:
            try:
                self.rng_randomize = bool(vals[0])
            except Exception:
                self.rng_randomize = bool(values)
            return
        if name == "emit_board":
            self.emit_board = bool(vals[0])
            return
        # Best-effort fallback: set an attribute on the instance.
        try:
            setattr(self, name, vals[0] if len(vals) == 1 else list(vals))
        except Exception:
            return

    # ------------------------------------------------------------------ internals
    def _wrap_obs(self, obs: np.ndarray, risk_tau: Any) -> Any:
        if not bool(self.include_risk_tau):
            return obs
        tau = self.default_risk_tau if risk_tau is None else float(risk_tau)
        return {"obs": obs, "risk_tau": np.asarray(tau, dtype=np.float32)}

    @staticmethod
    def _task_spec_for_level(level: int) -> Tuple[str, Optional[int], Optional[int]]:
        lvl = int(level)
        if lvl < 0:
            if lvl <= -4:
                return "matches", int(max(1, 16 + lvl)), 0
            return "viruses", None, int(max(0, 4 + lvl))
        return "viruses", None, None

    def _build_reset_specs(
        self, *, reset_mask: Optional[np.ndarray], options: Dict[str, Any]
    ) -> List[object]:
        rng_randomize = bool(options.get("randomize_rng", False))
        risk_tau = options.get("risk_tau", None)
        _ = risk_tau  # obs wrapper handles risk_tau; reset spec does not currently use it.

        rng_seed_bytes_opt = options.get("rng_seed_bytes")
        fixed_seed_bytes: Optional[Tuple[int, int]] = None
        if isinstance(rng_seed_bytes_opt, (list, tuple)) and len(rng_seed_bytes_opt) >= 2:
            try:
                fixed_seed_bytes = (int(rng_seed_bytes_opt[0]) & 0xFF, int(rng_seed_bytes_opt[1]) & 0xFF)
            except Exception:
                fixed_seed_bytes = None

        specs: List[object] = []
        for i in range(self.num_envs):
            if reset_mask is not None and int(reset_mask[i]) == 0:
                # Placeholder; ignored by native code.
                specs.append(build_reset_spec())
                continue

            lvl = int(self._levels[i])
            task_mode, match_target, synthetic_target = self._task_spec_for_level(lvl)
            _ = task_mode, match_target

            if fixed_seed_bytes is not None:
                seed_bytes = fixed_seed_bytes
                rng_override = True
            elif rng_randomize:
                seed_bytes = (int(self._rng.integers(0, 256)) & 0xFF, int(self._rng.integers(0, 256)) & 0xFF)
                rng_override = True
            else:
                seed_bytes = (0, 0)
                rng_override = False

            patch_counter = bool(synthetic_target is not None and int(synthetic_target) > 0)
            synthetic_seed = int(self._rng.integers(0, 2**32)) & 0xFFFFFFFF

            # Negative curriculum levels still run the ROM at level 0.
            rom_level = int(max(0, lvl))
            spec = build_reset_spec(
                level=rom_level,
                speed_setting=int(self.speed_setting),
                speed_ups=0,
                rng_state=seed_bytes,
                rng_override=rng_override,
                synthetic_virus_target=-1 if synthetic_target is None else int(synthetic_target),
                synthetic_patch_counter=patch_counter,
                synthetic_seed=synthetic_seed,
            )
            specs.append(spec)
        return specs

    def _apply_symmetry_reduction_in_place(self) -> None:
        # Symmetry reduction: identical colors => drop H-/V- duplicates (o=2,3).
        colors = self._runner.buffers.pill_colors
        if colors.shape != (self.num_envs, 2):
            return
        same = colors[:, 0] == colors[:, 1]
        if not bool(np.any(same)):
            return
        idxs = np.nonzero(same)[0]
        if idxs.size <= 0:
            return
        self._mask[idxs, 2, :, :] = 0
        self._mask[idxs, 3, :, :] = 0
        self._cost[idxs, 2, :, :] = np.uint16(0xFFFF)
        self._cost[idxs, 3, :, :] = np.uint16(0xFFFF)
        # If obs includes feasibility planes, keep them consistent.
        if self._obs.shape[1] >= 8:
            self._obs[idxs, 6, :, :] = 0.0
            self._obs[idxs, 7, :, :] = 0.0

    def _refresh_infos(self, *, step_tau_raw: np.ndarray, step_kind: str) -> None:
        _ = step_kind
        tiles_total = self._runner.buffers.tiles_cleared_total
        tiles_nonvirus = self._runner.buffers.tiles_cleared_nonvirus
        tiles_virus = self._runner.buffers.tiles_cleared_virus
        match_events = self._runner.buffers.match_events
        invalid = self._runner.buffers.invalid_action
        pill_colors = self._runner.buffers.pill_colors
        preview_colors = self._runner.buffers.preview_colors
        spawn_ids = self._runner.buffers.spawn_id
        viruses = self._runner.buffers.viruses_rem

        options_count = self._mask_1d.sum(axis=1).astype(np.int32, copy=False)

        for i in range(self.num_envs):
            info = self._infos[i]
            info.clear()

            # Decision-time outputs.
            info["placements/feasible_mask"] = self._mask[i]
            info["placements/cost_to_lock"] = self._cost[i]
            info["placements/options"] = int(options_count[i])
            info["placements/reach_backend"] = "cpp-pool"
            info["placements/needs_action"] = True
            info["placements/spawn_id"] = int(spawn_ids[i])
            info["pill/spawn_id"] = int(spawn_ids[i])

            info["next_pill_colors"] = np.asarray(pill_colors[i], dtype=np.int64)
            p_left = _canonical_to_raw_color(int(preview_colors[i, 0]))
            p_right = _canonical_to_raw_color(int(preview_colors[i, 1]))
            info["preview_pill"] = {"first_color": int(p_left), "second_color": int(p_right), "rotation": 0}

            # Debug board bytes (optional).
            if self.emit_board and self._runner.buffers.board_bytes is not None:
                info["board"] = self._runner.buffers.board_bytes[i]

            # Generic env info.
            lvl = int(self._levels[i])
            task_mode, match_target, synthetic_target = self._task_spec_for_level(lvl)
            info["level"] = lvl
            info["curriculum_level"] = lvl
            info["task_mode"] = str(task_mode)
            if synthetic_target is not None:
                info["synthetic_virus_target"] = int(synthetic_target)
            if match_target is not None:
                info["match_target"] = int(match_target)
            info["matches_completed"] = int(self._matches_completed[i])
            info["rng_randomize"] = bool(self.rng_randomize)
            info["viruses_remaining"] = int(viruses[i])
            info["drm/viruses_initial"] = int(self._viruses_initial[i])
            info["pill/speed_setting"] = int(self.speed_setting)
            info["speed_setting"] = int(self.speed_setting)

            # Task counters at decision points (mirrors `DrMarioPlacementEnv` keys).
            max_frames_raw = self._task_max_frames[i]
            max_spawns_raw = self._task_max_spawns[i]
            try:
                max_frames = None if max_frames_raw is None else int(max_frames_raw)
            except Exception:
                max_frames = None
            try:
                max_spawns = None if max_spawns_raw is None else int(max_spawns_raw)
            except Exception:
                max_spawns = None
            if max_frames is not None:
                info["task/max_frames"] = int(max_frames)
            if max_spawns is not None:
                info["task/max_spawns"] = int(max_spawns)
            info["task/frames_used"] = int(self._task_frames_used[i])
            info["task/spawns_used"] = int(self._task_spawns_used[i])

            # Step outputs (only meaningful for accepted actions; zeros for invalid/reset envs).
            tau_i_raw = int(step_tau_raw[i]) if i < int(step_tau_raw.shape[0]) else 0
            info["placements/tau"] = int(max(1, tau_i_raw))
            if int(invalid[i]) != -1:
                info["placements/invalid_action"] = int(invalid[i])
            info["tiles_cleared_total"] = int(tiles_total[i])
            info["tiles_cleared_non_virus"] = int(tiles_nonvirus[i])
            info["tiles_cleared_virus"] = int(tiles_virus[i])
            info["clear_events"] = int(match_events[i])
            info["match_event"] = bool(int(match_events[i]) > 0)

            # Reward keys populated after reward computation/budget shaping.
            info["delta_v"] = 0
            info["virus_clear_reward"] = 0.0
            info["non_virus_bonus"] = 0.0
            info["adjacency_bonus"] = 0.0
            info["virus_adjacency_bonus"] = 0.0
            info["terminal_bonus_reward"] = 0.0
            info["topout_penalty_reward"] = 0.0
            info["time_reward"] = 0.0
            info["r_env"] = 0.0
            info["r_shape"] = 0.0
            info["r_total"] = 0.0

            info["goal_achieved"] = False
            info["cleared"] = False
            info["topout"] = False
            info["terminal_reason"] = ""

            info["reward_config_loaded"] = bool(self._reward_cfg_path is not None)
            if self._reward_cfg_path is not None:
                info["reward_config_path"] = str(self._reward_cfg_path)

    def _apply_task_budgets_in_place(
        self,
        rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        *,
        step_tau_raw: np.ndarray,
    ) -> None:
        # Replicates the wrapper-level soft-budget semantics in
        # `envs/retro/placement_env.py::_finalize_step`.
        for i in range(self.num_envs):
            info = self._infos[i]

            tau_i_raw = int(step_tau_raw[i]) if i < int(step_tau_raw.shape[0]) else 0
            objective_met = bool(info.get("goal_achieved", False))

            max_frames_raw = self._task_max_frames[i]
            max_spawns_raw = self._task_max_spawns[i]
            try:
                max_frames = None if max_frames_raw is None else int(max_frames_raw)
            except Exception:
                max_frames = None
            try:
                max_spawns = None if max_spawns_raw is None else int(max_spawns_raw)
            except Exception:
                max_spawns = None

            frames_exceeded = bool(max_frames is not None and int(self._task_frames_used[i]) > int(max_frames))
            spawns_exceeded = bool(max_spawns is not None and int(self._task_spawns_used[i]) > int(max_spawns))
            budget_exceeded = bool(frames_exceeded or spawns_exceeded)
            within_budget = not budget_exceeded
            success = bool(objective_met and within_budget)

            if "goal_achieved" not in info:
                info["goal_achieved"] = False
            if "cleared" not in info:
                info["cleared"] = False

            # If objective met with budgets enabled, replace terminal clear bonus with time-goal reward.
            if objective_met and (max_frames is not None or max_spawns is not None):
                metric = "frames" if max_frames is not None else "spawns"
                goal = int(max_frames) if max_frames is not None else int(max_spawns or 0)
                used = int(self._task_frames_used[i]) if metric == "frames" else int(self._task_spawns_used[i])
                delta = int(goal) - int(used)

                old_bonus = 0.0
                try:
                    old_bonus = float(info.get("terminal_bonus_reward", 0.0) or 0.0)
                except Exception:
                    old_bonus = 0.0

                clear_bonus = float(old_bonus) if old_bonus != 0.0 else float(self._reward_cfg.terminal_clear_bonus)
                topout_penalty = float(self._reward_cfg.topout_penalty)
                if topout_penalty >= 0.0:
                    topout_penalty = -abs(float(topout_penalty))

                scale = max(1.0, 0.5 * float(max(1, abs(int(goal)))))
                y = float(np.tanh(float(delta) / float(scale)))
                new_bonus = float(y * clear_bonus) if y >= 0.0 else float(y * abs(topout_penalty))

                bonus_delta = float(new_bonus) - float(old_bonus)
                if bonus_delta != 0.0:
                    rewards[i] = float(rewards[i]) + float(bonus_delta)
                    for key in ("r_env", "r_total"):
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

                if not within_budget:
                    info["goal_achieved"] = False
                    info["cleared"] = False

            # Budgets are soft: never truncate just because budget was exceeded.
            _ = truncated, tau_i_raw

            info["task/objective_met"] = bool(objective_met)
            info["task/success"] = bool(success)
            info["task/within_budget"] = bool(within_budget)
            info["task/budget_exceeded"] = bool(budget_exceeded)
            info["task/budget_exceeded_frames"] = bool(frames_exceeded)
            info["task/budget_exceeded_spawns"] = bool(spawns_exceeded)
            if max_frames is not None:
                info["task/max_frames"] = int(max_frames)
            if max_spawns is not None:
                info["task/max_spawns"] = int(max_spawns)
            info["task/frames_used"] = int(self._task_frames_used[i])
            info["task/spawns_used"] = int(self._task_spawns_used[i])
            if budget_exceeded:
                reasons: List[str] = []
                if frames_exceeded:
                    reasons.append("frames")
                if spawns_exceeded:
                    reasons.append("spawns")
                if reasons:
                    info["task/budget_reason"] = "+".join(reasons)
