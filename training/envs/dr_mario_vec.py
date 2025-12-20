from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from gymnasium import spaces

__all__ = ["make_vec_env", "VecEnvConfig", "DummyVecEnv"]


@dataclass
class VecEnvConfig:
    """Configuration describing the vector environment."""

    id: str = "DrMario-v0"
    obs_mode: str = "pixels"
    num_envs: int = 1
    frame_stack: int = 1
    render: bool = False
    episode_length: int = 200
    randomize_rng: bool = False
    core: Optional[str] = None
    core_path: Optional[str] = None
    rom_path: Optional[str] = None
    backend: Optional[str] = None
    level: int = 0
    risk_tau: float = 1.0
    include_risk_tau: bool = False
    action_space: Optional[str] = None  # controller|intent|placement
    state_repr: Optional[str] = None
    vectorization: str = "auto"  # auto|sync|async
    emit_raw_ram: bool = True


class DummyVecEnv:
    """Lightweight vector environment used for smoke tests and dry runs.

    The implementation intentionally mimics the Gymnasium vector-API without
    pulling in the full retro Dr. Mario dependency tree. Each environment is a
    simple controllable Markov chain that terminates after a fixed horizon.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, cfg: VecEnvConfig, seed: Optional[int] = None) -> None:
        self.cfg = cfg
        self.num_envs = int(cfg.num_envs)
        self._episode_length = int(max(cfg.episode_length, 1))
        obs_channels = 3 if cfg.obs_mode == "pixels" else 2
        obs_shape = (cfg.frame_stack, obs_channels, 16, 16)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)
        self.single_action_space = spaces.Discrete(6)
        self._rng = np.random.default_rng(seed)
        self._episode_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._combo_max = np.zeros(self.num_envs, dtype=np.int32)
        self._viruses = np.zeros(self.num_envs, dtype=np.int32)
        self._lines = np.zeros(self.num_envs, dtype=np.int32)
        self._risk = np.zeros(self.num_envs, dtype=np.float32)
        self._drop_speed = np.zeros(self.num_envs, dtype=np.int32)
        self._top_out = np.zeros(self.num_envs, dtype=bool)
        self._last_actions = np.zeros(self.num_envs, dtype=np.int32)
        self._obs = self._sample_obs()

    def _sample_obs(self) -> np.ndarray:
        full_shape = (self.num_envs,) + self.observation_space.shape
        base = self._rng.random(full_shape, dtype=np.float32)
        return base.astype(np.float32)

    def reset(self, *, seed: Optional[int] = None) -> Tuple[np.ndarray, List[Dict[str, object]]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._episode_steps.fill(0)
        self._episode_returns.fill(0.0)
        self._combo_max.fill(0)
        self._viruses.fill(0)
        self._lines.fill(0)
        self._risk.fill(0.0)
        self._drop_speed.fill(0)
        self._top_out.fill(False)
        self._last_actions.fill(0)
        self._obs = self._sample_obs()
        return self._obs.copy(), [{} for _ in range(self.num_envs)]

    def step(
        self, actions: Iterable[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, object]]]:
        actions_arr = np.asarray(list(actions), dtype=np.int32)
        if actions_arr.shape[0] != self.num_envs:
            raise ValueError("Expected %d actions, received %d" % (self.num_envs, actions_arr.shape[0]))

        rewards = self._rng.normal(loc=0.5, scale=0.25, size=self.num_envs).astype(np.float32)
        self._episode_returns += rewards
        self._episode_steps += 1
        self._combo_max = np.maximum(self._combo_max, actions_arr % 4)
        self._viruses += (actions_arr % 3 == 0).astype(np.int32)
        self._lines += (actions_arr % 2 == 0).astype(np.int32)
        self._risk = np.clip(self._risk * 0.8 + 0.2 * (actions_arr / 5.0), 0.0, 1.0)
        self._drop_speed = (self._drop_speed + 1) % 20
        self._top_out |= actions_arr == self.single_action_space.n - 1
        self._last_actions = actions_arr

        terminated = self._episode_steps >= self._episode_length
        truncated = np.zeros_like(terminated, dtype=bool)
        next_obs = self._sample_obs()
        infos: List[Dict[str, object]] = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            if terminated[i]:
                infos[i] = {
                    "episode": {"r": float(self._episode_returns[i]), "l": int(self._episode_steps[i])},
                    "drm": {
                        "viruses_cleared": int(self._viruses[i]),
                        "lines_cleared": int(self._lines[i]),
                        "drop_speed_level": int(self._drop_speed[i]),
                        "top_out": bool(self._top_out[i]),
                        "combo_max": int(self._combo_max[i]),
                        "risk": float(self._risk[i]),
                    },
                }
        self._obs = next_obs
        reset_indices = np.where(terminated | truncated)[0]
        if reset_indices.size > 0:
            for idx in reset_indices:
                self._episode_steps[idx] = 0
                self._episode_returns[idx] = 0.0
                self._combo_max[idx] = 0
                self._viruses[idx] = 0
                self._lines[idx] = 0
                self._risk[idx] = 0.0
                self._drop_speed[idx] = 0
                self._top_out[idx] = False
        return next_obs.copy(), rewards, terminated, truncated, infos

    def render(self) -> np.ndarray:
        frame_stack = self._obs[0]
        frame = np.clip(frame_stack[-1] * 255.0, 0, 255).astype(np.uint8)
        # Arrange to (H, W, C)
        frame = np.transpose(frame, (1, 2, 0))
        return frame

    def close(self) -> None:  # pragma: no cover - trivial cleanup hook
        pass


def make_vec_env(cfg: VecEnvConfig | Dict[str, object] | object) -> DummyVecEnv:
    """Factory that returns a vectorised Dr. Mario environment instance.

    The training stack historically used a lightweight `DummyVecEnv` for unit
    tests and early prototyping. For actual training, we also support building a
    real Gymnasium VectorEnv backed by `envs.retro.DrMarioRetroEnv` (libretro or
    mock).

    Selection rule:
      - If `env.id` is one of the registered Dr. Mario env ids, build a real
        vector env.
      - Otherwise, fall back to `DummyVecEnv` (keeps unit tests fast and
        dependency-light).
    """

    if isinstance(cfg, VecEnvConfig):
        env_cfg = cfg
    else:
        env_dict = getattr(cfg, "env", cfg)
        if not isinstance(env_dict, dict):
            if hasattr(env_dict, "to_dict"):
                env_dict = env_dict.to_dict()  # type: ignore[assignment]
            else:
                env_dict = {k: getattr(env_dict, k) for k in dir(env_dict) if not k.startswith("_")}
        # Backwards-compatible config key.
        if "id" not in env_dict and "env_id" in env_dict:
            env_dict = dict(env_dict)
            env_dict["id"] = env_dict.get("env_id")
        env_cfg = VecEnvConfig(**{k: env_dict.get(k, getattr(VecEnvConfig, k)) for k in VecEnvConfig.__annotations__})
    env_id = str(env_cfg.id)
    real_ids = {
        "DrMarioRetroEnv-v0",
        "DrMarioIntentEnv-v0",
        "DrMarioPlacementEnv-v0",
        "DrMario-Placement-v0",  # legacy alias
    }
    if env_id not in real_ids:
        env: Any = DummyVecEnv(env_cfg)
    else:
        env = _make_real_vec_env(env_cfg, seed=getattr(cfg, "seed", None))

    curriculum_cfg = None
    try:
        curriculum_cfg = getattr(cfg, "curriculum", None)
    except Exception:
        curriculum_cfg = None
    if curriculum_cfg is None and isinstance(cfg, dict):
        curriculum_cfg = cfg.get("curriculum")
    if curriculum_cfg is not None:
        try:
            enabled = bool(getattr(curriculum_cfg, "enabled", False))
        except Exception:
            enabled = bool(curriculum_cfg.get("enabled", False)) if isinstance(curriculum_cfg, dict) else False
        if enabled and hasattr(env, "set_attr"):
            from training.envs.curriculum import CurriculumConfig, CurriculumVecEnv

            env = CurriculumVecEnv(env, CurriculumConfig.from_cfg(curriculum_cfg))

    return env


# --------------------------------------------------------------------------- real env


def _normalize_obs_mode(obs_mode: str) -> str:
    value = str(obs_mode or "").lower()
    if value in {"pixel", "pixels", "rgb"}:
        return "pixel"
    if value in {"state", "ram"}:
        return "state"
    return value or "state"


def _resolve_real_env_id(env_cfg: VecEnvConfig) -> str:
    action_space = str(env_cfg.action_space or "").lower()
    if action_space in {"controller", "ctrl"}:
        return "DrMarioRetroEnv-v0"
    if action_space in {"intent"}:
        return "DrMarioIntentEnv-v0"
    if action_space in {"placement"}:
        return "DrMarioPlacementEnv-v0"
    return str(env_cfg.id)


def _wrap_last_frame_if_needed(env: Any, *, frame_stack: int) -> Any:
    """If the env emits a fixed frame stack, optionally keep only the last frame."""

    if int(frame_stack) != 1:
        return env

    import gymnasium as gym

    class _LastFrameObsWrapper(gym.ObservationWrapper):
        def __init__(self, inner: gym.Env):
            super().__init__(inner)
            space = getattr(inner, "observation_space", None)
            if isinstance(space, spaces.Box) and len(space.shape) >= 1 and space.shape[0] > 1:
                self.observation_space = spaces.Box(
                    low=space.low[0],
                    high=space.high[0],
                    shape=space.shape[1:],
                    dtype=space.dtype,
                )

        def observation(self, observation: Any) -> Any:
            arr = np.asarray(observation)
            if arr.ndim >= 1 and arr.shape[0] > 1:
                return arr[-1]
            return observation

    try:
        space = getattr(env, "observation_space", None)
        if isinstance(space, spaces.Box) and len(space.shape) >= 1 and space.shape[0] > 1:
            return _LastFrameObsWrapper(env)
    except Exception:
        pass
    return env


def _make_real_vec_env(env_cfg: VecEnvConfig, seed: Optional[int] = None) -> Any:
    import gymnasium as gym
    from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

    env_id = _resolve_real_env_id(env_cfg)
    obs_mode = _normalize_obs_mode(env_cfg.obs_mode)

    # Placement env requires state access for planner snapshots/masks.
    if env_id in {"DrMarioPlacementEnv-v0", "DrMario-Placement-v0"}:
        obs_mode = "state"

    backend = str(env_cfg.backend) if env_cfg.backend else None
    kwargs: Dict[str, Any] = {
        "obs_mode": obs_mode,
        "level": int(env_cfg.level),
        "risk_tau": float(env_cfg.risk_tau),
        "include_risk_tau": bool(env_cfg.include_risk_tau),
        "backend": backend,
        "rng_randomize": bool(env_cfg.randomize_rng),
        "emit_raw_ram": bool(getattr(env_cfg, "emit_raw_ram", True)),
        # Always enable rgb_array rendering so debug UIs and video handlers can
        # call `env.render()` without needing a separate env instance.
        "render_mode": "rgb_array",
    }
    if env_cfg.core_path:
        kwargs["core_path"] = env_cfg.core_path
    if env_cfg.rom_path:
        kwargs["rom_path"] = env_cfg.rom_path
    if env_cfg.state_repr:
        kwargs["state_repr"] = env_cfg.state_repr

    def make_env(rank: int):
        def _init():
            # AsyncVectorEnv creates envs in subprocesses. Register the Dr. Mario env
            # ids inside each worker so `gym.make()` succeeds.
            from envs.retro.register_env import (
                register_env_id,
                register_intent_env_id,
                register_placement_env_id,
            )

            register_env_id()
            register_intent_env_id()
            register_placement_env_id()

            env = gym.make(env_id, **kwargs)
            env = _wrap_last_frame_if_needed(env, frame_stack=int(env_cfg.frame_stack))
            return env

        return _init

    num_envs = int(max(1, env_cfg.num_envs))
    vectorization = str(env_cfg.vectorization or "auto").lower()
    if vectorization == "sync" or num_envs == 1:
        base_env = SyncVectorEnv([make_env(i) for i in range(num_envs)])
    elif vectorization == "async" or vectorization == "auto":
        base_env = AsyncVectorEnv([make_env(i) for i in range(num_envs)])
    else:
        raise ValueError(f"Unknown vectorization mode: {env_cfg.vectorization}")

    return _InfoListWrapper(base_env)


# Gymnasium vector envs return `infos` as a dict-of-arrays. Our training adapters
# and tooling expect a list-of-dicts (one per env), matching the historical
# DummyVecEnv interface.
#
# While we adapt the info structure, we also attach standard episode statistics
# (`info["episode"] = {"r": return, "l": length}`) so training algorithms can
# report returns/lengths consistently across backends.
class _InfoListWrapper:
    def __init__(self, env: Any):
        self.env = env
        self._num_envs = int(getattr(env, "num_envs", 1))
        self._episode_returns = np.zeros(self._num_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(self._num_envs, dtype=np.int64)  # frames (uses placements/tau when present)
        self._episode_decisions = np.zeros(self._num_envs, dtype=np.int64)  # wrapper step calls

        self._viruses_prev = np.full(self._num_envs, -1, dtype=np.int32)
        self._episode_viruses_cleared = np.zeros(self._num_envs, dtype=np.int64)

    def reset(self, *args: Any, **kwargs: Any):
        obs, infos = self.env.reset(*args, **kwargs)
        infos_list = self._unbatch_infos(infos)
        self._episode_returns.fill(0.0)
        self._episode_lengths.fill(0)
        self._episode_decisions.fill(0)
        self._episode_viruses_cleared.fill(0)
        self._viruses_prev.fill(-1)
        for i, info in enumerate(infos_list):
            v = self._extract_int(info.get("viruses_remaining"))
            if v is not None:
                self._viruses_prev[i] = int(v)
        return obs, infos_list

    def step(self, actions: Any):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        infos_list = self._unbatch_infos(infos)

        rewards_arr = np.asarray(rewards, dtype=np.float64).reshape(self._num_envs)
        terminated_arr = np.asarray(terminated, dtype=bool).reshape(self._num_envs)
        truncated_arr = np.asarray(truncated, dtype=bool).reshape(self._num_envs)

        for i in range(self._num_envs):
            info = infos_list[i] if i < len(infos_list) else {}

            tau = self._extract_tau(info)
            self._episode_returns[i] += float(rewards_arr[i])
            self._episode_lengths[i] += int(tau)
            self._episode_decisions[i] += 1

            v_now = self._extract_int(info.get("viruses_remaining"))
            if v_now is not None:
                v_prev = int(self._viruses_prev[i])
                if v_prev >= 0:
                    self._episode_viruses_cleared[i] += max(0, int(v_prev) - int(v_now))
                self._viruses_prev[i] = int(v_now)

            if bool(terminated_arr[i] or truncated_arr[i]):
                info["episode"] = {
                    "r": float(self._episode_returns[i]),
                    "l": int(self._episode_lengths[i]),
                    "decisions": int(self._episode_decisions[i]),
                }
                drm: Dict[str, Any] = {
                    "viruses_cleared": int(self._episode_viruses_cleared[i]),
                }
                if v_now is not None:
                    drm["viruses_remaining"] = int(v_now)
                topout = info.get("topout")
                if topout is not None:
                    drm["top_out"] = bool(topout)
                cleared = info.get("cleared")
                if cleared is not None:
                    drm["cleared"] = bool(cleared)
                level = self._extract_int(info.get("level"))
                if level is not None:
                    drm["level"] = int(level)
                info["drm"] = drm

                self._episode_returns[i] = 0.0
                self._episode_lengths[i] = 0
                self._episode_decisions[i] = 0
                self._episode_viruses_cleared[i] = 0

        return obs, rewards, terminated, truncated, infos_list

    def render(self, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self.env, "render"):
            return self.env.render(*args, **kwargs)
        return None

    def close(self) -> None:
        env = getattr(self, "env", None)
        if env is None or not hasattr(env, "close"):
            return

        # Gymnasium's AsyncVectorEnv can hang on close if a worker crashed or a
        # pending call was interrupted mid-message. Prefer a force-terminate
        # shutdown path for robustness under long async runs.
        try:
            from gymnasium.vector.async_vector_env import AsyncState, AsyncVectorEnv

            if isinstance(env, AsyncVectorEnv):
                try:
                    if getattr(env, "_state", AsyncState.DEFAULT) != AsyncState.DEFAULT:
                        env._state = AsyncState.DEFAULT
                except Exception:
                    pass
                try:
                    env.close(terminate=True)
                    return
                except TypeError:
                    env.close()
                    return
        except Exception:
            pass

        try:
            env.close()
        except TypeError:
            try:
                env.close()
            except Exception:
                pass

    def _unbatch_infos(self, infos: Any) -> List[Dict[str, Any]]:
        n = self._num_envs
        if infos is None:
            return [{} for _ in range(n)]
        if isinstance(infos, (list, tuple)):
            return [dict(i) if isinstance(i, dict) else {} for i in infos]
        if not isinstance(infos, dict):
            return [{} for _ in range(n)]
        return self._split_info_dict(infos, n)

    def _split_info_dict(self, infos: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = [dict() for _ in range(n)]
        for key, value in infos.items():
            if isinstance(key, str) and key.startswith("_"):
                continue

            mask = infos.get(f"_{key}") if isinstance(key, str) else None
            has_mask = hasattr(mask, "__len__") and len(mask) == n

            if isinstance(value, dict):
                sub = self._split_info_dict(value, n)
                if has_mask:
                    for i, has in enumerate(mask):
                        if bool(has):
                            out[i][key] = sub[i]
                else:
                    for i in range(n):
                        out[i][key] = sub[i]
                continue

            try:
                arr = np.asarray(value)
            except Exception:
                arr = None
            if arr is not None and arr.shape[:1] == (n,):
                if has_mask:
                    for i, has in enumerate(mask):
                        if bool(has):
                            out[i][key] = value[i]
                else:
                    for i in range(n):
                        out[i][key] = value[i]
            else:
                for i in range(n):
                    out[i][key] = value
        return out

    @staticmethod
    def _extract_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            try:
                value = value.item()
            except Exception:
                return None
        try:
            return int(value)
        except Exception:
            return None

    @classmethod
    def _extract_tau(cls, info: Dict[str, Any]) -> int:
        tau = info.get("placements/tau", 1)
        tau_int = cls._extract_int(tau)
        return max(1, int(tau_int) if tau_int is not None else 1)

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self.env, name)
