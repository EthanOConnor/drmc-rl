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
        return DummyVecEnv(env_cfg)

    return _make_real_vec_env(env_cfg, seed=getattr(cfg, "seed", None))


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

    from envs.retro.register_env import register_env_id, register_intent_env_id, register_placement_env_id

    register_env_id()
    register_intent_env_id()
    register_placement_env_id()

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
            env = gym.make(env_id, **kwargs)
            env = _wrap_last_frame_if_needed(env, frame_stack=int(env_cfg.frame_stack))
            if seed is not None:
                env.reset(seed=int(seed) + int(rank))
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

    # Gymnasium vector envs return `infos` as a dict-of-arrays. Our training
    # adapters and tooling expect a list-of-dicts (one per env), matching the
    # historical DummyVecEnv interface.
    class _InfoListWrapper:
        def __init__(self, env: Any):
            self.env = env

        def reset(self, *args: Any, **kwargs: Any):
            obs, infos = self.env.reset(*args, **kwargs)
            return obs, self._unbatch_infos(infos)

        def step(self, actions: Any):
            obs, rewards, terminated, truncated, infos = self.env.step(actions)
            return obs, rewards, terminated, truncated, self._unbatch_infos(infos)

        def render(self, *args: Any, **kwargs: Any) -> Any:
            if hasattr(self.env, "render"):
                return self.env.render(*args, **kwargs)
            return None

        def close(self) -> None:
            if hasattr(self.env, "close"):
                self.env.close()

        def _unbatch_infos(self, infos: Any) -> List[Dict[str, Any]]:
            n = int(getattr(self.env, "num_envs", num_envs))
            if infos is None:
                return [{} for _ in range(n)]
            if isinstance(infos, (list, tuple)):
                return [dict(i) for i in infos]
            if not isinstance(infos, dict):
                return [{} for _ in range(n)]
            out: List[Dict[str, Any]] = [dict() for _ in range(n)]
            for key, value in infos.items():
                try:
                    arr = np.asarray(value)
                except Exception:
                    arr = None
                if arr is not None and arr.shape[:1] == (n,):
                    for i in range(n):
                        out[i][key] = value[i]
                else:
                    for i in range(n):
                        out[i][key] = value
            return out

        def __getattr__(self, name: str) -> Any:  # pragma: no cover - passthrough
            return getattr(self.env, name)

    return _InfoListWrapper(base_env)
