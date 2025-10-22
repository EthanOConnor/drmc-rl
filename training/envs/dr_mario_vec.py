from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

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
        base = self._rng.random(self.observation_space.shape, dtype=np.float32)
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
        frame = np.clip(self._obs[0] * 255.0, 0, 255).astype(np.uint8)
        # Arrange to (H, W, C)
        frame = np.transpose(frame[0], (1, 2, 0))
        return frame

    def close(self) -> None:  # pragma: no cover - trivial cleanup hook
        pass


def make_vec_env(cfg: VecEnvConfig | Dict[str, object] | object) -> DummyVecEnv:
    """Factory that returns a vectorised Dr. Mario environment instance."""

    if isinstance(cfg, VecEnvConfig):
        env_cfg = cfg
    else:
        env_dict = getattr(cfg, "env", cfg)
        if not isinstance(env_dict, dict):
            env_dict = {k: getattr(env_dict, k) for k in dir(env_dict) if not k.startswith("_")}
        env_cfg = VecEnvConfig(**{k: env_dict.get(k, getattr(VecEnvConfig, k)) for k in VecEnvConfig.__annotations__})
    return DummyVecEnv(env_cfg)
