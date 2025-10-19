from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

import envs.specs.ram_to_state as ram_specs

# Skeleton to avoid hard dependency on PettingZoo at repo bootstrap time
try:
    from pettingzoo.utils import ParallelEnv
    from gymnasium import spaces
except Exception:  # pragma: no cover
    ParallelEnv = object  # type: ignore
    spaces = None  # type: ignore


class DrMarioVsEnv(ParallelEnv):  # type: ignore[misc]
    metadata = {"name": "drmario_vs"}

    def __init__(self, obs_mode: str = "pixel") -> None:
        super().__init__()  # type: ignore[misc]
        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]
        self.obs_mode = obs_mode
        self._t = 0

        # Minimal spaces mock; replace with real Gym spaces if PettingZoo present
        if spaces is not None:
            if obs_mode == "pixel":
                self.observation_spaces = {a: spaces.Box(0.0, 1.0, (4, 128, 128, 3), dtype=np.float32) for a in self.agents}
            else:
                self.observation_spaces = {
                    a: spaces.Box(0.0, 1.0, (4, ram_specs.STATE_CHANNELS, 16, 8), dtype=np.float32)
                    for a in self.agents
                }
            self.action_spaces = {a: spaces.Discrete(10) for a in self.agents}
        else:
            self.observation_spaces = {}
            self.action_spaces = {}

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        self._t = 0
        obs = {a: self._mock_obs() for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions: Dict[str, int]):
        self._t += 1
        obs = {a: self._mock_obs() for a in self.agents}
        rews = {a: -1.0 for a in self.agents}
        terms = {a: False for a in self.agents}
        truncs = {a: self._t > 18000 for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, rews, terms, truncs, infos

    def _mock_obs(self):
        if self.obs_mode == "pixel":
            return np.zeros((4, 128, 128, 3), dtype=np.float32)
        return np.zeros((4, ram_specs.STATE_CHANNELS, 16, 8), dtype=np.float32)
