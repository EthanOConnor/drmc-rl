"""Example: env_ctor for seed_sweep to import without circular deps."""
from __future__ import annotations

import gymnasium as gym

from drmc_rl.envs.libretro.registration import register_env_id


def make_env(obs_mode='state', level=0, risk_tau=0.5, **kwargs):
    register_env_id()
    return gym.make('DrMarioLibretroEnv-v0', obs_mode=obs_mode, level=level, risk_tau=risk_tau, **kwargs)

