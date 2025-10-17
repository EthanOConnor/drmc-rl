"""Example: env_ctor for seed_sweep to import without circular deps."""
from __future__ import annotations
from envs.retro.register_env import register_env_id
import gymnasium as gym


def make_env(obs_mode='state', level=0, risk_tau=0.5, **kwargs):
    register_env_id()
    return gym.make('DrMarioRetroEnv-v0', obs_mode=obs_mode, level=level, risk_tau=risk_tau, **kwargs)

