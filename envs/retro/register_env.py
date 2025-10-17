"""Gymnasium env registration helper for Dr. Mario Retro env.

Usage:
    from envs.retro.register_env import register_env_id
    register_env_id()  # registers 'DrMarioRetroEnv-v0'

This expects your class to be available at envs.retro.drmario_env:DrMarioRetroEnv
and that the constructor accepts **kwargs (obs_mode, level, seed, etc.).
"""
from gymnasium.envs.registration import register

_ENV_ID = "DrMarioRetroEnv-v0"
_ENTRY_POINT = "envs.retro.drmario_env:DrMarioRetroEnv"


def register_env_id(env_id: str = _ENV_ID):
    try:
        register(
            id=env_id,
            entry_point=_ENTRY_POINT,
            kwargs={},  # you can pass defaults here if desired
            max_episode_steps=None,
        )
    except Exception:
        # It's fine if it's already registered in dev workflows.
        pass


__all__ = ["register_env_id", "_ENV_ID"]

