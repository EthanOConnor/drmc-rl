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

_INTENT_ENV_ID = "DrMarioIntentEnv-v0"
_INTENT_ENTRY_POINT = "envs.retro.intent_wrapper:DrMarioIntentEnv"

_PLACEMENT_ENV_ID = "DrMarioPlacementEnv-v0"
_PLACEMENT_ENTRY_POINT = "envs.retro.placement_wrapper:DrMarioPlacementEnv"


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


def register_intent_env_id(env_id: str = _INTENT_ENV_ID):
    try:
        register(
            id=env_id,
            entry_point=_INTENT_ENTRY_POINT,
            kwargs={},
            max_episode_steps=None,
        )
    except Exception:
        pass


def register_placement_env_id(env_id: str = _PLACEMENT_ENV_ID):
    try:
        register(
            id=env_id,
            entry_point=_PLACEMENT_ENTRY_POINT,
            kwargs={},
            max_episode_steps=None,
        )
    except Exception:
        pass


__all__ = [
    "register_env_id",
    "register_intent_env_id",
    "register_placement_env_id",
    "_ENV_ID",
    "_INTENT_ENV_ID",
    "_PLACEMENT_ENV_ID",
]
