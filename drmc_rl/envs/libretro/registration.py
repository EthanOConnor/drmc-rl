"""Gymnasium env registration helper for Dr. Mario Retro env.

Usage:
    from drmc_rl.envs.libretro.registration import register_env_id
    register_env_id()  # registers 'DrMarioLibretroEnv-v0'

This expects your class to be available at drmc_rl.envs.libretro.gym_env:DrMarioLibretroEnv
and that the constructor accepts **kwargs (obs_mode, level, seed, etc.).
"""

from gymnasium.envs.registration import register, registry

_ENV_ID = "DrMarioLibretroEnv-v0"
_ENTRY_POINT = "drmc_rl.envs.libretro.gym_env:DrMarioLibretroEnv"

_INTENT_ENV_ID = "DrMarioIntentEnv-v0"
_INTENT_ENTRY_POINT = "drmc_rl.envs.libretro.intent_env:DrMarioIntentEnv"

_PLACEMENT_ENV_ID = "DrMarioPlacementEnv-v0"
_PLACEMENT_ENV_ID_LEGACY = "DrMario-Placement-v0"
_PLACEMENT_ENTRY_POINT = "drmc_rl.envs.libretro.placement_env:make_placement_env"


def register_env_id(env_id: str = _ENV_ID):
    # `register()` warns (and overrides) when called repeatedly. Check the
    # registry first so workers and test harnesses can safely call this.
    try:
        if env_id in registry:
            return
    except Exception:
        # Fall back to attempting registration.
        pass
    register(
        id=env_id,
        entry_point=_ENTRY_POINT,
        kwargs={},  # you can pass defaults here if desired
        max_episode_steps=None,
    )


def register_intent_env_id(env_id: str = _INTENT_ENV_ID):
    try:
        if env_id in registry:
            return
    except Exception:
        pass
    register(
        id=env_id,
        entry_point=_INTENT_ENTRY_POINT,
        kwargs={},
        max_episode_steps=None,
    )


def register_placement_env_id(env_id: str = _PLACEMENT_ENV_ID):
    # Keep a legacy env id for older scripts/configs.
    for candidate_id in (env_id, _PLACEMENT_ENV_ID_LEGACY):
        try:
            if candidate_id in registry:
                continue
        except Exception:
            # If registry access fails, still try to register.
            pass
        register(
            id=candidate_id,
            entry_point=_PLACEMENT_ENTRY_POINT,
            kwargs={},
            max_episode_steps=None,
        )


__all__ = [
    "register_env_id",
    "register_intent_env_id",
    "register_placement_env_id",
    "_ENV_ID",
    "_INTENT_ENV_ID",
    "_PLACEMENT_ENV_ID",
    "_PLACEMENT_ENV_ID_LEGACY",
]
