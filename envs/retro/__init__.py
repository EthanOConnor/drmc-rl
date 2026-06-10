"""Dr. Mario placement helpers and emulator parity/debug wrappers.

This package provides:
- active placement planner/reachability modules used by training paths;
- DrMarioRetroEnv for emulator-backed parity/debug sessions;
- seed registry and libretro core configuration utilities.

Normal training uses `backend=cpp-pool` through `training.envs.drmario_pool_vec`.
"""

from .drmario_env import DrMarioRetroEnv, Action

__all__ = ["DrMarioRetroEnv", "Action"]
