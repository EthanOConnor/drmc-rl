"""Dr. Mario env wrappers with pluggable emulator backends (libretro default).

This package provides:
- DrMarioRetroEnv: single-agent Gymnasium Env with pixel/state observations
- Utilities for seed registry loading and core configuration
"""

from .drmario_env import DrMarioRetroEnv, Action

__all__ = ["DrMarioRetroEnv", "Action"]
