"""Stable-Retro based Dr. Mario env wrappers.

This package provides:
- DrMarioRetroEnv: single-agent Gymnasium Env with pixel/state observations
- Utilities for seed registry loading and core configuration
"""

from .drmario_env import DrMarioRetroEnv, Action

__all__ = ["DrMarioRetroEnv", "Action"]

