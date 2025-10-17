"""Stable-Retro helpers for Dr. Mario.

These helpers keep Stable-Retro specifics out of the main env class. They assume you've already
imported the game integration and have a libretro NES core available.

Environment variables respected:
- DRMARIO_CORE_PATH: explicit path to a libretro NES core (mesen/nestopia). Optional.
- DRMARIO_GAME: retro game name string (default 'DrMario-Nes', adjust to your import).
- DRMARIO_STATE: retro initial state (savestate name). Default 'LevelSelect'.
"""
from __future__ import annotations
import os
from typing import Optional, Tuple


def make_retro_env():
    import retro  # provided by stable-retro
    game = os.environ.get("DRMARIO_GAME", "DrMario-Nes")
    state = os.environ.get("DRMARIO_STATE", "LevelSelect")
    # NOTE: If your integration uses a different name/state, change here.
    kwargs = {}
    core_path = os.environ.get("DRMARIO_CORE_PATH")
    if core_path:
        kwargs["use_restricted_actions"] = retro.Actions.DISCRETE
        # stable-retro may accept core path via retro.make(..., inttype, **kwargs) or via env var.
        os.environ.setdefault("LIBRETRO_CORE", core_path)
    env = retro.make(game=game, state=state, **kwargs)
    return env


def get_buttons_layout() -> Tuple[str, ...]:
    """NES layout as used by most libretro cores in Retro: order matters for the button array.
    We'll assume the classic 8-button order: ['B','NULL','SELECT','START','UP','DOWN','LEFT','RIGHT','A']
    Adjust to your core if needed.
    """
    # Many cores expose 'B','A','SELECT','START','UP','DOWN','LEFT','RIGHT'; we use that here.
    return ("B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT")

