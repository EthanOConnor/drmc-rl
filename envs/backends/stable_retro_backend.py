"""Stable-Retro backend for compatibility with the previous implementation."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from . import register_backend
from .base import EmulatorBackend, NES_BUTTONS

try:
    from envs.retro.stable_retro_utils import make_retro_env
except Exception:  # pragma: no cover - stable-retro optional
    make_retro_env = None  # type: ignore[assignment]


class StableRetroBackend(EmulatorBackend):
    """Adapter wrapping the Gymnasium Stable-Retro environment."""

    def __init__(self, **kwargs) -> None:
        if make_retro_env is None:
            raise RuntimeError("stable-retro is not available. Install the `retro` extra.")
        self._env = None
        self._frame = np.zeros((240, 256, 3), dtype=np.uint8)

    def load(self) -> None:
        self._env = make_retro_env()
        self._frame = np.zeros((240, 256, 3), dtype=np.uint8)

    def reset(self) -> None:
        if self._env is None:
            raise RuntimeError("Backend not loaded.")
        obs, _ = self._env.reset()
        self._update_frame(obs)

    def step(self, buttons: Sequence[int], repeat: int = 1) -> None:
        if self._env is None:
            raise RuntimeError("Backend not loaded.")
        if len(buttons) != len(NES_BUTTONS):
            raise ValueError(f"Expected {len(NES_BUTTONS)} buttons, got {len(buttons)}.")
        action = np.asarray(buttons, dtype=np.uint8)
        for _ in range(max(1, int(repeat))):
            obs, _, _, _, _ = self._env.step(action)
            self._update_frame(obs)

    def get_frame(self) -> np.ndarray:
        return np.ascontiguousarray(self._frame)

    def get_ram(self) -> Optional[np.ndarray]:
        if self._env is None:
            return None
        try:
            ram = self._env.get_ram()
        except AttributeError:
            return None
        return np.asarray(ram, dtype=np.uint8)

    def serialize(self) -> Optional[bytes]:
        if self._env is None:
            return None
        try:
            return self._env.get_state()  # type: ignore[call-arg]
        except AttributeError:
            return None

    def deserialize(self, blob: bytes) -> None:
        if self._env is None:
            raise RuntimeError("Backend not loaded.")
        if not blob:
            return
        try:
            self._env.set_state(blob)  # type: ignore[call-arg]
        except AttributeError:
            pass

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _update_frame(self, obs: np.ndarray) -> None:
        if isinstance(obs, np.ndarray) and obs.ndim == 3:
            self._frame = np.ascontiguousarray(obs.astype(np.uint8))


register_backend("stable-retro", StableRetroBackend)
