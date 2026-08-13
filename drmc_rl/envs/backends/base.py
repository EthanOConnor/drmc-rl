"""Common backend utilities and protocol definitions."""

from __future__ import annotations

from typing import Optional, Protocol, Sequence

import numpy as np

NES_BUTTONS: tuple[str, ...] = ("B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT")


class EmulatorBackend(Protocol):
    """Protocol all emulator backends must satisfy."""

    def load(self) -> None:
        """Load core/ROM resources and prepare the emulator."""

    def reset(self) -> None:
        """Reset the underlying emulator to the initial state."""

    def step(self, buttons: Sequence[int], repeat: int = 1) -> None:
        """Advance the emulator by ``repeat`` frames using the provided NES button presses."""

    def get_frame(self) -> np.ndarray:
        """Return the most recent RGB frame as a ``uint8`` HxWx3 array."""

    def get_ram(self) -> Optional[np.ndarray]:
        """Return a view of 2 KB NES RAM as ``uint8`` array if available."""

    def serialize(self) -> Optional[bytes]:
        """Return a savestate blob representing the current core state."""

    def deserialize(self, blob: bytes) -> None:
        """Restore the emulator state from a savestate blob."""

    def close(self) -> None:
        """Release any resources held by the backend."""


__all__ = ["EmulatorBackend", "NES_BUTTONS"]
