from __future__ import annotations

"""Start-state bank sampling for the VS vector env (Go-Exploit style).

Loads a bank built by ``tools/build_start_bank.py`` (mid-game two-board
positions from the human VS corpus) and yields checkpoint keyword overlays
for ``envs.backends.drmario_vs_pool.build_vs_reset_spec``. The env merges
them with its own level / speed / RNG fields, so continuations from a bank
position use the env's randomized RNG (exact mid-game RNG is not
reconstructible from the corpus — boards are realistic, continuations are
stochastic).
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np

__all__ = ["StartBank"]


class StartBank:
    """Memory-resident start-state bank (npz from tools/build_start_bank.py)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        data = np.load(self.path, allow_pickle=False)
        self.boards = data["boards"]  # (N,2,16,8) uint8 NES tiles
        self.falling = data["falling"]  # (N,2,2) raw NES colors
        self.preview = data["preview"]  # (N,2,2) raw NES colors
        self.pill_counter = data["pill_counter"]  # (N,2) reserve index
        self.speed_ups = data["speed_ups"]  # (N,2)
        self.stratum = data["stratum"]  # (N,) 0 early 1 mid 2 late 3 crisis
        if self.boards.ndim != 4 or self.boards.shape[1:] != (2, 16, 8):
            raise ValueError(f"Bad start bank boards shape: {self.boards.shape!r}")
        if len(self) == 0:
            raise ValueError(f"Start bank {self.path} is empty")

    def __len__(self) -> int:
        return int(self.boards.shape[0])

    def sample_spec_kwargs(self, rng: np.random.Generator) -> Dict[str, Any]:
        """Checkpoint kwargs for one bank row (build_vs_reset_spec overlay)."""

        i = int(rng.integers(0, len(self)))
        return self.spec_kwargs(i)

    def spec_kwargs(self, i: int) -> Dict[str, Any]:
        return {
            "checkpoint_enabled": True,
            "checkpoint_board": self.boards[i].reshape(2, 128),
            "checkpoint_falling_colors": self.falling[i],
            "checkpoint_preview_colors": self.preview[i],
            "checkpoint_pill_counter": (
                int(self.pill_counter[i, 0]),
                int(self.pill_counter[i, 1]),
            ),
            "checkpoint_speed_ups": (
                int(self.speed_ups[i, 0]),
                int(self.speed_ups[i, 1]),
            ),
        }
