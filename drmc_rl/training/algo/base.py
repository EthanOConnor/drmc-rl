from __future__ import annotations

import abc
from typing import Any


class AlgoAdapter(abc.ABC):
    """Common interface implemented by all training algorithms."""

    def __init__(self, cfg: Any, env: Any, logger: Any, event_bus: Any, device: str | None = None) -> None:
        self.cfg = cfg
        self.env = env
        self.logger = logger
        self.event_bus = event_bus
        self.device = device or "cpu"

    @abc.abstractmethod
    def train_forever(self) -> None:
        """Run the optimisation loop until externally interrupted."""

    # Optional extension points -------------------------------------------------
    def close(self) -> None:  # pragma: no cover - trivial default
        """Release resources held by the adapter."""

    def __enter__(self) -> "AlgoAdapter":  # pragma: no cover - standard boilerplate
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # pragma: no cover - standard boilerplate
        self.close()
