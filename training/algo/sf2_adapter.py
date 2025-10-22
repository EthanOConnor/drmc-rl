from __future__ import annotations

from typing import Any

from .base import AlgoAdapter

try:  # pragma: no cover - optional dependency
    import sample_factory  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    sample_factory = None  # type: ignore


class SampleFactoryAdapter(AlgoAdapter):
    """Integration layer for running PPO/APPO via Sample Factory."""

    def __init__(self, cfg: Any, env: Any, logger: Any, event_bus: Any, device: str | None = None) -> None:
        super().__init__(cfg, env, logger, event_bus, device=device)
        if sample_factory is None:
            raise RuntimeError(
                "Sample Factory is not installed. Install extras with 'pip install .[rl]' before using"
                " the PPO/APPO adapter."
            )

    def train_forever(self) -> None:  # pragma: no cover - placeholder until SF integration lands
        raise NotImplementedError("SampleFactoryAdapter is not yet implemented in this lightweight build.")
