"""Optional CUDA reachability backend.

The package remains importable on non-CUDA machines; requesting a CUDA symbol
loads the platform-specific bindings.
"""

from __future__ import annotations

from typing import Any

__all__ = ["CudaReach", "INSTANCE_DTYPE", "N_POSES"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from drmc_rl.planning.cuda import host

    return getattr(host, name)
