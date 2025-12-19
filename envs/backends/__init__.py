"""Backend factory for Dr. Mario emulator integrations."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple, Type

from .base import EmulatorBackend

_BACKENDS: Dict[str, Type[EmulatorBackend]] = {}
_BUILTIN_SPECS: Dict[str, Tuple[str, str]] = {
    "libretro": ("envs.backends.libretro_backend", "LibretroBackend"),
    "stable-retro": ("envs.backends.stable_retro_backend", "StableRetroBackend"),
    "cpp-engine": ("envs.backends.cpp_engine_backend", "CppEngineBackend"),
}


def register_backend(name: str, backend_cls: Type[EmulatorBackend]) -> None:
    """Register a backend class under a string key."""
    key = name.lower()
    _BACKENDS[key] = backend_cls


def _ensure_backend_loaded(name: str) -> None:
    if name in _BACKENDS:
        return
    if name in _BUILTIN_SPECS:
        module_path, cls_name = _BUILTIN_SPECS[name]
        module = importlib.import_module(module_path)
        backend_cls = getattr(module, cls_name)
        register_backend(name, backend_cls)


def make_backend(name: str, **kwargs) -> EmulatorBackend:
    """Instantiate a backend by name."""
    key = name.lower()
    _ensure_backend_loaded(key)
    backend_cls = _BACKENDS.get(key)
    if backend_cls is None:
        available = sorted(set(_BACKENDS) | set(_BUILTIN_SPECS))
        raise ValueError(f"Unknown backend '{name}'. Available: {available}")
    return backend_cls(**kwargs)


def get_backend_names() -> Tuple[str, ...]:
    return tuple(sorted(set(_BACKENDS) | set(_BUILTIN_SPECS)))


__all__ = ["EmulatorBackend", "register_backend", "make_backend", "get_backend_names"]
