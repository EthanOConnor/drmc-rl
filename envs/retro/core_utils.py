"""Utilities for discovering and resolving bundled libretro cores."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Dict, Iterable, List

_LIBRETRO_PATTERNS: tuple[str, ...] = ("*_libretro.so", "*_libretro.dylib", "*_libretro.dll")


def _project_root() -> Path:
    """Return the repository root directory."""

    return Path(__file__).resolve().parents[2]


def _normalize_path(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    try:
        return candidate.resolve()
    except OSError:
        return candidate


def default_core_search_paths(extra_paths: Iterable[Path | str] | None = None) -> List[Path]:
    """Return search directories for libretro cores."""

    paths: List[Path] = []
    env_dir = os.environ.get("DRMARIO_CORES_DIR")
    if env_dir:
        paths.append(_normalize_path(env_dir))
    paths.append(_normalize_path(_project_root() / "cores"))
    if extra_paths:
        for entry in extra_paths:
            paths.append(_normalize_path(entry))
    unique: Dict[str, Path] = {}
    for path in paths:
        key = str(path)
        if key not in unique:
            unique[key] = path
    return list(unique.values())


def _canonical_core_name(path: Path) -> str:
    stem = path.stem.lower()
    if stem.endswith("_libretro"):
        stem = stem[: -len("_libretro")]
    return stem.replace("-", "_")


def discover_libretro_cores(extra_paths: Iterable[Path | str] | None = None) -> Dict[str, Path]:
    """Discover libretro cores available in known directories."""

    discovered: Dict[str, Path] = {}
    for directory in default_core_search_paths(extra_paths):
        if not directory.is_dir():
            continue
        for entry in sorted(directory.iterdir()):
            if not entry.is_file():
                continue
            if any(fnmatch.fnmatch(entry.name, pattern) for pattern in _LIBRETRO_PATTERNS):
                name = _canonical_core_name(entry)
                discovered.setdefault(name, entry)
    return discovered


def resolve_libretro_core(core: str, extra_paths: Iterable[Path | str] | None = None) -> Path:
    """Resolve a core name or path to an existing libretro core file."""

    candidate = _normalize_path(core)
    if candidate.is_file():
        return candidate

    available = discover_libretro_cores(extra_paths)
    lookup = core.lower()
    for name, path in available.items():
        if lookup in {name, path.stem.lower(), path.name.lower()}:
            return path

    raise FileNotFoundError(f"Unable to resolve libretro core '{core}'.")


def format_available_cores(extra_paths: Iterable[Path | str] | None = None) -> str:
    """Return a human readable list of available cores."""

    available = discover_libretro_cores(extra_paths)
    if not available:
        return "(no bundled cores found)"
    entries = [f"{name}: {path}" for name, path in sorted(available.items())]
    return ", ".join(entries)
