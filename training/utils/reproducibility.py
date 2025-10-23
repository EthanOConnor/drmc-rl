from __future__ import annotations

import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:  # pragma: no cover - torch optional in some deployments
    import torch
except Exception:  # pragma: no cover - keep torch optional
    torch = None  # type: ignore


@dataclass(slots=True)
class ReproContext:
    seed: int
    deterministic: bool = True
    numpy_seed: Optional[int] = None


def set_reproducibility(seed: int, *, deterministic: bool = True) -> ReproContext:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    return ReproContext(seed=seed, deterministic=deterministic, numpy_seed=seed)


def git_commit() -> str:
    """Return the current git commit hash with a dirty flag."""

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2])
        sha = sha.decode("utf-8").strip()
    except Exception:  # pragma: no cover - git optional in tests
        return "unknown"
    try:
        subprocess.check_call(["git", "diff", "--quiet"], cwd=Path(__file__).resolve().parents[2])
        dirty = False
    except Exception:  # pragma: no cover - dirty repo path
        dirty = True
    return f"{sha}{'*' if dirty else ''}"


def pick_device(device_pref: str | None = None) -> str:
    """Pick a torch device string based on availability and preference."""

    if torch is None:
        return "cpu"
    if device_pref in ("cuda", "gpu") and torch.cuda.is_available():
        return "cuda"
    if device_pref in ("mps",) and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if device_pref not in (None, "auto"):
        return device_pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_environment_file(path: Path) -> None:
    """Record installed packages for experiment provenance."""

    try:  # Python 3.8+
        import importlib.metadata as importlib_metadata
    except ImportError:  # pragma: no cover - Python <3.8 fallback
        try:
            import importlib_metadata  # type: ignore
        except Exception:  # pragma: no cover - metadata unavailable
            return

    def _distribution_name(dist: importlib_metadata.Distribution) -> str | None:
        metadata = getattr(dist, "metadata", None)
        if metadata is None:
            return getattr(dist, "name", None)
        name = metadata.get("Name")
        if name:
            return name
        try:
            return metadata["Name"]
        except Exception:  # pragma: no cover - metadata missing name key
            return getattr(dist, "name", None)

    distributions: dict[str, tuple[str, str]] = {}
    for dist in importlib_metadata.distributions():
        name = _distribution_name(dist)
        if not name:
            continue
        key = name.lower()
        distributions[key] = (name, dist.version)

    ordered_distributions = [distributions[key] for key in sorted(distributions)]

    with path.open("w", encoding="utf-8") as fp:
        for name, version in ordered_distributions:
            fp.write(f"{name}=={version}\n")
