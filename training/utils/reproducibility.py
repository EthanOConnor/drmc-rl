from __future__ import annotations

import gzip
import os
import random
import subprocess
from dataclasses import dataclass
from importlib.metadata import distributions
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

    def _dist_name(dist: object) -> str:
        try:
            name = getattr(dist, "metadata", {}).get("Name")
            if name:
                return str(name)
        except Exception:
            pass
        try:
            name = getattr(dist, "name", None)
            if name:
                return str(name)
        except Exception:
            pass
        return "unknown"

    dists = sorted(distributions(), key=lambda dist: _dist_name(dist).lower())
    if path.suffix == ".gz":
        fp_ctx = gzip.open(path, "wt", encoding="utf-8", compresslevel=9)
    else:
        fp_ctx = path.open("w", encoding="utf-8")
    with fp_ctx as fp:
        for dist in dists:
            name = _dist_name(dist)
            try:
                version = str(getattr(dist, "version", ""))
            except Exception:
                version = ""
            if not name or name == "unknown":
                continue
            fp.write(f"{name}=={version}\n")
