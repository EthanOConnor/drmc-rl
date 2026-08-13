from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Any, Optional

try:  # pragma: no cover - torch optional in some deployments
    import torch
except Exception:  # pragma: no cover - keep torch optional
    torch = None  # type: ignore

_COMPRESS_LEVEL = 9


def is_checkpoint_path(path: Path) -> bool:
    name = path.name
    return name.endswith(".pt") or name.endswith(".pt.gz")


def checkpoint_path(checkpoint_dir: Path, prefix: str, step: int, *, compress: bool = True) -> Path:
    suffix = ".pt.gz" if compress else ".pt"
    return checkpoint_dir / f"{prefix}_step{int(step)}{suffix}"


def load_checkpoint(path: Path, *, map_location: Optional[str] = None) -> Any:
    if torch is None:  # pragma: no cover - torch optional in some deployments
        raise RuntimeError("PyTorch is required to load checkpoints.")
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as fp:
            return torch.load(fp, map_location=map_location)
    return torch.load(str(path), map_location=map_location)


def save_checkpoint(payload: Any, path: Path) -> None:
    if torch is None:  # pragma: no cover - torch optional in some deployments
        raise RuntimeError("PyTorch is required to save checkpoints.")
    tmp_path = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass
    try:
        if path.suffix == ".gz":
            with gzip.open(tmp_path, "wb", compresslevel=_COMPRESS_LEVEL) as fp:
                torch.save(payload, fp, _use_new_zipfile_serialization=False)
        else:
            with tmp_path.open("wb") as fp:
                torch.save(payload, fp)
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
