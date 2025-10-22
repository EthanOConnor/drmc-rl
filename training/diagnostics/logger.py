from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore


class DiagLogger:
    """Unified diagnostics writer dispatching to TB, W&B, and JSONL."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.logdir = Path(getattr(cfg, "logdir", "runs/auto")).expanduser().resolve()
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.logdir / "metrics.jsonl"
        self._jsonl = self.metrics_path.open("a", encoding="utf-8")
        self._tb: Optional[SummaryWriter] = None
        self._wandb_run = None
        self._video_dir = self.logdir / "videos"
        self._video_dir.mkdir(exist_ok=True, parents=True)
        self._last_flush = time.time()

        viz = getattr(cfg, "viz", [])
        if isinstance(viz, str):
            viz = [viz]
        if SummaryWriter is not None and "tb" in viz:
            self._tb = SummaryWriter(log_dir=self.logdir / "tb")
        if wandb is not None and "wandb" in viz:
            wandb_kwargs = {
                "project": getattr(cfg, "wandb_project", "drmc"),
                "group": getattr(cfg, "wandb_group", getattr(cfg, "algo", "experiment")),
                "name": getattr(cfg, "wandb_name", self.logdir.name),
                "config": _extract_flattened_cfg(cfg),
            }
            self._wandb_run = wandb.init(**wandb_kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ logging
    def log_scalar(self, name: str, value: float, step: int) -> None:
        payload = {"step": int(step), "type": "scalar", "name": name, "value": float(value)}
        self._write_json(payload)
        if self._tb is not None:
            self._tb.add_scalar(name, value, step)
        if self._wandb_run is not None:
            self._wandb_run.log({name: value, "global_step": step})

    def log_hist(self, name: str, array: Iterable[float], step: int) -> None:
        data = np.asarray(list(array), dtype=np.float32)
        payload = {"step": int(step), "type": "hist", "name": name, "stats": _hist_stats(data)}
        self._write_json(payload)
        if self._tb is not None:
            self._tb.add_histogram(name, data, step)
        if self._wandb_run is not None:
            self._wandb_run.log({name: wandb.Histogram(data) if wandb is not None else data, "global_step": step})

    def log_text(self, name: str, text: str, step: int) -> None:
        payload = {"step": int(step), "type": "text", "name": name, "text": text}
        self._write_json(payload)
        if self._tb is not None:
            self._tb.add_text(name, text, step)
        if self._wandb_run is not None:
            self._wandb_run.log({name: text, "global_step": step})

    def log_video(self, tag: str, mp4_path: Path, step: int) -> None:
        payload = {"step": int(step), "type": "video", "tag": tag, "path": str(mp4_path)}
        self._write_json(payload)
        if self._wandb_run is not None and wandb is not None:
            self._wandb_run.log({tag: wandb.Video(str(mp4_path)), "global_step": step})

    # ------------------------------------------------------------------- helpers
    def flush(self) -> None:
        if self._tb is not None:
            self._tb.flush()
        self._jsonl.flush()
        if self._wandb_run is not None:
            self._wandb_run.flush()
        self._last_flush = time.time()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._jsonl.close()
            if self._tb is not None:
                self._tb.close()
            if self._wandb_run is not None:
                self._wandb_run.finish()

    # ------------------------------------------------------------------ metadata
    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        payload = {"step": 0, "type": "metadata", "data": metadata}
        self._write_json(payload)

    # ------------------------------------------------------------------ internals
    def _write_json(self, payload: Dict[str, Any]) -> None:
        json.dump(payload, self._jsonl)
        self._jsonl.write("\n")
        if time.time() - self._last_flush > 5.0:
            self.flush()


def _hist_stats(array: np.ndarray) -> Dict[str, float]:
    if array.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _extract_flattened_cfg(cfg: Any) -> Dict[str, Any]:
    if hasattr(cfg, "to_dict"):
        flat = cfg.to_dict()
    elif isinstance(cfg, dict):
        flat = cfg
    else:
        flat = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}
    return flat
