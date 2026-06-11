from __future__ import annotations

"""Frozen-opponent snapshot pool with PFSP sampling for VS self-play.

`OpponentPool` owns a directory (`<logdir>/opponent_pool/`) containing frozen
policy checkpoints plus a `manifest.json` with per-opponent learner-perspective
win/game counts, so a run is self-contained and restart-safe.

Sampling is prioritized fictitious self-play (PFSP): opponents the learner
beats ~50% of the time are sampled most, with a floor so no opponent is ever
starved. Unplayed opponents get the maximum weight.
"""

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

CHAMPION_CHECKPOINT = Path("runs/best_agents/smdp_ppo_step535164979.pt.gz")

_PFSP_FLOOR = 0.05
# weight(p) = (p*(1-p))**2 + floor is maximized at p=0.5.
_PFSP_MAX_WEIGHT = 0.25**2 + _PFSP_FLOOR
_MANIFEST_VERSION = 1


@dataclass
class OpponentEntry:
    """One frozen opponent. `wins`/`games` are learner-perspective."""

    id: str
    path: Optional[Path]  # None for injected (test-only) nets; not persisted
    protected: bool = False  # never evicted (seed champion)
    wins: float = 0.0  # draws count 0.5
    games: int = 0
    net: Any = field(default=None, repr=False)  # lazy-loaded CPU policy net
    aux_dim: int = 0
    candidate_max: int = 128

    def smoothed_winrate(self) -> float:
        """Laplace-smoothed learner win rate vs this opponent."""

        return (float(self.wins) + 1.0) / (float(self.games) + 2.0)

    def pfsp_weight(self) -> float:
        if int(self.games) <= 0:
            return _PFSP_MAX_WEIGHT
        p = self.smoothed_winrate()
        return float((p * (1.0 - p)) ** 2 + _PFSP_FLOOR)


def _step_from_name(path: Path) -> int:
    m = re.search(r"step(\d+)", path.name)
    return int(m.group(1)) if m else -1


def default_seed_paths(runs_dir: Path | str = Path("runs")) -> List[Path]:
    """Initial pool: newest vs2_* checkpoint + the 1P placement champion."""

    runs_dir = Path(runs_dir)
    paths: List[Path] = []
    ckpts = sorted(runs_dir.glob("vs2_*/checkpoints/*.pt.gz"), key=_step_from_name)
    if ckpts:
        paths.append(ckpts[-1])
    champion = runs_dir / CHAMPION_CHECKPOINT.relative_to("runs")
    if champion.is_file():
        paths.append(champion)
    return paths


class OpponentPool:
    """PFSP opponent pool persisted under `pool_dir`."""

    def __init__(self, pool_dir: Path | str, *, max_pool: int = 12) -> None:
        self.dir = Path(pool_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_pool = int(max(1, int(max_pool)))
        self.entries: List[OpponentEntry] = []
        if self.manifest_path.is_file():
            self._load_manifest()

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    # ------------------------------------------------------------------ seeding
    def seed(
        self,
        paths: Sequence[Path | str],
        *,
        protected: Optional[Sequence[bool]] = None,
    ) -> None:
        """Copy seed checkpoints into the pool dir and register them."""

        for k, src in enumerate(paths):
            src = Path(src)
            if not src.is_file():
                raise FileNotFoundError(f"Opponent seed checkpoint not found: {src}")
            dst = self.dir / src.name
            if not dst.is_file():
                shutil.copy2(src, dst)
            is_protected = bool(protected[k]) if protected is not None else False
            self._add_entry(
                OpponentEntry(id=self._unique_id(src.name.split(".")[0]), path=dst, protected=is_protected)
            )
        self._save_manifest()

    def add_loaded(
        self,
        opponent_id: str,
        net: Any,
        *,
        aux_dim: int = 0,
        candidate_max: int = 128,
        protected: bool = False,
    ) -> OpponentEntry:
        """Register an already-constructed net (tests); not persisted."""

        entry = OpponentEntry(
            id=self._unique_id(opponent_id),
            path=None,
            protected=protected,
            net=net,
            aux_dim=int(aux_dim),
            candidate_max=int(candidate_max),
        )
        self._add_entry(entry)
        return entry

    # ------------------------------------------------------------------ PFSP
    def pfsp_weights(self) -> np.ndarray:
        """Normalized PFSP sampling weights over `self.entries`."""

        if not self.entries:
            raise RuntimeError("Opponent pool is empty")
        w = np.asarray([e.pfsp_weight() for e in self.entries], dtype=np.float64)
        return w / w.sum()

    def sample(self, rng: np.random.Generator) -> OpponentEntry:
        idx = int(rng.choice(len(self.entries), p=self.pfsp_weights()))
        entry = self.entries[idx]
        self.ensure_loaded(entry)
        return entry

    def record(self, opponent_id: str, learner_won: bool | float) -> None:
        """Record one completed match (learner perspective; draws = 0.5)."""

        entry = self.get(opponent_id)
        if entry is None:
            return
        entry.wins += float(learner_won)
        entry.games += 1
        self._save_manifest()

    def get(self, opponent_id: str) -> Optional[OpponentEntry]:
        for entry in self.entries:
            if entry.id == opponent_id:
                return entry
        return None

    # ------------------------------------------------------------------ snapshots
    def snapshot(self, state_dict: Dict[str, Any], cfg: Dict[str, Any], step: int) -> OpponentEntry:
        """Freeze `state_dict` (EMA weights) as a new pool opponent."""

        import torch

        from training.utils.checkpoint_io import save_checkpoint

        sd_cpu = {k: v.detach().cpu().clone() if torch.is_tensor(v) else v for k, v in state_dict.items()}
        entry_id = self._unique_id(f"snap_step{int(step)}")
        path = self.dir / f"{entry_id}.pt.gz"
        save_checkpoint({"state_dict": sd_cpu, "cfg": dict(cfg or {}), "step": int(step)}, path)
        entry = OpponentEntry(id=entry_id, path=path)
        self._add_entry(entry)
        self._evict()
        self._save_manifest()
        return entry

    # ------------------------------------------------------------------ loading
    def ensure_loaded(self, entry: OpponentEntry) -> None:
        if entry.net is not None:
            return
        if entry.path is None:
            raise RuntimeError(f"Opponent {entry.id!r} has no checkpoint path and no net")
        import torch

        from tools.eval_policy import _build_net_from_cfg
        from training.utils.checkpoint_io import load_checkpoint

        payload = load_checkpoint(entry.path, map_location="cpu")
        net, aux_dim, candidate_max = _build_net_from_cfg(payload.get("cfg", {}), 12, "cpu")
        sd = payload.get("ema_state_dict") or payload["state_dict"]
        net.load_state_dict({k: v.detach().cpu() for k, v in sd.items()})
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        entry.net = net
        entry.aux_dim = int(aux_dim)
        entry.candidate_max = int(candidate_max)

    # ------------------------------------------------------------------ internals
    def _add_entry(self, entry: OpponentEntry) -> None:
        self.entries.append(entry)

    def _unique_id(self, base: str) -> str:
        if self.get(base) is None:
            return base
        k = 2
        while self.get(f"{base}_{k}") is not None:
            k += 1
        return f"{base}_{k}"

    def _evict(self) -> None:
        """Drop oldest unprotected entries beyond `max_pool`."""

        while len(self.entries) > self.max_pool:
            victim = next((e for e in self.entries if not e.protected), None)
            if victim is None:
                return
            self.entries.remove(victim)
            if victim.path is not None and victim.path.parent == self.dir:
                try:
                    victim.path.unlink()
                except OSError:
                    pass

    def _save_manifest(self) -> None:
        payload = {
            "version": _MANIFEST_VERSION,
            "entries": [
                {
                    "id": e.id,
                    "file": e.path.name,
                    "protected": bool(e.protected),
                    "wins": float(e.wins),
                    "games": int(e.games),
                }
                for e in self.entries
                if e.path is not None
            ],
        }
        tmp = self.dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def _load_manifest(self) -> None:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.entries = []
        for row in payload.get("entries", []):
            path = self.dir / str(row["file"])
            if not path.is_file():
                continue
            self._add_entry(
                OpponentEntry(
                    id=str(row["id"]),
                    path=path,
                    protected=bool(row.get("protected", False)),
                    wins=float(row.get("wins", 0.0)),
                    games=int(row.get("games", 0)),
                )
            )


__all__ = ["OpponentEntry", "OpponentPool", "default_seed_paths", "CHAMPION_CHECKPOINT"]
