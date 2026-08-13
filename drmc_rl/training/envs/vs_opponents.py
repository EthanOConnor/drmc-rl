from __future__ import annotations

"""Frozen-opponent snapshot pool with PFSP sampling for VS self-play.

`OpponentPool` owns a directory (`<logdir>/opponent_pool/`) containing frozen
policy checkpoints plus a `manifest.json` with per-opponent learner-perspective
win/game counts, so a run is self-contained and restart-safe.

Sampling is prioritized fictitious self-play (PFSP): opponents the learner
beats ~50% of the time are sampled most, with a floor so no opponent is ever
starved. Unplayed opponents get the maximum weight.

League roles (docs/LEAGUE.md) extend the pool with fixed external targets
("main agents") via `LeagueConfig`:

- ``pfsp`` (default): today's behavior — PFSP over the pool's own history.
- ``exploiter``: sample exclusively from the listed ``main_agents`` (PFSP
  weighting over per-target win rates); the learner never faces snapshots of
  itself.
- ``mixed``: with probability ``exploiter_fraction`` sample a main agent,
  otherwise the normal self-history pool.
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

_LEAGUE_MODES = ("pfsp", "exploiter", "mixed")


@dataclass
class LeagueConfig:
    """League roles for the opponent pool (see module docstring)."""

    mode: str = "pfsp"
    main_agents: List[Path] = field(default_factory=list)
    exploiter_fraction: float = 0.3


def parse_league_config(cfg: Any) -> LeagueConfig:
    """Parse/validate the `env.opponent_pool.league` config block."""

    if cfg is None:
        return LeagueConfig()
    if not isinstance(cfg, dict):
        raise ValueError(f"opponent_pool.league must be a mapping, got {type(cfg).__name__}")
    mode = str(cfg.get("mode", "pfsp")).strip().lower()
    if mode not in _LEAGUE_MODES:
        raise ValueError(f"opponent_pool.league.mode must be one of {_LEAGUE_MODES}, got {mode!r}")
    main_agents = [Path(p) for p in (cfg.get("main_agents") or [])]
    fraction = float(cfg.get("exploiter_fraction", 0.3))
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"opponent_pool.league.exploiter_fraction must be in [0, 1], got {fraction}")
    if mode in ("exploiter", "mixed") and not main_agents:
        raise ValueError(f"opponent_pool.league.mode={mode!r} requires non-empty main_agents")
    return LeagueConfig(mode=mode, main_agents=main_agents, exploiter_fraction=fraction)


@dataclass
class OpponentEntry:
    """One frozen opponent. `wins`/`games` are learner-perspective."""

    id: str
    path: Optional[Path]  # None for injected (test-only) nets; not persisted
    protected: bool = False  # never evicted (seed champion)
    league_target: bool = False  # fixed main agent (exploiter/mixed league modes)
    wins: float = 0.0  # draws count 0.5
    games: int = 0
    net: Any = field(default=None, repr=False)  # lazy-loaded policy net
    device: str = "cpu"  # where `net` lives (pool.device at load time)
    aux_dim: int = 0
    aux_spec: str = "v1"
    candidate_max: int = 128
    kind: str = "rl"
    rating: Optional[float] = None
    rating_sd: float = 0.0
    condition: Any = field(default=None, repr=False)

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

    def __init__(
        self,
        pool_dir: Path | str,
        *,
        max_pool: int = 12,
        league: Optional[LeagueConfig] = None,
        device: str = "auto",
    ) -> None:
        self.dir = Path(pool_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_pool = int(max(1, int(max_pool)))
        self.league = league if league is not None else LeagueConfig()
        # Where frozen opponent nets forward. "auto" = cuda when available:
        # on CPU-lean hosts the per-step opponent forwards are real serial
        # time in env.step (docs/NETWORKS_REPORT_2026-07.md rec 5).
        self.device = str(device)
        if self.device == "auto":
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
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

    def seed_league_targets(self, paths: Sequence[Path | str]) -> None:
        """Register fixed league targets (main agents); idempotent by filename.

        Targets are protected (never evicted). An existing entry with the same
        checkpoint filename is flagged in place rather than duplicated, so
        restarts keep the persisted per-target win/game counts.
        """

        by_name = {e.path.name: e for e in self.entries if e.path is not None}
        for src in paths:
            src = Path(src)
            if not src.is_file():
                raise FileNotFoundError(f"League target checkpoint not found: {src}")
            existing = by_name.get(src.name)
            if existing is not None:
                existing.league_target = True
                existing.protected = True
                continue
            dst = self.dir / src.name
            if not dst.is_file():
                shutil.copy2(src, dst)
            self._add_entry(
                OpponentEntry(
                    id=self._unique_id(src.name.split(".")[0]),
                    path=dst,
                    protected=True,
                    league_target=True,
                )
            )
        self._save_manifest()

    def seed_humans(self, specifications: Sequence[Dict[str, Any]]) -> None:
        """Register fixed-rating human-v2 checkpoints as protected opponents."""

        existing = {
            (e.path.name if e.path is not None else "", e.rating)
            for e in self.entries
            if e.kind == "human"
        }
        for specification in specifications:
            src = Path(str(specification["checkpoint"]))
            if not src.is_file():
                raise FileNotFoundError(f"Human opponent checkpoint not found: {src}")
            rating = float(specification["rating"])
            if (src.name, rating) in existing:
                continue
            dst = self.dir / src.name
            if not dst.is_file():
                shutil.copy2(src, dst)
            label = str(specification.get("id") or f"human_{int(round(rating))}")
            self._add_entry(
                OpponentEntry(
                    id=self._unique_id(label),
                    path=dst,
                    protected=bool(specification.get("protected", True)),
                    kind="human",
                    rating=rating,
                    rating_sd=float(specification.get("rating_sd", 0.0)),
                )
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
        league_target: bool = False,
    ) -> OpponentEntry:
        """Register an already-constructed net (tests); not persisted."""

        entry = OpponentEntry(
            id=self._unique_id(opponent_id),
            path=None,
            protected=protected,
            league_target=league_target,
            net=net,
            aux_dim=int(aux_dim),
            candidate_max=int(candidate_max),
        )
        self._add_entry(entry)
        return entry

    # ------------------------------------------------------------------ PFSP
    def pfsp_weights(self, entries: Optional[Sequence[OpponentEntry]] = None) -> np.ndarray:
        """Normalized PFSP sampling weights over `entries` (default: all)."""

        entries = self.entries if entries is None else list(entries)
        if not entries:
            raise RuntimeError("Opponent pool is empty")
        w = np.asarray([e.pfsp_weight() for e in entries], dtype=np.float64)
        return w / w.sum()

    def league_targets(self) -> List[OpponentEntry]:
        return [e for e in self.entries if e.league_target]

    def _sample_subset(self, rng: np.random.Generator) -> List[OpponentEntry]:
        """Entries eligible for this sample, per the league mode."""

        mode = self.league.mode
        if mode == "pfsp":
            return self.entries
        targets = self.league_targets()
        if mode == "exploiter":
            if not targets:
                raise RuntimeError("league mode 'exploiter' but the pool has no league targets")
            return targets
        # mixed: exploiter_fraction of assignments vs the targets, rest PFSP
        # over the self-history pool.
        others = [e for e in self.entries if not e.league_target]
        if targets and (not others or float(rng.random()) < float(self.league.exploiter_fraction)):
            return targets
        if not others:
            raise RuntimeError("league mode 'mixed' but the pool has no entries")
        return others

    def sample(self, rng: np.random.Generator) -> OpponentEntry:
        subset = self._sample_subset(rng)
        idx = int(rng.choice(len(subset), p=self.pfsp_weights(subset)))
        entry = subset[idx]
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

        from drmc_rl.training.utils.checkpoint_io import save_checkpoint

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

        from drmc_rl.training.utils.checkpoint_io import load_checkpoint

        payload = load_checkpoint(entry.path, map_location="cpu")
        if str(payload.get("schema", "")).startswith("drmc-human-policy-"):
            from drmc_rl.human.conditioning import HumanSkillCondition
            from drmc_rl.human.model import POLICY_CONDITION_DIM, build_human_policy

            cfg = payload["cfg"]
            net = build_human_policy(cfg, device="cpu")
            net.load_state_dict(payload["state_dict"])
            entry.kind = "human"
            entry.condition = HumanSkillCondition.from_dict(
                payload["human_meta"]["skill_condition"]
            )
            entry.aux_dim = POLICY_CONDITION_DIM
            entry.aux_spec = "human_v2"
            entry.candidate_max = int(
                cfg.get("smdp_ppo", cfg).get("candidate_max_candidates", 128)
            )
        else:
            from tools.eval_policy import _build_net_from_cfg

            cfg = payload.get("cfg", {})
            net, aux_dim, candidate_max = _build_net_from_cfg(cfg, 12, "cpu")
            entry.aux_dim = int(aux_dim)
            entry.aux_spec = str(cfg.get("smdp_ppo", cfg).get("aux_spec", "v1")).strip().lower()
            entry.candidate_max = int(candidate_max)
        sd = payload.get("ema_state_dict") or payload["state_dict"]
        if entry.kind != "human":
            net.load_state_dict({k: v.detach().cpu() for k, v in sd.items()})
        net = net.to(self.device)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        entry.net = net
        entry.device = self.device

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
                    "league_target": bool(e.league_target),
                    "wins": float(e.wins),
                    "games": int(e.games),
                    "kind": str(e.kind),
                    "rating": e.rating,
                    "rating_sd": float(e.rating_sd),
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
                    league_target=bool(row.get("league_target", False)),
                    wins=float(row.get("wins", 0.0)),
                    games=int(row.get("games", 0)),
                    kind=str(row.get("kind", "rl")),
                    rating=None if row.get("rating") is None else float(row["rating"]),
                    rating_sd=float(row.get("rating_sd", 0.0)),
                )
            )


__all__ = [
    "LeagueConfig",
    "OpponentEntry",
    "OpponentPool",
    "default_seed_paths",
    "parse_league_config",
    "CHAMPION_CHECKPOINT",
]
