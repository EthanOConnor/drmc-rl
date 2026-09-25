"""Evaluation-seed reserve and the shared training seed mixture.

Deployed play only ever meets the 32,767 console games per level and speed, so
training may use essentially the whole seed space; recognizing a seed is
acceptable. A small reserve of games that no training run may draw exists for
two purposes: the seen-versus-reserve memorization check and the decision games
of recipe A/B experiments.

Seed space
----------
An arena/trainer reset seed ``s`` (1..65535) is the NES RNG register itself:
``rng_state = (s & 0xFF, s >> 8)`` (``FrameVsPool.reset``). The game is fully
determined by ``(level, speed, rng_state)`` and both bottles share it, so at a
fixed level and speed there is no other source of game diversity.

The register step discards bit 0 of ``rng_state[1]``, and the engine steps the
register before every draw (128 pill ids, then virus placement) with nothing
reading the seeded value itself. So ``s`` and ``s ^ 0x100`` play
*byte-identical games* (same pill reserve, same virus board at every level):
the 65,535 nonzero seeds hold only 32,767 distinct games per (level, speed),
one per *game class*. Each class has one member on the console's orbit of
``0x89, 0x88`` (period 32,767, the only states real hardware visits) and one
transient twin; seed ``0x100`` steps into the ``0x0000`` lockup. Evaluation
diversity cannot be widened past 32,767 games without changing the level or
speed being evaluated.

The reserve works in game classes: it reserves whole classes, gives each study
the class's hardware-reachable orbit member, and blocks both members of every
reserved class from training at every level.

Training draws
--------------
New runs draw console-reachable seeds outside the reserve (and outside the twin
of every excluded seed) from a mixture: ``seed_mix`` (default 0.5) weighted by
real Fightcade play frequency (``seed_frequency.json``, from drmariostats
``/api/seeds``), the rest uniform. Every training seed source goes through
:func:`training_seed_pool`, :func:`require_training_seeds` or
:func:`draw_training_state`. Runs that predate the reserve are grandfathered by
output path (``legacy_training_outputs``) or ``"seed_reserve": "legacy"`` and
keep their original uniform draws. Studies take reserve seeds only through
:func:`allocate`, which hands out disjoint, recorded, contiguous slices.
Evaluation helpers: :func:`memorization_report`, :func:`strength_views`,
:func:`draw_mixture_seeds` and :func:`detectable_gap`.

CLI: ``python -m drmc_rl.program.seed_reserve {show,allocate,check,resize,frequency,build}``.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from dataclasses import dataclass
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable, Iterator

import numpy as np

RESERVE_PATH = Path(__file__).with_name("eval_seed_reserve.json")
ALLOCATIONS_PATH = Path(__file__).with_name("eval_seed_allocations.json")
FREQUENCY_PATH = Path(__file__).with_name("seed_frequency.json")
FREQUENCY_SCHEMA = "drmc-seed-frequency-v1"
FREQUENCY_API = "https://drmariostats.zudark.net/api/seeds?lvl={level}"
DEFAULT_SEED_MIX = 0.5  # share of new-run training draws weighted by real Fightcade play
RESERVE_SCHEMA = "drmc-eval-seed-reserve-v1"
ALLOCATION_SCHEMA = "drmc-eval-seed-allocations-v1"
SELECTION_KEY = "drmc-eval-seed-reserve-v1"
STATUSES = ("unseen", "evaluated", "trained_unrelated", "trained_lineage")
# Seed-list keys that may name allocated reserve seeds in order to hold them out.
EXCLUSION_KEYS = ("holdout_seeds", "seed_exclusions", "excluded_seeds", "excluded_reset_seeds")
_STUDY = re.compile(r"[a-z0-9][a-z0-9._-]*")


# --------------------------------------------------------------------------- RNG


def rng_state(seed: int) -> tuple[int, int]:
    """Arena reset seed -> engine ``rng_state`` bytes."""
    return int(seed) & 0xFF, (int(seed) >> 8) & 0xFF


def arena_seed(r0: int, r1: int) -> int:
    """Engine ``rng_state`` bytes -> arena reset seed."""
    return (int(r0) & 0xFF) | ((int(r1) & 0xFF) << 8)


def step_seeds(seeds: np.ndarray) -> np.ndarray:
    """One NES RNG step on arena-convention seeds (``GameLogic.cpp:rng_step``)."""
    s = np.asarray(seeds, dtype=np.int64)
    r0, r1 = s & 0xFF, s >> 8
    carry = ((r0 ^ r1) >> 1) & 1
    return ((r0 >> 1) | (carry << 7)) | (((r1 >> 1) | ((r0 & 1) << 7)) << 8)


@lru_cache(maxsize=1)
def orbit_seeds() -> frozenset[int]:
    """The 32,767 console-reachable states, as arena seeds."""
    seen, s = [], arena_seed(0x89, 0x88)
    while not seen or s != seen[0]:
        seen.append(s)
        s = int(step_seeds(np.asarray([s]))[0])
    return frozenset(seen)


def twin(seed: int) -> int:
    """The other seed of ``seed``'s game class (identical game); 0 for the lockup seed 0x100."""
    return int(seed) ^ 0x100


@lru_cache(maxsize=1)
def game_class_table() -> np.ndarray:
    """``table[s]`` = the register after one step, which alone determines the game."""
    return step_seeds(np.arange(65536, dtype=np.int64))


@lru_cache(maxsize=1)
def class_members() -> dict[int, tuple[int, ...]]:
    """Game class -> every nonzero arena seed that plays that game."""
    table, members = game_class_table(), {}
    for s in range(1, 65536):
        members.setdefault(int(table[s]), []).append(s)
    return {k: tuple(v) for k, v in members.items()}


def class_of(seed: int) -> int:
    return int(game_class_table()[int(seed)])


# ----------------------------------------------------------------------- reserve


@dataclass(frozen=True)
class Reserve:
    seeds: tuple[int, ...]  # evaluation seeds (orbit members) in allocation order
    status: tuple[str, ...]  # exposure of each seed's game class when the reserve was built
    blocked: frozenset[int]  # every arena seed sharing a game class with a reserve seed
    legacy_outputs: tuple[str, ...]
    sha256: str
    compatible_sha256: tuple[str, ...] = ()  # longer earlier files this reserve is a prefix of


@lru_cache(maxsize=4)
def load_reserve(path: Path | str = RESERVE_PATH) -> Reserve:
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("schema") != RESERVE_SCHEMA:
        raise ValueError(f"{path}: not a {RESERVE_SCHEMA} file")
    seeds = tuple(map(int, data["seeds"]))
    if set(data["status_codes"]) - set("0123"):
        raise ValueError(f"{path}: unknown status code")
    status = tuple(STATUSES[int(c)] for c in data["status_codes"])
    if len(seeds) != len(status):
        raise ValueError(f"{path}: seed and status lists disagree")
    orbit, classes = orbit_seeds(), {class_of(s) for s in seeds}
    if any(s not in orbit for s in seeds) or len(classes) != len(seeds):
        raise ValueError(f"{path}: reserve seeds must be orbit states of distinct games")
    blocked = frozenset(m for s in seeds for m in class_members()[class_of(s)])
    if data.get("blocked_count") != len(blocked):
        raise ValueError(f"{path}: blocked seed count does not match the game classes")
    return Reserve(seeds, status, blocked, tuple(data.get("legacy_training_outputs", ())),
                   hashlib.sha256(raw).hexdigest(),
                   tuple(entry["sha256"] for entry in data.get("truncated_from", ())))


def is_legacy(config: dict | None, reserve: Reserve | None = None) -> bool:
    """Grandfathered runs that predate the reserve keep their original seed draws."""
    if not config:
        return False
    if config.get("seed_reserve") == "legacy":
        return True
    reserve = reserve or load_reserve()
    output = str(config.get("output", "")).rstrip("/")
    return bool(output) and any(output.endswith(suffix) for suffix in reserve.legacy_outputs)


@dataclass(frozen=True)
class SeedPool:
    """Seeds a draw may use and their probabilities (``None`` = uniform)."""

    seeds: np.ndarray
    p: np.ndarray | None = None

    def choice(self, rng: np.random.Generator, count: int) -> np.ndarray:
        """``count`` distinct seeds; a uniform pool draws exactly like ``rng.choice(seeds, ...)``."""
        return rng.choice(self.seeds, count, replace=False, p=self.p)


@lru_cache(maxsize=4)
def load_seed_frequency(path: Path | str = FREQUENCY_PATH) -> dict[int, int]:
    """Real Fightcade games per console seed (arena convention), all levels pooled."""
    data = json.loads(Path(path).read_text())
    if data.get("schema") != FREQUENCY_SCHEMA:
        raise ValueError(f"{path}: not a {FREQUENCY_SCHEMA} file")
    table = dict(zip(map(int, data["seeds"]), map(int, data["games"])))
    if sum(table.values()) != data["total_games"] or not set(table) <= orbit_seeds():
        raise ValueError(f"{path}: frequency table is inconsistent")
    return table


def mixture_probabilities(seeds: Iterable[int] | None = None, *, seed_mix: float = DEFAULT_SEED_MIX,
                          frequency: dict[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(seeds, p)``: ``seed_mix`` by real play frequency plus the rest uniform over ``seeds``.

    ``seeds`` defaults to all 32,767 console-reachable states (the deployment
    distribution); both components are renormalized over the seeds given, and
    a transient seed counts as its orbit twin.
    """
    if not 0 <= seed_mix <= 1:
        raise ValueError("seed_mix must lie in [0, 1]")
    frequency = load_seed_frequency() if frequency is None else frequency
    pool = np.asarray(sorted(orbit_seeds()) if seeds is None else list(seeds), dtype=np.int64)
    orbit = orbit_seeds()
    counts = np.asarray([frequency.get(s if s in orbit else twin(s), 0) for s in pool.tolist()], float)
    p = np.full(len(pool), (1 - seed_mix) / len(pool))
    if counts.sum() > 0:
        p += seed_mix * counts / counts.sum()
    else:
        p += seed_mix / len(pool)
    return pool, p / p.sum()


def training_seed_pool(excluded: Iterable[int] = (), *, config: dict | None = None,
                       reserve: Reserve | None = None) -> SeedPool:
    """The seeds a training (or unregistered arena) draw may use, with their weights.

    Grandfathered runs keep their original uniform draw over 1..65535 minus
    ``excluded``. New runs draw console-reachable seeds outside the reserve and
    outside every excluded game (``s`` and ``s ^ 0x100`` are the same game):
    ``seed_mix`` (default 0.5) weighted by real play frequency, the rest uniform.
    """
    reserve = reserve or load_reserve()
    removed = set(map(int, excluded))
    if is_legacy(config, reserve):
        return SeedPool(np.setdiff1d(np.arange(1, 65536), np.fromiter(removed, np.int64, len(removed))))
    removed |= {twin(s) for s in removed} | reserve.blocked
    allowed = sorted(orbit_seeds() - removed)
    seeds, p = mixture_probabilities(allowed, seed_mix=float((config or {}).get("seed_mix", DEFAULT_SEED_MIX)))
    return SeedPool(seeds, p)


def require_training_seeds(seeds: Iterable[int], *, config: dict | None = None,
                           what: str = "training seeds", reserve: Reserve | None = None) -> None:
    """Fail loudly when explicit training seeds touch the evaluation reserve."""
    reserve = reserve or load_reserve()
    if is_legacy(config, reserve):
        return
    hit = sorted(set(map(int, seeds)) & reserve.blocked)
    if hit:
        raise ValueError(f"{what} use {len(hit)} evaluation-reserve seeds (e.g. {hit[:5]}); "
                         "see drmc_rl/program/seed_reserve.py")


def is_reserved_state(r0: int, r1: int, reserve: Reserve | None = None) -> bool:
    return arena_seed(r0, r1) in (reserve or load_reserve()).blocked


@lru_cache(maxsize=8)
def _state_pool(seed_mix: float, reserve_sha: str) -> SeedPool:
    return training_seed_pool(config={"seed_mix": seed_mix})


def draw_training_state(rng: np.random.Generator, *, seed_mix: float = DEFAULT_SEED_MIX) -> tuple[int, int]:
    """Random ``rng_state`` bytes for a training reset, from the new-run mixture outside the reserve."""
    pool = _state_pool(float(seed_mix), load_reserve().sha256)
    return rng_state(int(rng.choice(pool.seeds, p=pool.p)))


# ------------------------------------------------------------------- allocation


def load_allocations(path: Path | str = ALLOCATIONS_PATH) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if data.get("schema") != ALLOCATION_SCHEMA:
        raise ValueError(f"{path}: not a {ALLOCATION_SCHEMA} file")
    return list(data["allocations"])


def validate_allocations(allocations: list[dict], reserve: Reserve) -> None:
    names, end = set(), 0
    for entry in allocations:
        start, count = int(entry["start"]), int(entry["count"])
        if entry["study"] in names or start != end or count < 1 or start + count > len(reserve.seeds):
            raise ValueError(f"allocation {entry['study']!r} is not a fresh contiguous reserve slice")
        if entry["reserve_sha256"] not in (reserve.sha256, *reserve.compatible_sha256):
            raise ValueError(f"allocation {entry['study']!r} was made from a different reserve file")
        if entry["seeds_sha256"] != _seeds_sha(reserve.seeds[start:start + count]):
            raise ValueError(f"allocation {entry['study']!r} does not match its reserve slice")
        names.add(entry["study"])
        end = start + count


def allocated_seeds(study: str | None = None, *, reserve: Reserve | None = None,
                    path: Path | str = ALLOCATIONS_PATH) -> list[int]:
    """Seeds allocated to ``study`` (or to every study when ``study`` is None)."""
    reserve = reserve or load_reserve()
    allocations = load_allocations(path)
    validate_allocations(allocations, reserve)
    chosen = [a for a in allocations if study is None or a["study"] == study]
    if study is not None and not chosen:
        raise KeyError(f"no evaluation-seed allocation for study {study!r}")
    return [s for a in chosen for s in reserve.seeds[a["start"]:a["start"] + a["count"]]]


def allocate(study: str, count: int, purpose: str, *, reserve: Reserve | None = None,
             path: Path | str = ALLOCATIONS_PATH, now: datetime | None = None) -> list[int]:
    """Record the next ``count`` unallocated reserve seeds for ``study`` and return them."""
    reserve = reserve or load_reserve()
    if not _STUDY.fullmatch(study):
        raise ValueError("study names are lowercase [a-z0-9._-]")
    if count < 1 or not purpose.strip():
        raise ValueError("allocations need a positive count and a purpose")
    path = Path(path)
    allocations = load_allocations(path)
    validate_allocations(allocations, reserve)
    if any(a["study"] == study for a in allocations):
        raise ValueError(f"study {study!r} already holds an allocation; seeds are never reissued")
    start = sum(int(a["count"]) for a in allocations)
    if start + count > len(reserve.seeds):
        raise ValueError(f"only {len(reserve.seeds) - start} reserve seeds remain unallocated")
    seeds = list(reserve.seeds[start:start + count])
    status = reserve.status[start:start + count]
    allocations.append(dict(
        study=study, purpose=purpose.strip(), start=start, count=count,
        created_at=(now or datetime.now(UTC)).isoformat(timespec="seconds"),
        reserve_sha256=reserve.sha256, seeds_sha256=_seeds_sha(seeds),
        status_at_creation={k: status.count(k) for k in STATUSES if k in status},
    ))
    path.write_text(json.dumps(dict(schema=ALLOCATION_SCHEMA, allocations=allocations), indent=1) + "\n")
    return seeds


def _seeds_sha(seeds: Iterable[int]) -> str:
    return hashlib.sha256(json.dumps([int(s) for s in seeds]).encode()).hexdigest()


# ------------------------------------------------------------------- evaluation
# Pair-mean variance of a side-swapped seed pair is ~0.165 in the afterstate
# panels; ~0.04 of it persists across related candidates on the same seed (the
# seed x entrant interaction), the rest is per-play noise.
SEED_VARIANCE, PLAY_VARIANCE = 0.04, 0.125


def detectable_gap(reserve_seeds: int, plays_per_seed: float = 1.0, seen_seeds: int | None = None, *,
                   seed_variance: float = SEED_VARIANCE, play_variance: float = PLAY_VARIANCE,
                   z_alpha: float = 1.96, z_power: float = 0.8416) -> float:
    """Smallest seen-minus-reserve score gap found at 5% two-sided with 80% power.

    ``plays_per_seed`` counts side-swapped pairs per seed per entrant (one per
    opponent); ``seen_seeds`` defaults to as many seen seeds as reserve seeds.
    """
    per_seed = seed_variance + play_variance / plays_per_seed
    seen = reserve_seeds if seen_seeds is None else seen_seeds
    return (z_alpha + z_power) * float(np.sqrt(per_seed * (1 / reserve_seeds + 1 / seen)))


def _seed_means(rows: Iterable[dict]) -> dict[int, tuple[float, int]]:
    total: dict[int, list[float]] = {}
    for row in rows:
        if row.get("score") is not None:
            total.setdefault(int(row["seed"]), []).append(float(row["score"]))
    return {s: (sum(v) / len(v), len(v)) for s, v in total.items()}


def _clustered(means: list[float], weights: np.ndarray | None = None) -> dict:
    x = np.asarray(means, float)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    mean = float(w @ x)
    ess = float(1 / np.sum(w * w))
    se = float(np.sqrt(np.sum(w * w * (x - mean) ** 2) * len(x) / max(len(x) - 1, 1)))
    return dict(seeds=len(x), mean=mean, se=se, effective_seeds=ess)


def memorization_report(rows: Iterable[dict], *, reserve: Reserve | None = None) -> dict:
    """Seen-minus-reserve score gap for one entrant, per condition and pooled.

    ``rows`` are that entrant's games: ``seed`` (arena), ``score`` (entrant's
    score, 1/0.5/0) and an optional ``condition`` label such as ``"L14-HI"``.
    Seeds are the clustering unit; a positive gap means better play on games
    that training could have seen.
    """
    reserve = reserve or load_reserve()
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get("condition", "all")), []).append(row)
    out, gaps = {}, []
    for condition, items in sorted(groups.items()):
        means = _seed_means(items)
        held = [m for s, (m, _) in means.items() if s in reserve.blocked]
        seen = [m for s, (m, _) in means.items() if s not in reserve.blocked]
        entry = dict(reserve=_clustered(held) if held else None, seen=_clustered(seen) if seen else None)
        if len(held) > 1 and len(seen) > 1:
            gap = entry["seen"]["mean"] - entry["reserve"]["mean"]
            se = float(np.hypot(entry["seen"]["se"], entry["reserve"]["se"]))
            entry.update(gap=gap, se=se, ci95=(gap - 1.96 * se, gap + 1.96 * se))
            gaps.append((gap, se))
        out[condition] = entry
    if gaps:
        w = np.asarray([1 / se**2 if se > 0 else 0.0 for _, se in gaps])
        if w.sum() > 0:
            gap = float(w @ np.asarray([g for g, _ in gaps]) / w.sum())
            se = float(1 / np.sqrt(w.sum()))
            out["pooled"] = dict(gap=gap, se=se, ci95=(gap - 1.96 * se, gap + 1.96 * se))
    return out


def strength_views(rows: Iterable[dict], *, seed_mix: float = DEFAULT_SEED_MIX) -> dict:
    """Uniform and real-play-weighted mean score over the seeds in ``rows``.

    The weighted view reweights each seed by its probability under the
    ``seed_mix`` mixture over all 32,767 console games relative to uniform
    (self-normalized). It is only as good as its ``effective_seeds``: a bank
    of rarely played seeds carries little real-play weight, so for this view
    play seeds drawn by :func:`draw_mixture_seeds` (their plain mean is the
    weighted view).
    """
    means = _seed_means(rows)
    seeds = sorted(means)
    _, p = mixture_probabilities(seed_mix=seed_mix)
    index = {s: i for i, s in enumerate(sorted(orbit_seeds()))}
    orbit = orbit_seeds()
    weights = np.asarray([p[index[s if s in orbit else twin(s)]] for s in seeds]) * len(orbit)
    values = [means[s][0] for s in seeds]
    return dict(uniform=_clustered(values), weighted=_clustered(values, weights), seed_mix=seed_mix)


def draw_mixture_seeds(rng: np.random.Generator, count: int, *, seed_mix: float = DEFAULT_SEED_MIX,
                       excluded: Iterable[int] = ()) -> list[int]:
    """Distinct non-reserve seeds from the training mixture (real-play-weighted evaluation games)."""
    return [int(s) for s in training_seed_pool(excluded, config={"seed_mix": seed_mix}).choice(rng, count)]


# ------------------------------------------------------------------------ check


def seed_lists(value, *, seed_file: bool = False, key: str = "") -> Iterator[tuple[str, list[int]]]:
    """Integer lists stored under a ``*seed*`` key (or anywhere in a seeds file)."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield from seed_lists(v, seed_file=seed_file, key=f"{key}.{k}" if key else str(k))
    elif isinstance(value, list):
        if value and all(type(x) is int for x in value):
            if seed_file or "seed" in key.rsplit(".", 1)[-1].lower():
                yield key, value
        elif value and all(isinstance(x, list) and len(x) == 2 and all(type(b) is int for b in x)
                           for x in value) and "seed" in key.rsplit(".", 1)[-1].lower():
            yield key, [arena_seed(r0, r1) for r0, r1 in value]  # byte-pair reset seeds
        else:
            for i, v in enumerate(value):
                yield from seed_lists(v, seed_file=seed_file, key=f"{key}[{i}]")


def check_config(value, *, name: str = "config", study: str | None = None, seed_file: bool = False,
                 reserve: Reserve | None = None, allocations_path: Path | str = ALLOCATIONS_PATH) -> list[str]:
    """Problems with any seed list that touches the reserve outside a recorded allocation.

    Hold-out/exclusion lists may name any allocated seed; other lists (explicit
    match seeds, study seed files, anchor seeds) may name only seeds allocated to
    ``study`` (every allocation when no study is given).
    """
    reserve = reserve or load_reserve()
    every = set(allocated_seeds(reserve=reserve, path=allocations_path))
    own = every if study is None else set(allocated_seeds(study, reserve=reserve, path=allocations_path))
    problems = []
    for key, seeds in seed_lists(value, seed_file=seed_file):
        hit = set(seeds) & reserve.blocked
        allowed = every if key.rsplit(".", 1)[-1] in EXCLUSION_KEYS else own
        bad = sorted(hit - allowed)
        if bad:
            problems.append(f"{name}:{key} uses {len(bad)} unallocated evaluation-reserve seeds "
                            f"(e.g. {bad[:5]})")
    return problems


def check_paths(paths: Iterable[Path | str], *, study: str | None = None,
                reserve: Reserve | None = None, allocations_path: Path | str = ALLOCATIONS_PATH) -> list[str]:
    problems = []
    for path in map(Path, paths):
        files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
        for file in files:
            try:
                value = json.loads(file.read_text())
            except (ValueError, UnicodeDecodeError):
                continue
            problems += check_config(value, name=str(file), study=study, reserve=reserve,
                                     seed_file="seeds" in file.name, allocations_path=allocations_path)
    return problems


# ------------------------------------------------------------------------ build


def training_journal(path: str) -> bool:
    """Journals whose games are learned from (PPO journals and retention anchor banks)."""
    return "training-games" in path or "controller-retention-bank" in path


def build_reserve(journals: dict[str, set[int]], registered: set[int], *, size: int,
                  lineage: tuple[str, ...], legacy_outputs: tuple[str, ...], sources: dict,
                  created_at: str) -> dict:
    """Rank game classes by exposure, then by a keyed hash, and keep the first ``size``.

    Classes holding a registered study seed or a retention-anchor seed are left
    out entirely (they already belong to a study or to reusable training data).
    """
    members, orbit = class_members(), orbit_seeds()
    anchors = set().union(*[s for p, s in journals.items() if "controller-retention-bank" in p])
    training = [s for p, s in journals.items() if training_journal(p)]
    trained = set().union(*training)
    related = set().union(*[s for p, s in journals.items()
                            if training_journal(p) and any(k in p for k in lineage)])
    seen = set().union(*journals.values())
    ranked, excluded = [], 0
    for cls, seeds in members.items():
        group = set(seeds)
        if len(seeds) != 2 or group & (registered | anchors):  # 0x100 is the lockup seed
            excluded += 1
            continue
        tier = 3 if group & related else 2 if group & trained else 1 if group & seen else 0
        exposure = sum(1 for s in training if group & s) if tier >= 2 else 0
        (evaluation,) = [s for s in seeds if s in orbit]
        key = hashlib.sha256(f"{SELECTION_KEY}:{evaluation}".encode()).hexdigest()
        ranked.append((tier, exposure, key, evaluation))
    ranked.sort()
    if len(ranked) < size:
        raise ValueError(f"only {len(ranked)} eligible game classes")
    chosen = ranked[:size]
    blocked = sum(len(members[class_of(r[-1])]) for r in chosen)
    status = [STATUSES[r[0]] for r in chosen]
    eligible = {STATUSES[t]: sum(1 for r in ranked if r[0] == t) for t in range(len(STATUSES))}
    # Every status is recorded per game class: a seed counts as seen when either twin was.
    return dict(
        schema=RESERVE_SCHEMA, created_at=created_at, selection_key=SELECTION_KEY,
        seed_convention="arena reset seed s -> rng_state (s & 0xFF, s >> 8); evaluation seeds are "
                        "hardware-reachable orbit states, one per game class",
        game_class_rule="seeds s and s ^ 0x100 step to the same register and play byte-identical games",
        rule="exclude classes holding any registered study/holdout seed or retention-anchor seed; rank the "
             "rest unseen < evaluated-only < trained only by unrelated arms < trained by the champion/afterstate "
             "lineage, then by the number of training journals holding the game, then by "
             "sha256(selection_key:seed); allocate in this order",
        size=size, blocked_count=blocked,
        status_at_creation={k: status.count(k) for k in STATUSES},
        eligible_classes=eligible, excluded_classes=excluded, lineage_markers=list(lineage),
        legacy_training_outputs=list(legacy_outputs), sources=sources,
        limitation="Exposure reflects only the audited journals; the grandfathered runs keep drawing "
                   "reserve seeds until they finish, so their checkpoints and descendants have seen them.",
        status_legend={str(i): s for i, s in enumerate(STATUSES)},
        seeds=[r[-1] for r in chosen], status_codes="".join(str(r[0]) for r in chosen),
        training_journals=[r[1] for r in chosen],
    )


def _registered_from(paths: Iterable[Path], limit: int = 4 << 20) -> set[int]:
    found = set()
    for file in paths:
        if file.stat().st_size > limit:
            continue
        try:
            value = json.loads(file.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        for _, seeds in seed_lists(value, seed_file="seeds" in file.name):
            found.update(s for s in seeds if 1 <= s <= 65535)
    return found


def _cli_build(args) -> None:
    if RESERVE_PATH.exists() and not args.force:
        raise SystemExit(f"{RESERVE_PATH} exists; the reserve is permanent (use --force only to rebuild "
                         "before any allocation)")
    journals, sources, registered = {}, {}, set()
    for audit in args.audit:
        data = json.load(gzip.open(audit, "rt"))
        for p, v in data["journals"].items():
            journals.setdefault(p, set()).update(map(int, v["seeds"]))
        registered.update(map(int, data.get("mixed_v2_holdout", [])))
        sources[Path(audit).name + ":" + hashlib.sha256(Path(audit).read_bytes()).hexdigest()[:16]] = \
            len(data["journals"])
    files = sorted({f for pattern in args.registered for f in Path("/").glob(pattern.lstrip("/"))})
    registered |= _registered_from(files)
    reserve = build_reserve(journals, registered, size=args.size, lineage=tuple(args.lineage),
                            legacy_outputs=tuple(args.legacy_output), created_at=datetime.now(UTC).isoformat(
                                timespec="seconds"),
                            sources=dict(audits=sources, journals=len(journals),
                                         registered_files=len(files), registered_seeds=len(registered)))
    RESERVE_PATH.write_text(json.dumps(reserve, separators=(",", ":")) + "\n")
    load_reserve.cache_clear()
    print(json.dumps({k: v for k, v in reserve.items()
                      if k not in ("seeds", "status_codes", "training_journals")}, indent=1))


def resize_reserve(size: int, *, path: Path | str = RESERVE_PATH,
                   allocations_path: Path | str = ALLOCATIONS_PATH, now: datetime | None = None) -> dict:
    """Keep only the first ``size`` reserve games; the rest return to training.

    Truncation never reorders, so every allocation made from the longer file
    stays a valid slice; the old file hash is kept as compatible.
    """
    path = Path(path)
    raw = path.read_bytes()
    data = json.loads(raw)
    used = sum(int(a["count"]) for a in load_allocations(allocations_path))
    if not used <= size < len(data["seeds"]):
        raise ValueError(f"size must cover the {used} allocated games and shrink the reserve")
    data["truncated_from"] = [*data.get("truncated_from", []), dict(
        sha256=hashlib.sha256(raw).hexdigest(), size=len(data["seeds"]),
        at=(now or datetime.now(UTC)).isoformat(timespec="seconds"))]
    data["seeds"], data["status_codes"] = data["seeds"][:size], data["status_codes"][:size]
    data["training_journals"] = data["training_journals"][:size]
    data["size"] = size
    data["blocked_count"] = sum(len(class_members()[class_of(s)]) for s in data["seeds"])
    data["status_at_creation"] = {k: data["status_codes"].count(str(i)) for i, k in enumerate(STATUSES)}
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    load_reserve.cache_clear()
    return data


def build_seed_frequency(levels: Iterable[int] = range(21), *, source_dir: Path | None = None) -> dict:
    """Pool drmariostats ``/api/seeds`` counts over levels into an arena-seed frequency table.

    The API reports seeds as ``rng_state[0] << 8 | rng_state[1]`` (the seedlab
    convention); stored seeds are arena seeds. ``source_dir`` reads saved
    ``seeds-<level>.json`` responses instead of fetching.
    """
    import urllib.request

    counts: dict[int, int] = {}
    per_level = {}
    for level in levels:
        if source_dir is not None:
            payload = json.loads((Path(source_dir) / f"seeds-{level}.json").read_text())
        else:
            with urllib.request.urlopen(FREQUENCY_API.format(level=level), timeout=120) as response:
                payload = json.load(response)
        if payload.get("lvl") != level:
            raise ValueError(f"level {level}: unexpected response")
        per_level[str(level)] = sum(int(r["n"]) for r in payload["rows"])
        for row in payload["rows"]:
            s = int(row["s"])
            seed = arena_seed(s >> 8, s & 0xFF)
            counts[seed] = counts.get(seed, 0) + int(row["n"])
    if not set(counts) <= orbit_seeds():
        raise ValueError("API seeds are not console-reachable states; check the byte convention")
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return dict(schema=FREQUENCY_SCHEMA, source=FREQUENCY_API.format(level="0..20"),
                filter="drmariostats /api/seeds rows: status-ok replays, garbage crowns removed",
                fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
                convention="arena reset seed (rng_state[0] | rng_state[1] << 8), all levels pooled",
                games_by_level=per_level, total_games=sum(counts.values()), distinct_seeds=len(counts),
                seeds=[s for s, _ in ordered], games=[n for _, n in ordered])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show", help="summarize the reserve and its allocations")
    a = sub.add_parser("allocate", help="record a disjoint reserve slice for a named study")
    a.add_argument("study")
    a.add_argument("count", type=int)
    a.add_argument("--purpose", required=True)
    a.add_argument("--out", type=Path, help="also write the seeds to this JSON file")
    c = sub.add_parser("check", help="fail when configs or seed lists use unallocated reserve seeds")
    c.add_argument("paths", nargs="+", type=Path)
    c.add_argument("--study", help="non-exclusion lists may use only this study's allocation")
    b = sub.add_parser("build", help="create the reserve from cached journal audits (once)")
    b.add_argument("--audit", nargs="+", required=True, help="seed-audit-*.json.gz files")
    b.add_argument("--registered", nargs="*", default=[], help="glob(s) of JSON files holding registered seed lists")
    b.add_argument("--size", type=int, default=4096)
    b.add_argument("--lineage", nargs="*", default=["controller-core-live-v4/", "controller-retention-mixed-v2/",
                                                    "afterstate"])
    b.add_argument("--legacy-output", nargs="*", default=[])
    b.add_argument("--force", action="store_true")
    r = sub.add_parser("resize", help="truncate the reserve; released games return to training")
    r.add_argument("size", type=int)
    f = sub.add_parser("frequency", help="rebuild seed_frequency.json from drmariostats /api/seeds")
    f.add_argument("--source-dir", type=Path, help="saved seeds-<level>.json responses")
    args = parser.parse_args(argv)
    if args.command == "build":
        return _cli_build(args)
    if args.command == "resize":
        data = resize_reserve(args.size)
        print(json.dumps({k: data[k] for k in ("size", "blocked_count", "truncated_from")}, indent=1))
        return
    if args.command == "frequency":
        data = build_seed_frequency(source_dir=args.source_dir)
        FREQUENCY_PATH.write_text(json.dumps(data, separators=(",", ":")) + "\n")
        load_seed_frequency.cache_clear()
        print(json.dumps({k: data[k] for k in ("total_games", "distinct_seeds", "games_by_level")}))
        return
    reserve = load_reserve()
    if args.command == "show":
        allocations = load_allocations()
        validate_allocations(allocations, reserve)
        used = sum(a["count"] for a in allocations)
        print(json.dumps(dict(size=len(reserve.seeds), blocked_training_seeds=len(reserve.blocked),
                              allocated=used, remaining=len(reserve.seeds) - used,
                              legacy_training_outputs=reserve.legacy_outputs,
                              allocations=[{k: a[k] for k in ("study", "start", "count", "created_at", "purpose")}
                                           for a in allocations]), indent=1))
    elif args.command == "allocate":
        seeds = allocate(args.study, args.count, args.purpose, reserve=reserve)
        if args.out:
            args.out.write_text(json.dumps(dict(study=args.study, seeds=seeds), indent=1) + "\n")
        print(json.dumps(dict(study=args.study, count=len(seeds), first=seeds[:4])))
    elif args.command == "check":
        problems = check_paths(args.paths, study=args.study, reserve=reserve)
        for p in problems:
            print(p, file=sys.stderr)
        if problems:
            raise SystemExit(f"{len(problems)} seed list(s) use unallocated evaluation-reserve seeds")
        print("ok: no unallocated evaluation-reserve seeds")


if __name__ == "__main__":
    main()
