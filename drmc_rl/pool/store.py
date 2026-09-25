"""Append-only pool journal and the state folded from it.

Data directory layout (everything a rating needs is in the first two files)::

    pool.json        settings (anchor rating, scheduler constants, fidelity)
    games.jsonl      one compact row per played or imported game: the rating input
    registry.jsonl   typed events: entrant, condition, condition_set, job,
                     intention, admission, block, unblock
    batches.jsonl    one row per accepted batch (throughput, worker, digest)
    audit.jsonl      fidelity comparisons (calibration and replicate audits)
    traces/          sampled gzip move traces (bounded by pool.json trace_cap_mb)
    artifacts/       content-addressed checkpoints served to workers
    spool/           replicate-audit references awaiting a second worker

Every file is append-only JSON lines written with a single ``write`` and
``fsync``. A torn last line from a crash is truncated on open. Game rows are
idempotent by id: ``condition/a/b/seed/side`` with ``a < b``. Replaying the
files reproduces the coordinator's state exactly.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import fnmatch
import json
import os
from pathlib import Path
import threading
import time

from drmc_rl.pool.conditions import check_name, condition_key, default_name, requirements
from drmc_rl.pool.ratings import PairStats

SCHEMA = "drmc-rating-pool-v1"
ENTRANT_STATUSES = ("active", "benched", "retired")
JOB_STATUSES = ("active", "paused", "done", "cancelled")
LOADERS = ("plain", "pace_adapter")

DEFAULT_SETTINGS = dict(
    schema=SCHEMA,
    anchor_rating=1500.0,
    prior_sd=4.0,
    seed_allocation="rating-pool-v1",
    batch_games=dict(events=32, frames=16),
    min_rated_games=64,           # below this an entrant is "new" under a condition
    target_games=512,             # background coverage target per entrant and condition
    new_entrant_boost=8.0,
    underplayed_boost=3.0,
    nearest_opponents=4,
    background_priority=10,
    background_min_share=0.1,     # this share of leases goes to background even while jobs run
    max_inflight_per_pairing=2,
    trace_every=16,               # keep move traces for one seed pair in this many
    trace_cap_mb=2048,
    lease_ttl=1800.0,
    replicate_every=16,
    calibration_games=8,
    fidelity="tolerant",
    min_agreement=0.99,
    trust=["mps/"],
    refit_seconds=20.0,
    default_anchor="champion-retention-mixed-v2",
    # Lineages (snapshots of one training run): older snapshots of an active run get a thin
    # maintenance share until their per-condition 95% half-width is below maintenance_ci
    # (75 per condition is about +/-28 pooled over seven paces), then only occasional games.
    maintenance_ci=75.0,
    maintenance_share=0.1,
    maintenance_idle=0.01,
    max_download_streams=2,
    download_mb_per_second=20.0,
)


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Journal:
    """One append-only JSON-lines file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._repair()
        self.lock = threading.Lock()

    def _repair(self):
        with open(self.path, "rb+") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            if not size:
                return
            stream.seek(size - 1)
            if stream.read(1) == b"\n":
                return
            # Torn last line from a crash: drop it (its writer never got an acknowledgement).
            stream.seek(0)
            data = stream.read()
            keep = data.rfind(b"\n") + 1
            stream.truncate(keep)

    def read(self):
        with open(self.path) as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)

    def append(self, rows):
        if isinstance(rows, dict):
            rows = [rows]
        if not rows:
            return
        data = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)
        with self.lock, open(self.path, "a") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())


def game_id(condition, a, b, seed, side):
    return f"{condition}/{a}/{b}/{seed}/{side}"


def canonical_pairing(a, b, side, score):
    """Order a pairing as (a < b), flipping side and score when needed."""
    if a < b:
        return a, b, side, score
    return b, a, 1 - side, None if score is None else 1.0 - score


def pattern_match(patterns, ids):
    out = []
    for pattern in patterns:
        hits = sorted(i for i in ids if fnmatch.fnmatchcase(i, pattern))
        out.extend(h for h in hits if h not in out)
    return out


class PoolState:
    """Everything folded from the journals. Single-threaded use (the coordinator's state thread)."""

    def __init__(self, data_dir, *, settings=None):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        settings_path = self.dir / "pool.json"
        if settings_path.exists():
            stored = json.loads(settings_path.read_text())
        else:
            stored = dict(DEFAULT_SETTINGS, created_at=now_iso())
            settings_path.write_text(json.dumps(stored, indent=1) + "\n")
        self.settings = {**DEFAULT_SETTINGS, **stored, **(settings or {})}
        self.games_journal = Journal(self.dir / "games.jsonl")
        self.registry_journal = Journal(self.dir / "registry.jsonl")
        self.batches_journal = Journal(self.dir / "batches.jsonl")
        self.audit_journal = Journal(self.dir / "audit.jsonl")
        (self.dir / "traces").mkdir(exist_ok=True)
        (self.dir / "artifacts").mkdir(exist_ok=True)
        (self.dir / "spool").mkdir(exist_ok=True)
        self.entrants, self.conditions, self.condition_names = {}, {}, {}
        self.condition_sets, self.jobs, self.intentions = {}, {}, {}
        self.lineages = {}                     # run -> {status, best, final, stop_rule, reason}
        self.admitted, self.blocks = {}, {}
        self.history = defaultdict(list)       # (kind, id) -> events, for reports
        self.games = {}                        # id -> row
        # condition -> (a, b) -> seed -> {side: row}
        self.by_pairing = defaultdict(lambda: defaultdict(dict))
        self.dirty = set()
        self._stats = {}
        for event in self.registry_journal.read():
            self._apply(event)
        for row in self.games_journal.read():
            self._index_game(row)

    # -- registry events --------------------------------------------------------
    def record(self, event_type, body, *, by="coordinator"):
        event = dict(type=event_type, time=now_iso(), by=by, **body)
        self._apply(event)             # validate before persisting
        self.registry_journal.append(event)
        return event

    def _apply(self, event):
        kind = event["type"]
        if kind == "entrant":
            record = {k: v for k, v in event.items() if k not in ("type", "time", "by")}
            current = self.entrants.get(record["id"])
            merged = {**(current or {}), **record}
            validate_entrant(merged)
            if current is None:
                merged.setdefault("added_at", event["time"])
            merged["updated_at"] = event["time"]
            self.entrants[record["id"]] = merged
            self.history[("entrant", record["id"])].append(event)
        elif kind == "condition":
            spec = event["spec"]
            key = condition_key(spec)
            name = check_name(event.get("name") or default_name(spec))
            if key in self.conditions and self.conditions[key]["name"] != name:
                raise ValueError(f"condition {key} is already registered as {self.conditions[key]['name']}")
            if name in self.condition_names and self.condition_names[name] != key:
                raise ValueError(f"condition name {name} already names another condition")
            self.conditions[key] = dict(key=key, name=name, spec=spec, notes=event.get("notes", ""),
                                        added_at=self.conditions.get(key, {}).get("added_at", event["time"]))
            self.condition_names[name] = key
        elif kind == "condition_set":
            name = check_name(event["name"])
            keys = [self.resolve_condition(c) for c in event["conditions"]]
            anchor = event["anchor"]
            for key in keys:
                for other in self.condition_sets.values():
                    if other["name"] != name and key in other["conditions"] and other["anchor"] != anchor:
                        raise ValueError(f"condition {key} already belongs to set {other['name']} "
                                         f"anchored on {other['anchor']}")
            self.condition_sets[name] = dict(name=name, conditions=keys, anchor=anchor,
                                             weight=float(event.get("weight", 1.0)),
                                             primary=bool(event.get("primary", False)),
                                             notes=event.get("notes", ""))
        elif kind == "job":
            record = {k: v for k, v in event.items() if k not in ("type", "time", "by")}
            merged = {**self.jobs.get(record["id"], {}), **record}
            validate_job(merged)
            merged.setdefault("created_at", event["time"])
            merged["updated_at"] = event["time"]
            self.jobs[record["id"]] = merged
            self.history[("job", record["id"])].append(event)
        elif kind == "intention":
            from drmc_rl.pool.intentions import validate_intention
            record = {k: v for k, v in event.items() if k not in ("type", "time", "by")}
            merged = {**self.intentions.get(record["id"], {}), **record}
            validate_intention(merged)
            merged.setdefault("created_at", event["time"])
            merged["updated_at"] = event["time"]
            self.intentions[record["id"]] = merged
            self.history[("intention", record["id"])].append(event)
        elif kind == "lineage":
            record = {k: v for k, v in event.items() if k not in ("type", "time", "by")}
            check_name(record["run"])
            merged = {**self.lineages.get(record["run"], dict(status="active")), **record}
            if merged["status"] not in ("active", "concluded"):
                raise ValueError("lineage status must be active or concluded")
            rule = merged.get("stop_rule")
            if rule is not None and (not isinstance(rule, dict) or "set" not in rule):
                raise ValueError("lineage stop_rule needs at least a condition set")
            merged["updated_at"] = event["time"]
            self.lineages[record["run"]] = merged
            self.history[("lineage", record["run"])].append(event)
        elif kind == "admission":
            self.admitted[event["numerics"]] = event["verdict"]
            self._stats.clear()
            self.dirty.update(self.conditions)
        elif kind == "block":
            self.blocks[block_key(event)] = dict(event)
        elif kind == "unblock":
            self.blocks.pop(block_key(event), None)
        else:
            raise ValueError(f"unknown registry event {kind}")

    def resolve_condition(self, ref):
        if ref in self.conditions:
            return ref
        if ref in self.condition_names:
            return self.condition_names[ref]
        raise KeyError(f"unknown condition {ref!r}")

    def expand_conditions(self, refs):
        keys = []
        for ref in refs:
            if isinstance(ref, str) and ref.startswith("set:"):
                keys.extend(self.condition_sets[ref[4:]]["conditions"])
            else:
                keys.append(self.resolve_condition(ref))
        return list(dict.fromkeys(keys))

    def anchor_for(self, condition):
        for s in self.condition_sets.values():
            if condition in s["conditions"]:
                return s["anchor"]
        return self.settings.get("default_anchor")

    # -- games ------------------------------------------------------------------
    def _index_game(self, row):
        if row["id"] in self.games:
            return False
        self.games[row["id"]] = row
        self.by_pairing[row["condition"]][(row["a"], row["b"])].setdefault(row["seed"], {})[row["side"]] = row
        self.dirty.add(row["condition"])
        self._stats.pop(row["condition"], None)
        return True

    def add_games(self, rows):
        """Append new rows (duplicates by id are ignored). Returns the rows written."""
        fresh, seen = [], set()
        for row in rows:
            if row["id"] in self.games or row["id"] in seen:
                continue
            if row["condition"] not in self.conditions:
                raise ValueError(f"game {row['id']} names unknown condition {row['condition']}")
            for e in (row["a"], row["b"]):
                if e not in self.entrants:
                    raise ValueError(f"game {row['id']} names unknown entrant {e}")
            if not row["a"] < row["b"]:
                raise ValueError("game rows must be canonical (a < b)")
            seen.add(row["id"])
            fresh.append(row)
        self.games_journal.append(fresh)
        for row in fresh:
            self._index_game(row)
        return fresh

    def counts(self, row):
        """Whether a game row may enter ratings: its numerics class is not rejected."""
        verdict = self.admitted.get(row.get("numerics"))
        return verdict is None or verdict is True

    def pair_stats(self, condition):
        """PairStats per canonical pairing: complete, uncensored, admitted seed pairs only."""
        if condition in self._stats:
            return self._stats[condition]
        stats = {}
        for pairing, seeds in self.by_pairing.get(condition, {}).items():
            s = PairStats()
            for sides in seeds.values():
                if len(sides) != 2 or any(r["score"] is None or not self.counts(r) for r in sides.values()):
                    continue
                s.add(sides[0]["score"] + sides[1]["score"],
                      draws=sum(r["score"] == 0.5 for r in sides.values()))
            if s.pairs:
                stats[pairing] = s
        self._stats[condition] = stats
        return stats

    def played_seeds(self, condition, a, b):
        """Seeds whose side-swapped pair is complete for this pairing (censored pairs included)."""
        seeds = self.by_pairing.get(condition, {}).get((a, b), {})
        return {seed for seed, sides in seeds.items() if len(sides) == 2}

    def pairing_games(self, condition, a, b, seeds=None):
        rows = self.by_pairing.get(condition, {}).get((a, b), {})
        return sum(len(sides) for seed, sides in rows.items() if seeds is None or seed in seeds)

    # -- entrants ---------------------------------------------------------------
    def entrant_ids(self, statuses=ENTRANT_STATUSES):
        return sorted(e for e, r in self.entrants.items() if r["status"] in statuses)

    def resolve_entrants(self, patterns, statuses=ENTRANT_STATUSES):
        return pattern_match(patterns, self.entrant_ids(statuses))

    # -- lineages (snapshots of one training run) -----------------------------------
    def lineage_of(self, entrant):
        return ((self.entrants.get(entrant) or {}).get("lineage") or {}).get("run")

    def step_of(self, entrant):
        return int(((self.entrants.get(entrant) or {}).get("lineage") or {}).get("step") or 0)

    def lineage_members(self, run):
        return sorted((e for e in self.entrants if self.lineage_of(e) == run), key=lambda e: (self.step_of(e), e))

    def lineage_status(self, run):
        return self.lineages.get(run, {}).get("status", "active")

    def lineage_runs(self):
        return sorted({self.lineage_of(e) for e in self.entrants} - {None})

    def newest(self, run):
        members = [e for e in self.lineage_members(run) if self.entrants[e]["status"] != "retired"]
        return members[-1] if members else None

    def snapshot_role(self, entrant):
        """None (not a lineage snapshot), 'newest', or 'older' (an earlier snapshot of an active run)."""
        run = self.lineage_of(entrant)
        if run is None or self.lineage_status(run) != "active":
            return None
        return "newest" if self.newest(run) == entrant else "older"

    def blocked(self, condition, a, b=None):
        keys = [(condition, a, None)]
        if b is not None:
            keys += [(condition, b, None), (condition, *sorted((a, b)))]
        for key in keys:
            if key in self.blocks:
                return self.blocks[key]["reason"]
        return None

    def snapshot(self):
        return copy.deepcopy(dict(entrants=self.entrants, conditions=self.conditions,
                                  condition_sets=self.condition_sets, jobs=self.jobs,
                                  intentions=self.intentions, admitted=self.admitted,
                                  blocks=list(self.blocks.values())))


def block_key(event):
    if event.get("b"):
        a, b = sorted((event["a"], event["b"]))
        return (event["condition"], a, b)
    return (event["condition"], event["a"], None)


def validate_entrant(record):
    check_name(record["id"])
    if "/" in record["id"]:
        raise ValueError("entrant ids may not contain '/'")
    if record.get("status") not in ENTRANT_STATUSES:
        raise ValueError(f"entrant status must be one of {ENTRANT_STATUSES}")
    if record.get("loader") not in LOADERS:
        raise ValueError(f"entrant loader must be one of {LOADERS}")
    checkpoint = record.get("checkpoint") or {}
    if len(str(checkpoint.get("sha256", ""))) != 64:
        raise ValueError("entrant checkpoint needs a sha256")
    if record["loader"] == "pace_adapter" and len(str((record.get("adapter") or {}).get("sha256", ""))) != 64:
        raise ValueError("a pace_adapter entrant needs adapter.sha256")
    settings = record.get("settings", {})
    forbidden = {"delay", "decision_point", "early_preview", "preview_input", "compute_input_frames",
                 "early_delay_input", "movement", "checkpoint", "adapter_checkpoint", "anticipation"}
    if set(settings) & forbidden:
        raise ValueError(f"entrant settings may not set condition or loader keys: {sorted(set(settings) & forbidden)}")
    if not record.get("era"):
        raise ValueError("entrant needs an era tag")


def validate_job(job):
    check_name(job["id"])
    if job.get("status") not in JOB_STATUSES:
        raise ValueError(f"job status must be one of {JOB_STATUSES}")
    games = job.get("games")
    if type(games) is not int or games < 2 or games % 2:
        raise ValueError("job games (per pairing and condition) must be a positive even integer")
    if not job.get("conditions"):
        raise ValueError("job needs conditions")
    if not job.get("entrants") and not job.get("pairings"):
        raise ValueError("job needs entrants or explicit pairings")
    if job.get("mode", "vs") not in ("vs", "round_robin", "vs_parent", "explicit"):
        raise ValueError("job mode must be vs, round_robin, vs_parent or explicit")
    seeds = job.get("seeds", "bank")
    if seeds != "bank" and not (isinstance(seeds, dict) and ("allocation" in seeds or "explicit" in seeds)):
        raise ValueError("job seeds must be 'bank', {'allocation': study} or {'explicit': {condition: [seeds]}}")
    if job.get("step_every") is not None and (type(job["step_every"]) is not int or job["step_every"] < 1):
        raise ValueError("job step_every must be a positive integer (frames)")
    if not isinstance(job.get("priority", 50), (int, float)):
        raise ValueError("job priority must be a number")


def entrant_requirements(record):
    return {f"loader:{record['loader']}", *record.get("requires", [])}


def condition_requirements(state, key):
    return requirements(state.conditions[key]["spec"])

