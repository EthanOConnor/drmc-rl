"""The pool coordinator: leases rating batches forever and journals their games.

Transport, authentication, checkpoint-by-hash serving, source-revision pinning,
numerics classes, calibration replays and replicate audits are those of
``tools.trainer_arena_distributed``; this class replaces its fixed study plan
with the scheduler's priority list. Torch-free: it never loads a model.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile
import time

from drmc_rl.pool import intentions as intent
from drmc_rl.pool.conditions import canonical, runtime_capabilities
from drmc_rl.pool.ratings import fit, pooled
from drmc_rl.pool.scheduler import Scheduler
from drmc_rl.pool.store import PoolState, game_id, now_iso

PROTOCOL = "drmc-rating-pool-v1"
REPO = Path(__file__).resolve().parents[2]
# Runtime settings shared by every condition (not rules; see conditions.py).
DEFAULT_RUNTIME = dict(reactive_compute_frames=4, preparation_compute_frames=6, max_game_frames=120000,
                       strict_fp32=True, memoize=True, async_planning=True, planner_workers=3, replay_games=0)
ROW_FIELDS = ("seed", "side", "index", "score", "winner", "reason", "frames")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Artifacts:
    """Content-addressed checkpoints: ``artifacts/<sha>`` plus verified local paths named by entrants."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        try:
            self.index = json.loads(self.index_path.read_text())
        except (OSError, ValueError):
            self.index = {}

    def _save(self):
        temporary = self.index_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.index))
        os.replace(temporary, self.index_path)

    def verified(self, path):
        path = Path(path)
        try:
            stat = path.stat()
        except OSError:
            return None
        key = f"{path}|{stat.st_size}|{int(stat.st_mtime)}"
        if key not in self.index:
            self.index[key] = sha256_file(path)
            self._save()
        return self.index[key]

    def path(self, digest, candidates=()):
        stored = self.root / digest
        if stored.is_file():
            return stored
        for candidate in candidates:
            if Path(candidate).is_file() and self.verified(candidate) == digest:
                return Path(candidate)
        return None

    def store(self, digest, stream, length, *, limit=4 << 30):
        if length > limit:
            raise ValueError("artifact too large")
        target = self.root / digest
        if target.is_file():
            stream.read(length)
            return target
        fd, temporary = tempfile.mkstemp(dir=self.root, prefix=f".{digest}.")
        sha, remaining = hashlib.sha256(), length
        try:
            with os.fdopen(fd, "wb") as out:
                while remaining:
                    block = stream.read(min(1 << 20, remaining))
                    if not block:
                        raise ValueError("truncated upload")
                    remaining -= len(block)
                    sha.update(block)
                    out.write(block)
            if sha.hexdigest() != digest:
                raise ValueError("uploaded artifact hash mismatch")
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return target


class PoolCoordinator:
    """All pool state. Not thread-safe: the server calls it from one thread."""

    def __init__(self, data_dir, *, source=None, capabilities=None, settings=None, log=print,
                 allow_source_mismatch=False, clock=time.time):
        self.log, self.clock = log, clock
        self.state = PoolState(data_dir, settings=settings)
        self.settings = self.state.settings
        self.dir = self.state.dir
        self.runtime = {**DEFAULT_RUNTIME, **self.settings.get("runtime", {})}
        self.source = source or _source_revision()
        self.capabilities = set(capabilities) if capabilities is not None else runtime_capabilities(REPO)
        self.allow_source_mismatch = allow_source_mismatch
        self.artifacts = Artifacts(self.dir / "artifacts")
        self.started = clock()
        self.fatal = None
        self.leases, self.workers, self.failures = {}, {}, {}
        self.inflight = {}
        self.fits, self.fitted_at = {}, {}
        self.trace_bytes = sum(p.stat().st_size for p in (self.dir / "traces").rglob("*.json.gz"))
        self.fidelity_stats = {}
        self.audits = {}
        for path in sorted((self.dir / "spool").glob("*.json.gz")):
            reference = json.loads(gzip.decompress(path.read_bytes()))
            self.audits[reference["batch"]["key"]] = reference
        self.issued = sum(1 for _ in self.state.batches_journal.read())
        self._seed_cache = {}
        self._view_cache = {}
        self.scheduler = Scheduler(self.state, seed_source=self.seed_source, available=self.available,
                                   capabilities=self.capabilities, background_seeds=self.background_seeds,
                                   roles=self.lineage_roles)
        self.tick()

    # -- seeds ----------------------------------------------------------------------
    def bank(self):
        if "bank" not in self._seed_cache:
            if self.settings.get("seed_bank"):       # tests only
                self._seed_cache["bank"] = list(self.settings["seed_bank"])
            else:
                from drmc_rl.program.seed_reserve import allocated_seeds
                self._seed_cache["bank"] = allocated_seeds(self.settings["seed_allocation"])
        return self._seed_cache["bank"]

    def seed_sets(self):
        """name -> seeds: the reserve bank (ratings history, memorization reference), ``uniform``
        (non-reserve console seeds, uniform: the fair seen-seed comparison) and ``mixture`` (50/50
        real-play frequency and uniform: its plain mean is real-play-weighted strength).

        The drawn sets are persisted in pool.json on first use, so they never change underneath
        the journal when the frequency table is refreshed."""
        if "sets" in self._seed_cache:
            return self._seed_cache["sets"]
        stored = self.settings.get("seed_sets")
        if not stored:
            import numpy as np
            from drmc_rl.program.seed_reserve import draw_mixture_seeds
            rng = np.random.default_rng(int(self.settings["seed_set_rng"]))
            size = int(self.settings["seed_set_size"])
            uniform = draw_mixture_seeds(rng, size, seed_mix=0.0, excluded=self.bank())
            mixture = draw_mixture_seeds(rng, size, seed_mix=0.5, excluded=self.bank() + uniform)
            stored = dict(uniform=dict(seed_mix=0.0, seeds=uniform), mixture=dict(seed_mix=0.5, seeds=mixture))
            self.settings["seed_sets"] = stored
            path = self.dir / "pool.json"
            saved = json.loads(path.read_text())
            saved["seed_sets"] = stored
            path.write_text(json.dumps(saved, indent=1) + "\n")
            self.log(f"pool: drew seed sets uniform ({len(uniform)}) and mixture ({len(mixture)})")
        sets = dict(reserve=list(self.bank()), **{k: list(v["seeds"]) for k, v in stored.items()})
        self._seed_cache["sets"] = sets
        self._seed_cache["set_of"] = {s: name for name, seeds in sets.items() for s in seeds}
        return sets

    def set_of(self, seed):
        self.seed_sets()
        return self._seed_cache["set_of"].get(seed)

    def background_seeds(self, condition, a, b, inflight):
        """The seed set this background pair plays next: the one furthest below its share."""
        sets = self.seed_sets()
        shares = self.settings["background_seed_shares"]
        played = self.state.played_seeds(condition, a, b) | inflight.get((condition, a, b), set())
        best = None
        for name in sorted(shares):
            seeds = sets.get(name) or []
            if not shares[name] or not seeds:
                continue
            used = sum(1 for s in seeds if s in played)
            if used >= len(seeds):
                continue
            key = (used / shares[name], name)
            if best is None or key < best[0]:
                best = (key, name, seeds)
        return ("reserve", sets["reserve"]) if best is None else (best[1], best[2])

    def seed_source(self, job, condition):
        seeds = "bank" if job is None else job.get("seeds", "bank")
        if seeds == "bank":
            return self.bank()
        if isinstance(job, str):
            job = self.state.jobs[job]
        explicit = seeds.get("explicit")
        if isinstance(explicit, dict):
            name = self.state.conditions[condition]["name"]
            return explicit.get(condition) or explicit.get(name) or []
        return seeds.get("seeds") or []

    def validate_job_seeds(self, job):
        """Pool games never use training seeds: explicit job seeds must be evaluation-reserve seeds."""
        seeds = job.get("seeds", "bank")
        if seeds == "bank":
            return
        if self.settings.get("seed_bank"):           # tests only
            blocked = set(range(1, 65536))
        else:
            from drmc_rl.program.seed_reserve import load_reserve
            blocked = load_reserve().blocked
        lists = list(seeds.get("explicit", {}).values()) if isinstance(seeds.get("explicit"), dict) \
            else [seeds.get("seeds") or []]
        for values in lists:
            bad = [s for s in values if type(s) is not int or s not in blocked]
            if bad:
                raise ValueError(f"job seeds {bad[:4]} are not evaluation-reserve seeds")
            if len(set(values)) != len(values):
                raise ValueError("job seed lists must be unique")

    # -- artifacts -------------------------------------------------------------------
    def _entrant_artifacts(self, entrant):
        record = self.state.entrants[entrant]
        items = [record["checkpoint"]]
        if record["loader"] == "pace_adapter":
            items.append(record["adapter"])
        return items

    def artifact_path(self, digest):
        for record in self.state.entrants.values():
            for item in (record["checkpoint"], record.get("adapter") or {}):
                if item.get("sha256") == digest:
                    return self.artifacts.path(digest, item.get("paths", []))
        return self.artifacts.path(digest)

    def available(self, entrant):
        return all(self.artifact_path(item["sha256"]) is not None for item in self._entrant_artifacts(entrant))

    # -- ratings ---------------------------------------------------------------------
    def tick(self):
        """Periodic bookkeeping: lease expiry, refits, intention pickup, job completion."""
        now = self.clock()
        self._expire(now)
        refitted = not getattr(self, "_ticked", False)
        self._ticked = True
        for condition in sorted(self.state.dirty):
            if now - self.fitted_at.get(condition, 0) >= self.settings["refit_seconds"] or condition not in self.fits:
                self.refit(condition)
                refitted = True
        if refitted:
            self._auto_conclude()
        for record in list(intent.ready_to_start(self.state, self.capabilities)):
            job = dict(record["job"], id=record["id"], intention=record["id"])
            job.setdefault("status", "active")
            job.setdefault("title", record["title"])
            try:
                self.validate_job_seeds(job)
                self.state.record("job", job)
                self.state.record("intention", dict(id=record["id"], status="running",
                                                    notes=_append(record.get("notes", ""),
                                                                  f"{now_iso()} job submitted by the pool")))
                self.log(f"pool: intention {record['id']} is ready; submitted its job")
            except (KeyError, ValueError) as error:
                self.log(f"pool: intention {record['id']} job rejected: {error}")
        for job in list(self.state.jobs.values()):
            if job["status"] == "active" and self.scheduler.job_complete(job):
                self.state.record("job", dict(id=job["id"], status="done"))
                self.log(f"pool: job {job['id']} complete")
        for key, reference in list(self.audits.items()):
            if now - reference["created"] > 6 * 3600:
                self._drop_audit(key, "no second worker within six hours")

    def refit(self, condition):
        anchor = self.state.anchor_for(condition)
        if anchor is None or anchor not in self.state.entrants:
            return None
        self.fits[condition] = fit(self.state.pair_stats(condition), anchor,
                                   anchor_rating=self.settings["anchor_rating"], prior_sd=self.settings["prior_sd"])
        self.fitted_at[condition] = self.clock()
        self.state.dirty.discard(condition)
        return self.fits[condition]

    def all_fits(self):
        for condition in list(self.state.dirty):
            self.refit(condition)
        for condition in self.state.conditions:
            if condition not in self.fits:
                self.refit(condition)
        return self.fits

    # -- workers and leases ------------------------------------------------------------
    def _register(self, worker):
        if worker.get("protocol") != PROTOCOL:
            raise PermissionError("protocol mismatch")
        if worker.get("source") != self.source and not self.allow_source_mismatch:
            raise PermissionError(f"worker source {worker.get('source')} != coordinator {self.source}")
        identity = {k: worker[k] for k in ("worker_id", "host", "numerics", "device", "threads", "planner_workers",
                                           "source", "native", "engine", "capabilities") if k in worker}
        previous = self.workers.get(identity["worker_id"], {})
        identity.update(seen=self.clock(), batches=previous.get("batches", 0), games=previous.get("games", 0),
                        seconds=previous.get("seconds", 0.0), failures=previous.get("failures", 0),
                        last_error=previous.get("last_error"), status=previous.get("status", "idle"))
        self.workers[identity["worker_id"]] = identity
        return identity

    def _trusted(self, numerics):
        trust = tuple(self.settings["trust"])
        return bool(trust) and numerics.startswith(trust)

    def admission(self, numerics):
        verdict = self.state.admitted.get(numerics)
        if verdict is None and self._trusted(numerics):
            return True
        return verdict

    def _expire(self, now):
        for lease_id, lease in list(self.leases.items()):
            if lease["expires"] < now:
                self._drop(lease_id, "expired")

    def _drop(self, lease_id, reason):
        lease = self.leases.pop(lease_id)
        spec = lease["batch"]
        slot = (spec["condition"], spec["a"], spec["b"])
        if lease["purpose"] == "play":
            self.inflight.get(slot, set()).difference_update(spec["seeds"])
            self.inflight.get((*slot, "leases"), set()).discard(lease_id)
        self.log(f"pool: lease {lease_id[:8]} {spec['key']} by {lease['worker']} {reason}")

    def heartbeat(self, worker):
        """A worker that is alive but not leasing (e.g. paused for the host budget or disk)."""
        identity = self._register(worker)
        identity["status"] = f"paused ({str(worker.get('reason', ''))[:120]})"
        return dict(ok=True)

    def lease(self, worker):
        if self.fatal:
            return dict(status="wait", reason="coordinator stopped after an internal error", retry=60)
        now = self.clock()
        self._expire(now)
        identity = self._register(worker)
        claimed = set(identity.get("capabilities") or ())
        # Source capabilities are the coordinator's (same revision); the engine build is the worker's claim.
        caps = (claimed & self.capabilities) | {c for c in claimed if c.startswith("engine:")}
        numerics = identity["numerics"]
        verdict = self.admission(numerics)
        if verdict is not True and verdict is not None:
            return dict(status="rejected", reason=f"numerics class {numerics} failed fidelity: {verdict}")
        if verdict is None:
            if not self.settings["calibration_games"]:
                verdict = True
            else:
                if any(l["purpose"] == "calibrate" and l["numerics"] == numerics for l in self.leases.values()):
                    return dict(status="wait", reason="calibration of this numerics class is in progress", retry=30)
                spec = self._calibration_batch(caps)
                if spec is None:
                    return dict(status="wait", reason="calibration needs traced games from an admitted host", retry=60)
                return self._issue(identity, spec, "calibrate", now)
        # Replicate audits first, so a sampled batch is replayed while its reference is fresh.
        for key, reference in sorted(self.audits.items()):
            if reference.get("leased_until", 0) > now or reference["worker"]["worker_id"] == identity["worker_id"]:
                continue
            if reference["worker"]["numerics"] == numerics and self._other_class_available(reference):
                continue
            reference["leased_until"] = now + self.settings["lease_ttl"]
            return self._issue(identity, reference["batch"], "replicate", now)
        batch = self.scheduler.next_batch(self.all_fits(), caps, self.inflight)
        if batch is None:
            return dict(status="wait", reason="nothing to play for this worker's capabilities", retry=60)
        self.issued += 1
        batch["audit"] = bool(self.settings["replicate_every"]) and self.issued % self.settings["replicate_every"] == 0
        return self._issue(identity, self._spec(batch), "play", now)

    def _other_class_available(self, reference):
        own = reference["worker"]["numerics"]
        return any(w["numerics"] != own and self.clock() - w["seen"] < 3 * self.settings["lease_ttl"]
                   and self.admission(w["numerics"]) is True for w in self.workers.values())

    def _spec(self, batch):
        """Everything a worker needs to play one batch, with checkpoints by hash."""
        condition = self.state.conditions[batch["condition"]]
        spec = condition["spec"]
        if batch.get("job") is None:
            sets = self.seed_sets()
            order = sets.get(self.set_of(batch["seeds"][0]) or "reserve", sets["reserve"])
        else:
            order = self.seed_source(self.state.jobs[batch["job"]], batch["condition"])
        position = {s: i for i, s in enumerate(order)}
        jobs = [(seed, side, 2 * position.get(seed, 0) + side) for seed in batch["seeds"] for side in (0, 1)]
        key = f"{condition['name']}.{batch['a']}.{batch['b']}.{batch['seeds'][0]}.{len(jobs)}"
        variants = {}
        for entrant in (batch["a"], batch["b"]):
            record = self.state.entrants[entrant]
            params = dict(record.get("settings", {}), name=record.get("name", entrant), **spec["decision"])
            if spec["movement"] != "exact":
                params["movement"] = spec["movement"]
            if record["loader"] == "pace_adapter":
                params["checkpoint"] = "sha256:" + record["checkpoint"]["sha256"]
                params["adapter_checkpoint"] = "sha256:" + record["adapter"]["sha256"]
            else:
                params["checkpoint"] = "sha256:" + record["checkpoint"]["sha256"]
            variants[entrant] = params
        from drmc_rl.pool.conditions import pace_profile
        match = dict(id=key, a=batch["a"], b=batch["b"], games=len(jobs), level=spec["level"], pace=spec["pace"],
                     execution_profile=pace_profile(spec["pace"]), execution_key=spec["execution_key"])
        anchor = self.state.anchor_for(batch["condition"])
        artifacts = {}
        for entrant in (batch["a"], batch["b"], anchor):
            for item in self._entrant_artifacts(entrant):
                artifacts[item["sha256"]] = dict(name=item.get("name") or item["sha256"][:12] + ".pt",
                                                 size=item.get("size"), paths=item.get("paths", []))
        return dict(key=key, condition=batch["condition"], a=batch["a"], b=batch["b"], seeds=batch["seeds"],
                    job=batch.get("job"), why=batch.get("why", ""), audit=batch.get("audit", False),
                    match=match, variants=variants, jobs=jobs, engine=spec["engine"], backend=spec["backend"],
                    runtime=dict(self.runtime, rollout_backend=spec["backend"],
                                 anchor_checkpoint="sha256:" + self.state.entrants[anchor]["checkpoint"]["sha256"]),
                    artifacts=artifacts)

    def _calibration_batch(self, caps):
        """Journaled, traced whole seed pairs of one pairing, for a new numerics class to replay."""
        wanted = max(1, self.settings["calibration_games"] // 2)
        for condition in sorted(self.state.by_pairing):
            for (a, b), seeds in sorted(self.state.by_pairing[condition].items()):
                if not all(self.scheduler.playable(condition, e, caps) for e in (a, b)):
                    continue
                traced = [s for s, sides in sorted(seeds.items())
                          if len(sides) == 2 and all(r.get("trace") and self.admission(r.get("numerics")) is True
                                                     for r in sides.values())]
                if len(traced) >= wanted:
                    spec = self._spec(dict(condition=condition, a=a, b=b, seeds=traced[:wanted]))
                    spec["key"] = "calibration." + spec["key"]
                    return spec
        return None

    def _issue(self, identity, spec, purpose, now):
        lease_id, claim = secrets.token_hex(12), secrets.token_hex(16)
        self.leases[lease_id] = dict(batch=spec, claim=claim, worker=identity["worker_id"], purpose=purpose,
                                     numerics=identity["numerics"], expires=now + self.settings["lease_ttl"],
                                     issued=now)
        if purpose == "play":
            slot = (spec["condition"], spec["a"], spec["b"])
            self.inflight.setdefault(slot, set()).update(spec["seeds"])
            self.inflight.setdefault((*slot, "leases"), set()).add(lease_id)
        self.workers[identity["worker_id"]]["status"] = f"{purpose} {spec['key']}"
        self.log(f"pool: {purpose} {spec['key']} ({len(spec['jobs'])} games; {spec.get('why', '')}) -> "
                 f"{identity['worker_id']}")
        return dict(status="lease", lease_id=lease_id, claim_token=claim, purpose=purpose, batch=spec,
                    ttl_seconds=self.settings["lease_ttl"])

    def renew(self, lease_id, claim):
        lease = self.leases.get(lease_id)
        if lease is None or not secrets.compare_digest(lease["claim"], claim):
            return dict(renewed=False)
        lease["expires"] = self.clock() + self.settings["lease_ttl"]
        return dict(renewed=True, ttl_seconds=self.settings["lease_ttl"])

    def release(self, lease_id, claim):
        lease = self.leases.get(lease_id)
        if lease is None or not secrets.compare_digest(lease["claim"], claim):
            return dict(released=False)
        if lease["purpose"] == "replicate":
            self.audits.get(lease["batch"]["key"], {}).pop("leased_until", None)
        self._drop(lease_id, "released by its worker")
        return dict(released=True)

    def release_batch(self, key):
        dropped = [i for i, l in self.leases.items() if l["batch"]["key"] == key]
        for lease_id in dropped:
            self._drop(lease_id, "released by operator")
        return dict(released=len(dropped), batch=key)

    def fail(self, lease_id, payload):
        """A worker could not play its batch. Deterministic failures block the entrant under that condition."""
        lease = self.leases.get(lease_id)
        if lease is None or not secrets.compare_digest(lease["claim"], str(payload.get("claim_token", ""))):
            return dict(recorded=False)
        spec = lease["batch"]
        error, kind = str(payload.get("error", ""))[:2000], payload.get("kind", "transient")
        worker = self.workers.get(lease["worker"], {})
        worker.update(failures=worker.get("failures", 0) + 1, last_error=error[:300])
        self._drop(lease_id, f"failed ({kind}): {error[:200]}")
        if lease["purpose"] == "replicate":
            self.audits.get(spec["key"], {}).pop("leased_until", None)
        if kind == "incompatible":
            slot = (spec["condition"], spec["a"], spec["b"])
            self.failures.setdefault(slot, []).append(dict(worker=lease["worker"], error=error, time=self.clock()))
            if len(self.failures[slot]) >= 2:
                self._block(spec, error)
        return dict(recorded=True)

    def _block(self, spec, error):
        condition, a, b = spec["condition"], spec["a"], spec["b"]
        rated = {e for pair in self.state.by_pairing.get(condition, {}) for e in pair}
        anchor = self.state.anchor_for(condition)
        # The anchor, and entrants that have already played this condition, are not the suspects.
        suspects = [e for e in (a, b) if e not in rated and e != anchor]
        if suspects:
            for e in suspects:
                self.state.record("block", dict(condition=condition, a=e, reason=f"two failed batches: {error[:300]}"))
        else:
            self.state.record("block", dict(condition=condition, a=a, b=b,
                                            reason=f"two failed batches: {error[:300]}"))
        self.log(f"pool: blocked {suspects or [a, b]} under {self.state.conditions[condition]['name']}")

    # -- results -------------------------------------------------------------------------
    def submit(self, lease_id, payload):
        lease = self.leases.get(lease_id)
        if lease is not None and not secrets.compare_digest(lease["claim"], str(payload.get("claim_token", ""))):
            raise PermissionError("claim token mismatch")
        spec = lease["batch"] if lease is not None else payload.get("batch")
        purpose = lease["purpose"] if lease is not None else payload.get("purpose", "play")
        if not isinstance(spec, dict) or spec.get("condition") not in self.state.conditions:
            raise ValueError("submission does not name a known batch")
        jobs = [tuple(j) for j in spec["jobs"]]
        rows, moves = payload["rows"], payload["moves"]
        if len(rows) != len(jobs) or any((r["seed"], r["side"], r["index"]) != j for r, j in zip(rows, jobs)):
            raise ValueError("submission does not match the leased games")
        if len(moves) != len(rows):
            raise ValueError("submission needs one move journal per game")
        elapsed = payload.get("elapsed")
        if not isinstance(elapsed, (int, float)) or not 0 < elapsed < float("inf"):
            raise ValueError("elapsed must be a positive number of seconds")
        worker = payload.get("worker", {})
        if not {"worker_id", "numerics", "device", "threads"} <= set(worker):
            raise ValueError("submission needs the worker identity it leased with")
        for entrant in (spec["a"], spec["b"]):
            if entrant not in self.state.entrants:
                raise ValueError(f"unknown entrant {entrant}")
        digest = hashlib.sha256(canonical(dict(rows=rows, moves=moves)).encode()).hexdigest()
        if lease is not None:
            self.leases.pop(lease_id)
            if purpose == "play":
                slot = (spec["condition"], spec["a"], spec["b"])
                self.inflight.get(slot, set()).difference_update(spec["seeds"])
                self.inflight.get((*slot, "leases"), set()).discard(lease_id)
        if purpose == "calibrate":
            return self._calibrate(spec, payload, worker)
        if purpose == "replicate":
            return self._replicate(spec, payload, worker, digest)
        verdict = self.admission(worker["numerics"])
        if verdict is not True and verdict is not None:
            return dict(accepted=False, reason=f"numerics class {worker['numerics']} is rejected")
        fresh = self._journal(spec, rows, moves, worker, elapsed, digest, source="pool")
        record = self.workers.get(worker["worker_id"])
        if record is not None:
            record.update(batches=record.get("batches", 0) + 1, games=record.get("games", 0) + len(rows),
                          seconds=record.get("seconds", 0.0) + elapsed, status="idle")
        if spec.get("audit") and fresh:
            reference = dict(batch=spec, worker=worker, rows=rows, moves=moves, created=self.clock())
            path = self.dir / "spool" / f"{_safe(spec['key'])}.json.gz"
            path.write_bytes(gzip.compress(canonical(reference).encode(), mtime=0))
            self.audits[spec["key"]] = reference
        return dict(accepted=True, sha256=digest, new_games=len(fresh), duplicate_games=len(rows) - len(fresh))

    def _journal(self, spec, rows, moves, worker, elapsed, digest, *, source):
        condition, a, b = spec["condition"], spec["a"], spec["b"]
        out, traced = [], []
        stamp = now_iso()
        every = max(1, int(self.settings["trace_every"]))
        cap = self.settings["trace_cap_mb"] << 20
        for row, move in zip(rows, moves):
            gid = game_id(condition, a, b, row["seed"], row["side"])
            if gid in self.state.games:
                continue
            compact = dict(id=gid, condition=condition, a=a, b=b, seed=row["seed"], side=row["side"],
                           score=row["score"], winner=row["winner"], reason=row["reason"], frames=row["frames"],
                           decisions=[row.get("a_stats", {}).get("decisions", 0),
                                      row.get("b_stats", {}).get("decisions", 0)],
                           source=source, batch=spec["key"], job=spec.get("job"), worker=worker["worker_id"],
                           numerics=worker["numerics"], time=stamp)
            if isinstance(row.get("style"), list) and len(row["style"]) == 2:
                compact["style"] = row["style"]          # combo/showiness counters [a, b]; visibility only
            pair_hash = int(hashlib.sha1(f"{condition}/{a}/{b}/{row['seed']}".encode()).hexdigest()[:8], 16)
            if pair_hash % every == 0 and self.trace_bytes < cap:
                path = self._write_trace(gid, row, move)
                compact["trace"] = str(path.relative_to(self.dir))
                traced.append(path)
            out.append(compact)
        fresh = self.state.add_games(out)
        self.state.batches_journal.append(dict(batch=spec["key"], condition=condition, a=a, b=b, job=spec.get("job"),
                                               games=len(rows), new_games=len(fresh), elapsed=elapsed,
                                               worker=worker, sha256=digest, time=stamp, unix=self.clock(),
                                               source=source))
        return fresh

    def _write_trace(self, gid, row, moves):
        name = hashlib.sha1(gid.encode()).hexdigest()[:20]
        path = self.dir / "traces" / gid.split("/", 1)[0] / f"{name}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = gzip.compress(json.dumps(dict(id=gid, game=row, moves=moves)).encode(), mtime=0)
        path.write_bytes(data)
        self.trace_bytes += len(data)
        return path

    def _compare(self, rows_a, moves_a, rows_b, moves_b):
        from tools.trainer_arena_distributed import StudyCoordinator
        strip = lambda rows: [{k: r.get(k) for k in ROW_FIELDS} for r in rows]  # noqa: E731
        return StudyCoordinator._compare(None, strip(rows_a), moves_a, strip(rows_b), moves_b)

    def _acceptable(self, comparison):
        if self.settings["fidelity"] == "strict":
            return not comparison["differing"]
        return comparison["agreement"] >= self.settings["min_agreement"]

    def _record_audit(self, body, comparison, numerics):
        stats = self.fidelity_stats.setdefault(numerics, dict(games=0, divergent_games=0, compared_decisions=0,
                                                              agreed_decisions=0))
        for key in ("games", "divergent_games", "compared_decisions", "agreed_decisions"):
            stats[key] += comparison[key]
        stats.update(agreement=stats["agreed_decisions"] / max(stats["compared_decisions"], 1),
                     divergent_game_rate=stats["divergent_games"] / max(stats["games"], 1))
        self.state.audit_journal.append(dict(body, **comparison, time=now_iso(), fidelity=self.settings["fidelity"],
                                             min_agreement=self.settings["min_agreement"],
                                             acceptable=self._acceptable(comparison), class_totals=dict(stats)))

    def _calibrate(self, spec, payload, worker):
        numerics = worker["numerics"]
        reference_rows, reference_moves = [], []
        for row in payload["rows"]:
            gid = game_id(spec["condition"], spec["a"], spec["b"], row["seed"], row["side"])
            trace = json.loads(gzip.decompress((self.dir / self.state.games[gid]["trace"]).read_bytes()))
            reference_rows.append(trace["game"])
            reference_moves.append(trace["moves"])
        comparison = self._compare(reference_rows, reference_moves, payload["rows"], payload["moves"])
        ok = self._acceptable(comparison)
        verdict = True if ok else (f"calibration failed: agreement {comparison['agreement']:.4f}, "
                                   f"{comparison['divergent_games']} of {comparison['games']} games diverge")
        self.state.record("admission", dict(numerics=numerics, verdict=verdict, batch=spec["key"],
                                            worker=worker["worker_id"]))
        self._record_audit(dict(batch=spec["key"], kind="calibration", replica=worker), comparison, numerics)
        self.log(f"pool: calibration {numerics}: {'admitted' if ok else verdict}")
        return dict(accepted=ok, calibration=verdict, agreement=comparison["agreement"])

    def _replicate(self, spec, payload, worker, digest):
        reference = self.audits.get(spec["key"])
        if reference is None:
            return dict(accepted=False, reason="no audit reference for this batch")
        comparison = self._compare(reference["rows"], reference["moves"], payload["rows"], payload["moves"])
        first, second = reference["worker"]["numerics"], worker["numerics"]
        self._record_audit(dict(batch=spec["key"], kind="replicate", reference=reference["worker"], replica=worker),
                           comparison, second)
        self._drop_audit(spec["key"], None)
        if not self._acceptable(comparison) and first != second:
            suspect = first if self._trusted(second) and not self._trusted(first) else second
            if not self._trusted(suspect):
                self.state.record("admission", dict(
                    numerics=suspect, batch=spec["key"],
                    verdict=f"replica of {spec['key']} failed: agreement {comparison['agreement']:.4f}"))
                self.log(f"pool: rejected numerics class {suspect}; its games leave the ratings")
        return dict(accepted=False, reason="replica recorded for audit", identical=not comparison["differing"],
                    agreement=comparison["agreement"], sha256=digest)

    def _drop_audit(self, key, reason):
        self.audits.pop(key, None)
        path = self.dir / "spool" / f"{_safe(key)}.json.gz"
        if path.exists():
            path.unlink()
        if reason:
            self.log(f"pool: audit {key} dropped: {reason}")

    # -- registry API --------------------------------------------------------------------
    def register(self, event):
        """Validated registry writes from the CLI (entrant, condition, condition_set, job, intention, block, unblock)."""
        kind = event.get("type")
        body = {k: v for k, v in event.items() if k not in ("type", "time")}
        by = body.pop("by", "cli")
        if kind == "conclude":
            return self.conclude(body["run"], best=body.get("best"), reason=body.get("reason", "marked done"), by=by)
        if kind not in ("entrant", "condition", "condition_set", "job", "intention", "block", "unblock", "lineage"):
            raise ValueError(f"unsupported registry event {kind}")
        if kind == "job":
            merged = {**self.state.jobs.get(body["id"], {}), **body}
            merged.setdefault("status", "active")
            body.setdefault("status", merged["status"])
            self.validate_job_seeds(merged)
        if kind == "entrant":
            body.setdefault("status", self.state.entrants.get(body["id"], {}).get("status", "active"))
        if kind == "intention":
            body.setdefault("status", self.state.intentions.get(body["id"], {}).get("status", "planned"))
        event = self.state.record(kind, body, by=by)
        self.tick()
        return dict(recorded=True, event=event)

    def conclude(self, run, *, best=None, reason="marked done", by="cli"):
        """End a run: keep its best and final snapshots active, retire the intermediates (still rated)."""
        state = self.state
        members = state.lineage_members(run)
        if not members:
            raise KeyError(f"no snapshots with lineage.run {run!r}")
        final = members[-1]
        if best is None:
            rule = state.lineages.get(run, {}).get("stop_rule") or {}
            try:
                from drmc_rl.pool.report import stop_rule
                result = stop_rule(self, run=run, condition_set=rule.get("set"),
                                   min_games=int(rule.get("min_games", 1)), patience=int(rule.get("patience", 2)),
                                   step_every=rule.get("step_every"), weighting=rule.get("weighting", "pace"))
                best = (result.get("selected") or {}).get("entrant")
            except KeyError:
                best = None
            if best is None or best not in state.entrants:
                best = final
        keep = {best, final}
        for e in members:
            if e not in keep and state.entrants[e]["status"] != "retired":
                state.record("entrant", dict(id=e, status="retired"), by=f"lineage:{run}")
        state.record("lineage", dict(run=run, status="concluded", best=best, final=final, reason=reason), by=by)
        self.log(f"pool: lineage {run} concluded ({reason}); best {best}, final {final}")
        return dict(run=run, best=best, final=final, retired=[e for e in members if e not in keep])

    def _auto_conclude(self):
        """Lineages that opted into the pool stop rule conclude when it fires."""
        from drmc_rl.pool.report import stop_rule
        for run, record in list(self.state.lineages.items()):
            rule = record.get("stop_rule")
            if record["status"] != "active" or not rule or not rule.get("auto"):
                continue
            try:
                result = stop_rule(self, run=run, condition_set=rule["set"], min_games=int(rule.get("min_games", 128)),
                                   patience=int(rule.get("patience", 2)), step_every=rule.get("step_every"),
                                   weighting=rule.get("weighting", "pace"))
            except KeyError:
                continue
            if result["fired"]:
                self.conclude(run, best=result["selected"]["entrant"], reason="pool stop rule fired", by="stop-rule")

    def import_games(self, payload):
        """Rows from pre-pool studies (``source`` = ``import:<name>``), idempotent by game id."""
        rows, out = payload["rows"], []
        traces = payload.get("traces") or {}
        for row in rows:
            if not str(row.get("source", "")).startswith("import:"):
                raise ValueError("imported rows need source import:<name>")
            if not row["a"] < row["b"]:
                raise ValueError("imported rows must be canonical (a < b)")
            gid = game_id(row["condition"], row["a"], row["b"], row["seed"], row["side"])
            row = dict(row, id=gid, numerics=row.get("numerics", "import"))
            if gid in traces and gid not in self.state.games and self.trace_bytes < self.settings["trace_cap_mb"] << 20:
                path = self._write_trace(gid, traces[gid]["game"], traces[gid]["moves"])
                row["trace"] = str(path.relative_to(self.dir))
            out.append(row)
        fresh = self.state.add_games(out)
        return dict(imported=len(fresh), duplicates=len(rows) - len(fresh))

    # -- views -------------------------------------------------------------------------
    def status(self):
        now = self.clock()
        return dict(protocol=PROTOCOL, source=self.source, capabilities=sorted(self.capabilities),
                    uptime=now - self.started, games=len(self.state.games),
                    leases=[dict(batch=l["batch"]["key"], worker=l["worker"], purpose=l["purpose"],
                                 expires_in=round(l["expires"] - now)) for l in self.leases.values()],
                    workers=list(self.workers.values()),
                    admitted={k: v for k, v in self.state.admitted.items()},
                    fidelity=dict(mode=self.settings["fidelity"], min_agreement=self.settings["min_agreement"],
                                  classes=self.fidelity_stats, pending_audits=len(self.audits)),
                    trace_mb=round(self.trace_bytes / 2 ** 20, 1))

    def study(self):
        return dict(protocol=PROTOCOL, source=self.source, capabilities=sorted(self.capabilities),
                    runtime=self.runtime)

    def report(self):
        from drmc_rl.pool.report import build_report
        return build_report(self)

    def stop_rule(self, query):
        from drmc_rl.pool.report import stop_rule
        return stop_rule(self, **query)

    def lineage_roles(self):
        """entrant -> (kind, detail, targets) for non-retired snapshots of active runs.

        ``newest`` and ``best`` (best so far by the weighted pooled rating on the primary set)
        get full new-entrant priority. Every other snapshot is ``resolving`` while its comparison
        with an adjacent snapshot (by frames) or the best is undecided, else ``maintenance``.
        """
        state = self.state
        stamp = (len(state.games), len(state.entrants), tuple(sorted(state.dirty)),
                 tuple(sorted((r, v.get("status")) for r, v in state.lineages.items())))
        if getattr(self, "_roles", None) and self._roles[0] == stamp:
            return self._roles[1]
        from drmc_rl.pool.report import primary_set
        from drmc_rl.pool.ratings import pooled_difference_se, superiority
        cset = primary_set(state)
        fits = self.fits
        keys = cset["conditions"] if cset else []
        weights = self.pace_weights(keys) if keys else []
        view = pooled(fits, keys, min_games=1, weights=weights) if keys and all(k in fits for k in keys) else {}
        s = self.settings

        def games(e):
            return sum(fits[k].ratings[e].games for k in keys if k in fits and e in fits[k].ratings)

        def resolved(e, o):
            if e not in view or o not in view:
                return False
            se = pooled_difference_se(fits, keys, e, o, weights)
            if se is None:
                return False
            los = superiority(view[e]["rating"] - view[o]["rating"], se)
            return los >= s["resolve_los"] or los <= 1 - s["resolve_los"] or 1.96 * se <= s["resolve_ci"]
        roles = {}
        for run in state.lineage_runs():
            if state.lineage_status(run) != "active":
                continue
            members = [e for e in state.lineage_members(run) if state.entrants[e]["status"] != "retired"]
            if not members:
                continue
            newest = members[-1]
            rated = [e for e in members if e in view]
            best = max(rated, key=lambda e: (view[e]["rating"], state.step_of(e))) if rated else newest
            roles[newest] = ("newest", "", [])
            roles.setdefault(best, ("best", "best so far", []))
            for i, e in enumerate(members):
                if e in roles:
                    continue
                if games(e) >= s["snapshot_game_cap"]:
                    roles[e] = ("maintenance", "game cap", [])
                    continue
                around = [members[j] for j in (i - 1, i + 1) if 0 <= j < len(members)] + [best]
                open_ = [o for o in dict.fromkeys(around) if o != e and not resolved(e, o)]
                roles[e] = ("resolving", "vs " + ", ".join(open_), open_) if open_ else ("maintenance", "resolved", [])
        self._roles = (stamp, roles)
        return roles

    def pace_weights(self, conditions, weighting="pace"):
        """Per-condition weights of a pooled view: confirmed pace weights, or all equal."""
        if weighting == "equal":
            return [1.0] * len(conditions)
        table = self.settings["pace_weights"]
        return [float(table.get(self.state.conditions[c]["spec"]["pace"], 1.0)) for c in conditions]

    def view_fits(self, view="all"):
        """Per-condition fits on all comparable games, or on one seed set's games only
        (``real_play`` = the mixture set, ``uniform``, ``reserve``)."""
        if view == "all":
            return self.all_fits()
        name = dict(real_play="mixture").get(view, view)
        cached = self._view_cache.get(view)
        if cached and cached[0] == len(self.state.games):
            return cached[1]
        seeds = frozenset(self.seed_sets()[name])
        fits = {}
        for condition in self.state.conditions:
            anchor = self.state.anchor_for(condition)
            if anchor in self.state.entrants:
                fits[condition] = fit(self.state.pair_stats(condition, (name, seeds)), anchor,
                                      anchor_rating=self.settings["anchor_rating"], prior_sd=self.settings["prior_sd"])
        self._view_cache[view] = (len(self.state.games), fits)
        return fits

    def ratings_for(self, condition_set=None, weighting="pace", view="all"):
        fits = self.view_fits(view)
        if condition_set is None:
            return fits
        cset = self.state.condition_sets[condition_set]
        return pooled(fits, cset["conditions"], min_games=1, weights=self.pace_weights(cset["conditions"], weighting))

    def export_registry(self):
        return self.state.snapshot()

    def close(self):
        pass


def _safe(key):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in key)[:180]


def _append(notes, line):
    return (notes + "\n" + line).strip()


def _source_revision():
    from tools.trainer_arena_distributed import source_revision
    return source_revision(REPO)

