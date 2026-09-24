"""Split one trainer-planning-arena study across processes and hosts.

The coordinator owns the study output (``games.jsonl``, ``moves/``,
``arena.sqlite``, ``results.json``) exactly as ``tools.trainer_planning_arena``
would write it. It leases the *same* seed-pair batches the single-host arena
would play (same ``pairs``, same sequential ``look_games`` and identity-probe
sizes) to workers over authenticated HTTP. Results are accepted per comparison
in batch order, sequential stopping is evaluated after each accepted batch, and
the journal is written in schedule order. Batches played speculatively beyond
a stopping decision are discarded. The journal therefore equals a single-host
run whenever each batch's games equal the single-host games; see
docs/ARENA_HOSTS.md for what that does and does not guarantee across devices.

    serve   coordinator for one study config
    worker  lease, play and upload batches (any host with the source + native libs)
    local   coordinator plus N local worker processes (single-host sharding)
    compare byte-compare two study outputs (journal and every move trace)

Workers must run the coordinator's source revision. Checkpoints are fetched by
SHA-256 into a local cache. ``--calibration-games`` replays already-journaled
games on each new numerics class (device/arch/torch) before it may contribute;
``--replicate-every`` re-plays sampled batches on a second worker and records
any divergence in ``distributed/audit.jsonl``.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import random
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL = "drmc-trainer-study-v1"
REPO = Path(__file__).resolve().parents[1]
CHECKPOINT_KEYS = ("checkpoint", "adapter_checkpoint")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def source_revision(root=REPO):
    """Commit plus a digest of uncommitted tracked changes to executable code."""
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                              check=False).stdout.strip()
    commit = git("rev-parse", "HEAD") or "unknown"
    diff = git("diff", "HEAD", "--", "drmc_rl", "tools", "reach_native")
    return commit + ("+dirty-" + hashlib.sha256(diff.encode()).hexdigest()[:12] if diff else "")


# ---------------------------------------------------------------------------
# Deterministic batch plan and ordered commit (no networking).


class Batch:
    __slots__ = ("key", "match", "ordinal", "jobs", "result", "status", "leases", "replicas", "audit")

    def __init__(self, match, ordinal, jobs):
        self.key = f"{match['id']}.{jobs[0][2]:05d}.{len(jobs)}"
        self.match, self.ordinal, self.jobs = match, ordinal, jobs
        self.result = None          # first accepted submission
        self.status = "open"        # open | done | accepted | journaled | discarded
        self.leases = set()
        self.replicas = []          # later submissions, compared with result
        self.audit = False


class StudyCoordinator:
    """All study state. Not thread-safe: the server calls it from one thread."""

    def __init__(self, config, *, lease_ttl=1800.0, replicate_every=0, calibration_games=0,
                 max_ahead=0, allow_source_mismatch=False, trust=(), fidelity="tolerant", min_agreement=0.99,
                 log=print):
        from drmc_rl.arena.store import ArenaStore
        from tools.trainer_arena_stopping import ComparisonStopping
        from tools.trainer_planning_arena import (
            batch_size, bind_execution_profiles, load_journal, paired_jobs, publish, register_variants,
        )
        self.fatal = None
        self.log = log
        self.publish_fn = publish
        self.config = config
        if config.get("watch", False):
            raise ValueError("distributed studies play a fixed schedule; watch mode is single-host only")
        bind_execution_profiles(config)
        self.output = Path(config["output"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.spool = self.output / "distributed" / "spool"
        self.spool.mkdir(parents=True, exist_ok=True)
        self.store = ArenaStore(config["working_db"], replay_dir=self.output / "replays")
        register_variants(config, self.store)
        self.records = self.output / "games.jsonl"
        self.results = load_journal(config, self.records)
        self.stopping = ComparisonStopping(config, self.output)
        config["_stopping"] = self.stopping.verdicts
        for match in config["schedule"]:
            self.stopping.verdict(match, self.results.get(match["id"], []))
        self.lease_ttl, self.replicate_every = float(lease_ttl), int(replicate_every)
        self.calibration_games, self.max_ahead = int(calibration_games), int(max_ahead)
        self.allow_source_mismatch = allow_source_mismatch
        self.source = source_revision()
        self.started = time.time()

        # Batches exactly as trainer_planning_arena.main slices them.
        pairs = batch_size(config)
        self.matches = [m for m in config["schedule"]]
        self.batches = {}
        self.plan = {}
        for order, match in enumerate(self.matches):
            batches = []
            if match["id"] not in self.stopping.verdicts:
                done = {row["index"] for row in self.results.get(match["id"], [])}
                jobs = [job for job in paired_jobs(config, match) if job[2] not in done]
                start = 0
                size = self.stopping.batch_games(match, self.results.get(match["id"], []), pairs)
                while start < len(jobs):
                    batch = Batch(match, len(batches), jobs[start:start + size])
                    batch.audit = bool(self.replicate_every) and (len(batches) * len(self.matches) + order) % self.replicate_every == 0
                    batches.append(batch)
                    self.batches[batch.key] = batch
                    start += size
                    size = self.stopping.steady_batch_games(match, pairs)
            self.plan[match["id"]] = batches
        self.accepted_rows = {m["id"]: [dict(r) for r in self.results.get(m["id"], [])] for m in self.matches}
        self.next_accept = {m["id"]: 0 for m in self.matches}
        self.leases = {}
        self.workers = {}
        if fidelity not in ("tolerant", "strict"):
            raise ValueError("fidelity must be tolerant or strict")
        self.fidelity, self.min_agreement = fidelity, float(min_agreement)
        self.fidelity_stats = {}
        self.trust = tuple(trust)      # numerics prefixes admitted without calibration
        self.admitted = {}           # numerics class -> calibration verdict
        self.calibration = self._calibration_jobs()

        # Checkpoints are served by content hash.
        self.artifacts = {}
        self.wire = self._wire_config()
        for path in list(self.spool.glob("*.json.gz")):
            payload = json.loads(gzip.decompress(path.read_bytes()))
            batch = self.batches.get(payload["batch"])
            if batch is not None and batch.result is None:
                batch.result, batch.status = payload, "done"
        for match in self.matches:
            self._advance(match)
        self._flush()

    # -- wire -----------------------------------------------------------------
    def _wire_config(self):
        wire = {k: copy.deepcopy(v) for k, v in self.config.items()
                if not k.startswith("_") and k not in ("output", "working_db")}
        def reference(path):
            path = str(Path(path).expanduser().resolve())
            digest = sha256_file(path)
            self.artifacts[digest] = dict(path=path, name=Path(path).name, size=os.path.getsize(path))
            return "sha256:" + digest
        wire["checkpoint"] = reference(wire["checkpoint"])
        for params in wire["variants"].values():
            for key in CHECKPOINT_KEYS:
                if key in params:
                    params[key] = reference(params[key])
        return wire

    def study(self):
        return dict(protocol=PROTOCOL, source=self.source, config=self.wire,
                    artifacts={k: dict(name=v["name"], size=v["size"], path=v["path"]) for k, v in self.artifacts.items()},
                    study_sha256=hashlib.sha256(canonical(self.wire)).hexdigest())

    def artifact_path(self, digest):
        entry = self.artifacts.get(digest)
        return None if entry is None else Path(entry["path"])

    # -- plan state -----------------------------------------------------------
    def _final(self, match):
        return match["id"] in self.stopping.verdicts or self.next_accept[match["id"]] >= len(self.plan[match["id"]])

    def complete(self):
        return all(self._final(m) and all(b.status in ("journaled", "discarded") for b in self.plan[m["id"]])
                   for m in self.matches)

    def _calibration_jobs(self):
        """Already-journaled whole seed pairs, replayed by each new numerics class."""
        if not self.calibration_games:
            return None
        for match in self.matches:
            rows = self.results.get(match["id"], [])
            by_seed = {}
            for row in rows:
                by_seed.setdefault(row["seed"], []).append(row)
            chosen = [r for seed_rows in by_seed.values() if len(seed_rows) == 2 for r in sorted(seed_rows, key=lambda r: r["index"])]
            chosen = chosen[:self.calibration_games - self.calibration_games % 2]
            if chosen:
                return dict(match=match["id"], jobs=[(r["seed"], r["side"], r["index"]) for r in chosen])
        return None

    def _expire(self, now):
        for lease_id, lease in list(self.leases.items()):
            if lease["expires"] < now:
                self.leases.pop(lease_id)
                batch = self.batches.get(lease["batch"])
                if batch is not None:
                    batch.leases.discard(lease_id)
                    self.log(f"coordinator: lease {lease_id[:8]} for {lease['batch']} by {lease['worker']} expired")

    def _candidates(self):
        """Open batches, breadth-first across comparisons in schedule order."""
        depth = max((len(v) for v in self.plan.values()), default=0)
        for ordinal in range(depth):
            for match in self.matches:
                batches = self.plan[match["id"]]
                if ordinal >= len(batches) or match["id"] in self.stopping.verdicts:
                    continue
                if self.max_ahead and ordinal - self.next_accept[match["id"]] >= self.max_ahead:
                    continue
                yield batches[ordinal]

    def lease(self, worker):
        if self.fatal:
            return dict(status="wait", reason="coordinator stopped after an internal error", retry=60)
        now = time.time()
        self._expire(now)
        identity = self._register(worker)
        numerics = identity["numerics"]
        if self.calibration_games:
            if numerics.startswith(self.trust) and self.trust:
                self.admitted.setdefault(numerics, True)
            state = self.admitted.get(numerics)
            if state is None and self.calibration is None:
                self.calibration = self._calibration_jobs()
            if state is None and self.calibration is not None:
                if not any(l["purpose"] == "calibrate" and l["numerics"] == numerics for l in self.leases.values()):
                    return self._issue(identity, None, "calibrate", now)
                return dict(status="wait", reason="calibration of this numerics class is in progress", retry=30)
            if state is None:
                return dict(status="wait", reason="calibration needs journaled games from an admitted host", retry=60)
            if state is not True:
                return dict(status="rejected", reason=f"numerics class {numerics} failed calibration: {state}")
        if self.complete():
            return dict(status="done")
        # Audits go first so a sampled batch is replayed while the study is still running.
        for batch in self.batches.values():
            if (not batch.audit or batch.result is None or batch.replicas or batch.leases
                    or batch.status == "discarded"
                    or batch.result["worker"]["worker_id"] == identity["worker_id"]):
                continue
            if batch.result["worker"]["numerics"] == numerics and self._other_class_available(batch):
                continue
            return self._issue(identity, batch, "replicate", now)
        for batch in self._candidates():
            if batch.status == "open" and not batch.leases:
                return self._issue(identity, batch, "play", now)
        return dict(status="wait", reason="every open batch is leased", retry=15)

    def _other_class_available(self, batch):
        own = batch.result["worker"]["numerics"]
        return any(w["numerics"] != own and time.time() - w["seen"] < 3 * self.lease_ttl
                   for w in self.workers.values())

    def _register(self, worker):
        if worker.get("protocol") != PROTOCOL:
            raise PermissionError("protocol mismatch")
        if worker.get("source") != self.source and not self.allow_source_mismatch:
            raise PermissionError(f"worker source {worker.get('source')} != coordinator {self.source}")
        identity = {k: worker[k] for k in ("worker_id", "host", "numerics", "device", "threads",
                                           "planner_workers", "source", "native") if k in worker}
        identity["seen"] = time.time()
        self.workers[identity["worker_id"]] = identity
        return identity

    def _issue(self, identity, batch, purpose, now):
        lease_id, claim = secrets.token_hex(12), secrets.token_hex(16)
        if purpose == "calibrate":
            match_id, jobs, key = self.calibration["match"], self.calibration["jobs"], "calibration"
        else:
            match_id, jobs, key = batch.match["id"], batch.jobs, batch.key
            batch.leases.add(lease_id)
        self.leases[lease_id] = dict(batch=key, claim=claim, worker=identity["worker_id"], purpose=purpose,
                                     numerics=identity["numerics"], expires=now + self.lease_ttl, jobs=jobs,
                                     match=match_id)
        self.log(f"coordinator: {purpose} {key} ({len(jobs)} games) -> {identity['worker_id']}")
        return dict(status="lease", lease_id=lease_id, claim_token=claim, batch=key, purpose=purpose,
                    match=match_id, jobs=[list(j) for j in jobs], ttl_seconds=self.lease_ttl)

    def renew(self, lease_id, claim):
        lease = self.leases.get(lease_id)
        if lease is None or not secrets.compare_digest(lease["claim"], claim):
            return dict(renewed=False)
        lease["expires"] = time.time() + self.lease_ttl
        return dict(renewed=True, ttl_seconds=self.lease_ttl)

    def release(self, lease_id, claim):
        """A worker gives a lease back unfinished so the batch is re-leased at once."""
        lease = self.leases.get(lease_id)
        if lease is None or not secrets.compare_digest(lease["claim"], claim):
            return dict(released=False)
        self._drop(lease_id, "released by its worker")
        return dict(released=True)

    def release_batch(self, key):
        """Operator action: drop every live lease on one batch (e.g. a hung or killed worker)."""
        dropped = [lease_id for lease_id, lease in self.leases.items() if lease["batch"] == key]
        for lease_id in dropped:
            self._drop(lease_id, "released by operator")
        return dict(released=len(dropped), batch=key)

    def _drop(self, lease_id, reason):
        lease = self.leases.pop(lease_id)
        batch = self.batches.get(lease["batch"])
        if batch is not None:
            batch.leases.discard(lease_id)
        self.log(f"coordinator: lease {lease_id[:8]} for {lease['batch']} by {lease['worker']} {reason}")

    def submit(self, lease_id, payload):
        lease = self.leases.pop(lease_id, None)
        if lease is not None and not secrets.compare_digest(lease["claim"], str(payload.get("claim_token", ""))):
            raise PermissionError("claim token mismatch")
        key = payload["batch"]
        jobs = [tuple(j) for j in (lease["jobs"] if lease else self.batches[key].jobs)]
        rows = payload["rows"]
        if len(rows) != len(jobs) or any((r["seed"], r["side"], r["index"]) != tuple(j) for r, j in zip(rows, jobs)):
            raise ValueError("submission does not match the leased games")
        if len(payload["moves"]) != len(rows) or len(payload.get("replays", [])) != len(rows):
            raise ValueError("submission needs one move journal and replay per game")
        elapsed = payload.get("elapsed")
        if not isinstance(elapsed, (int, float)) or not 0 < elapsed < float("inf"):
            raise ValueError("elapsed must be a positive number of seconds")
        worker = payload.get("worker", {})
        if not {"worker_id", "numerics", "device", "threads"} <= set(worker) or int(worker["threads"]) < 1:
            raise ValueError("submission needs the worker identity it leased with")
        digest = hashlib.sha256(canonical(dict(rows=rows, moves=payload["moves"]))).hexdigest()
        payload["sha256"] = digest
        if key == "calibration":
            return self._calibrate(payload)
        batch = self.batches.get(key)
        if batch is None:
            raise ValueError(f"unknown batch {key}")
        batch.leases.discard(lease_id)
        if batch.result is None:
            if batch.status == "discarded":
                return dict(accepted=False, reason="batch discarded after a stopping decision")
            spool = self.spool / f"{key}.json.gz"
            with tempfile.NamedTemporaryFile(dir=self.spool, delete=False) as stream:
                stream.write(gzip.compress(canonical(payload), mtime=0))
            os.replace(stream.name, spool)
            batch.result, batch.status = payload, "done"
            self._advance(batch.match)
            self._flush()
            return dict(accepted=True, sha256=digest)
        replica = lease is not None and lease["purpose"] == "replicate"
        if batch.result["sha256"] == digest and not replica:
            return dict(accepted=True, duplicate=True, sha256=digest)
        batch.replicas.append(dict(worker=payload["worker"], sha256=digest))
        self._audit(batch, payload)
        return dict(accepted=False, reason="replica recorded for audit", sha256=digest,
                    identical=batch.result["sha256"] == digest)

    def _compare(self, rows_a, moves_a, rows_b, moves_b):
        """Per-game divergence plus decision agreement up to each first divergence."""
        differing, agreed, compared = [], 0, 0
        for ra, ma, rb, mb in zip(rows_a, moves_a, rows_b, moves_b):
            if canonical(ra) == canonical(rb) and canonical(ma) == canonical(mb):
                agreed += len(ma)
                compared += len(ma)
                continue
            first = next((i for i, (x, y) in enumerate(zip(ma, mb)) if canonical(x) != canonical(y)),
                         min(len(ma), len(mb)))
            # Decisions before the first divergence agree and the divergent one
            # does not; after it the two are different games and are not compared.
            diverged = first < min(len(ma), len(mb))
            agreed += first
            compared += first + int(diverged)
            differing.append(dict(index=ra["index"], seed=ra["seed"], first_move=first,
                                  moves=[len(ma), len(mb)], scores=[ra["score"], rb["score"]]))
        return dict(differing=differing, agreed_decisions=agreed, compared_decisions=compared,
                    agreement=agreed / compared if compared else 1.0, divergent_games=len(differing),
                    games=len(rows_a))

    def _acceptable(self, comparison):
        if self.fidelity == "strict":
            return not comparison["differing"]
        return comparison["agreement"] >= self.min_agreement

    def _trusted(self, numerics):
        return bool(self.trust) and numerics.startswith(self.trust)

    def _record(self, record, comparison, numerics):
        stats = self.fidelity_stats.setdefault(numerics, dict(games=0, divergent_games=0, compared_decisions=0,
                                                              agreed_decisions=0))
        for key in list(stats):
            stats[key] += comparison[key]
        stats.update(agreement=stats["agreed_decisions"] / max(stats["compared_decisions"], 1),
                     divergent_game_rate=stats["divergent_games"] / max(stats["games"], 1))
        record.update(comparison, equal=not comparison["differing"], fidelity=self.fidelity,
                      min_agreement=self.min_agreement, acceptable=self._acceptable(comparison),
                      class_totals=dict(stats))
        with (self.output / "distributed" / "audit.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def _audit(self, batch, payload):
        comparison = self._compare(batch.result["rows"], batch.result["moves"], payload["rows"], payload["moves"])
        reference, replica = batch.result["worker"]["numerics"], payload["worker"]["numerics"]
        self._record(dict(batch=batch.key, time=time.time(), reference=batch.result["worker"],
                          replica=payload["worker"]), comparison, replica)
        self.log(f"coordinator: audit {batch.key}: {comparison['divergent_games']}/{comparison['games']} games "
                 f"diverge, decision agreement {comparison['agreement']:.4f}")
        if not self._acceptable(comparison) and reference != replica:
            # Blame whichever class is not the trusted reference.
            suspect = reference if self._trusted(replica) and not self._trusted(reference) else replica
            self.admitted[suspect] = f"replica of {batch.key} failed: agreement {comparison['agreement']:.4f}, " \
                                     f"{comparison['divergent_games']} divergent games ({self.fidelity})"
            self.log(f"coordinator: rejecting numerics class {suspect}")

    def _calibrate(self, payload):
        numerics = payload["worker"]["numerics"]
        match_id = self.calibration["match"]
        by_index = {r["index"]: r for r in self.results[match_id]}
        from drmc_rl.arena.identity import load_journals
        reference = [by_index[r["index"]] for r in payload["rows"]]
        journals = load_journals(self.output, match_id, reference)
        strip = ("comparison", "level", "pace", "execution_key")
        reference_rows = [{k: v for k, v in r.items() if k not in strip} for r in reference]
        comparison = self._compare(reference_rows, [journals[r["index"]] for r in reference],
                                   payload["rows"], payload["moves"])
        ok = self._acceptable(comparison)
        self.admitted[numerics] = True if ok else (
            f"calibration failed: agreement {comparison['agreement']:.4f}, "
            f"{comparison['divergent_games']} of {len(reference)} games diverge ({self.fidelity})")
        self._record(dict(batch="calibration", time=time.time(), replica=payload["worker"], numerics=numerics),
                     comparison, numerics)
        self.log(f"coordinator: calibration {numerics}: {'admitted' if ok else self.admitted[numerics]} "
                 f"({comparison['divergent_games']}/{comparison['games']} games diverge, "
                 f"agreement {comparison['agreement']:.4f})")
        return dict(accepted=ok, calibration=self.admitted[numerics], agreement=comparison["agreement"])

    def _advance(self, match):
        """Accept finished batches in order; evaluate stopping after each."""
        from tools.trainer_planning_arena import write_trace
        id = match["id"]
        batches = self.plan[id]
        while (id not in self.stopping.verdicts and self.next_accept[id] < len(batches)
               and batches[self.next_accept[id]].result is not None):
            batch = batches[self.next_accept[id]]
            for row, moves in zip(batch.result["rows"], batch.result["moves"]):
                row = dict(row, comparison=id, level=match["level"], pace=match.get("pace", "frame_perfect"),
                           execution_key=match["execution_key"])
                write_trace(self.output, match, row, moves)
                self.accepted_rows[id].append(row)
            batch.status = "accepted"
            self.next_accept[id] += 1
            if self.stopping.verdict(match, self.accepted_rows[id]) is not None:
                for later in batches[self.next_accept[id]:]:
                    if later.status != "journaled":
                        later.status = "discarded"
                self.log(f"coordinator: {id} decided after {len(self.accepted_rows[id])} games: "
                         f"{self.stopping.verdicts[id]['decision']}")

    def _flush(self):
        """Journal accepted batches in schedule order, as the single-host arena does."""
        from tools.trainer_planning_arena import commit_batch
        changed = False
        for match in self.matches:
            for batch in self.plan[match["id"]]:
                if batch.status != "accepted":
                    continue
                result = batch.result
                played = [(dict(r), m, p) for r, m, p in zip(result["rows"], result["moves"], result["replays"])]
                commit_batch(self.config, match, played, result["elapsed"], output=self.output, store=self.store,
                             records=self.records, results=self.results, worker=result["worker"])
                batch.status = "journaled"
                changed = True
                with (self.output / "distributed" / "batches.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(batch=batch.key, sha256=result["sha256"], elapsed=result["elapsed"],
                                                 worker=result["worker"], journaled=time.time())) + "\n")
                rows = self.results[match["id"]]
                print(json.dumps(dict(comparison=match["id"], games=len(rows), target=match["games"],
                                      batch=batch.key, worker=result["worker"]["worker_id"],
                                      batch_seconds=round(result["elapsed"], 2))), flush=True)
            if not self._final(match):
                break
        if changed or self.complete():
            self.config.update(_worker_status="Complete" if self.complete() else "Playing (distributed)",
                               _current_match=None)
            self.publish_fn(self.config, self.results, self.output, self.store)

    def status(self):
        now = time.time()
        return dict(protocol=PROTOCOL, complete=self.complete(), source=self.source,
                    elapsed=now - self.started,
                    journaled=sum(len(v) for v in self.results.values()),
                    batches={s: sum(b.status == s for b in self.batches.values())
                             for s in ("open", "done", "accepted", "journaled", "discarded")},
                    leases=[dict(batch=l["batch"], worker=l["worker"], purpose=l["purpose"],
                                 expires_in=round(l["expires"] - now)) for l in self.leases.values()],
                    workers=list(self.workers.values()), admitted=self.admitted,
                    fidelity=dict(mode=self.fidelity, min_agreement=self.min_agreement, classes=self.fidelity_stats),
                    verdicts=self.stopping.verdicts)

    def close(self):
        self.store.close()


class Handler(BaseHTTPRequestHandler):
    """Requests are threaded; coordinator state (and its SQLite handle) lives on one thread."""
    coordinator: StudyCoordinator
    state: ThreadPoolExecutor
    token: str

    def _call(self, function, *args):
        return self.state.submit(function, *args).result()

    def _authorized(self):
        return secrets.compare_digest(self.headers.get("Authorization", ""), f"Bearer {self.token}")

    def _json(self, status, value):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        data = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
        return json.loads(data)

    def do_GET(self):  # noqa: N802
        if not self._authorized():
            return self._json(401, dict(error="unauthorized"))
        path = self.path.split("?", 1)[0]
        c = self.coordinator
        if path == "/api/v1/study":
            return self._json(200, self._call(c.study))
        if path == "/api/v1/study/status":
            return self._json(200, self._call(c.status))
        prefix = "/api/v1/checkpoints/"
        if path.startswith(prefix):
            source = self._call(c.artifact_path, urllib.parse.unquote(path[len(prefix):]))
            if source is None:
                return self._json(404, dict(error="unknown artifact"))
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(source.stat().st_size))
            self.end_headers()
            with source.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    self.wfile.write(block)
            return None
        return self._json(404, dict(error="not found"))

    def do_POST(self):  # noqa: N802
        if not self._authorized():
            return self._json(401, dict(error="unauthorized"))
        path = self.path.split("?", 1)[0]
        c = self.coordinator
        try:
            request = self._body()
            if path == "/api/v1/study/leases":
                return self._json(200, self._call(c.lease, request))
            if path == "/api/v1/study/release":
                return self._json(200, self._call(c.release_batch, str(request["batch"])))
            parts = path.split("/")
            if len(parts) == 7 and parts[:5] == ["", "api", "v1", "study", "leases"]:
                lease_id = urllib.parse.unquote(parts[5])
                if parts[6] == "renew":
                    return self._json(200, self._call(c.renew, lease_id, str(request.get("claim_token", ""))))
                if parts[6] == "release":
                    return self._json(200, self._call(c.release, lease_id, str(request.get("claim_token", ""))))
                if parts[6] == "results":
                    return self._json(200, self._call(c.submit, lease_id, request))
            return self._json(404, dict(error="not found"))
        except PermissionError as error:
            return self._json(409, dict(error=str(error)))
        except (KeyError, TypeError, ValueError) as error:
            return self._json(400, dict(error=f"{type(error).__name__}: {error}"))
        except Exception as error:
            # A failed journal write leaves no trustworthy state: stop the study.
            traceback.print_exc()
            c.fatal = f"{type(error).__name__}: {error}"
            return self._json(500, dict(error=c.fatal))

    def log_message(self, fmt, *args):
        pass


class StudyServer(ThreadingHTTPServer):
    request_queue_size = 128   # a fleet starting at once must not overflow the accept backlog
    daemon_threads = True


def start_server(coordinator, state, host, port, token):
    handler = type("StudyHandler", (Handler,), dict(coordinator=coordinator, state=state, token=token))
    server = StudyServer((host, port), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name="study-coordinator", daemon=True)
    thread.start()
    return server


# ---------------------------------------------------------------------------
# Worker.


class Abandoned(Exception):
    """Raised inside a batch when the worker was asked to stop."""


class CoordinatorError(RuntimeError):
    def __init__(self, code, detail):
        super().__init__(f"coordinator HTTP {code}: {detail}")
        self.code = code


TRANSIENT = (urllib.error.URLError, OSError, TimeoutError, http.client.HTTPException)


class StudyClient:
    def __init__(self, base_url, token, timeout=120.0):
        self.base, self.token, self.timeout = base_url.rstrip("/"), token.strip(), timeout
        if not self.token:
            raise ValueError("empty worker token")

    def request(self, method, path, payload=None, *, compress=False):
        body, headers = None, {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
            if compress:
                body = gzip.compress(body, compresslevel=6)
                headers["Content-Encoding"] = "gzip"
        request = urllib.request.Request(self.base + path, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise CoordinatorError(error.code, error.read().decode(errors="replace")) from error

    def retrying(self, method, path, payload=None, *, compress=False, patience=900.0):
        """Ride out coordinator restarts and brief network loss."""
        return self._retry(lambda: self.request(method, path, payload, compress=compress), patience)

    @staticmethod
    def _retry(call, patience):
        # Every endpoint is idempotent (reads, leases, renewals, hash-checked uploads).
        deadline, delay = time.monotonic() + patience, 1.0
        while True:
            try:
                return call()
            except (CoordinatorError, *TRANSIENT) as error:
                if isinstance(error, CoordinatorError) and error.code < 500 or time.monotonic() > deadline:
                    raise
                wait = delay * (0.5 + random.random())
                print(f"worker: coordinator request failed ({error}); retrying in {wait:.1f}s", flush=True)
                time.sleep(wait)
                delay = min(60.0, delay * 2)

    def download(self, digest, name, cache, patience=3600.0):
        return self._retry(lambda: self._download(digest, name, cache), patience)

    def _download(self, digest, name, cache):
        cache = Path(cache).expanduser()
        target = cache / f"{digest}-{name}"
        if target.is_file() and sha256_file(target) == digest:
            return target
        cache.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(f"{self.base}/api/v1/checkpoints/{digest}",
                                         headers={"Authorization": f"Bearer {self.token}"})
        fd, temporary = tempfile.mkstemp(dir=cache, prefix=f".{digest}.")
        try:
            sha = hashlib.sha256()
            with os.fdopen(fd, "wb") as stream, urllib.request.urlopen(request, timeout=self.timeout) as response:
                for block in iter(lambda: response.read(1 << 20), b""):
                    sha.update(block)
                    stream.write(block)
            if sha.hexdigest() != digest:
                raise OSError(f"artifact {name} hash mismatch (truncated or corrupted transfer)")
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return target


def cpu_name():
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, check=False).stdout.strip().replace(" ", "_") or platform.machine()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip().replace(" ", "_")
    except OSError:
        pass
    return platform.machine()


def numerics_class(device):
    import torch
    kind = str(device).split(":")[0]
    if kind == "cuda":
        index = torch.device(device).index or 0
        name = torch.cuda.get_device_name(index).replace(" ", "_")
        capability = "sm_%d%d" % torch.cuda.get_device_capability(index)
        detail = f"{name}/{capability}/cuda{torch.version.cuda}"
    else:
        detail = cpu_name()
    return f"{kind}/{platform.system().lower()}-{platform.machine()}/{detail}/torch{torch.__version__}"


def localize(study, args):
    """The coordinator's config with this host's checkpoints, libraries and device."""
    config = copy.deepcopy(study["config"])
    paths = {}
    for digest, entry in study["artifacts"].items():
        shared = Path(entry["path"])
        if args.shared_artifacts and shared.is_file() and sha256_file(shared) == digest:
            paths["sha256:" + digest] = str(shared)  # same filesystem as the coordinator
        else:
            paths["sha256:" + digest] = str(args.client.download(digest, entry["name"], args.cache))
    config["checkpoint"] = paths[config["checkpoint"]]
    for params in config["variants"].values():
        for key in CHECKPOINT_KEYS:
            if key in params:
                params[key] = paths[params[key]]
    native = args.native_library or os.environ.get("DRMC_POOL_LIB") or config.get("native_library")
    if not native or not Path(native).is_file():
        raise FileNotFoundError("pass --native-library (this host's libdrmario_pool build)")
    config["native_library"] = str(Path(native).resolve())
    reach = args.reach_library or os.environ.get("DRMARIO_REACH_LIB") or config.get("reach_library")
    if reach:
        if not Path(reach).is_file():
            raise FileNotFoundError(f"reach library {reach} is missing")
        os.environ["DRMARIO_REACH_LIB"] = str(Path(reach).resolve())
    for key in ("device", "threads", "planner_workers"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    work = Path(args.cache).expanduser() / "work" / args.worker_id
    config.update(output=str(work), working_db=str(work / "unused.sqlite"))
    return config


def run_worker(args):
    from tools.trainer_planning_arena import ArenaRuntime, bind_execution_profiles
    token = (args.token or Path(args.token_file).expanduser().read_text()).strip()
    args.worker_id = args.worker_id or f"{socket.gethostname()}-{args.device or 'cfg'}-{os.getpid()}"
    args.client = client = StudyClient(args.coordinator, token)
    study = client.retrying("GET", "/api/v1/study")
    if study["protocol"] != PROTOCOL:
        raise RuntimeError(f"coordinator protocol {study['protocol']} != {PROTOCOL}")
    source = source_revision()
    if source != study["source"] and not args.allow_source_mismatch:
        raise RuntimeError(f"worker source {source} != coordinator {study['source']}; check out the same commit")
    config = localize(study, args)
    expected = {m["id"]: m["execution_key"] for m in config["schedule"]}
    bind_execution_profiles(config)
    if any(expected[m["id"]] != m["execution_key"] for m in config["schedule"]):
        raise RuntimeError("this host's motor limits differ from the coordinator's")
    matches = {m["id"]: m for m in config["schedule"]}
    runtime = ArenaRuntime(config)
    identity = dict(protocol=PROTOCOL, worker_id=args.worker_id, host=socket.gethostname(),
                    device=config.get("device", "cuda"), threads=config.get("threads", 1),
                    planner_workers=config.get("planner_workers"), source=source,
                    numerics=numerics_class(config.get("device", "cuda")),
                    native=dict(pool=sha256_file(config["native_library"]),
                                reach=sha256_file(os.environ["DRMARIO_REACH_LIB"])
                                if os.environ.get("DRMARIO_REACH_LIB") else None))
    print(json.dumps(dict(worker=identity, study=study["study_sha256"])), flush=True)
    stopped = threading.Event()

    def interrupt(*_):
        if stopped.is_set():
            print("worker: second signal, exiting now", flush=True)
            os._exit(130)
        print("worker: stopping; the current batch is abandoned and its lease released "
              "(signal again to exit immediately)", flush=True)
        stopped.set()

    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, interrupt)

    def activity(_progress):
        if stopped.is_set():
            raise Abandoned()
    played = 0
    try:
        while not stopped.is_set():
            lease = client.retrying("POST", "/api/v1/study/leases", identity)
            if lease["status"] in ("done", "rejected"):
                print(json.dumps(dict(worker=args.worker_id, status=lease["status"], reason=lease.get("reason"),
                                      batches=played)), flush=True)
                if lease["status"] == "rejected":
                    sys.exit(3)
                return
            if lease["status"] != "lease":
                stopped.wait(min(args.poll, lease.get("retry", args.poll)))
                continue
            renewal_stop = threading.Event()

            def renew():
                while not renewal_stop.wait(max(5.0, lease["ttl_seconds"] / 3)):
                    try:
                        client.retrying("POST", f"/api/v1/study/leases/{lease['lease_id']}/renew",
                                        dict(claim_token=lease["claim_token"]), patience=lease["ttl_seconds"] / 3)
                    except Exception as error:
                        print(f"worker: renewal failed: {error}", flush=True)
            renewer = threading.Thread(target=renew, daemon=True)
            renewer.start()
            try:
                batch, elapsed = runtime.play(matches[lease["match"]], [tuple(j) for j in lease["jobs"]],
                                              activity=activity)
            except Abandoned:
                try:
                    client.request("POST", f"/api/v1/study/leases/{lease['lease_id']}/release",
                                   dict(claim_token=lease["claim_token"]))
                    print(f"worker: released {lease['batch']}", flush=True)
                except Exception as error:
                    print(f"worker: release failed ({error}); the lease will expire", flush=True)
                return
            finally:
                renewal_stop.set()
            submission = dict(claim_token=lease["claim_token"], batch=lease["batch"], elapsed=elapsed,
                              worker=identity, rows=[b[0] for b in batch], moves=[b[1] for b in batch],
                              replays=[b[2] for b in batch])
            # Canonical JSON round trip: the coordinator journals exactly these bytes.
            submission = json.loads(json.dumps(submission))
            # A lost reply is safe to resend: identical uploads are idempotent.
            reply = client.retrying("POST", f"/api/v1/study/leases/{lease['lease_id']}/results",
                                    submission, compress=True, patience=3600.0)
            played += 1
            games = len(batch)
            print(json.dumps(dict(worker=args.worker_id, batch=lease["batch"], purpose=lease["purpose"],
                                  games=games, seconds=round(elapsed, 2),
                                  games_per_hour=round(games / max(elapsed, 1e-9) * 3600), reply=reply)), flush=True)
            if args.max_batches and played >= args.max_batches:
                return
    finally:
        runtime.close()


def add_worker_arguments(parser):
    parser.add_argument("--device", help="override the study device (cpu, mps, cuda, cuda:1)")
    parser.add_argument("--threads", type=int, help="torch intra-op threads for this worker")
    parser.add_argument("--planner-workers", dest="planner_workers", type=int)
    parser.add_argument("--native-library", help="this host's libdrmario_pool build")
    parser.add_argument("--reach-library", help="this host's libdrm_reach_full build")
    parser.add_argument("--cache", default="~/.cache/drmc-rl/study-worker")
    parser.add_argument("--poll", type=float, default=20.0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--shared-artifacts", action="store_true",
                        help="read checkpoints from the coordinator's paths when they exist here with the same hash")


def serve(args, *, token=None, on_ready=None, stopped=None):
    config = json.loads(Path(args.config).read_text())
    token = token or (args.token or Path(args.token_file).expanduser().read_text()).strip()
    state = ThreadPoolExecutor(max_workers=1, thread_name_prefix="study-state")
    coordinator = state.submit(lambda: StudyCoordinator(
        config, lease_ttl=args.lease_ttl, replicate_every=args.replicate_every,
        calibration_games=args.calibration_games, max_ahead=args.max_ahead,
        allow_source_mismatch=args.allow_source_mismatch, trust=args.trust,
        fidelity=args.fidelity, min_agreement=args.min_agreement)).result()
    server = start_server(coordinator, state, args.host, args.port, token)
    print(json.dumps(dict(coordinator=f"http://{args.host}:{server.server_address[1]}",
                          batches=len(coordinator.batches), source=coordinator.source)), flush=True)
    if on_ready is not None:
        on_ready(server.server_address[1])
    stopped = stopped or threading.Event()
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stopped.set())
    try:
        while not stopped.wait(1):
            if coordinator.fatal:
                raise RuntimeError(f"study stopped after an internal error: {coordinator.fatal}")
            done = state.submit(lambda: (coordinator._expire(time.time()), coordinator.complete())[1]).result()
            if done and args.exit_when_done:
                stopped.wait(args.linger)
                break
    finally:
        server.shutdown()
        state.submit(coordinator.close).result()
        state.shutdown()
    return coordinator


def run_local(args):
    """Coordinator on loopback plus N worker processes of this checkout."""
    token = secrets.token_hex(24)
    ready, finished = threading.Event(), threading.Event()
    port = {}
    # Workers leave once the study is complete; the coordinator outlives them.
    args.host, args.port, args.exit_when_done = "127.0.0.1", 0, False
    thread = threading.Thread(target=serve, args=(args,), kwargs=dict(
        token=token, stopped=finished,
        on_ready=lambda p: (port.setdefault("port", p), ready.set())), daemon=True)
    thread.start()
    if not ready.wait(600):
        raise RuntimeError("coordinator did not start")
    procs = []
    env = dict(os.environ, DRMC_STUDY_TOKEN=token)
    for i in range(args.workers):
        command = [sys.executable, "-m", "tools.trainer_arena_distributed", "worker",
                   "--coordinator", f"http://127.0.0.1:{port['port']}", "--token-env", "DRMC_STUDY_TOKEN",
                   "--worker-id", f"{socket.gethostname()}-local{i}", "--cache", args.cache,
                   "--shared-artifacts"]
        for key in ("device", "threads", "planner_workers", "native_library", "reach_library"):
            value = getattr(args, key)
            if value is not None:
                command += [f"--{key.replace('_', '-')}", str(value)]
        if args.allow_source_mismatch:
            command.append("--allow-source-mismatch")
        procs.append(subprocess.Popen(command, cwd=REPO, env=env))
    try:
        codes = [p.wait() for p in procs]
        finished.set()
        thread.join()
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
    if any(codes):
        raise SystemExit(f"local workers exited with {codes}")


def compare_outputs(a, b):
    """Byte-compare two study outputs: journal and every decompressed move trace."""
    a, b = Path(a), Path(b)
    report = dict(journal_identical=(a / "games.jsonl").read_bytes() == (b / "games.jsonl").read_bytes())
    names_a = {p.name for p in (a / "moves").glob("*.json.gz")}
    names_b = {p.name for p in (b / "moves").glob("*.json.gz")}
    differing = [n for n in sorted(names_a & names_b)
                 if gzip.decompress((a / "moves" / n).read_bytes()) != gzip.decompress((b / "moves" / n).read_bytes())]
    report.update(games=len((a / "games.jsonl").read_text().splitlines()), traces=len(names_a),
                  only_in_a=sorted(names_a - names_b), only_in_b=sorted(names_b - names_a),
                  differing_traces=differing)
    report["identical"] = report["journal_identical"] and not differing and names_a == names_b
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def coordinator_arguments(p):
        p.add_argument("--config", required=True, type=Path)
        p.add_argument("--lease-ttl", type=float, default=1800.0)
        p.add_argument("--replicate-every", type=int, default=16,
                       help="replay every Nth planned batch on a second worker, preferring another numerics "
                            "class (0 disables)")
        p.add_argument("--fidelity", choices=("tolerant", "strict"), default="tolerant",
                       help="tolerant: admit a class whose decisions agree >= --min-agreement up to each "
                            "first divergence; strict: require byte-identical games (confirmation runs)")
        p.add_argument("--min-agreement", type=float, default=0.99)
        p.add_argument("--calibration-games", type=int, default=0,
                       help="journaled games each new numerics class must reproduce before contributing")
        p.add_argument("--max-ahead", type=int, default=0,
                       help="cap on leased batches beyond a comparison's accepted prefix (0: no cap)")
        p.add_argument("--trust", action="append", default=[],
                       help="numerics-class prefix admitted without calibration (repeatable), e.g. mps/")
        p.add_argument("--allow-source-mismatch", action="store_true")

    s = commands.add_parser("serve", help="run the study coordinator")
    coordinator_arguments(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8099)
    s.add_argument("--token-file", default="~/.config/drmc-rl/study-worker.token")
    s.add_argument("--token")
    s.add_argument("--exit-when-done", action="store_true")
    s.add_argument("--linger", type=float, default=60.0)

    w = commands.add_parser("worker", help="lease and play batches")
    w.add_argument("--coordinator", required=True)
    w.add_argument("--token-file", default="~/.config/drmc-rl/study-worker.token")
    w.add_argument("--token-env")
    w.add_argument("--worker-id")
    w.add_argument("--allow-source-mismatch", action="store_true")
    add_worker_arguments(w)

    l = commands.add_parser("local", help="coordinator plus N local worker processes")
    coordinator_arguments(l)
    l.add_argument("--workers", type=int, default=2)
    add_worker_arguments(l)

    c = commands.add_parser("compare", help="byte-compare two study outputs")
    c.add_argument("a")
    c.add_argument("b")

    args = parser.parse_args(argv)
    if args.command == "serve":
        serve(args)
    elif args.command == "worker":
        args.token = os.environ.get(args.token_env) if args.token_env else None
        run_worker(args)
    elif args.command == "local":
        run_local(args)
    else:
        report = compare_outputs(args.a, args.b)
        print(json.dumps(report, indent=1))
        sys.exit(0 if report["identical"] else 1)


if __name__ == "__main__":
    main()
