"""Pool worker: lease a batch, play it with the arena runtime, upload rows and move journals.

The games are played by ``tools.trainer_planning_arena.ArenaRuntime.play`` —
the same call the single-host arena and distributed study workers make — so a
pool game is an arena game. One runtime per (backend, anchor checkpoint) is
kept; entrant policies are loaded once and kept in a small LRU.

Host etiquette: ``--host-budget`` caps arena worker processes on this host
(pool workers yield to other distributed-study workers, e.g. a stop-rule
panel), ``--min-free-gb`` pauses when the disk is tight, and ``--cache-gb``
bounds downloaded checkpoints. SIGTERM/SIGINT abandons the current batch and
releases its lease within about five seconds; a second signal exits at once.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

from drmc_rl.pool.conditions import runtime_capabilities
from drmc_rl.pool.coordinator import PROTOCOL, REPO

FOREIGN_WORKERS = r"tools\.trainer_arena_distributed (worker|local)|tools\.trainer_planning_arena "


class Abandoned(Exception):
    pass


def foreign_arena_processes():
    try:
        out = subprocess.run(["pgrep", "-f", FOREIGN_WORKERS], capture_output=True, text=True, check=False).stdout
    except OSError:
        return 0
    return len([line for line in out.split() if line.strip() and int(line) != os.getpid()])


class LocalArtifacts:
    """Resolve checkpoints by hash: verified local paths, --artifact-dir files, then the download cache."""

    def __init__(self, client, cache, dirs=(), cache_gb=4.0):
        self.client, self.cache, self.cache_gb = client, Path(cache).expanduser(), cache_gb
        self.cache.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache / "index.json"
        try:
            self.index = json.loads(self.index_path.read_text())
        except (OSError, ValueError):
            self.index = {}
        self.dirs = [Path(d).expanduser() for d in dirs]

    def _sha(self, path):
        from drmc_rl.pool.client import sha256_file
        stat = path.stat()
        key = f"{path}|{stat.st_size}|{int(stat.st_mtime)}"
        if key not in self.index:
            self.index[key] = sha256_file(path)
            temporary = self.index_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.index))
            os.replace(temporary, self.index_path)
        return self.index[key]

    def resolve(self, digest, entry):
        size = entry.get("size")
        candidates = [Path(p) for p in entry.get("paths", [])]
        for root in self.dirs:
            if entry.get("name"):
                candidates += list(root.rglob(entry["name"]))
        for path in candidates:
            if path.is_file() and (size is None or path.stat().st_size == size) and self._sha(path) == digest:
                return path
        self._evict(size or 0)
        return self.client.study.download(digest, entry.get("name") or "checkpoint.pt", self.cache)

    def _evict(self, incoming):
        files = sorted((p for p in self.cache.glob("*-*") if p.is_file() and not p.name.startswith(".")),
                       key=lambda p: p.stat().st_atime)
        total = sum(p.stat().st_size for p in files) + incoming
        while files and total > self.cache_gb * 2 ** 30:
            victim = files.pop(0)
            total -= victim.stat().st_size
            victim.unlink()


class Runtimes:
    """ArenaRuntime per (backend, anchor) and an LRU of loaded entrant policies."""

    def __init__(self, args, max_loaded=4):
        self.args, self.max_loaded = args, max_loaded
        self.runtimes, self.policies, self.bases = {}, OrderedDict(), OrderedDict()
        self.built = False

    def config(self, spec, paths):
        runtime = spec["runtime"]
        anchor = paths[runtime["anchor_checkpoint"]]
        config = {k: v for k, v in runtime.items() if k != "anchor_checkpoint"}
        config.update(checkpoint=anchor, device=self.args.device, threads=self.args.threads,
                      native_library=self.args.native_library, output=str(Path(self.args.cache).expanduser() / "work"),
                      working_db=str(Path(self.args.cache).expanduser() / "work" / "unused.sqlite"))
        if self.args.planner_workers:
            config["planner_workers"] = self.args.planner_workers
        # Host-local scheduling of inference (not part of the condition): one forward per
        # shared network for both entrants, and filled asynchronous decision batches.
        batching = getattr(self.args, "batching", "off") == "on"
        config.update(share_base_forwards=batching, fill_inference_batches=batching)
        # A placeholder mixed variant makes the runtime score each entrant with its own policy.
        config["variants"] = {"_anchor": dict(name="anchor", delay=4, checkpoint=anchor)}
        return config

    def get(self, spec, paths):
        from tools.trainer_planning_arena import ArenaRuntime
        key = (spec["runtime"]["rollout_backend"], spec["runtime"]["anchor_checkpoint"])
        if key not in self.runtimes:
            import torch
            for other in list(self.runtimes):
                self.runtimes.pop(other).close()     # one live runtime (planner threads, parent policy)
            # torch accepts set_num_interop_threads once per process; later runtimes keep the first setting.
            original = torch.set_num_interop_threads
            if self.built:
                torch.set_num_interop_threads = lambda n: None
            try:
                self.runtimes[key] = ArenaRuntime(self.config(spec, paths))
            finally:
                torch.set_num_interop_threads = original
            self.built = True
        return self.runtimes[key]

    def policy(self, runtime, params, identity):
        from tools.trainer_planning_arena import _variant_actor, variant_policy
        if identity in self.policies:
            self.policies.move_to_end(identity)
            return self.policies[identity]
        # Knob variants of one checkpoint share its loaded network (and so, with
        # share_base_forwards, its forwards); only the knob wrapper differs.
        weights = json.dumps({k: v for k, v in json.loads(identity).items() if k != "knobs"}, sort_keys=True)
        base = self.bases.get(weights)
        if base is None:
            base = self.bases[weights] = _variant_actor(runtime.config, {k: v for k, v in params.items() if k != "knobs"},
                                                        runtime.policy)
        self.bases.move_to_end(weights)
        policy = variant_policy(runtime.config, params, runtime.policy, base=base)
        self.policies[identity] = policy
        while len(self.bases) > self.max_loaded:
            self.bases.popitem(last=False)
        while len(self.policies) > self.max_loaded:
            self.policies.popitem(last=False)
            try:
                import torch
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass
        return policy

    def play(self, spec, paths, activity):
        from tools.trainer_planning_arena import bind_execution_profiles
        runtime = self.get(spec, paths)
        variants = {}
        for entrant, params in spec["variants"].items():
            local = dict(params)
            for key in ("checkpoint", "adapter_checkpoint"):
                if key in local:
                    local[key] = str(paths[local[key]])
            variants[entrant] = local
        match = copy.deepcopy(spec["match"])
        check = dict(schedule=[dict(match)], variants=variants)
        bind_execution_profiles(check)
        if check["schedule"][0]["execution_key"] != spec["match"]["execution_key"]:
            raise RuntimeError("this host's motor limits differ from the coordinator's condition")
        runtime.config["variants"] = variants
        from drmc_rl.pool.conditions import DECISION_DEFAULTS
        contract = {"delay", "name", "movement", *DECISION_DEFAULTS}
        # Policies are keyed by what is loaded (weights and player settings), not by the condition.
        runtime.policies = {e: self.policy(runtime, p, json.dumps({k: v for k, v in spec["variants"][e].items()
                                                                   if k not in contract}, sort_keys=True))
                            for e, p in variants.items()}
        return runtime.play(match, [tuple(j) for j in spec["jobs"]], activity=activity)

    def close(self):
        for runtime in self.runtimes.values():
            runtime.close()


class FakeRuntimes:
    """Deterministic stand-in for tests: entrant strength comes from ``settings.fake_strength``."""

    def __init__(self, strengths):
        self.strengths = strengths

    def play(self, spec, paths, activity):
        import hashlib
        import math
        begun = time.perf_counter()
        a, b = spec["match"]["a"], spec["match"]["b"]
        diff = self.strengths.get(a, 0.0) - self.strengths.get(b, 0.0)
        p = 1 / (1 + math.exp(-diff))
        out = []
        for seed, side, index in spec["jobs"]:
            activity({})
            u = int(hashlib.sha256(f"{spec['condition']}/{a}/{b}/{seed}/{side}".encode()).hexdigest()[:8], 16) / 2 ** 32
            score = 1.0 if u < p else 0.0
            row = dict(seed=seed, side=side, index=index, score=score, winner="a" if score else "b",
                       reason="topout", frames=1000 + seed % 100, a_stats=dict(decisions=10), b_stats=dict(decisions=10))
            out.append((row, [dict(frame=i, side=i % 2, placement=dict(action=(seed + i) % 512)) for i in range(4)], []))
        return out, max(time.perf_counter() - begun, 1e-3)

    def close(self):
        pass


def identity(args):
    numerics = args.numerics or _numerics(args.device)
    native = {}
    for key, path in (("pool", args.native_library), ("reach", os.environ.get("DRMARIO_REACH_LIB"))):
        if path and Path(path).is_file():
            from drmc_rl.pool.client import sha256_file
            native[key] = sha256_file(path)
    from tools.trainer_arena_distributed import source_revision
    return dict(protocol=PROTOCOL, worker_id=args.worker_id, host=socket.gethostname(), device=args.device,
                threads=args.threads, planner_workers=args.planner_workers, source=source_revision(REPO),
                batching=getattr(args, "batching", "off"),
                numerics=numerics, native=native, engine=args.engine,
                capabilities=sorted(runtime_capabilities(REPO) | {f"engine:{args.engine}"}))


def _numerics(device):
    from tools.trainer_arena_distributed import numerics_class
    return numerics_class(device)


def run_worker(args, client):
    args.worker_id = args.worker_id or f"{socket.gethostname()}-pool-{args.device}-{args.slot}"
    if args.reach_library:
        os.environ["DRMARIO_REACH_LIB"] = str(Path(args.reach_library).resolve())
    if not args.fake and (not args.native_library or not Path(args.native_library).is_file()):
        raise FileNotFoundError("pass --native-library (this host's libdrmario_pool build)")
    me = identity(args)
    study = client.get("/api/v1/pool/study")
    if study["protocol"] != PROTOCOL:
        raise RuntimeError(f"coordinator protocol {study['protocol']} != {PROTOCOL}")
    if study["source"] != me["source"] and not args.allow_source_mismatch:
        raise RuntimeError(f"worker source {me['source']} != coordinator {study['source']}; check out the same commit")
    artifacts = LocalArtifacts(client, args.cache, args.artifact_dir, args.cache_gb)
    runtimes = FakeRuntimes(json.loads(args.fake)) if args.fake else Runtimes(args, args.max_loaded)
    print(json.dumps(dict(worker=me)), flush=True)
    stopped = threading.Event()

    def interrupt(*_):
        if stopped.is_set():
            os._exit(130)
        print("worker: stopping; the current batch is abandoned and its lease released", flush=True)
        stopped.set()
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, interrupt)

    def activity(_progress):
        if stopped.is_set():
            raise Abandoned()
    played, failures = 0, 0
    try:
        while not stopped.is_set():
            pause = host_pause(args)
            if pause:
                print(f"worker: pausing ({pause})", flush=True)
                try:
                    client.post("/api/v1/pool/heartbeat", dict(me, reason=pause), patience=30)
                except Exception as error:
                    print(f"worker: heartbeat failed: {error}", flush=True)
                stopped.wait(args.poll)
                continue
            lease = client.post("/api/v1/pool/leases", me, patience=900)
            if lease["status"] == "rejected":
                print(json.dumps(dict(worker=args.worker_id, status="rejected", reason=lease.get("reason"))), flush=True)
                sys.exit(3)
            if lease["status"] != "lease":
                stopped.wait(min(args.poll, lease.get("retry", args.poll)))
                continue
            spec, base = lease["batch"], f"/api/v1/pool/leases/{lease['lease_id']}"
            renewal = threading.Event()

            def renew():
                while not renewal.wait(max(5.0, lease["ttl_seconds"] / 3)):
                    try:
                        client.post(base + "/renew", dict(claim_token=lease["claim_token"]),
                                    patience=lease["ttl_seconds"] / 3)
                    except Exception as error:
                        print(f"worker: renewal failed: {error}", flush=True)
            threading.Thread(target=renew, daemon=True).start()
            try:
                paths = {("sha256:" + d): (None if args.fake else artifacts.resolve(d, e))
                         for d, e in spec["artifacts"].items()}
                batch, elapsed = runtimes.play(spec, paths, activity)
            except Abandoned:
                try:
                    client.study.request("POST", base + "/release", dict(claim_token=lease["claim_token"]))
                except Exception as error:
                    print(f"worker: release failed ({error}); the lease will expire", flush=True)
                return
            except Exception as error:
                # Contract errors (a loader or decision contract this entrant cannot meet) are
                # deterministic; anything else (device, memory, network) is treated as transient.
                kind = "incompatible" if isinstance(error, (ValueError, KeyError)) else "transient"
                detail = f"{type(error).__name__}: {error}"
                traceback.print_exc()
                client.post(base + "/fail", dict(claim_token=lease["claim_token"], error=detail, kind=kind))
                failures += 1
                if failures >= args.max_failures:
                    raise SystemExit(f"worker: {failures} consecutive failures; last: {detail}")
                stopped.wait(min(300, 10 * failures))
                continue
            finally:
                renewal.set()
            failures = 0
            if not args.fake:
                from drmc_rl.pool.style import game_style
                for row, moves, _ in batch:
                    row["style"] = game_style(row, moves)
            submission = json.loads(json.dumps(dict(
                claim_token=lease["claim_token"], batch=spec, purpose=lease["purpose"], elapsed=elapsed,
                worker=me, rows=[b[0] for b in batch], moves=[b[1] for b in batch])))
            reply = client.post(base + "/results", submission, compress=True, patience=3600)
            played += 1
            print(json.dumps(dict(worker=args.worker_id, batch=spec["key"], purpose=lease["purpose"], games=len(batch),
                                  seconds=round(elapsed, 2), games_per_hour=round(len(batch) / max(elapsed, 1e-9) * 3600),
                                  reply={k: reply.get(k) for k in ("accepted", "new_games", "reason", "agreement")})),
                  flush=True)
            if args.max_batches and played >= args.max_batches:
                return
    finally:
        runtimes.close()


def host_pause(args):
    if args.min_free_gb:
        free = shutil.disk_usage(Path(args.cache).expanduser()).free / 2 ** 30
        if free < args.min_free_gb:
            return f"{free:.1f} GB free < {args.min_free_gb} GB"
    if args.host_budget:
        foreign = foreign_arena_processes()
        if foreign + args.slot >= args.host_budget:
            return f"{foreign} other arena workers on this host; slot {args.slot} of budget {args.host_budget}"
    return None


def add_worker_arguments(parser):
    parser.add_argument("--device", default="mps")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--planner-workers", dest="planner_workers", type=int)
    parser.add_argument("--native-library", default=os.environ.get("DRMARIO_POOL_LIB"))
    parser.add_argument("--reach-library", default=os.environ.get("DRMARIO_REACH_LIB"))
    parser.add_argument("--engine", default="19f292c", help="native engine commit these libraries were built from")
    parser.add_argument("--cache", default="~/.cache/drmc-rl/pool-worker")
    parser.add_argument("--cache-gb", type=float, default=4.0)
    parser.add_argument("--artifact-dir", action="append", default=[],
                        help="directory searched (by file name, then hash) before downloading a checkpoint")
    parser.add_argument("--max-loaded", type=int, default=4, help="entrant policies kept in memory")
    parser.add_argument("--batching", choices=("on", "off"), default="off",
                        help="share one forward per network between entrants and fill decision batches")
    parser.add_argument("--slot", type=int, default=0, help="this worker's index on the host (for --host-budget)")
    parser.add_argument("--host-budget", type=int, default=0,
                        help="pause while other arena workers + slot >= budget (0: no limit)")
    parser.add_argument("--min-free-gb", type=float, default=0.0)
    parser.add_argument("--poll", type=float, default=30.0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-failures", type=int, default=10)
    parser.add_argument("--numerics", help=argparse_suppress())
    parser.add_argument("--fake", help=argparse_suppress())
    parser.add_argument("--worker-id")
    parser.add_argument("--allow-source-mismatch", action="store_true")


def argparse_suppress():
    import argparse
    return argparse.SUPPRESS
