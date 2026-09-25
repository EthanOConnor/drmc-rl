"""Continuously running rating pool: coordinator, workers, registry, intentions and reports.

    serve          coordinator (API with bearer token; optional read-only LAN report port)
    worker         lease and play pool batches (Mac MPS, green CUDA, ...)
    local          coordinator on loopback plus N local workers (tests, single host)
    bootstrap      register conditions, sets, entrants and intentions from a JSON file
    entrant        add | set | list
    condition      add | list;   condition-set add
    job            submit | set | list      (focused experiments with a priority spec)
    intention      add | set | import | list (the project roadmap)
    import-study   feed a finished trainer-planning-arena study journal into the pool
    watch-run      register a training run's snapshots as they are written (training hook)
    stop-rule      a run's snapshot stop rule from pool ratings
    summary        CLI summary;  report --json FILE: full JSON export
    rate           recompute ratings offline from a copy of games.jsonl + registry.jsonl

See docs/RATING_POOL.md.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import fnmatch
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from drmc_rl.pool.client import DEFAULT_TOKEN_FILE, DEFAULT_URL, PoolClient, artifact_record, register_snapshot  # noqa: E402
from tools.trainer_arena_distributed import Handler  # noqa: E402

DEFAULT_PORT = 8097


# ---------------------------------------------------------------------------- server


class PoolHandler(Handler):
    """The distributed-study transport (bearer token, JSON, gzip uploads) with pool routes."""
    downloads: threading.BoundedSemaphore
    mb_per_second: float

    def end_headers(self):
        # API answers are live state: never cached by Cloudflare or any proxy.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):  # noqa: N802
        if not self._authorized():
            return self._json(401, dict(error="unauthorized"))
        url = urllib.parse.urlsplit(self.path)
        path, query = url.path, dict(urllib.parse.parse_qsl(url.query))
        c = self.coordinator
        try:
            if path == "/api/v1/pool/study":
                return self._json(200, self._call(c.study))
            if path == "/api/v1/pool/status":
                return self._json(200, self._call(c.status))
            if path == "/api/v1/pool/report":
                return self._json(200, self._call(c.report))
            if path == "/api/v1/pool/registry":
                return self._json(200, self._call(c.export_registry))
            if path == "/api/v1/pool/stop-rule":
                q = dict(run=query["run"], condition_set=query.get("set"),
                         min_games=int(query.get("min_games", 128)), patience=int(query.get("patience", 2)),
                         step_every=int(query["step_every"]) if query.get("step_every") else None,
                         weighting=query.get("weighting", "pace"))
                return self._json(200, self._call(c.stop_rule, q))
            m = re.fullmatch(r"/api/v1/pool/artifacts/([0-9a-f]{64})/exists", path)
            if m:
                return self._json(200, dict(exists=self._call(c.artifact_path, m.group(1)) is not None))
            m = re.fullmatch(r"/api/v1/checkpoints/([0-9a-f]{64})", path)
            if m:
                return self._send_artifact(m.group(1))
        except KeyError as error:
            return self._json(404, dict(error=f"not found: {error}"))
        return self._json(404, dict(error="not found"))

    def _send_artifact(self, digest):
        source = self._call(self.coordinator.artifact_path, digest)
        if source is None:
            return self._json(404, dict(error="unknown artifact"))
        # Few concurrent, rate-limited transfers: the coordinator host may serve a website.
        with self.downloads:
            size = source.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            started, sent = time.monotonic(), 0
            with source.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    self.wfile.write(block)
                    sent += len(block)
                    ahead = sent / (self.mb_per_second * 2 ** 20) - (time.monotonic() - started)
                    if ahead > 0:
                        time.sleep(ahead)
        return None

    def do_PUT(self):  # noqa: N802
        if not self._authorized():
            return self._json(401, dict(error="unauthorized"))
        path = self.path.split("?", 1)[0]
        part = re.fullmatch(r"/api/v1/pool/artifacts/([0-9a-f]{64})/parts/(\d+)", path)
        if part:
            try:
                self.coordinator.artifacts.store_part(part.group(1), int(part.group(2)), self.rfile,
                                                      int(self.headers["Content-Length"]))
                return self._json(200, dict(part=int(part.group(2))))
            except (TypeError, ValueError) as error:
                return self._json(400, dict(error=str(error)))
        m = re.fullmatch(r"/api/v1/pool/artifacts/([0-9a-f]{64})", path)
        if not m:
            return self._json(404, dict(error="not found"))
        try:
            length = int(self.headers["Content-Length"])
            # Streaming to disk happens on this request thread; only the final rename is state.
            self.coordinator.artifacts.store(m.group(1), self.rfile, length)
            return self._json(200, dict(sha256=m.group(1)))
        except (TypeError, ValueError) as error:
            return self._json(400, dict(error=str(error)))

    def do_POST(self):  # noqa: N802
        if not self._authorized():
            return self._json(401, dict(error="unauthorized"))
        path = self.path.split("?", 1)[0]
        c = self.coordinator
        try:
            request = self._body()
            if path == "/api/v1/pool/leases":
                return self._json(200, self._call(c.lease, request))
            done = re.fullmatch(r"/api/v1/pool/artifacts/([0-9a-f]{64})/complete", path)
            if done:
                # Joining and hashing run on this request thread, outside coordinator state.
                c.artifacts.complete(done.group(1), int(request["parts"]), int(request["size"]))
                return self._json(200, dict(sha256=done.group(1)))
            if path == "/api/v1/pool/heartbeat":
                return self._json(200, self._call(c.heartbeat, request))
            if path == "/api/v1/pool/release":
                return self._json(200, self._call(c.release_batch, str(request["batch"])))
            if path == "/api/v1/pool/registry":
                return self._json(200, self._call(c.register, request))
            if path == "/api/v1/pool/import":
                return self._json(200, self._call(c.import_games, request))
            parts = path.split("/")
            if len(parts) == 7 and parts[:5] == ["", "api", "v1", "pool", "leases"]:
                lease_id, action = urllib.parse.unquote(parts[5]), parts[6]
                claim = str(request.get("claim_token", ""))
                if action == "renew":
                    return self._json(200, self._call(c.renew, lease_id, claim))
                if action == "release":
                    return self._json(200, self._call(c.release, lease_id, claim))
                if action == "fail":
                    return self._json(200, self._call(c.fail, lease_id, request))
                if action == "results":
                    return self._json(200, self._call(c.submit, lease_id, request))
            return self._json(404, dict(error="not found"))
        except PermissionError as error:
            return self._json(409, dict(error=str(error)))
        except (KeyError, TypeError, ValueError) as error:
            return self._json(400, dict(error=f"{type(error).__name__}: {error}"))
        except Exception as error:
            import traceback
            traceback.print_exc()
            c.fatal = f"{type(error).__name__}: {error}"
            return self._json(500, dict(error=c.fatal))


class ReportHandler(BaseHTTPRequestHandler):
    """Read-only, unauthenticated LAN report: the page and its JSON (cached)."""
    coordinator = None
    state: ThreadPoolExecutor
    cache: dict

    def do_GET(self):  # noqa: N802
        from drmc_rl.pool.report import PAGE
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            body, kind = PAGE.encode(), "text/html; charset=utf-8"
        elif path == "/report.json":
            if time.monotonic() - self.cache.get("at", -1e9) > 30:
                self.cache.update(at=time.monotonic(), body=json.dumps(
                    self.state.submit(self.coordinator.report).result()).encode())
            body, kind = self.cache["body"], "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache" if kind.startswith("text/html") else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


class Server(ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True


def serve(args, *, token=None, on_ready=None, stopped=None):
    from drmc_rl.pool.coordinator import PoolCoordinator
    token = token or Path(args.token_file).expanduser().read_text().strip()
    settings = json.loads(args.settings) if args.settings else None
    state = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pool-state")
    coordinator = state.submit(lambda: PoolCoordinator(
        args.data, settings=settings, allow_source_mismatch=args.allow_source_mismatch,
        log=lambda message: print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True))).result()
    s = coordinator.settings
    handler = type("PoolAPI", (PoolHandler,), dict(
        coordinator=coordinator, state=state, token=token,
        downloads=threading.BoundedSemaphore(int(s["max_download_streams"])),
        mb_per_second=float(s["download_mb_per_second"])))
    server = Server((args.host, args.port), handler)
    threading.Thread(target=server.serve_forever, name="pool-api", daemon=True).start()
    report = None
    if args.report_port:
        report_handler = type("PoolReport", (ReportHandler,), dict(coordinator=coordinator, state=state, cache={}))
        report = Server((args.report_host or args.host, args.report_port), report_handler)
        threading.Thread(target=report.serve_forever, name="pool-report", daemon=True).start()
    print(json.dumps(dict(coordinator=f"http://{args.host}:{server.server_address[1]}", source=coordinator.source,
                          report=None if report is None else f"http://{args.report_host or args.host}:"
                                                              f"{report.server_address[1]}/",
                          games=len(coordinator.state.games), entrants=len(coordinator.state.entrants))), flush=True)
    if on_ready is not None:
        on_ready(server.server_address[1])
    stopped = stopped or threading.Event()
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stopped.set())
    try:
        while not stopped.wait(5):
            if coordinator.fatal:
                raise RuntimeError(f"pool stopped after an internal error: {coordinator.fatal}")
            state.submit(coordinator.tick).result()
    finally:
        server.shutdown()
        if report is not None:
            report.shutdown()
        state.shutdown()
    return coordinator


def run_local(args):
    """Coordinator on loopback plus N worker processes of this checkout."""
    token = secrets.token_hex(24)
    ready, finished = threading.Event(), threading.Event()
    port = {}
    args.host, args.port = "127.0.0.1", 0
    thread = threading.Thread(target=serve, args=(args,), kwargs=dict(
        token=token, stopped=finished, on_ready=lambda p: (port.setdefault("port", p), ready.set())), daemon=True)
    thread.start()
    if not ready.wait(600):
        raise RuntimeError("coordinator did not start")
    env = dict(os.environ, DRMC_POOL_TOKEN=token)
    procs = []
    for i in range(args.workers):
        command = [sys.executable, "-m", "tools.rating_pool", "worker", "--coordinator",
                   f"http://127.0.0.1:{port['port']}", "--slot", str(i), "--worker-id", f"{socket.gethostname()}-local{i}"]
        for key in ("device", "threads", "planner_workers", "native_library", "reach_library", "engine", "cache",
                    "cache_gb", "max_loaded", "host_budget", "min_free_gb", "max_batches", "numerics", "fake"):
            value = getattr(args, key, None)
            if value not in (None, "", 0):
                command += [f"--{key.replace('_', '-')}", str(value)]
        for directory in args.artifact_dir:
            command += ["--artifact-dir", directory]
        procs.append(subprocess.Popen(command, cwd=REPO, env=env))
    try:
        codes = [p.wait() for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        finished.set()
        thread.join(30)
    if any(codes):
        raise SystemExit(f"local workers exited with {codes}")


# ---------------------------------------------------------------------------- registry commands


def client_from(args):
    return PoolClient(args.coordinator, token=getattr(args, "token", None), token_file=args.token_file)


def csv(value):
    return [v for v in (value or "").split(",") if v]


def cmd_entrant(args, client):
    if args.action == "add-knob":
        # A knob variant of an existing entrant: same weights, its own identity, parent = the base.
        from drmc_rl.style import knobs
        registry = client.get("/api/v1/pool/registry")
        base = registry["entrants"][args.id]
        entries = [knobs.parse(k) for k in args.knob]
        knobs.validate_knobs(entries)
        if not knobs.active(entries):
            raise SystemExit("every lambda is 0: that is the base entrant itself")
        settings = {k: v for k, v in (base.get("settings") or {}).items() if k != "knobs"}
        settings["knobs"] = entries
        eid = args.variant_id or (args.id + knobs.suffix(entries))
        event = dict(type="entrant", id=eid, name=f"{base.get('name', args.id)} {knobs.suffix(entries)}",
                     loader=base["loader"], checkpoint=base["checkpoint"], era=base["era"], status="active",
                     settings=settings, tags=sorted(set(base.get("tags", [])) | {"knob"}),
                     lineage=dict(parent=args.id, recipe="knob:" + "+".join(f"{k['id']}@{k['version']}" for k in entries)),
                     notes=args.notes or f"knob variant of {args.id}")
        if base.get("adapter"):
            event["adapter"] = base["adapter"]
        print(json.dumps(client.register(event)["event"]["id"]))
        return
    if args.action == "list":
        registry = client.get("/api/v1/pool/registry")
        for e in sorted(registry["entrants"].values(), key=lambda r: (r["era"], r["id"])):
            print(f"{e['status']:<8} {e['era']:<16} {e['id']:<36} {e['checkpoint']['sha256'][:12]}  {e.get('name', '')}")
        return
    event = dict(type="entrant", id=args.id)
    if args.action == "add":
        checkpoint = artifact_record(args.checkpoint)
        if not args.no_upload:
            client.upload(args.checkpoint, checkpoint["sha256"])
        event.update(checkpoint=checkpoint, loader="pace_adapter" if args.adapter else "plain",
                     status=args.status or "active", era=args.era, name=args.name or args.id)
        if args.adapter:
            event["adapter"] = artifact_record(args.adapter)
            if not args.no_upload:
                client.upload(args.adapter, event["adapter"]["sha256"])
    lineage = {k: v for k, v in dict(parent=args.parent, recipe=args.recipe, run=args.run,
                                     step=args.step).items() if v is not None}
    if lineage:
        event["lineage"] = lineage
    for key in ("status", "era", "name", "notes"):
        if getattr(args, key, None) and key not in event:
            event[key] = getattr(args, key)
    if args.tags:
        event["tags"] = csv(args.tags)
    if args.requires:
        event["requires"] = csv(args.requires)
    print(json.dumps(client.register(event)["event"], indent=1))


def cmd_condition(args, client):
    from drmc_rl.pool.conditions import make_condition
    if args.action == "list":
        registry = client.get("/api/v1/pool/registry")
        for key, c in sorted(registry["conditions"].items(), key=lambda kv: kv[1]["name"]):
            print(f"{key}  {c['name']:<40} {json.dumps(c['spec']['decision'])}")
        for s in registry["condition_sets"].values():
            print(f"set {s['name']}: anchor {s['anchor']} weight {s['weight']} primary {s['primary']} "
                  f"({len(s['conditions'])} conditions)")
        return
    decision = dict(delay=args.delay)
    for key in ("decision_point", "early_preview", "preview_input"):
        if getattr(args, key):
            decision[key] = getattr(args, key)
    spec = make_condition(backend=args.backend, engine=args.engine, level=args.level, pace=args.pace,
                          decision=decision, movement=args.movement)
    print(json.dumps(client.register(dict(type="condition", spec=spec, name=args.name, notes=args.notes or "")),
                     indent=1))


def cmd_condition_set(args, client):
    print(json.dumps(client.register(dict(type="condition_set", name=args.name, conditions=csv(args.conditions),
                                          anchor=args.anchor, weight=args.weight, primary=args.primary,
                                          notes=args.notes or "")), indent=1))


def job_seeds(value):
    if value in (None, "bank"):
        return "bank"
    if value.startswith("allocation:"):
        from drmc_rl.program.seed_reserve import allocated_seeds
        study = value.split(":", 1)[1]
        return dict(allocation=study, seeds=allocated_seeds(study))
    if value.startswith("file:"):
        return dict(explicit=json.loads(Path(value.split(":", 1)[1]).read_text()))
    raise ValueError("seeds must be bank, allocation:STUDY or file:PATH (a {condition: [seeds]} JSON)")


def cmd_job(args, client):
    if args.action == "list":
        report = client.get("/api/v1/pool/report")
        for j in report["jobs"]:
            print(f"{j['status']:<9} p{j['priority']:<4} {j['id']:<36} {j['games']}/{j['target']}  {j['title']}")
        return
    event = dict(type="job", id=args.id)
    if args.action == "submit":
        event.update(title=args.title or args.id, mode=args.mode, entrants=csv(args.entrants),
                     conditions=csv(args.conditions), games=args.games,
                     priority=50 if args.priority is None else args.priority,
                     seeds=job_seeds(args.seeds), status="active", owner=args.owner or os.environ.get("USER", ""),
                     open=args.open)
        if args.opponents:
            event["opponents"] = csv(args.opponents)
        if args.pairings:
            event["pairings"] = [p.split(":") for p in csv(args.pairings)]
        if args.deadline:
            event["deadline"] = args.deadline
    else:
        for key in ("status", "priority", "deadline", "title"):
            if getattr(args, key, None) is not None:
                event[key] = getattr(args, key)
    print(json.dumps(client.register(event)["event"], indent=1))


def intention_event(args):
    event = dict(type="intention", id=args.id)
    for key in ("title", "hypothesis", "decision_rule", "owner", "due", "status", "notes"):
        if getattr(args, key, None) is not None:
            event[key] = getattr(args, key)
    for key in ("entrants", "conditions", "metrics", "depends", "resolved"):
        if getattr(args, key, None):
            event[key] = csv(getattr(args, key)) if key != "metrics" else getattr(args, key).split(";")
    if getattr(args, "job", None):
        event["job"] = json.loads(Path(args.job).read_text()) if Path(args.job).is_file() else json.loads(args.job)
    return event


def cmd_intention(args, client):
    if args.action == "list":
        report = client.get("/api/v1/pool/report")
        for i in report["intentions"]:
            wait = f"  waiting on {', '.join(i['waiting_on'])}" if i["waiting_on"] else ""
            print(f"{i['view']:<8}{' OVERDUE' if i['overdue'] else ''} {i['id']:<34} {i['title']}{wait}")
        return
    if args.action == "import":
        for record in json.loads(Path(args.file).read_text())["intentions"]:
            client.register(dict(type="intention", **record))
            print("recorded", record["id"])
        return
    print(json.dumps(client.register(intention_event(args))["event"], indent=1))


def cmd_bootstrap(args, client):
    """Idempotently register a declarative pool setup (conditions, sets, entrants, intentions)."""
    from drmc_rl.pool.conditions import make_condition
    spec = json.loads(Path(args.file).read_text())
    registry = client.get("/api/v1/pool/registry")
    names = {c["name"] for c in registry["conditions"].values()}
    for c in spec.get("conditions", []):
        if c["name"] in names:
            continue
        condition = make_condition(**{k: v for k, v in c.items() if k not in ("name", "notes")})
        client.register(dict(type="condition", spec=condition, name=c["name"], notes=c.get("notes", "")))
        print("condition", c["name"])
    for e in spec.get("entrants", []):
        record = dict(e)
        for key in ("checkpoint", "adapter"):
            if key in record and isinstance(record[key], str):
                path = Path(record[key]).expanduser()
                expected = record.pop(f"{key}_sha256", None)
                if path.exists():
                    record[key] = artifact_record(path)
                    if expected and record[key]["sha256"] != expected:
                        raise ValueError(f"{path}: sha256 {record[key]['sha256']} != declared {expected}")
                    if not args.no_upload:
                        client.upload(path, record[key]["sha256"])
                elif expected:
                    record[key] = dict(sha256=expected, paths=[str(path)], name=path.name)
                else:
                    raise FileNotFoundError(path)
        current = registry["entrants"].get(record["id"])
        if current and current["checkpoint"]["sha256"] == record["checkpoint"]["sha256"]:
            record = {k: v for k, v in record.items() if k in ("id", "notes", "tags", "era", "name", "lineage")}
        client.register(dict(type="entrant", by="bootstrap", **record))
        print("entrant", record["id"])
    for s in spec.get("condition_sets", []):
        client.register(dict(type="condition_set", **s))
        print("condition set", s["name"])
    if spec.get("blocks"):
        keys = {c["name"]: k for k, c in client.get("/api/v1/pool/registry")["conditions"].items()}
        for block in spec["blocks"]:
            client.register(dict(type="block", by="bootstrap", **dict(block, condition=keys.get(block["condition"],
                                                                                              block["condition"]))))
        print("blocks", len(spec["blocks"]))
    for record in spec.get("intentions", []):
        if record["id"] not in registry["intentions"]:
            client.register(dict(type="intention", by="bootstrap", **record))
            print("intention", record["id"])
    for record in spec.get("jobs", []):
        if record["id"] not in registry["jobs"]:
            client.register(dict(type="job", by="bootstrap", **record))
            print("job", record["id"])


# ---------------------------------------------------------------------------- import and watch


def cmd_import_study(args, client):
    """Import a trainer-planning-arena study journal where its conditions are pool conditions."""
    from drmc_rl.pool.conditions import condition_key, default_name, study_condition
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    name = args.name or config_path.stem
    output = Path(config["output"])
    if not (output / "games.jsonl").exists():
        output = config_path.with_suffix("")
    registry = client.get("/api/v1/pool/registry")
    known = {}
    for e in registry["entrants"].values():
        known[identity_key(e["checkpoint"]["sha256"], (e.get("adapter") or {}).get("sha256"),
                           e.get("settings", {}))] = e["id"]
    hashes = dict(config.get("model_sha256", {}))

    def digest(path):
        if path in hashes:
            return hashes[path]
        if Path(path).is_file():
            hashes[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            return hashes[path]
        return None
    parent = config["checkpoint"]
    entrants, skipped = {}, {}
    contract = {"delay", "decision_point", "early_preview", "preview_input", "compute_input_frames",
                "early_delay_input", "movement", "checkpoint", "adapter_checkpoint", "name", "ready_when"}
    for vid, params in config["variants"].items():
        base = params.get("checkpoint", parent)
        ckpt, adapter = digest(base), digest(params["adapter_checkpoint"]) if "adapter_checkpoint" in params else None
        if ckpt is None or ("adapter_checkpoint" in params and adapter is None):
            skipped[vid] = "checkpoint file and recorded hash unavailable"
            continue
        settings = {k: v for k, v in params.items() if k not in contract}
        key = identity_key(ckpt, adapter, settings)
        if key not in known:
            if not args.register_unknown:
                skipped[vid] = f"no pool entrant with checkpoint {ckpt[:12]}" + (f"+adapter {adapter[:12]}" if adapter else "")
                continue
            eid = slug(f"imp-{params.get('name', vid)}-{(adapter or ckpt)[:8]}")
            record = dict(type="entrant", id=eid, name=params.get("name", vid), era=args.era,
                          status="retired", loader="pace_adapter" if adapter else "plain",
                          checkpoint=dict(sha256=ckpt, paths=[base], name=Path(base).name),
                          notes=f"registered by import of {name}", by=f"import:{name}")
            if adapter:
                record["adapter"] = dict(sha256=adapter, paths=[params["adapter_checkpoint"]],
                                         name=Path(params["adapter_checkpoint"]).name)
            if settings:
                record["settings"] = settings
            client.register(record)
            known[key] = eid
        entrants[vid] = known[key]
    conditions = {c["key"] if "key" in c else k: c for k, c in registry["conditions"].items()}
    names = {c["name"]: k for k, c in conditions.items()}
    matches, reasons = {}, {}
    for match in config["schedule"]:
        a, b = match["a"], match["b"]
        if a not in entrants or b not in entrants:
            reasons[match["id"]] = f"entrant unavailable: {skipped.get(a) or skipped.get(b)}"
            continue
        if entrants[a] == entrants[b]:
            reasons[match["id"]] = "both sides are the same pool entrant"
            continue
        spec, why = study_condition(config, match, config["variants"][a], config["variants"][b])
        if spec is None:
            reasons[match["id"]] = why
            continue
        key = condition_key(spec)
        if key not in conditions:
            cname = default_name(spec)
            if cname in names:
                cname = f"{cname}-{key[1:7]}"
            client.register(dict(type="condition", spec=spec, name=cname, notes=f"first seen in import {name}"))
            conditions[key] = dict(name=cname, spec=spec)
            names[cname] = key
        matches[match["id"]] = (key, entrants[a], entrants[b], match)
    rows, traces, counted = [], {}, {}
    device = config.get("device", "unknown")
    every = max(1, args.trace_every)
    for line in (output / "games.jsonl").read_text().splitlines():
        row = json.loads(line)
        hit = matches.get(row["comparison"])
        if hit is None:
            counted["skipped"] = counted.get("skipped", 0) + 1
            continue
        key, ea, eb, match = hit
        if row.get("execution_key") and row["execution_key"] != match.get("execution_key", row["execution_key"]):
            raise ValueError(f"{row['comparison']}: journal execution key differs from its schedule")
        a, b, side, score = (ea, eb, row["side"], row["score"]) if ea < eb else \
            (eb, ea, 1 - row["side"], None if row["score"] is None else 1.0 - row["score"])
        winner = row["winner"] if ea < eb or row["winner"] in (None, "draw") else {"a": "b", "b": "a"}[row["winner"]]
        out = dict(condition=key, a=a, b=b, seed=row["seed"], side=side, score=score, winner=winner,
                   reason=row["reason"], frames=row["frames"],
                   decisions=[row.get("a_stats", {}).get("decisions", 0), row.get("b_stats", {}).get("decisions", 0)],
                   source=f"import:{name}", batch=row["comparison"], numerics=f"import/{device}",
                   time=date.today().isoformat())
        if ea > eb:
            out["decisions"].reverse()
        rows.append(out)
        pair_hash = int(hashlib.sha1(f"{key}/{a}/{b}/{row['seed']}".encode()).hexdigest()[:8], 16)
        trace = output / "moves" / f"{row['comparison']}-{row['index']:04d}.json.gz"
        if pair_hash % every == 0 and trace.exists():
            gid = f"{key}/{a}/{b}/{row['seed']}/{side}"
            traces[gid] = json.loads(gzip.decompress(trace.read_bytes()))
    result = dict(study=name, rows=len(rows), entrants=entrants, skipped_variants=skipped,
                  skipped_comparisons=reasons, imported=0, duplicates=0)
    if args.dry_run:
        print(json.dumps(result, indent=1))
        return result
    for start in range(0, len(rows), 2000):
        chunk = rows[start:start + 2000]
        ids = {f"{r['condition']}/{r['a']}/{r['b']}/{r['seed']}/{r['side']}" for r in chunk}
        reply = client.post("/api/v1/pool/import", dict(rows=chunk, traces={k: v for k, v in traces.items() if k in ids}),
                            compress=True, patience=600)
        result["imported"] += reply["imported"]
        result["duplicates"] += reply["duplicates"]
    print(json.dumps(result, indent=1))
    return result


def identity_key(checkpoint, adapter, settings):
    return json.dumps([checkpoint, adapter, settings or {}], sort_keys=True)


def slug(text):
    out = re.sub(r"[^A-Za-z0-9._+-]+", "-", text).strip("-")
    return out[:60] or "entrant"


def cmd_watch_run(args, client):
    """The training hook: register each new snapshot of a run as it lands (stable for --settle seconds)."""
    directory = Path(args.dir).expanduser()
    seen = {}
    step_re = re.compile(args.step_regex)
    configured = False
    while True:
        registered_now = False
        registry = client.get("/api/v1/pool/registry")
        present = {e["checkpoint"]["sha256"] for e in registry["entrants"].values()}
        for path in sorted(directory.glob(args.pattern)):
            if not path.is_file() or fnmatch.fnmatch(path.name, args.exclude or "\0"):
                continue
            stat = path.stat()
            signature = (stat.st_size, int(stat.st_mtime))
            if seen.get(path) != signature:
                seen[path] = signature
                continue
            if time.time() - stat.st_mtime < args.settle:
                continue
            m = step_re.search(path.name)
            step = int(m.group(1)) if m else int(stat.st_mtime)
            record = artifact_record(path)
            if record["sha256"] in present:
                continue
            entrant = args.id_format.format(run=args.run, step=step, stem=path.stem)
            register_snapshot(client, path, run=args.run, entrant_id=entrant, era=args.era, step=step,
                              parent=args.parent, recipe=args.recipe, panel_set=args.panel_set,
                              panel_games=args.panel_games, panel_priority=args.panel_priority,
                              panel_step_every=args.panel_step_every, anchor=args.anchor)
            present.add(record["sha256"])
            registered_now = True
            print(json.dumps(dict(registered=entrant, step=step, sha256=record["sha256"])), flush=True)
        if args.auto_conclude and args.panel_set and not configured:
            client.register(dict(type="lineage", run=args.run, stop_rule=dict(
                set=args.panel_set, min_games=args.panel_games, patience=2, step_every=args.panel_step_every,
                auto=True), by=f"watch-run:{args.run}"))
            configured = True
        if args.final_marker and not registered_now and list(directory.glob(args.final_marker)):
            print(json.dumps(client.register(dict(type="conclude", run=args.run,
                                                  reason=f"final marker {args.final_marker} present"))), flush=True)
            return
        if args.once:
            return
        time.sleep(args.poll)


# ---------------------------------------------------------------------------- reports


def cmd_summary(args, client):
    from drmc_rl.pool.report import summary_text
    report = client.get("/api/v1/pool/report")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n")
        print(f"wrote {args.json}")
    else:
        print(summary_text(report))


def cmd_stop_rule(args, client):
    result = client.get("/api/v1/pool/stop-rule", run=args.run, set=args.set, min_games=args.min_games,
                        patience=args.patience, step_every=args.step_every,
                        weighting="equal" if args.equal else "pace")
    print(json.dumps(result, indent=1))
    if args.exit_code:
        sys.exit(10 if result["fired"] else 0)


def cmd_lineage(args, client):
    if args.action == "list":
        for row in client.get("/api/v1/pool/report")["lineages"]:
            print(f"{row['status']:<10} {row['run']:<28} {row['members']:>3} snapshots  newest {row['newest']}"
                  + (f"  best {row['best']}" if row.get("best") else ""))
        return
    if args.action == "conclude":
        event = dict(type="conclude", run=args.run, reason=args.reason)
        if args.best:
            event["best"] = args.best
        print(json.dumps(client.register(event), indent=1))
        return
    rule = dict(set=args.stop_set, min_games=args.min_games, patience=args.patience, auto=args.auto)
    if args.step_every:
        rule["step_every"] = args.step_every
    print(json.dumps(client.register(dict(type="lineage", run=args.run, stop_rule=rule))["event"], indent=1))


def cmd_rate(args):
    """Offline recomputation from copied journals: proves the ratings depend on nothing else."""
    from drmc_rl.pool.coordinator import PoolCoordinator
    from drmc_rl.pool.report import summary_text
    coordinator = PoolCoordinator(args.data, source="offline", capabilities=set(), log=lambda *_: None)
    report = coordinator.report()
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n")
    print(summary_text(report))


# ---------------------------------------------------------------------------- CLI


def main(argv=None):
    from drmc_rl.pool.worker import add_worker_arguments, run_worker
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def remote(p):
        p.add_argument("--coordinator", default=DEFAULT_URL)
        p.add_argument("--token-file", default=DEFAULT_TOKEN_FILE)

    def serving(p, mismatch=True):
        p.add_argument("--data", required=True, type=Path)
        p.add_argument("--settings", help="JSON overrides of pool.json settings for this run")
        if mismatch:
            p.add_argument("--allow-source-mismatch", action="store_true")
        p.add_argument("--report-port", type=int, default=0)
        p.add_argument("--report-host")

    s = commands.add_parser("serve", help="run the pool coordinator")
    serving(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    s.add_argument("--token-file", default=DEFAULT_TOKEN_FILE)

    w = commands.add_parser("worker", help="lease and play pool batches")
    remote(w)
    add_worker_arguments(w)

    lo = commands.add_parser("local", help="coordinator on loopback plus N local workers")
    serving(lo, mismatch=False)
    lo.add_argument("--workers", type=int, default=2)
    add_worker_arguments(lo)

    b = commands.add_parser("bootstrap", help="register a declarative pool setup")
    remote(b)
    b.add_argument("file")
    b.add_argument("--no-upload", action="store_true")

    e = commands.add_parser("entrant", help="add, update or list entrants")
    remote(e)
    e.add_argument("action", choices=("add", "set", "list", "add-knob"))
    e.add_argument("--knob", action="append", default=[], help="add-knob: id@version:lambda (repeatable, applied in order)")
    e.add_argument("--variant-id", help="add-knob: override the id (default: base id + knob suffix)")
    e.add_argument("id", nargs="?")
    e.add_argument("--checkpoint")
    e.add_argument("--adapter")
    e.add_argument("--era")
    e.add_argument("--name")
    e.add_argument("--status", choices=("active", "benched", "retired"))
    e.add_argument("--parent")
    e.add_argument("--recipe")
    e.add_argument("--run")
    e.add_argument("--step", type=int)
    e.add_argument("--tags")
    e.add_argument("--requires", help="comma-separated extra runtime capabilities")
    e.add_argument("--notes")
    e.add_argument("--no-upload", action="store_true")

    c = commands.add_parser("condition", help="add or list rating conditions")
    remote(c)
    c.add_argument("action", choices=("add", "list"))
    c.add_argument("--name")
    c.add_argument("--backend", default="events", choices=("events", "frames"))
    c.add_argument("--engine", default="19f292c")
    c.add_argument("--level", type=int, default=14)
    c.add_argument("--pace", default="frame_perfect")
    c.add_argument("--delay", type=int, default=4)
    c.add_argument("--decision-point")
    c.add_argument("--early-preview")
    c.add_argument("--preview-input")
    c.add_argument("--movement", default="exact", choices=("exact", "human"))
    c.add_argument("--notes")

    cs = commands.add_parser("condition-set", help="define a named, anchored condition set")
    remote(cs)
    cs.add_argument("action", choices=("add",))
    cs.add_argument("name")
    cs.add_argument("--conditions", required=True)
    cs.add_argument("--anchor", required=True)
    cs.add_argument("--weight", type=float, default=1.0)
    cs.add_argument("--primary", action="store_true")
    cs.add_argument("--notes")

    j = commands.add_parser("job", help="submit or change a focused experiment")
    remote(j)
    j.add_argument("action", choices=("submit", "set", "list"))
    j.add_argument("id", nargs="?")
    j.add_argument("--title")
    j.add_argument("--mode", default="vs", choices=("vs", "round_robin", "vs_parent", "explicit"))
    j.add_argument("--entrants", help="comma-separated ids or glob patterns")
    j.add_argument("--opponents", help="for mode vs (default: each condition's anchor)")
    j.add_argument("--pairings", help="for mode explicit: a:b,c:d")
    j.add_argument("--conditions", help="names, keys or set:NAME")
    j.add_argument("--games", type=int, default=128, help="games per pairing and condition (even)")
    j.add_argument("--priority", type=int, default=None)
    j.add_argument("--deadline")
    j.add_argument("--seeds", default="bank", help="bank | allocation:STUDY | file:PATH")
    j.add_argument("--open", action="store_true", help="keep active as new matching entrants appear")
    j.add_argument("--status", choices=("active", "paused", "done", "cancelled"))
    j.add_argument("--owner")

    i = commands.add_parser("intention", help="record planned experiments (the roadmap)")
    remote(i)
    i.add_argument("action", choices=("add", "set", "import", "list"))
    i.add_argument("id", nargs="?")
    i.add_argument("--file", help="for import: JSON with an 'intentions' list")
    i.add_argument("--title")
    i.add_argument("--hypothesis")
    i.add_argument("--entrants")
    i.add_argument("--conditions")
    i.add_argument("--metrics", help="semicolon-separated")
    i.add_argument("--decision-rule", dest="decision_rule")
    i.add_argument("--depends", help="comma-separated intention:ID, capability:CAP, entrant:PATTERN, external:TEXT")
    i.add_argument("--resolved", help="dependencies now met (external ones)")
    i.add_argument("--status", choices=("planned", "blocked", "running", "done", "dropped"))
    i.add_argument("--owner")
    i.add_argument("--due")
    i.add_argument("--job", help="job spec JSON (or a file) submitted when the intention is ready")
    i.add_argument("--notes")

    im = commands.add_parser("import-study", help="import a finished trainer-planning-arena study")
    remote(im)
    im.add_argument("config")
    im.add_argument("--name")
    im.add_argument("--trace-every", type=int, default=16)
    im.add_argument("--register-unknown", action="store_true",
                    help="register unknown checkpoints as retired entrants instead of skipping them")
    im.add_argument("--era", default="imported")
    im.add_argument("--dry-run", action="store_true")

    wr = commands.add_parser("watch-run", help="register a training run's snapshots as they are written")
    remote(wr)
    wr.add_argument("--dir", required=True)
    wr.add_argument("--pattern", default="core-f*.pt")
    wr.add_argument("--exclude")
    wr.add_argument("--run", required=True)
    wr.add_argument("--era", required=True)
    wr.add_argument("--parent", help="entrant id of the run's initialization")
    wr.add_argument("--recipe")
    wr.add_argument("--id-format", default="{run}-f{step:011d}")
    wr.add_argument("--step-regex", default=r"f(\d+)")
    wr.add_argument("--panel-set", help="condition set of the run's open stop-rule panel job")
    wr.add_argument("--panel-games", type=int, default=128)
    wr.add_argument("--panel-priority", type=int, default=60)
    wr.add_argument("--panel-step-every", type=int,
                    help="panel and stop rule only on snapshots at multiples of this many frames (e.g. 50000000)")
    wr.add_argument("--auto-conclude", action="store_true",
                    help="let the pool stop rule on --panel-set conclude the run when it fires")
    wr.add_argument("--final-marker", help="glob in --dir whose appearance means the run is finished")
    wr.add_argument("--anchor")
    wr.add_argument("--settle", type=float, default=120.0)
    wr.add_argument("--poll", type=float, default=120.0)
    wr.add_argument("--once", action="store_true")

    sr = commands.add_parser("stop-rule", help="a run's snapshot stop rule from pool ratings")
    remote(sr)
    sr.add_argument("--run", required=True)
    sr.add_argument("--set")
    sr.add_argument("--min-games", type=int, default=128)
    sr.add_argument("--patience", type=int, default=2)
    sr.add_argument("--step-every", type=int, help="only snapshots at multiples of this many frames")
    sr.add_argument("--equal", action="store_true", help="equal pace weights instead of the confirmed pace weights")
    sr.add_argument("--exit-code", action="store_true", help="exit 10 when the rule has fired")

    lg = commands.add_parser("lineage", help="training-run lineages: list, conclude, or set the pool stop rule")
    remote(lg)
    lg.add_argument("action", choices=("list", "conclude", "set"))
    lg.add_argument("run", nargs="?")
    lg.add_argument("--best", help="conclude: the snapshot to keep (default: stop-rule selection, else final)")
    lg.add_argument("--reason", default="marked done")
    lg.add_argument("--stop-set", help="set: condition set of the pool stop rule")
    lg.add_argument("--step-every", type=int)
    lg.add_argument("--min-games", type=int, default=128)
    lg.add_argument("--patience", type=int, default=2)
    lg.add_argument("--auto", action="store_true", help="set: conclude automatically when the stop rule fires")

    su = commands.add_parser("summary", help="CLI summary (or --json FILE for the full export)")
    remote(su)
    su.add_argument("--json")

    ra = commands.add_parser("rate", help="recompute ratings offline from a data directory")
    ra.add_argument("--data", required=True)
    ra.add_argument("--json")

    rel = commands.add_parser("release", help="drop live leases of a batch (hung worker)")
    remote(rel)
    rel.add_argument("batch")

    args = parser.parse_args(argv)
    if args.command == "serve":
        serve(args)
    elif args.command == "local":
        run_local(args)
    elif args.command == "rate":
        cmd_rate(args)
    else:
        client = client_from(args)
        dispatch = dict(worker=lambda: run_worker(args, client), bootstrap=lambda: cmd_bootstrap(args, client),
                        entrant=lambda: cmd_entrant(args, client), condition=lambda: cmd_condition(args, client),
                        job=lambda: cmd_job(args, client), intention=lambda: cmd_intention(args, client),
                        summary=lambda: cmd_summary(args, client), release=lambda: print(client.post(
                            "/api/v1/pool/release", dict(batch=args.batch))),
                        **{"condition-set": lambda: cmd_condition_set(args, client),
                           "import-study": lambda: cmd_import_study(args, client),
                           "watch-run": lambda: cmd_watch_run(args, client),
                           "stop-rule": lambda: cmd_stop_rule(args, client),
                           "lineage": lambda: cmd_lineage(args, client)})
        dispatch[args.command]()


if __name__ == "__main__":
    main()
