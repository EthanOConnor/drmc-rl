"""Continuous lineage tournament, promotion gate, and web dashboard.

The arena deliberately separates immutable agent identity from checkpoint paths.
Every promoted champion remains an active ``lineage`` entrant forever; the
scheduler gives underserved historical matchups priority while concentrating
most games on candidates and the current champion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import random
import signal
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from drmc_rl.arena.store import Agent, ArenaStore
from tools.tournament import GameSpec, VsMatchRunner, sprt_bounds, sprt_llr

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "runs" / "arena" / "arena.sqlite"
STATIC = Path(__file__).with_name("arena_web")


def stable_id(name: str, checkpoint: str) -> str:
    digest = hashlib.sha256(f"{name}\0{checkpoint}".encode()).hexdigest()[:10]
    return f"{name.lower().replace(' ', '-')}-{digest}"


def register_manifest(store: ArenaStore, path: Path) -> None:
    manifest = json.loads(path.read_text())
    for item in manifest["agents"]:
        checkpoint = str(Path(item["checkpoint"]).expanduser().resolve())
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        name = item["name"]
        store.register(
            agent_id=item.get("id", stable_id(name, checkpoint)),
            name=name,
            family=item.get("family", "central"),
            generation=int(item.get("generation", 0)),
            parent_id=item.get("parent_id"), checkpoint=checkpoint,
            mode=item.get("mode", "plain"), params=item.get("params", {}),
            status=item.get("status", "candidate"), metadata=item.get("metadata", {}),
        )


def checkpoint_step(path: Path) -> int:
    values = re.findall(r"(?:step)?(\d{5,})", path.stem)
    return int(values[-1]) if values else int(path.stat().st_mtime)


def discover_once(store: ArenaStore, config_path: Path) -> int:
    """Register stable campaign checkpoints as named candidate generations."""
    config = json.loads(config_path.read_text())
    known = {Path(agent.checkpoint).resolve() for agent in store.agents()}
    added = 0
    for campaign in config["campaigns"]:
        root = Path(campaign.get("root", REPO_ROOT)).expanduser().resolve()
        paths = sorted(root.glob(campaign["glob"]), key=checkpoint_step)
        family = campaign["family"]
        family_agents = [agent for agent in store.agents() if agent.family == family]
        generation = max((agent.generation for agent in family_agents), default=-1) + 1
        parent = next((agent.id for agent in family_agents if agent.status == "champion"), None)
        for path in paths:
            resolved = path.resolve()
            if resolved in known:
                continue
            # A checkpoint writer may use its final filename before the stream
            # closes. Ignore very recent files until the next scan.
            if time.time() - path.stat().st_mtime < float(campaign.get("settle_seconds", 60)):
                continue
            step = checkpoint_step(path)
            name = campaign["name"].format(generation=generation, step=step)
            store.register(
                agent_id=stable_id(name, str(resolved)), name=name, family=family,
                generation=generation, parent_id=parent, checkpoint=str(resolved),
                mode=campaign.get("mode", "plain"), params=campaign.get("params", {}),
                status="candidate", metadata={"training_step": step, "campaign": campaign.get("id")},
            )
            known.add(resolved)
            generation += 1
            added += 1
    return added


def run_registrar(args: argparse.Namespace) -> None:
    store = ArenaStore(args.db)
    stopped = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while not stopped:
            added = discover_once(store, args.config)
            if added:
                print(f"registered {added} new checkpoint(s)", flush=True)
            if args.once:
                break
            time.sleep(args.poll)
    finally:
        store.close()


def run_telemetry(args: argparse.Namespace) -> None:
    """Mirror the remote trainer's scalar stream without moving run artifacts."""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stopped = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    remote = (
        f"cd {args.remote_repo} && "
        "latest=$(find runs -name metrics.jsonl.gz -printf '%T@ %p\\n' | sort -n | tail -1 | cut -d' ' -f2-); "
        "printf '%s\\n' \"$latest\"; gzip -cd \"$latest\" 2>/dev/null"
    )
    while not stopped:
        result = subprocess.run(
            ["ssh", args.ssh, remote], capture_output=True, text=True, timeout=args.timeout,
            check=False,
        )
        lines = result.stdout.splitlines()
        # A live gzip stream legitimately exits nonzero because its writer has
        # not emitted the final trailer yet; every complete JSONL row is valid.
        if lines:
            run = lines[0]
            latest: dict[str, Any] = {}
            history: dict[str, list[list[float]]] = {}
            for line in lines[1:]:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("type") != "scalar":
                    continue
                name = str(row["name"])
                latest[name] = row["value"]
                if name in {"perf/sps", "perf/dps", "train/return_mean",
                            "search_distill/searched_fraction_actual"}:
                    history.setdefault(name, []).append([row["step"], row["value"]])
            payload = {
                "run": run.removeprefix("runs/"), "updated": utc_timestamp(),
                "latest": latest,
                "history": {name: points[-60:] for name, points in history.items()},
                "source": args.ssh,
            }
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, separators=(",", ":")))
            temporary.replace(output)
        if args.once:
            break
        time.sleep(args.poll)


def utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pair_priority(store: ArenaStore, agents: list[Agent]) -> tuple[Agent, Agent]:
    """Choose the most useful matchup, retaining coverage of every lineage era."""
    champion = next((a for a in agents if a.status == "champion"), None)
    scored: list[tuple[float, Agent, Agent]] = []
    now = time.time()
    del now  # reserved for age weighting once timestamps become material
    for i, a in enumerate(agents):
        for b in agents[i + 1:]:
            n = len(store.matchup_games(a.id, b.id))
            priority = 1.0 / (1.0 + n)
            statuses = {a.status, b.status}
            # Establish a connected comparison graph before spending heavily on
            # repeated gates or expensive search-vs-search matches.
            if n == 0:
                priority += 1_000.0
                if "provisional" in statuses:
                    priority += 1_000.0
            if "candidate" in statuses:
                priority += 80.0 / (1.0 + n)
            if champion and champion.id in {a.id, b.id} and "candidate" in {a.status, b.status}:
                priority += 160.0 / (1.0 + n)
            if n and (a.mode != "plain" or b.mode != "plain"):
                priority *= 0.15
            priority *= random.uniform(0.98, 1.02)
            scored.append((priority, a, b))
    if not scored:
        raise RuntimeError("arena needs at least two active agents")
    _, a, b = max(scored, key=lambda item: item[0])
    return a, b


def maybe_promote(store: ArenaStore, candidate: Agent, champion: Agent, *, elo0: float,
                  elo1: float, alpha: float, beta: float, max_games: int) -> str | None:
    scores = store.matchup_games(candidate.id, champion.id)
    wins = sum(score == 1 for score in scores)
    draws = sum(score == 0.5 for score in scores)
    losses = sum(score == 0 for score in scores)
    llr = sprt_llr(wins, draws, losses, elo0, elo1)
    lower, upper = sprt_bounds(alpha, beta)
    detail = {"champion": champion.id, "games": len(scores), "wins": wins,
              "draws": draws, "losses": losses, "llr": llr,
              "bounds": [lower, upper], "elo": [elo0, elo1]}
    if llr >= upper:
        store.promote(candidate.id, detail=detail)
        return "promoted"
    if llr <= lower or len(scores) >= max_games:
        return "rejected" if llr <= lower else "inconclusive"
    return None


def run_worker(args: argparse.Namespace) -> None:
    store = ArenaStore(args.db)
    stopped = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    runner = VsMatchRunner(level=args.level, speed_setting=args.speed_setting,
                           num_pairs=args.batch, device=args.device, threads=args.threads,
                           run_seed=args.seed, state_repr=args.state_repr,
                           replay_sample_rate=args.replay_sample_rate,
                           max_decisions_per_side=args.max_decisions_per_side)
    serial = int(store.conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0])
    try:
        while not stopped:
            agents = store.agents(("candidate", "champion", "provisional", "lineage", "anchor"))
            if len(agents) < 2:
                time.sleep(args.poll)
                continue
            a, b = pair_priority(store, agents)
            specs = [GameSpec(game_idx=serial + i, seed=(args.seed + serial + i) & 0xFFFF,
                              a_side=(serial + i) % 2) for i in range(args.batch)]
            results = runner.play(a.entry(), b.entry(), specs)
            for result in results:
                store.record(a.id, b.id, seed=result.spec.seed, side=result.spec.a_side,
                             winner=result.winner, match_len_sec=result.match_len_sec,
                             decisions=result.decisions,
                             terminal_reason=result.terminal_reason, replay=result.replay)
                serial += 1
            results.close()
            champion = next((x for x in store.agents(("champion",)) if x.family == a.family), None)
            for candidate in (a, b):
                if candidate.status == "candidate" and champion and candidate.family == champion.family:
                    maybe_promote(store, candidate, champion, elo0=args.elo0, elo1=args.elo1,
                                  alpha=args.alpha, beta=args.beta, max_games=args.max_gate_games)
    finally:
        runner.close()
        store.close()


class DashboardHandler(BaseHTTPRequestHandler):
    db: Path

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/snapshot":
            store = ArenaStore(self.db)
            try:
                payload = json.dumps(store.snapshot(), separators=(",", ":")).encode()
            finally:
                store.close()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if path.startswith("/api/replay/"):
            try:
                match_id = int(path.rsplit("/", 1)[1])
            except ValueError:
                self.send_error(400)
                return
            store = ArenaStore(self.db)
            try:
                replay = store.replay(match_id)
            finally:
                store.close()
            if replay is None:
                self.send_error(404)
                return
            payload = json.dumps(replay, separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        target = STATIC / ("index.html" if path == "/" else path.lstrip("/"))
        if not target.is_file() or STATIC not in target.resolve().parents:
            self.send_error(404)
            return
        payload = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"dashboard: {fmt % args}", flush=True)


def serve(args: argparse.Namespace) -> None:
    DashboardHandler.db = Path(args.db)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"arena dashboard: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register")
    reg.add_argument("manifest", type=Path)
    discover = sub.add_parser("discover")
    discover.add_argument("config", type=Path)
    discover.add_argument("--poll", type=float, default=60)
    discover.add_argument("--once", action="store_true")
    telemetry = sub.add_parser("telemetry")
    telemetry.add_argument("--ssh", default="tf3090")
    telemetry.add_argument("--remote-repo", default="/home/ethan/drmario/drmc-rl")
    telemetry.add_argument("--output", default=str(DEFAULT_DB.parent / "training.json"))
    telemetry.add_argument("--poll", type=float, default=10)
    telemetry.add_argument("--timeout", type=float, default=20)
    telemetry.add_argument("--once", action="store_true")
    web = sub.add_parser("serve")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8097)
    worker = sub.add_parser("worker")
    worker.add_argument("--device", default="cuda")
    worker.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    worker.add_argument("--batch", type=int, default=8)
    worker.add_argument("--level", type=int, default=14)
    worker.add_argument("--speed-setting", type=int, default=2)
    worker.add_argument("--state-repr", default="bitplane_bottle_conn_mask")
    worker.add_argument("--seed", type=int, default=27182)
    worker.add_argument("--poll", type=float, default=10)
    worker.add_argument("--elo0", type=float, default=0)
    worker.add_argument("--elo1", type=float, default=10)
    worker.add_argument("--alpha", type=float, default=0.05)
    worker.add_argument("--beta", type=float, default=0.05)
    worker.add_argument("--max-gate-games", type=int, default=400)
    worker.add_argument("--replay-sample-rate", type=float, default=0.2)
    worker.add_argument("--max-decisions-per-side", type=int, default=500,
                        help="adjudicate unresolved games as draws at this placement horizon (0 disables)")
    args = parser.parse_args()
    if args.command == "register":
        store = ArenaStore(args.db)
        try:
            register_manifest(store, args.manifest)
        finally:
            store.close()
    elif args.command == "discover":
        run_registrar(args)
    elif args.command == "telemetry":
        run_telemetry(args)
    elif args.command == "serve":
        serve(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
