"""Continuous lineage tournament, promotion gate, and web dashboard.

The arena deliberately separates immutable agent identity from checkpoint paths.
Every promoted champion remains an active ``lineage`` entrant forever; the
scheduler gives underserved historical matchups priority while concentrating
most games on candidates and the current champion.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import mimetypes
import os
import random
import secrets
import signal
import re
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from drmc_rl.arena.ratings import RatingConfig, RatingConvergenceError
from drmc_rl.arena.remote import (
    ArenaRemoteClient,
    PROTOCOL_VERSION,
    content_sha256,
)
from drmc_rl.arena.store import DEFAULT_SCHEDULER_BOOST, Agent, ArenaStore
from tools.tournament import (
    ARENA_MAX_DECISIONS_PER_SIDE,
    NES_FPS,
    GameSpec,
    VsMatchRunner,
    sprt_bounds,
    sprt_llr,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "runs" / "arena" / "arena.sqlite"
STATIC = Path(__file__).with_name("arena_web")
SCHEDULER_TEMPERATURE = 1.25
SCHEDULER_COVERAGE_MIX = 0.10
SEARCH_COMPUTE_FACTOR = 0.15


def paired_specs(
    *,
    schedule_seed: int,
    agent_a: str,
    agent_b: str,
    start: int,
    count: int,
    level: int,
    speed_setting: int,
    state_repr: str,
    max_decisions_per_side: int,
    policy_run_seed: int,
    used: set[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic, complete reset specs in side-swapped pairs."""
    if count <= 0 or count % 2:
        raise ValueError("arena batches must contain a positive even number of games")
    used = set() if used is None else used
    specs: list[dict[str, Any]] = []
    for offset in range(0, count, 2):
        pair_serial = start + offset // 2
        digest = hashlib.sha256(
            f"arena-v2\0{schedule_seed}\0{agent_a}\0{agent_b}\0{pair_serial}".encode()
        ).digest()
        seed = int.from_bytes(digest[:2], "little") or 0x55AA
        for _candidate in range(1 << 16):
            if (seed, 0) not in used and (seed, 1) not in used:
                break
            seed = (seed + 1) & 0xFFFF or 1
        else:
            raise RuntimeError(f"seed space exhausted for {agent_a} vs {agent_b}")
        frame_counter_base = int.from_bytes(digest[2:4], "little")
        for side in (0, 1):
            game_index = start + offset + side
            spec = {
                "game_idx": game_index,
                "seed": seed,
                "a_side": side,
                "frame_counter_base": frame_counter_base,
                "level": int(level),
                "speed_setting": int(speed_setting),
                "state_repr": str(state_repr),
                "max_decisions_per_side": int(max_decisions_per_side),
                "policy_run_seed": int(policy_run_seed),
            }
            identity = {"agent_a": agent_a, "agent_b": agent_b, **spec}
            spec["match_id"] = content_sha256(identity)
            specs.append(spec)
            used.add((seed, side))
    return specs


def game_spec_from_wire(spec: dict[str, Any]) -> GameSpec:
    required = {
        "game_idx",
        "seed",
        "a_side",
        "frame_counter_base",
        "level",
        "speed_setting",
        "state_repr",
        "max_decisions_per_side",
        "policy_run_seed",
    }
    missing = required - spec.keys()
    if missing:
        raise ValueError("incomplete reset spec: " + ", ".join(sorted(missing)))
    return GameSpec(**{key: spec[key] for key in required})


@functools.lru_cache(maxsize=256)
def _file_sha256_cached(path: str, size: int, modified_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    stat = path.stat()
    return _file_sha256_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def agent_wire(agent: Agent) -> dict[str, Any]:
    checkpoint = Path(agent.checkpoint)
    return {
        "id": agent.id,
        "name": agent.name,
        "mode": agent.mode,
        "params": agent.params,
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": file_sha256(checkpoint),
    }


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
            parent_id=item.get("parent_id"),
            checkpoint=checkpoint,
            mode=item.get("mode", "plain"),
            params=item.get("params", {}),
            status=item.get("status", "candidate"),
            metadata=item.get("metadata", {}),
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
                agent_id=stable_id(name, str(resolved)),
                name=name,
                family=family,
                generation=generation,
                parent_id=parent,
                checkpoint=str(resolved),
                mode=campaign.get("mode", "plain"),
                params=campaign.get("params", {}),
                status="candidate",
                metadata={"training_step": step, "campaign": campaign.get("id")},
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
        # PPO emits every few seconds.  A short freshness window prevents a
        # completed smoke run (or a stopped service) from being summed into
        # the dashboard's live throughput for another quarter hour.
        "find runs -name metrics.jsonl.gz -mmin -3 -printf '%T@ %p\\n' | sort -n | "
        "while read -r stamp path; do printf '@@JSON %s\\n' \"$path\"; "
        'gzip -cd "$path" 2>/dev/null; done; '
        "find runs/human_policy -name '*_train.log' -mmin -15 -print 2>/dev/null | "
        "while read -r path; do printf '@@AFTERSTATE %s\\n' \"$path\"; "
        'tail -400 "$path"; done; '
        "find /home/ethan/.cache/drmc-rl/training-store/human-v5/full-corpus "
        "-maxdepth 1 -name 'extract*.log' -mmin -15 -print 2>/dev/null | "
        "while read -r path; do printf '@@CORPUS %s\n' \"$path\"; "
        'tail -200 "$path"; done'
    )
    while not stopped:
        result = subprocess.run(
            ["ssh", args.ssh, remote],
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
        tasks = parse_telemetry(result.stdout)
        # A live gzip stream legitimately exits nonzero because its writer has
        # not emitted the final trailer yet; every complete JSONL row is valid.
        if tasks:
            for item in tasks:
                item["history"] = {name: points[-60:] for name, points in item["history"].items()}
            # Preserve the original top-level shape for existing dashboard
            # consumers while exposing every recently active campaign.
            latest: dict[str, Any] = {}
            for name in ("perf/sps", "perf/dps"):
                latest[name] = sum(float(item["latest"].get(name, 0.0)) for item in tasks)
            if tasks:
                primary = max(tasks, key=lambda item: float(item["latest"].get("perf/sps", 0.0)))
                for name, value in primary["latest"].items():
                    latest.setdefault(name, value)
            payload = {
                "run": f"{len(tasks)} active campaign{'s' if len(tasks) != 1 else ''}",
                "updated": utc_timestamp(),
                "latest": latest,
                "history": {},
                "tasks": tasks,
                "source": args.ssh,
            }
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, separators=(",", ":")))
            temporary.replace(output)
        if args.once:
            break
        time.sleep(args.poll)


def run_externalize_replays(args: argparse.Namespace) -> None:
    store = ArenaStore(args.db, replay_dir=args.replay_dir)
    moved = 0
    try:
        while True:
            count = store.externalize_replays(limit=args.batch)
            moved += count
            if count:
                print(f"externalized {moved:,} replay rows", flush=True)
            if count < args.batch:
                break
        if args.vacuum:
            store.conn.execute("VACUUM")
    finally:
        store.close()
    print(f"externalized {moved:,} replay rows total", flush=True)


_AFTERSTATE_PROGRESS = re.compile(
    r"epoch=(?P<epoch>\d+) step=(?P<step>\d+) decisions/s=(?P<dps>[\d,]+) "
    r"loss=(?P<loss>[\d.eE+-]+)(?P<parts>.*)"
)
_SCALAR_PART = re.compile(r"(?P<name>[a-z_]+)=(?P<value>[\d.eE+-]+)")
_VALIDATION_SCALAR = re.compile(
    r'"(?P<name>validation_(?:objective|top1|quality_top1|outcome_brier|mean_regret|regret_q90|'
    r'low_rating_regret_q90|high_rating_regret_q90|regret_tail_gap|rows))"\s*:\s*'
    r"(?P<value>[\d.eE+-]+)"
)
_CORPUS_PROGRESS = re.compile(
    r"scanned=(?P<scanned>[\d,]+) sampled=(?P<sampled>[\d,]+) "
    r"kept=(?P<kept>[\d,]+) rate=(?P<rate>[\d,]+)/s"
)


def parse_telemetry(text: str) -> list[dict[str, Any]]:
    """Parse both standard scalar streams and the live V3 corpus trainer log."""
    tasks: list[dict[str, Any]] = []
    kind: str | None = None
    task: dict[str, Any] | None = None
    afterstate_lines: list[str] = []

    def finish() -> None:
        nonlocal task, afterstate_lines
        if task is None:
            return
        if kind == "afterstate":
            _parse_afterstate_lines(task, afterstate_lines)
        elif kind == "corpus":
            _parse_corpus_lines(task, afterstate_lines)
        task["history"] = {name: points[-60:] for name, points in task["history"].items()}
        if task["latest"]:
            tasks.append(task)
        task = None
        afterstate_lines = []

    for line in text.splitlines():
        if line.startswith("@@JSON ") or line.startswith("@@RUN "):
            finish()
            prefix = "@@JSON " if line.startswith("@@JSON ") else "@@RUN "
            kind = "json"
            task = {"run": line[len(prefix) :].removeprefix("runs/"), "latest": {}, "history": {}}
            continue
        if line.startswith("@@AFTERSTATE "):
            finish()
            kind = "afterstate"
            task = {"run": line[13:].removeprefix("runs/"), "latest": {}, "history": {}}
            continue
        if line.startswith("@@CORPUS "):
            finish()
            kind = "corpus"
            task = {"run": line[9:], "latest": {}, "history": {}}
            continue
        if task is None:
            continue
        if kind in {"afterstate", "corpus"}:
            afterstate_lines.append(line)
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("type") != "scalar":
            continue
        if row.get("step") is not None:
            task["latest"]["global_step"] = max(
                int(task["latest"].get("global_step", 0)), int(row["step"])
            )
        name = str(row["name"])
        task["latest"][name] = row["value"]
        if name in {
            "perf/sps",
            "perf/dps",
            "train/return_mean",
            "search_distill/searched_fraction_actual",
        }:
            task["history"].setdefault(name, []).append([row["step"], row["value"]])
    finish()
    return tasks


def _parse_afterstate_lines(task: dict[str, Any], lines: list[str]) -> None:
    latest = task["latest"]
    history = task["history"]
    for line in lines:
        match = _AFTERSTATE_PROGRESS.search(line)
        if match is None:
            continue
        epoch = int(match["epoch"])
        step = int(match["step"])
        dps = float(match["dps"].replace(",", ""))
        latest.update(
            {
                "train/epoch": epoch,
                "train/step": step,
                "perf/dps": dps,
                "train/loss": float(match["loss"]),
            }
        )
        history.setdefault("perf/dps", []).append([step, dps])
        history.setdefault("train/loss", []).append([step, float(match["loss"])])
        for part in _SCALAR_PART.finditer(match["parts"]):
            latest[f"train/{part['name']}"] = float(part["value"])
    joined = "\n".join(lines)
    for match in _VALIDATION_SCALAR.finditer(joined):
        name = match["name"].removeprefix("validation_")
        latest[f"validation/{name}"] = float(match["value"])


def _parse_corpus_lines(task: dict[str, Any], lines: list[str]) -> None:
    latest = task["latest"]
    history = task["history"]
    for line in lines:
        match = _CORPUS_PROGRESS.search(line)
        if match is None:
            continue
        scanned = int(match["scanned"].replace(",", ""))
        rate = float(match["rate"].replace(",", ""))
        latest.update(
            {
                "global_step": scanned,
                "perf/dps": rate,
                "corpus/sampled": int(match["sampled"].replace(",", "")),
                "corpus/kept": int(match["kept"].replace(",", "")),
                "corpus/total": 54_873_706,
            }
        )
        history.setdefault("perf/dps", []).append([scanned, rate])


def utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_entry_boosts(
    agents: tuple[Agent, ...],
    agent_games: dict[str, int],
    superiority: dict[str, dict[str, float]],
) -> tuple[float, list[dict[str, Any]]]:
    """Return bounded boosts for entrants that have not settled yet."""

    multiplier = 1.0
    factors: list[dict[str, Any]] = []
    for agent in agents:
        config = agent.metadata.get("scheduler_boost")
        if not isinstance(config, dict):
            continue
        cap = int(config.get("max_games", DEFAULT_SCHEDULER_BOOST["max_games"]))
        target = float(config.get("los_target", DEFAULT_SCHEDULER_BOOST["los_target"]))
        games = agent_games.get(agent.id, 0)
        parent_los = (
            None if agent.parent_id is None else superiority.get(agent.id, {}).get(agent.parent_id)
        )
        los_resolved = parent_los is not None and max(parent_los, 1.0 - parent_los) >= target
        if games >= cap or los_resolved:
            continue
        boost = float(config.get("multiplier", DEFAULT_SCHEDULER_BOOST["multiplier"]))
        # Multiple unsettled entrants in the same matchup share one bounded
        # boost. Compounding two 6x boosts into 36x over-prioritizes a weakly
        # identified new-vs-new edge instead of connecting each entrant to the
        # established graph.
        multiplier = max(multiplier, boost)
        factors.append({"label": "new entrant", "factor": boost})
    return multiplier, factors


def matchup_schedule(store: ArenaStore, agents: list[Agent]) -> list[dict[str, Any]]:
    """Return the scheduler's current weighted matchup distribution.

    Information gain is the only ordinary matchup signal. Temperature flattens
    the distribution slightly, while bounded metadata on a newly registered
    entrant supplies a short initial boost. Status labels do not affect mature
    pairings.
    """

    information = store.matchup_information()
    counts = store.matchup_counts()
    superiority = store.matchup_superiority()
    agent_games: dict[str, int] = {agent.id: 0 for agent in agents}
    for (a_id, b_id), games in counts.items():
        if a_id in agent_games:
            agent_games[a_id] += games
        if b_id in agent_games:
            agent_games[b_id] += games
    if information:
        maximum_information = max(information.values(), default=1e-6)
        scheduled: list[dict[str, Any]] = []
        for i, a in enumerate(agents):
            for b in agents[i + 1 :]:
                key = tuple(sorted((a.id, b.id)))
                games = counts.get(key, 0)
                # A missing posterior edge means a newly registered entrant;
                # give it an intentionally high provisional information value
                # until the roster-change HMC fit publishes.
                gain = information.get(key, maximum_information * 2.0 / (1.0 + games))
                normalized_gain = max(gain / maximum_information, 1e-4)
                weight = normalized_gain ** (1.0 / SCHEDULER_TEMPERATURE)
                factors: list[dict[str, Any]] = [
                    {
                        "label": "temperature",
                        "factor": SCHEDULER_TEMPERATURE,
                        "display_only": True,
                    }
                ]
                # Search games are useful but materially more expensive; use
                # information per approximate compute cost rather than count.
                if a.mode != "plain" or b.mode != "plain":
                    weight *= SEARCH_COMPUTE_FACTOR
                    factors.append({"label": "search cost", "factor": SEARCH_COMPUTE_FACTOR})
                boost, boost_factors = _new_entry_boosts((a, b), agent_games, superiority)
                weight *= boost
                factors.extend(boost_factors)
                scheduled.append(
                    {
                        "a": a,
                        "b": b,
                        "games": games,
                        "information_gain": gain,
                        "weight": weight,
                        "factors": factors,
                        "posterior": key in information,
                    }
                )
        if not scheduled:
            raise RuntimeError("arena needs at least two active agents")
        total = sum(item["weight"] for item in scheduled)
        for item in scheduled:
            information_share = item["weight"] / total
            item["selection_probability"] = (
                1.0 - SCHEDULER_COVERAGE_MIX
            ) * information_share + SCHEDULER_COVERAGE_MIX / len(scheduled)
            item["weight"] = item["selection_probability"]
            item["factors"].append(
                {
                    "label": "coverage floor",
                    "factor": SCHEDULER_COVERAGE_MIX,
                    "display_only": True,
                }
            )
        scheduled.sort(key=lambda item: item["weight"], reverse=True)
        return scheduled

    # Bootstrap fallback before the first posterior exists.
    scheduled = []
    for i, a in enumerate(agents):
        for b in agents[i + 1 :]:
            n = counts.get(tuple(sorted((a.id, b.id))), 0)
            priority = 1.0 / (1.0 + n)
            # Establish a connected comparison graph before spending heavily on
            # repeated gates or expensive search-vs-search matches.
            if n == 0:
                priority += 1_000.0
            if n and (a.mode != "plain" or b.mode != "plain"):
                priority *= SEARCH_COMPUTE_FACTOR
            boost, boost_factors = _new_entry_boosts((a, b), agent_games, superiority)
            priority *= boost
            scheduled.append(
                {
                    "a": a,
                    "b": b,
                    "games": n,
                    "information_gain": None,
                    "weight": priority,
                    "factors": [{"label": "posterior pending", "factor": None}, *boost_factors],
                    "posterior": False,
                }
            )
    if not scheduled:
        raise RuntimeError("arena needs at least two active agents")
    total = sum(item["weight"] for item in scheduled)
    for item in scheduled:
        information_share = item["weight"] / total
        item["selection_probability"] = (
            1.0 - SCHEDULER_COVERAGE_MIX
        ) * information_share + SCHEDULER_COVERAGE_MIX / len(scheduled)
        item["weight"] = item["selection_probability"]
    scheduled.sort(key=lambda item: item["weight"], reverse=True)
    return scheduled


def pair_priority(store: ArenaStore, agents: list[Agent]) -> tuple[Agent, Agent]:
    """Sample matchups by expected posterior information gain per unit cost."""

    schedule = matchup_schedule(store, agents)
    if schedule[0]["information_gain"] is None:
        return schedule[0]["a"], schedule[0]["b"]
    selected = random.choices(schedule, weights=[item["weight"] for item in schedule], k=1)[0]
    return selected["a"], selected["b"]


def eligible_agents(store: ArenaStore) -> list[Agent]:
    active = store.agents(("candidate", "champion", "provisional", "lineage", "anchor"))
    focused = [agent for agent in active if agent.metadata.get("scheduler_focus") is True]
    return focused or active


def scheduler_snapshot(store: ArenaStore) -> dict[str, Any]:
    """Serialize the most useful portion of the live worker schedule."""

    agents = eligible_agents(store)
    if len(agents) < 2:
        return {"mode": "waiting", "matchups": []}
    schedule = matchup_schedule(store, agents)
    posterior = schedule[0]["information_gain"] is not None
    return {
        "mode": "bayesian_information" if posterior else "bootstrap",
        "rating_pending_games": store.rating_backlog(),
        "temperature": SCHEDULER_TEMPERATURE,
        "coverage_mix": SCHEDULER_COVERAGE_MIX,
        "new_entry_boost": dict(DEFAULT_SCHEDULER_BOOST),
        "eligible_agents": len(agents),
        "eligible_pairs": len(schedule),
        "matchups": [
            {
                "a": item["a"].id,
                "b": item["b"].id,
                "games": item["games"],
                "information_bits": (
                    None
                    if item["information_gain"] is None
                    else item["information_gain"] / math.log(2.0)
                ),
                "selection_probability": item["selection_probability"],
                "factors": item["factors"],
            }
            for item in schedule[:12]
        ],
    }


def maybe_promote(
    store: ArenaStore,
    candidate: Agent,
    champion: Agent,
    *,
    elo0: float,
    elo1: float,
    alpha: float,
    beta: float,
    max_games: int,
) -> str | None:
    scores = store.matchup_games(candidate.id, champion.id)
    wins = sum(score == 1 for score in scores)
    draws = sum(score == 0.5 for score in scores)
    losses = sum(score == 0 for score in scores)
    llr = sprt_llr(wins, draws, losses, elo0, elo1)
    lower, upper = sprt_bounds(alpha, beta)
    detail = {
        "champion": champion.id,
        "games": len(scores),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "llr": llr,
        "bounds": [lower, upper],
        "elo": [elo0, elo1],
    }
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
    runner = VsMatchRunner(
        level=args.level,
        speed_setting=args.speed_setting,
        num_pairs=args.batch,
        device=args.device,
        threads=args.threads,
        run_seed=args.seed,
        state_repr=args.state_repr,
        replay_sample_rate=args.replay_sample_rate,
        max_decisions_per_side=args.max_decisions_per_side,
    )
    worker_id = f"{runner.device}-{os.getpid()}"
    try:
        while not stopped:
            agents = eligible_agents(store)
            if len(agents) < 2:
                time.sleep(args.poll)
                continue
            a, b = pair_priority(store, agents)
            serial = store.reserve_serials(args.batch)
            wire_specs = paired_specs(
                schedule_seed=args.seed,
                agent_a=a.id,
                agent_b=b.id,
                start=serial,
                count=args.batch,
                level=args.level,
                speed_setting=args.speed_setting,
                state_repr=args.state_repr,
                max_decisions_per_side=args.max_decisions_per_side,
                policy_run_seed=args.seed,
                used=store.pair_seed_assignments(a.id, b.id),
            )
            specs = [game_spec_from_wire(spec) for spec in wire_specs]
            match_ids = [str(spec["match_id"]) for spec in wire_specs]
            batch_started = time.perf_counter()
            batch_games = 0
            batch_frames = 0
            batch_decisions = 0
            results = runner.play(a.entry(), b.entry(), specs)
            for match_id, result in zip(match_ids, results, strict=True):
                inserted = store.record(
                    a.id,
                    b.id,
                    seed=result.spec.seed,
                    side=result.spec.a_side,
                    winner=result.winner,
                    match_len_sec=result.match_len_sec,
                    decisions=result.decisions,
                    terminal_reason=result.terminal_reason,
                    replay=result.replay,
                    match_key=match_id,
                    game_index=result.spec.game_idx,
                    frame_counter_base=result.spec.frame_counter_base,
                    level=result.spec.level,
                    speed_setting=result.spec.speed_setting,
                    state_repr=result.spec.state_repr,
                    max_decisions_per_side=result.spec.max_decisions_per_side,
                    provenance={
                        "protocol_version": PROTOCOL_VERSION,
                        "worker_id": worker_id,
                        "device": runner.device,
                    },
                )
                if not inserted:
                    raise RuntimeError(f"arena match was not committed: {match_id}")
                batch_games += 1
                batch_frames += int(result.frames or round(result.match_len_sec * NES_FPS))
                batch_decisions += int(result.decisions)
            results.close()
            store.record_worker_sample(
                worker_id=worker_id,
                device=runner.device,
                threads=args.threads,
                batch_size=args.batch,
                agent_a=a.id,
                agent_b=b.id,
                games=batch_games,
                simulated_frames=batch_frames,
                decisions=batch_decisions,
                wall_seconds=time.perf_counter() - batch_started,
            )
            champion = next((x for x in store.agents(("champion",)) if x.family == a.family), None)
            for candidate in (a, b):
                if (
                    candidate.status == "candidate"
                    and champion
                    and candidate.family == champion.family
                ):
                    maybe_promote(
                        store,
                        candidate,
                        champion,
                        elo0=args.elo0,
                        elo1=args.elo1,
                        alpha=args.alpha,
                        beta=args.beta,
                        max_games=args.max_gate_games,
                    )
    finally:
        runner.close()
        store.close()


def run_remote_worker(args: argparse.Namespace) -> None:
    token = Path(args.token_file).expanduser().read_text().strip()
    client = ArenaRemoteClient(
        args.coordinator,
        token,
        checkpoint_cache=args.checkpoint_cache,
        timeout=args.request_timeout,
    )
    client.capabilities()
    runner = VsMatchRunner(
        level=args.level,
        speed_setting=args.speed_setting,
        num_pairs=args.batch,
        device=args.device,
        threads=args.threads,
        run_seed=args.seed,
        state_repr=args.state_repr,
        replay_sample_rate=args.replay_sample_rate,
        max_decisions_per_side=args.max_decisions_per_side,
    )
    worker_id = args.worker_id or f"{socket.gethostname()}-{runner.device}-{os.getpid()}"
    stopped = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while not stopped:
            lease = client.lease(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "worker_id": worker_id,
                    "device": runner.device,
                    "threads": args.threads,
                    "batch_size": args.batch,
                    "arena_config": {
                        "level": args.level,
                        "speed_setting": args.speed_setting,
                        "state_repr": args.state_repr,
                        "max_decisions_per_side": args.max_decisions_per_side,
                        "policy_run_seed": args.seed,
                    },
                }
            )
            if lease is None:
                time.sleep(args.poll)
                continue
            renewal_stop = threading.Event()
            renewal_errors: list[Exception] = []

            def renew() -> None:
                interval = max(5.0, float(lease.get("ttl_seconds", 600)) / 3.0)
                while not renewal_stop.wait(interval):
                    try:
                        client.renew(str(lease["lease_id"]), str(lease["claim_token"]))
                    except Exception as error:
                        renewal_errors.append(error)
                        return

            renewal_thread = threading.Thread(target=renew, name="arena-lease-renewal", daemon=True)
            renewal_thread.start()
            try:
                entries = []
                for key in ("agent_a", "agent_b"):
                    wire = lease[key]
                    checkpoint = client.materialize_checkpoint(wire)
                    entries.append(
                        {
                            "name": wire["name"],
                            "checkpoint": str(checkpoint),
                            "mode": wire["mode"],
                            "params": wire["params"],
                        }
                    )
                specs = [game_spec_from_wire(spec) for spec in lease["specs"]]
                match_ids = [str(spec["match_id"]) for spec in lease["specs"]]
                started = time.perf_counter()
                played = runner.play(entries[0], entries[1], specs)
                try:
                    results = [
                        {
                            "match_id": match_id,
                            "winner": result.winner,
                            "match_len_sec": result.match_len_sec,
                            "decisions": result.decisions,
                            "terminal_reason": result.terminal_reason,
                            "frames": int(result.frames or round(result.match_len_sec * NES_FPS)),
                            "replay": result.replay,
                        }
                        for match_id, result in zip(match_ids, played, strict=True)
                    ]
                finally:
                    played.close()
            finally:
                renewal_stop.set()
                renewal_thread.join(timeout=5)
            if renewal_errors:
                print(f"worker: lease renewal warning: {renewal_errors[-1]}", flush=True)
            wall = time.perf_counter() - started
            submission = {
                "protocol_version": PROTOCOL_VERSION,
                "claim_token": lease["claim_token"],
                "results": results,
                "worker_sample": {
                    "worker_id": worker_id,
                    "device": runner.device,
                    "threads": args.threads,
                    "batch_size": args.batch,
                    "agent_a": lease["agent_a"]["id"],
                    "agent_b": lease["agent_b"]["id"],
                    "games": len(results),
                    "simulated_frames": sum(int(item["frames"]) for item in results),
                    "decisions": sum(int(item["decisions"]) for item in results),
                    "wall_seconds": wall,
                },
            }
            # A lost response is safe to retry because the coordinator hashes
            # the canonical submission and treats an identical replay as done.
            for attempt in range(5):
                try:
                    client.submit(str(lease["lease_id"]), submission)
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(min(2**attempt, 8))
    finally:
        runner.close()


def rating_config(args: argparse.Namespace, *, seed: int | None = None) -> RatingConfig:
    return RatingConfig(
        chains=args.rating_chains,
        warmup=args.rating_warmup,
        samples=args.rating_samples,
        seed=args.rating_seed if seed is None else seed,
    )


def rating_loop(args: argparse.Namespace, stopped: threading.Event | None = None) -> None:
    """Keep a converged posterior cache current without blocking match play."""

    stopped = stopped or threading.Event()
    retry_seed = args.rating_seed
    while not stopped.is_set():
        store = ArenaStore(args.db)
        try:
            needs_refresh = store.ratings_need_refresh(min_new_matches=args.rating_refresh_games)
            full_hmc = bool(getattr(args, "rating_full_hmc", False))
            if needs_refresh or (full_hmc and getattr(args, "once", False)):
                started = time.monotonic()
                config = rating_config(args, seed=retry_seed)
                result = (
                    store.refit_ratings(config)
                    if full_hmc
                    else store.update_ratings(
                        config,
                        min_new_matches=args.rating_refresh_games,
                        laplace_samples=args.rating_laplace_samples,
                    )
                )
                if result is not None:
                    retry_seed = args.rating_seed
                    diagnostics = result["diagnostics"]
                    quality = (
                        f"mode gradient {diagnostics['mode_gradient_max']:.2g}"
                        if result["method"] == "laplace"
                        else f"R-hat {diagnostics['max_rhat']:.3f}, "
                        f"ESS {diagnostics['min_ess']:.0f}"
                    )
                    print(
                        f"ratings: {result['method']} fit {result['id']} over "
                        f"{result['match_count']} games in "
                        f"{time.monotonic() - started:.3f}s; {quality}",
                        flush=True,
                    )
        except RatingConvergenceError as error:
            # Never publish a suspect posterior. The last converged fit remains
            # visible and the next polling interval retries from genuinely
            # fresh chains rather than deterministically repeating a bad draw.
            retry_seed += 1
            print(
                f"ratings: fit rejected: {error}; retry seed {retry_seed}",
                flush=True,
            )
        except Exception as error:
            print(f"ratings: update failed: {type(error).__name__}: {error}", flush=True)
        finally:
            store.close()
        if getattr(args, "once", False):
            return
        stopped.wait(args.rating_poll)


class DashboardHandler(BaseHTTPRequestHandler):
    db: Path
    replay_dir: Path | None = None
    worker_token: str | None = None
    lease_ttl: float = 600.0
    lease_seed: int = 0xA8E4
    arena_config: dict[str, Any] = {}
    lease_lock = threading.Lock()
    snapshot_lock = threading.Lock()
    snapshot_payload: bytes | None = None
    snapshot_created_monotonic: float = 0.0

    def _authorized(self) -> bool:
        expected = self.worker_token
        if expected is None:
            return False
        supplied = self.headers.get("Authorization", "")
        return secrets.compare_digest(supplied, f"Bearer {expected}")

    def _send_json(self, status: int, value: Any) -> None:
        payload = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_json(self, *, limit: int = 64 << 20) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > limit:
            raise ValueError("invalid request body size")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("request body must be an object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/v1/capabilities":
            self._send_json(
                200,
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "leases": True,
                    "checkpoint_delivery": True,
                    "idempotent_submission": True,
                    "external_replays": True,
                    "complete_reset_specs": True,
                    "paired_side_swaps": True,
                    "committed_insert_accounting": True,
                },
            )
            return
        if path.startswith("/api/v1/checkpoints/"):
            if not self._authorized():
                self._send_json(401, {"error": "unauthorized"})
                return
            agent_id = unquote(path.rsplit("/", 1)[1])
            store = ArenaStore(self.db, replay_dir=self.replay_dir)
            try:
                try:
                    checkpoint = Path(store.agent(agent_id).checkpoint)
                except KeyError:
                    self.send_error(404)
                    return
                payload = checkpoint.read_bytes()
            finally:
                store.close()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Content-SHA256", hashlib.sha256(payload).hexdigest())
            self.end_headers()
            self.wfile.write(payload)
            return
        if path == "/api/snapshot":
            handler = type(self)
            with handler.snapshot_lock:
                payload = handler.snapshot_payload
                created = handler.snapshot_created_monotonic
            if payload is None:
                self._send_json(503, {"error": "dashboard snapshot is warming up"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Snapshot-Age", f"{max(0.0, time.monotonic() - created):.3f}")
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
            store = ArenaStore(self.db, replay_dir=self.replay_dir)
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
        self.send_header(
            "Content-Type", mimetypes.guess_type(target)[0] or "application/octet-stream"
        )
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return
        try:
            request = self._read_json()
            if int(request.get("protocol_version", -1)) != PROTOCOL_VERSION:
                self._send_json(409, {"error": "protocol version mismatch"})
                return
            if path == "/api/v1/leases":
                self._lease(request)
                return
            renew_suffix = "/renew"
            if path.startswith("/api/v1/leases/") and path.endswith(renew_suffix):
                lease_id = unquote(path[len("/api/v1/leases/") : -len(renew_suffix)])
                self._renew(lease_id, request)
                return
            prefix, suffix = "/api/v1/leases/", "/results"
            if path.startswith(prefix) and path.endswith(suffix):
                lease_id = unquote(path[len(prefix) : -len(suffix)])
                self._submit(lease_id, request)
                return
            self.send_error(404)
        except (KeyError, TypeError, ValueError) as error:
            self._send_json(400, {"error": str(error)})
        except PermissionError as error:
            self._send_json(409, {"error": str(error)})
        except Exception as error:
            print(f"coordinator: {type(error).__name__}: {error}", flush=True)
            self._send_json(500, {"error": "internal coordinator error"})

    def _lease(self, request: dict[str, Any]) -> None:
        with self.lease_lock:
            self._lease_locked(request)

    def _lease_locked(self, request: dict[str, Any]) -> None:
        worker_id = str(request["worker_id"]).strip()
        batch_size = int(request["batch_size"])
        if not worker_id or not 2 <= batch_size <= 64 or batch_size % 2:
            raise ValueError("invalid worker_id or batch_size")
        requested_config = request.get("arena_config")
        if requested_config != self.arena_config:
            raise ValueError(
                f"worker arena_config {requested_config!r} does not match "
                f"coordinator {self.arena_config!r}"
            )
        now = time.time()
        store = ArenaStore(self.db, replay_dir=self.replay_dir)
        try:
            reclaimed = store.claim_expired_lease(
                worker_id=worker_id,
                now=now,
                ttl_seconds=self.lease_ttl,
                required_protocol_version=PROTOCOL_VERSION,
            )
            if reclaimed is not None:
                payload, _token = reclaimed
                self._send_json(200, payload)
                return
            agents = eligible_agents(store)
            if len(agents) < 2:
                self.send_response(204)
                self.end_headers()
                return
            a, b = pair_priority(store, agents)
            serial = store.reserve_serials(batch_size)
            used = store.pair_seed_assignments(a.id, b.id)
            specs = paired_specs(
                schedule_seed=self.lease_seed,
                agent_a=a.id,
                agent_b=b.id,
                start=serial,
                count=batch_size,
                used=used,
                **self.arena_config,
            )
            provenance = {
                "protocol_version": PROTOCOL_VERSION,
                "worker_id": worker_id,
                "device": str(request.get("device", "unknown")),
                "checkpoint_a_sha256": agent_wire(a)["checkpoint_sha256"],
                "checkpoint_b_sha256": agent_wire(b)["checkpoint_sha256"],
            }
            for spec in specs:
                spec["provenance"] = provenance
            payload = {
                "protocol_version": PROTOCOL_VERSION,
                "agent_a": agent_wire(a),
                "agent_b": agent_wire(b),
                "specs": specs,
                "ttl_seconds": self.lease_ttl,
            }
            lease = store.create_lease(
                lease_id=secrets.token_hex(16),
                worker_id=worker_id,
                agent_a=a.id,
                agent_b=b.id,
                payload=payload,
                now=now,
                ttl_seconds=self.lease_ttl,
            )
            self._send_json(200, lease)
        finally:
            store.close()

    def _submit(self, lease_id: str, request: dict[str, Any]) -> None:
        results = request["results"]
        sample = request["worker_sample"]
        if not isinstance(results, list) or not isinstance(sample, dict):
            raise ValueError("invalid result submission")
        digest = content_sha256({"results": results, "worker_sample": sample})
        store = ArenaStore(self.db, replay_dir=self.replay_dir)
        try:
            accepted = store.submit_lease(
                lease_id=lease_id,
                claim_token=str(request["claim_token"]),
                submission_sha256=digest,
                results=results,
                worker_sample=sample,
                now=time.time(),
            )
        finally:
            store.close()
        self._send_json(200, {"accepted": accepted, "submission_sha256": digest})

    def _renew(self, lease_id: str, request: dict[str, Any]) -> None:
        store = ArenaStore(self.db, replay_dir=self.replay_dir)
        try:
            expires = store.renew_lease(
                lease_id=lease_id,
                claim_token=str(request["claim_token"]),
                now=time.time(),
                ttl_seconds=self.lease_ttl,
            )
        finally:
            store.close()
        self._send_json(200, {"expires": expires, "ttl_seconds": self.lease_ttl})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"dashboard: {fmt % args}", flush=True)


def refresh_dashboard_snapshot(db: Path, replay_dir: Path | None) -> bytes:
    """Build one immutable dashboard response outside HTTP request threads."""

    store = ArenaStore(db, replay_dir=replay_dir)
    try:
        snapshot = store.snapshot()
        snapshot["scheduler"] = scheduler_snapshot(store)
        return json.dumps(snapshot, separators=(",", ":")).encode()
    finally:
        store.close()


def dashboard_snapshot_loop(
    db: Path,
    replay_dir: Path | None,
    *,
    interval: float,
    stopped: threading.Event,
) -> None:
    """Refresh the materialized dashboard response without blocking serving."""

    handler = DashboardHandler
    while not stopped.is_set():
        started = time.monotonic()
        try:
            payload = refresh_dashboard_snapshot(db, replay_dir)
            with handler.snapshot_lock:
                handler.snapshot_payload = payload
                handler.snapshot_created_monotonic = time.monotonic()
        except Exception as error:
            # Keep serving the last complete snapshot.  A transient SQLite
            # read failure must not create request storms or affect workers.
            print(f"dashboard snapshot: {type(error).__name__}: {error}", flush=True)
        stopped.wait(max(0.0, interval - (time.monotonic() - started)))


def serve(args: argparse.Namespace) -> None:
    DashboardHandler.db = Path(args.db)
    DashboardHandler.replay_dir = None if args.replay_dir is None else Path(args.replay_dir)
    DashboardHandler.lease_ttl = float(args.lease_ttl)
    DashboardHandler.lease_seed = int(args.lease_seed)
    DashboardHandler.arena_config = {
        "level": int(args.level),
        "speed_setting": int(args.speed_setting),
        "state_repr": str(args.state_repr),
        "max_decisions_per_side": int(args.max_decisions_per_side),
        "policy_run_seed": int(args.policy_run_seed),
    }
    DashboardHandler.worker_token = (
        None
        if args.worker_token_file is None
        else Path(args.worker_token_file).expanduser().read_text().strip()
    )
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not DashboardHandler.worker_token:
        raise ValueError("a worker token file is required when serving beyond loopback")
    if args.snapshot_refresh <= 0:
        raise ValueError("snapshot refresh interval must be positive")
    DashboardHandler.snapshot_payload = refresh_dashboard_snapshot(
        DashboardHandler.db, DashboardHandler.replay_dir
    )
    DashboardHandler.snapshot_created_monotonic = time.monotonic()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    rating_stop = threading.Event()
    snapshot_stop = threading.Event()
    snapshot_thread = threading.Thread(
        target=dashboard_snapshot_loop,
        args=(DashboardHandler.db, DashboardHandler.replay_dir),
        kwargs={"interval": float(args.snapshot_refresh), "stopped": snapshot_stop},
        name="arena-dashboard-snapshot",
        daemon=True,
    )
    snapshot_thread.start()
    rating_thread = None
    if args.ratings:
        rating_thread = threading.Thread(
            target=rating_loop, args=(args, rating_stop), name="arena-ratings", daemon=True
        )
        rating_thread.start()
    print(f"arena dashboard: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        rating_stop.set()
        snapshot_stop.set()
        if rating_thread is not None:
            rating_thread.join(timeout=5)
        snapshot_thread.join(timeout=5)


def add_rating_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rating-chains", type=int, default=4)
    parser.add_argument("--rating-warmup", type=int, default=800)
    parser.add_argument("--rating-samples", type=int, default=1_200)
    parser.add_argument("--rating-seed", type=int, default=0xD0C70A11)
    parser.add_argument("--rating-refresh-games", type=int, default=128)
    parser.add_argument("--rating-laplace-samples", type=int, default=4_096)
    parser.add_argument("--rating-poll", type=float, default=5.0)


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
    externalize = sub.add_parser("externalize-replays")
    externalize.add_argument("--replay-dir", required=True)
    externalize.add_argument("--batch", type=int, default=1000)
    externalize.add_argument("--vacuum", action="store_true")
    web = sub.add_parser("serve")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8097)
    web.add_argument("--worker-token-file")
    web.add_argument(
        "--snapshot-refresh",
        type=float,
        default=5.0,
        help="seconds between materialized dashboard snapshots",
    )
    web.add_argument("--lease-ttl", type=float, default=600)
    web.add_argument("--lease-seed", type=int, default=0xA8E4)
    web.add_argument("--replay-dir")
    web.add_argument("--level", type=int, default=14)
    web.add_argument("--speed-setting", type=int, default=2)
    web.add_argument("--state-repr", default="bitplane_bottle_conn_mask_vs")
    web.add_argument("--max-decisions-per-side", type=int, default=ARENA_MAX_DECISIONS_PER_SIDE)
    web.add_argument("--policy-run-seed", type=int, default=27182)
    web.add_argument("--ratings", action=argparse.BooleanOptionalAction, default=True)
    add_rating_arguments(web)
    rate = sub.add_parser("rate")
    rate.add_argument("--once", action="store_true")
    rate.add_argument(
        "--rating-full-hmc",
        action="store_true",
        help="run an explicit offline multi-chain HMC audit",
    )
    add_rating_arguments(rate)
    focus = sub.add_parser("focus")
    focus.add_argument("agents", nargs="*", help="agent IDs; omit to clear focus")
    status = sub.add_parser("status")
    status.add_argument("agent")
    status.add_argument("status")
    status.add_argument("--reason", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--device", default="cuda")
    worker.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    worker.add_argument("--batch", type=int, default=8)
    worker.add_argument("--level", type=int, default=14)
    worker.add_argument("--speed-setting", type=int, default=2)
    worker.add_argument("--state-repr", default="bitplane_bottle_conn_mask_vs")
    worker.add_argument("--seed", type=int, default=27182)
    worker.add_argument("--poll", type=float, default=10)
    worker.add_argument("--elo0", type=float, default=0)
    worker.add_argument("--elo1", type=float, default=10)
    worker.add_argument("--alpha", type=float, default=0.05)
    worker.add_argument("--beta", type=float, default=0.05)
    worker.add_argument("--max-gate-games", type=int, default=400)
    worker.add_argument("--replay-sample-rate", type=float, default=0.2)
    worker.add_argument(
        "--max-decisions-per-side",
        type=int,
        default=ARENA_MAX_DECISIONS_PER_SIDE,
        help="adjudicate unresolved games as draws at this placement horizon (0 disables)",
    )
    worker.add_argument("--coordinator", help="remote arena coordinator base URL")
    worker.add_argument("--token-file", help="shared worker token file for remote mode")
    worker.add_argument(
        "--checkpoint-cache",
        default="~/.cache/drmc-rl/arena-checkpoints",
        help="content-addressed checkpoint cache used by remote workers",
    )
    worker.add_argument("--request-timeout", type=float, default=60)
    worker.add_argument("--worker-id")
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
    elif args.command == "externalize-replays":
        run_externalize_replays(args)
    elif args.command == "serve":
        serve(args)
    elif args.command == "rate":
        rating_loop(args)
    elif args.command == "focus":
        store = ArenaStore(args.db)
        try:
            store.set_scheduler_focus(args.agents)
        finally:
            store.close()
    elif args.command == "status":
        store = ArenaStore(args.db)
        try:
            store.set_status(args.agent, args.status, reason=args.reason)
        finally:
            store.close()
    else:
        if args.coordinator:
            if not args.token_file:
                parser.error("worker --coordinator requires --token-file")
            run_remote_worker(args)
        else:
            run_worker(args)


if __name__ == "__main__":
    main()
