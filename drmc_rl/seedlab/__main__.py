"""seedlab CLI: exhaustive per-seed clear-time catalog (docs/SEED_CATALOG.md).

Examples:
    python -m drmc_rl.seedlab init --levels 0-20 --speed 2 --chunk 512
    python -m drmc_rl.seedlab worker --policy checkpoint \
        --checkpoint runs/best_agents/smdp_ppo_step535164979.pt.gz \
        --device mps --num-envs 32 --attempts-per-seed 4
    python -m drmc_rl.seedlab report --speed 2
    python -m drmc_rl.seedlab grade --level 14 --frames 4200 --seed 8988
    python -m drmc_rl.seedlab verify --level 14 --seed 8988
    python -m drmc_rl.seedlab tui
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

from drmc_rl.seedlab import rng as slrng
from drmc_rl.seedlab.db import CatalogDB


def _parse_levels(spec: str) -> List[int]:
    out: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _parse_seed(text: str) -> int:
    return int(text, 16) if any(c in text.lower() for c in "abcdef") or len(text) == 4 else int(text)


def _open_db(args) -> CatalogDB:
    return CatalogDB(Path(args.db).expanduser() if args.db else CatalogDB.default_path())


def cmd_init(args) -> None:
    db = _open_db(args)
    levels = _parse_levels(args.levels)
    try:
        if not args.skip_census:
            from drmc_rl.seedlab.census import run_census

            run_census(db, levels=levels)
        for level in levels:
            n = db.enqueue_units(
                level=level, speed=args.speed, pass_idx=args.pass_idx,
                total_seeds=slrng.ORBIT_PERIOD, chunk=args.chunk,
            )
            if n:
                print(f"[init] level {level}: enqueued {n} units (pass {args.pass_idx})")
        print(f"[init] queue: {db.unit_counts()}")
    finally:
        db.close()


def cmd_worker(args) -> None:
    from drmc_rl.seedlab.worker import CatalogWorker

    db = _open_db(args)
    try:
        stale_cutoff = (
            datetime.now(timezone.utc) - timedelta(minutes=args.lease_ttl_min)
        ).isoformat(timespec="seconds")
        reclaimed = db.reclaim_stale_leases(older_than_iso=stale_cutoff)
        if reclaimed:
            print(f"[seedlab] reclaimed {reclaimed} stale leases")

        worker_id = args.worker_id or f"{socket.gethostname()}:{os.getpid()}"
        worker = CatalogWorker(
            db=db,
            worker_id=worker_id,
            policy=args.policy,
            checkpoint=args.checkpoint,
            device=args.device,
            temperature=args.temperature,
            attempts_per_seed=args.attempts_per_seed,
            num_envs=args.num_envs,
            state_repr=args.state_repr,
            max_decisions=args.max_decisions,
            levels=_parse_levels(args.levels) if args.levels else None,
            seed=args.seed,
            max_units=args.max_units,
        )
        worker.install_signal_handlers()
        print(f"[seedlab] worker {worker_id} starting (policy={worker.solver.solver_id})")
        worker.run()
        print(
            f"[seedlab] worker done: attempts={worker.total_attempts} "
            f"clears={worker.total_clears} new_bests={worker.total_new_bests}"
        )
    finally:
        db.close()


def cmd_report(args) -> None:
    from drmc_rl.seedlab.report import print_report

    db = _open_db(args)
    try:
        print_report(
            db, speed=args.speed,
            levels=_parse_levels(args.levels) if args.levels else None,
            top=args.top,
        )
    finally:
        db.close()


def cmd_grade(args) -> None:
    from drmc_rl.seedlab.report import grade

    db = _open_db(args)
    try:
        result = grade(
            db, level=args.level, speed=args.speed, frames=args.frames,
            seed=_parse_seed(args.seed) if args.seed else None,
        )
        print(json.dumps(result, indent=2))
    finally:
        db.close()


def cmd_verify(args) -> None:
    from drmc_rl.seedlab.verify import verify_and_mark

    db = _open_db(args)
    try:
        if args.all:
            cur = db._conn.execute(
                "SELECT level, speed, seed FROM solutions WHERE verified=0 ORDER BY level, seed;"
            )
            targets = [(int(l), int(sp), int(se)) for l, sp, se in cur.fetchall()]
        else:
            if args.level is None or args.seed is None:
                raise SystemExit("--level and --seed required (or use --all)")
            targets = [(args.level, args.speed, _parse_seed(args.seed))]
        n_ok = 0
        for level, speed, seed in targets:
            ok, msg = verify_and_mark(db, level=level, speed=speed, seed=seed)
            n_ok += int(ok)
            print(f"[verify] level={level} speed={speed} seed={seed:04x}: {msg}")
        print(f"[verify] {n_ok}/{len(targets)} ok")
        if targets and n_ok < len(targets):
            raise SystemExit(1)
    finally:
        db.close()


def cmd_explore(args) -> None:
    # Thread discipline: explorer coexists with training/interactive use.
    # The pool gets the full thread budget; torch stays single-threaded —
    # measured 2026-06-11: CPU conv2d at beam batch sizes gets SLOWER with
    # more torch threads, and wide-beam forwards run on MPS when --device mps.
    os.environ.setdefault("DRMARIO_POOL_WORKERS", str(max(1, args.threads)))
    import torch

    torch.set_num_threads(1)

    from drmc_rl.seedlab.explore import Explorer

    db = _open_db(args)
    try:
        explorer = Explorer(
            db=db,
            levels=_parse_levels(args.levels),
            speed=args.speed,
            checkpoint=args.checkpoint,
            device=args.device,
            num_envs=args.num_envs,
            exact_max_level=args.exact_max_level,
            seed=args.seed,
            priority_level=None if args.priority_level < 0 else args.priority_level,
        )
        explorer.install_signal_handlers()
        try:
            if args.bench:
                _run_bench(explorer)
            else:
                explorer.run(
                    iterations=args.iterations,
                    duration_sec=None if args.duration_min is None else args.duration_min * 60.0,
                )
        finally:
            explorer.close()
    finally:
        db.close()


def _run_bench(explorer) -> None:
    """Fixed benchmark suite; prints a table for docs/SEEDLAB_SEARCH.md."""

    from drmc_rl.seedlab.search import beam_search

    print(f"{'level':>5} {'width':>5} {'frames':>7} {'nodes':>7} {'wall_s':>7} {'nodes/s':>8}")
    for level in (0, 7, 14):
        for width in (8, 32):
            res = beam_search(
                explorer.beam_engine,
                level=level, speed=explorer.speed, seed=0x8988,
                width=width, top_m=8,
                solver=explorer.solver if explorer.solver.policy == "checkpoint" else None,
                lambda_frames=explorer._lambda_for(level),
            )
            print(
                f"{level:>5} {width:>5} {str(res.frames):>7} {res.nodes:>7} "
                f"{res.wall_sec:>7.1f} {res.nodes / max(res.wall_sec, 1e-9):>8.0f}"
            )


def cmd_tui(args) -> None:
    from drmc_rl.seedlab.dashboard import run_dashboard

    run_dashboard(
        Path(args.db).expanduser() if args.db else CatalogDB.default_path(),
        speed=args.speed, refresh=args.refresh,
    )


def main() -> None:
    ap = argparse.ArgumentParser(prog="seedlab", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=str, default=None,
                    help="catalog DB path (default: $DRMARIO_SEED_CATALOG_DB or data/seed_catalog.sqlite3)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="run the census and enqueue work units")
    p.add_argument("--levels", type=str, default="0-20")
    p.add_argument("--speed", type=int, default=2)
    p.add_argument("--pass-idx", type=int, default=0)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--skip-census", action="store_true")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("worker", help="run a search worker until the queue drains")
    p.add_argument("--policy", choices=["checkpoint", "greedy-cost", "random"],
                   default="greedy-cost")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--attempts-per-seed", type=int, default=1)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--state-repr", type=str, default="bitplane_bottle_conn_mask")
    p.add_argument("--max-decisions", type=int, default=600)
    p.add_argument("--levels", type=str, default=None,
                   help="restrict to these levels (e.g. 0-5,20)")
    p.add_argument("--worker-id", type=str, default=None)
    p.add_argument("--lease-ttl-min", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-units", type=int, default=None,
                   help="stop after completing this many units")
    p.set_defaults(fn=cmd_worker)

    p = sub.add_parser("explore", help="jagged random-depth search across seeds")
    p.add_argument("--levels", type=str, default="0-20")
    p.add_argument("--speed", type=int, default=2)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="policy checkpoint (strongly recommended; guides the search)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--duration-min", type=float, default=None)
    p.add_argument("--threads", type=int, default=2,
                   help="total thread budget (pool workers + torch)")
    p.add_argument("--exact-max-level", type=int, default=0,
                   help="exact B&B ceiling; benchmarked 2026-06-11: cannot close "
                        "level >=1 within budget until the step bound improves")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--priority-level", type=int, default=4,
                   help="anchor width-first build-up here; lower levels get "
                        "random backfill only (negative disables)")
    p.add_argument("--bench", action="store_true", help="run the benchmark suite and exit")
    p.set_defaults(fn=cmd_explore)

    p = sub.add_parser("report", help="print coverage and best-time tables")
    p.add_argument("--speed", type=int, default=2)
    p.add_argument("--levels", type=str, default=None)
    p.add_argument("--top", type=int, default=5)
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("grade", help="percentile of a clear time vs the catalog")
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--speed", type=int, default=2)
    p.add_argument("--frames", type=int, required=True)
    p.add_argument("--seed", type=str, default=None, help="hex seed (e.g. 8988)")
    p.set_defaults(fn=cmd_grade)

    p = sub.add_parser("verify", help="replay stored best solutions and check frames")
    p.add_argument("--level", type=int, default=None)
    p.add_argument("--speed", type=int, default=2)
    p.add_argument("--seed", type=str, default=None, help="hex seed (e.g. 8988)")
    p.add_argument("--all", action="store_true", help="verify all unverified solutions")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("tui", help="live dashboard")
    p.add_argument("--speed", type=int, default=2)
    p.add_argument("--refresh", type=float, default=5.0)
    p.set_defaults(fn=cmd_tui)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
