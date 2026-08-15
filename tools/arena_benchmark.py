"""Fixed-workload arena throughput benchmark.

This deliberately reports simulated frames and decisions per wall second as
the primary rates. Games per minute is retained only as workload context: it
changes substantially with the selected agents, seeds, and terminal outcomes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from drmc_rl.arena.store import ArenaStore
from tools.arena import DEFAULT_DB
from tools.tournament import GameSpec, VsMatchRunner, make_specs, pick_state_repr


def _play(runner: VsMatchRunner, a: dict[str, Any], b: dict[str, Any], specs: list[GameSpec]) -> dict[str, int]:
    games = frames = decisions = 0
    results = runner.play(a, b, specs)
    try:
        for result in results:
            games += 1
            frames += result.frames
            decisions += result.decisions
    finally:
        results.close()
    return {"games": games, "frames": frames, "decisions": decisions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--agent-a", default="outcome-hardening-g4-1p4b")
    parser.add_argument("--agent-b", default="outcome-hardening-g4-1p5b")
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--warmup-games", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8675309)
    parser.add_argument("--max-decisions-per-side", type=int, default=500)
    args = parser.parse_args()

    store = ArenaStore(args.db)
    try:
        a = store.agent(args.agent_a).entry()
        b = store.agent(args.agent_b).entry()
    finally:
        store.close()
    state_repr = pick_state_repr((a, b))
    runner = VsMatchRunner(
        level=14,
        speed_setting=2,
        num_pairs=args.batch,
        device=args.device,
        threads=args.threads,
        run_seed=args.seed,
        state_repr=state_repr,
        replay_sample_rate=0.0,
        max_decisions_per_side=args.max_decisions_per_side,
    )
    try:
        warmup = make_specs(args.seed, a["name"], b["name"], range(-args.warmup_games, 0))
        _play(runner, a, b, warmup)
        specs = make_specs(args.seed, a["name"], b["name"], range(args.games))
        started = time.perf_counter()
        totals = _play(runner, a, b, specs)
        wall = time.perf_counter() - started
    finally:
        runner.close()

    print(json.dumps({
        "device": args.device,
        "threads": args.threads,
        "batch": args.batch,
        "wall_seconds": wall,
        **totals,
        "frames_per_second": totals["frames"] / wall,
        "frames_per_minute": 60.0 * totals["frames"] / wall,
        "decisions_per_second": totals["decisions"] / wall,
        "games_per_minute": 60.0 * totals["games"] / wall,
        "frames_per_game": totals["frames"] / totals["games"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
