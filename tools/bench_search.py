"""Bench SearchPolicy.decide() latency on real mid-game states.

Plays 1P pool episodes with the search's own decisions (the states it would
actually visit) and reports decide() latency percentiles plus the anytime
stage histogram. Target: p95 <= 80 ms at beam 8.

``--ponder`` benches `PonderingSearchPolicy` instead: after every decision
the next-decision ponder job runs to completion (simulating dead time, as in
the live bridge between spawns), and the report splits decide() latency by
cache hit/miss plus the ponder job wall-time percentiles. Targets: hit
latency < 5 ms, job wall p95 < 1.0 s.

Usage:
  .venv/bin/python -m tools.bench_search --checkpoint runs/vs2_02/checkpoints/<ck>.pt.gz \
      [--level 14] [--decisions 200] [--beam 8] [--deadline-ms 60] [--device auto] [--ponder]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from drmc_rl.envs.backends.drmario_pool import DrMarioPoolRunner, build_reset_spec
from drmc_rl.models.policy.search_policy import PonderingSearchPolicy, SearchPolicy

_CANON_TO_RAW = (1, 0, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--level", type=int, default=14)
    ap.add_argument("--speed-setting", type=int, default=2)
    ap.add_argument("--decisions", type=int, default=200)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--deadline-ms", type=float, default=60.0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--ponder", action="store_true",
                    help="bench PonderingSearchPolicy (job wall time + hit/miss latency)")
    ap.add_argument("--ponder-budget-s", type=float, default=1.0)
    args = ap.parse_args()

    torch.set_num_threads(int(args.threads))
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    cls = PonderingSearchPolicy if args.ponder else SearchPolicy
    kw = {"ponder_budget_s": float(args.ponder_budget_s)} if args.ponder else {}
    sp = cls(
        args.checkpoint,
        beam=int(args.beam),
        deadline_ms=float(args.deadline_ms),
        device=device,
        seed=int(args.seed),
        **kw,
    )
    env = DrMarioPoolRunner(num_envs=1, obs_spec=4, obs_channels=12, emit_board=True)
    rng = np.random.default_rng(int(args.seed))

    def reset() -> None:
        env.reset(
            None,
            [
                build_reset_spec(
                    level=int(args.level),
                    speed_setting=int(args.speed_setting),
                    rng_state=(int(rng.integers(256)), int(rng.integers(256))),
                    rng_override=True,
                )
            ],
        )

    reset()
    v0 = int(env.buffers.viruses_rem[0])
    lat: list[float] = []
    lat_hit: list[float] = []
    lat_miss: list[float] = []
    stages: dict[str, int] = {}
    agreed = 0
    frames = 0
    decisions = 0
    while len(lat) < int(args.decisions):
        mask = env.buffers.feasible_mask[0].copy()
        if not mask.any():
            reset()
            v0 = int(env.buffers.viruses_rem[0])
            frames = decisions = 0
            continue
        pill = env.buffers.pill_colors[0]
        prev = env.buffers.preview_colors[0]
        pill_raw = (_CANON_TO_RAW[int(pill[0])], _CANON_TO_RAW[int(pill[1])])
        prev_raw = (_CANON_TO_RAW[int(prev[0])], _CANON_TO_RAW[int(prev[1])])
        board = env.buffers.board_bytes[0].copy()
        cost = env.buffers.cost_to_lock[0].copy()
        spd_ups = min(0x31, max(0, int(args.level) - 20) + decisions // 10)
        a, info = sp.decide(
            board,
            pill_raw,
            prev_raw,
            mask,
            cost,
            int(args.speed_setting),
            spd_ups,
            int(args.level),
            viruses_initial=v0,
            frames_used=frames,
        )
        lat.append(float(info["elapsed_ms"]))
        if args.ponder:
            (lat_hit if info.get("ponder_hit") else lat_miss).append(float(info["elapsed_ms"]))
        stages[info["stage"]] = stages.get(info["stage"], 0) + 1
        agreed += int(bool(info["agreed_with_policy"]))
        if args.ponder and a >= 0:
            sp.start_ponder(
                board,
                int(a),
                pill_raw,
                prev_raw,
                feasible_mask512=mask,
                cost_to_lock512=cost,
                speed_setting=int(args.speed_setting),
                speed_ups=spd_ups,
                level=int(args.level),
                viruses_initial=v0,
                frames_used=frames,
            )
            sp.wait_ponder_idle()  # simulate dead time >> ponder compute
        env.step(np.array([a], dtype=np.int32), None, None)
        frames += int(env.buffers.tau_frames[0])
        decisions += 1
        if env.buffers.terminated[0] or env.buffers.truncated[0]:
            reset()
            v0 = int(env.buffers.viruses_rem[0])
            frames = decisions = 0

    arr = np.asarray(lat)
    print(
        f"decide() latency over {arr.size} real decisions "
        f"(level {args.level}, beam {args.beam}, deadline {args.deadline_ms}ms, device {device}):\n"
        f"  p50={np.percentile(arr, 50):.1f}ms p95={np.percentile(arr, 95):.1f}ms "
        f"max={arr.max():.1f}ms\n"
        f"  stages={stages} agreement={agreed / arr.size:.2f}"
    )
    if args.ponder:
        def _pct(xs: list[float]) -> str:
            if not xs:
                return "n=0"
            a2 = np.asarray(xs)
            return (
                f"n={a2.size} p50={np.percentile(a2, 50):.2f}ms "
                f"p95={np.percentile(a2, 95):.2f}ms max={a2.max():.2f}ms"
            )

        stats = sp.ponder_stats()
        print(
            f"ponder: hit_rate={stats['hit_rate']} hits={stats['hits']} misses={stats['misses']} "
            f"aborts={stats['aborts']} jobs={stats['jobs_completed']} "
            f"pairs_depth2={stats['pairs_depth2']}/{stats['pairs_total']}\n"
            f"  job wall: p50={stats.get('job_wall_s_p50', float('nan')):.3f}s "
            f"p95={stats.get('job_wall_s_p95', float('nan')):.3f}s "
            f"max={stats.get('job_wall_s_max', float('nan')):.3f}s\n"
            f"  decide hit:  {_pct(lat_hit)}\n"
            f"  decide miss: {_pct(lat_miss)}"
        )
    env.close()
    sp.close()


if __name__ == "__main__":
    main()
