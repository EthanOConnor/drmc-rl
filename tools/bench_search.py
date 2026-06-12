"""Bench SearchPolicy.decide() latency on real mid-game states.

Plays 1P pool episodes with the search's own decisions (the states it would
actually visit) and reports decide() latency percentiles plus the anytime
stage histogram. Target: p95 <= 80 ms at beam 8.

Usage:
  .venv/bin/python -m tools.bench_search --checkpoint runs/vs2_02/checkpoints/<ck>.pt.gz \
      [--level 14] [--decisions 200] [--beam 8] [--deadline-ms 60] [--device auto]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from envs.backends.drmario_pool import DrMarioPoolRunner, build_reset_spec
from models.policy.search_policy import SearchPolicy

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
    args = ap.parse_args()

    torch.set_num_threads(int(args.threads))
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    sp = SearchPolicy(
        args.checkpoint,
        beam=int(args.beam),
        deadline_ms=float(args.deadline_ms),
        device=device,
        seed=int(args.seed),
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
        a, info = sp.decide(
            env.buffers.board_bytes[0],
            (_CANON_TO_RAW[int(pill[0])], _CANON_TO_RAW[int(pill[1])]),
            (_CANON_TO_RAW[int(prev[0])], _CANON_TO_RAW[int(prev[1])]),
            mask,
            env.buffers.cost_to_lock[0],
            int(args.speed_setting),
            min(0x31, max(0, int(args.level) - 20) + decisions // 10),
            int(args.level),
            viruses_initial=v0,
            frames_used=frames,
        )
        lat.append(float(info["elapsed_ms"]))
        stages[info["stage"]] = stages.get(info["stage"], 0) + 1
        agreed += int(bool(info["agreed_with_policy"]))
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
    env.close()
    sp.close()


if __name__ == "__main__":
    main()
