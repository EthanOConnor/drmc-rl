"""Watch a run's checkpoints and evaluate each on fixed levels as it lands.

Appends one JSON line per evaluated checkpoint to <run_dir>/eval_history.jsonl,
giving a learning curve on real levels (curriculum-free), independent of the
shaped training reward. Kept deliberately light (small env counts, CPU) so it
can run beside a live training process.

    python -m tools.eval_watch runs/human_push_01 --levels 0,5,10 \
        --episodes 12 --min-gap-frames 10000000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tools.eval_policy import _build_net_from_cfg, _make_aux_builder, evaluate_level
from training.utils.checkpoint_io import load_checkpoint

_CHANNELS = {
    "bitplane_bottle": 4,
    "bitplane_bottle_mask": 8,
    "bitplane_bottle_conn": 8,
    "bitplane_bottle_conn_mask": 12,
}


def _ckpt_step(path: Path) -> int:
    name = path.name
    try:
        return int(name.split("step")[-1].split(".")[0])
    except Exception:
        return -1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--levels", type=str, default="0,5,10")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--speed-setting", type=int, default=2)
    ap.add_argument("--state-repr", type=str, default="bitplane_bottle_conn_mask")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--min-gap-frames", type=int, default=10_000_000)
    ap.add_argument("--poll-sec", type=float, default=30.0)
    ap.add_argument("--once", action="store_true", help="evaluate pending checkpoints and exit")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    history_path = run_dir / "eval_history.jsonl"
    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    evaluated: set[int] = set()
    last_step = -(10**18)
    if history_path.is_file():
        for line in history_path.read_text().splitlines():
            try:
                row = json.loads(line)
                evaluated.add(int(row["step"]))
                last_step = max(last_step, int(row["step"]))
            except Exception:
                continue

    while True:
        ckpts = sorted(ckpt_dir.glob("*.pt.gz"), key=_ckpt_step)
        pending = [
            p
            for p in ckpts
            if _ckpt_step(p) not in evaluated and _ckpt_step(p) >= last_step + args.min_gap_frames
        ]
        for path in pending:
            step = _ckpt_step(path)
            try:
                payload = load_checkpoint(path, map_location="cpu")
                in_ch = _CHANNELS.get(args.state_repr, 12)
                net, aux_dim, cand_max = _build_net_from_cfg(payload.get("cfg", {}), in_ch, args.device)
                net.load_state_dict(payload["state_dict"])
                aux_shim = _make_aux_builder(aux_dim)
            except Exception as exc:  # partial writes from the live trainer
                print(f"skip {path.name}: {exc}")
                continue

            row = {"step": step, "checkpoint": path.name, "levels": {}}
            for level in levels:
                res = evaluate_level(
                    level=level,
                    episodes=args.episodes,
                    num_envs=args.num_envs,
                    speed_setting=args.speed_setting,
                    state_repr=args.state_repr,
                    policy="checkpoint",
                    net=net,
                    aux_shim=aux_shim,
                    candidate_max=cand_max,
                    device=args.device,
                    temperature=0.0,
                    seed=777,
                )
                row["levels"][str(level)] = {
                    "clear_rate": res["clear_rate"],
                    "frames_to_clear_p50": res["frames_to_clear_p50"],
                    "viruses_cleared_mean": res["viruses_cleared_mean"],
                }
            with history_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            evaluated.add(step)
            last_step = max(last_step, step)
            summary = " ".join(
                f"L{lvl}:{row['levels'][str(lvl)]['clear_rate']*100:.0f}%/"
                f"{row['levels'][str(lvl)]['viruses_cleared_mean']:.1f}v"
                for lvl in levels
            )
            print(f"step={step:,} {summary}", flush=True)

        if args.once:
            break
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
