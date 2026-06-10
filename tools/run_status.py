"""Summarize a training run directory: throughput, curriculum, losses.

Tolerates partially-written gzip logs from live runs.

    python -m tools.run_status runs/human_push_01 [--tail-minutes 10]
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Dict, List


def read_scalars(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with gzip.open(path, "rt") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    break
    except (EOFError, OSError):
        pass
    return [r for r in rows if r.get("type") == "scalar"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--history", action="store_true", help="show curriculum stage history")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    metrics = run_dir / "metrics.jsonl.gz"
    if not metrics.is_file():
        # allow pointing at the parent (auto run-id dirs)
        candidates = sorted(run_dir.glob("*/metrics.jsonl.gz"))
        if not candidates:
            raise SystemExit(f"no metrics.jsonl.gz under {run_dir}")
        metrics = candidates[-1]
        run_dir = metrics.parent

    sc = read_scalars(metrics)
    if not sc:
        raise SystemExit("no scalar rows yet")

    latest: Dict[str, Any] = {}
    for r in sc:
        latest[r["name"]] = r["value"]
    last_step = sc[-1]["step"]

    def g(name: str, fmt: str = "{:.3f}") -> str:
        v = latest.get(name)
        if v is None:
            return "-"
        try:
            return fmt.format(v)
        except Exception:
            return str(v)

    print(f"run: {run_dir}")
    print(f"frames: {last_step:,}")
    print(
        f"throughput: {g('perf/sps', '{:.0f}')} frames/s, {g('perf/dps', '{:.0f}')} dec/s, "
        f"update {g('perf/update_sec', '{:.2f}')}s, inference {g('perf/inference_ms_avg', '{:.2f}')}ms"
    )
    print(
        f"curriculum: level {g('curriculum/current_level', '{:.0f}')} "
        f"(stage {g('curriculum/stage_index', '{:.0f}')}, "
        f"rate {g('curriculum/rate_current', '{:.2f}')}, "
        f"threshold {g('curriculum/success_threshold', '{:.2f}')}, "
        f"LB {g('curriculum/confidence_lower_bound', '{:.2f}')})"
    )
    print(
        f"losses: policy {g('loss/policy')}, value {g('loss/value')}, "
        f"entropy {g('policy/entropy')}, kl {g('policy/kl', '{:.4f}')}, "
        f"clip {g('policy/clip_frac', '{:.3f}')}"
    )
    print(
        f"episodes: return {g('train/return_mean', '{:.2f}')}, "
        f"viruses/ep {g('drm/viruses_per_ep', '{:.2f}')}"
    )

    if args.history:
        print("\nstage advancements (frames -> level):")
        for r in sc:
            if r["name"] == "curriculum/advanced_to":
                print(f"  {r['step']:>12,} -> {int(r['value'])}")

    ckpts = sorted((run_dir / "checkpoints").glob("*.pt.gz"))
    if ckpts:
        print(f"\ncheckpoints: {len(ckpts)} (latest {ckpts[-1].name})")


if __name__ == "__main__":
    main()
