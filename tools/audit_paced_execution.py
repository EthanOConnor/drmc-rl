"""Audit complete named-pace feasibility, frame replay, and local CPU latency.

This engineering diagnostic does not certify human percentile calibration.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from drmc_rl.execution.pace import PACES
from drmc_rl.planning.fast_reach import compute_speed_threshold, simulate_frame
from drmc_rl.planning.native_reach import NativeReachabilityRunner, resolve_library_path
from tools.audit_execution_control import columns, digest, initial_state, replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-states", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists() or args.max_states < 1:
        parser.error("use a new output and positive sample size")
    manifest = json.loads(Path(str(args.input) + ".manifest.json").read_text())
    if (manifest.get("schema") != "drmc-execution-corpus-sample-v2"
            or not manifest.get("recorded_lock_replay_verified")
            or digest(args.input) != manifest["output_sha256"]):
        parser.error("requires hash-verified covered replay v2 sample")
    groups = defaultdict(list)
    for line in args.input.read_text().splitlines():
        row = json.loads(line)
        if not row["recorded_lock_verified"]:
            raise ValueError("unverified row")
        if row["player_fold"] == 0 or row["split"] != "train":
            key = (row["speed"], int(row["rating"] // 400), bool(np.any(columns(row) & 15)))
            groups[key].append(row)
    rng = np.random.default_rng(20260908)
    for group in groups.values():
        rng.shuffle(group)
    chosen = []
    while len(chosen) < args.max_states and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(chosen) < args.max_states:
                chosen.append(groups[key].pop())
    runner = NativeReachabilityRunner(max_frames=2048)
    results, skipped, witnesses = [], 0, 0
    started = time.monotonic()
    for number, row in enumerate(chosen):
        cols, spawn = columns(row), initial_state(row)
        threshold = compute_speed_threshold(row["speed"], row["speed_ups"])
        locked, tau = replay(cols, spawn, row["script"], threshold)
        if not locked.locked or tau != len(row["script"]) or (locked.x, locked.y, locked.rot) != tuple(row["lock_pose"]):
            raise ValueError("source no longer replays")
        for _ in range(8):
            if spawn.locked:
                break
            spawn = simulate_frame(cols, spawn, 0, speed_threshold=threshold)
        if spawn.locked:
            skipped += 1
            continue
        previous = set()
        record = {"high_board": bool(np.any(cols & 15)), "speed": row["speed"], "paces": {}}
        for pace in PACES:
            tick = time.perf_counter()
            reach = runner.bfs_full(cols, spawn, speed_threshold=threshold, **pace.planner_args(8))
            elapsed = (time.perf_counter() - tick) * 1000
            # Match the placement ABI: both pill halves must be on-screen.
            poses = {int(p) for p in np.flatnonzero(reach.costs_u16 != 65535)
                     if ((p >> 7) % 2 == 0 and (p & 7) < 7)
                     or ((p >> 7) % 2 == 1 and ((p >> 3) & 15) >= 1)}
            if not previous <= poses:
                raise ValueError(f"pace feasible-set inversion: {pace.id}, columns={cols.tolist()}, spawn={spawn}, threshold={threshold}, missing={sorted(previous - poses)}")
            previous = poses
            durations = []
            for pose in poses:
                target = (int(pose) & 7, (int(pose) >> 3) & 15, int(pose) >> 7)
                script = reach.script_for_pose(*target)
                audit = pace.validate(cols, spawn, script, speed_threshold=threshold, execution_delay=8)
                if (audit["x"], audit["y"], audit["rotation"]) != target or len(script) != reach.costs_u16[pose]:
                    raise ValueError("witness target or duration mismatch")
                durations.append(len(script))
                witnesses += 1
            record["paces"][pace.id] = {"ms": elapsed, "candidates": len(poses),
                "mean_frames": float(np.mean(durations)) if durations else None}
        results.append(record)
        if (number + 1) % 8 == 0:
            print(json.dumps({"audited": number + 1, "elapsed_seconds": round(time.monotonic() - started, 1),
                              "latest_ms": {k: round(v["ms"], 2) for k, v in record["paces"].items()}}), flush=True)
    summary = {}
    for pace in PACES:
        rows = [r["paces"][pace.id] for r in results]
        summary[pace.id] = {"limits": pace.to_dict(),
            "latency_ms": dict(zip(("p50", "p95", "max"), np.quantile([r["ms"] for r in rows], [.5, .95, 1]).tolist())),
            "mean_candidates": float(np.mean([r["candidates"] for r in rows]))}
    report = {"schema": "drmc-paced-execution-audit-v1", "diagnostic_only": True,
        "source_sha256": manifest["output_sha256"], "native_library_sha256": digest(resolve_library_path()),
        "code_sha256": {p: digest(p) for p in ("reach_native/drm_reach_full.c", "drmc_rl/execution/pace.py", __file__)},
        "selection": "seeded heldout rating/speed/high-board round robin", "states": len(results),
        "natural_locks_during_host_delay": skipped, "witnesses": witnesses, "violations": 0,
        "feasible_set_inversions": 0, "summary": summary, "rows": results}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
