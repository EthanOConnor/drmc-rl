"""Record real arena planning requests and replay them through both planners.

``capture`` runs an event arena config unchanged and stores every
``plan_candidates`` request (public state, charged delay, pace) that reached a
planner. ``compare`` replays those requests through the CPU planner
(``drm_reach_bfs_full``) and the batched CUDA planner and requires, for every
request, byte-identical native reachability arrays (costs, offsets, lengths,
script bytes), identical packed candidate sets and costs, and identical
``execution_for_action`` output (placement, controller frames, controller
states) for every feasible action. Requests the CUDA planner declines are
counted as explicit CPU fallbacks, never compared as CUDA answers.

usage:
  python -m tools.trainer_planner_parity capture --config ARENA.json --out REQ.pkl.gz
  python -m tools.trainer_planner_parity compare --requests REQ.pkl.gz [...] --report OUT.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
import threading
import time
from pathlib import Path

import numpy as np


def capture(config_path: Path, out: Path, limit: int) -> None:
    import tools.trainer_event_rollout as events
    import tools.trainer_planning_arena as arena

    lock = threading.Lock()
    records = []
    original = events.ParallelPlanning._plan

    def recording(self, request):
        state, delay, pace = request
        with lock:
            if len(records) < limit:
                records.append(dict(state=state, delay=int(delay), pace=pace.id))
        return original(self, request)

    events.ParallelPlanning._plan = recording
    sys.argv = ["arena", "--config", str(config_path)]
    try:
        arena.main()
    finally:
        with gzip.open(out, "wb") as stream:
            pickle.dump(records, stream, protocol=pickle.HIGHEST_PROTOCOL)
        print(json.dumps(dict(captured=len(records), out=str(out))), flush=True)


def _same_candidate(cpu, gpu) -> str | None:
    names = ("own", "opponent", "pill", "preview", "speed", "speed_ups", "frame", "reach", "packed", "costs")
    for name, a, b in zip(names, cpu, gpu):
        if name == "reach":
            for field in ("costs_u16", "offsets_u16", "lengths_u16", "script_buf"):
                x, y = getattr(a, field), getattr(b, field)
                if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                    return f"reach.{field}"
        elif name == "packed":
            if a.count != b.count or any(getattr(a, f).tobytes() != getattr(b, f).tobytes()
                                         for f in ("actions", "mask", "cost")):
                return "packed"
        elif isinstance(a, np.ndarray):
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                return name
        elif a != b:
            return name
    return None


def compare(paths: list[Path], report: Path, batch: int, executions: str, stride: int) -> None:
    from drmc_rl.execution.pace import BY_ID
    from drmc_rl.human.anticipation import execution_for_action
    from drmc_rl.human.backend import NoReachablePlacement, plan_candidates
    from drmc_rl.planning.native_reach import NativeReachabilityRunner
    from tools.trainer_event_rollout import CudaPlanning

    requests = []
    for path in paths:
        with gzip.open(path, "rb") as stream:
            requests.extend(pickle.load(stream))
    cpu = NativeReachabilityRunner()
    gpu = CudaPlanning(cpu_workers=1, cache=False)
    counts = dict(requests=len(requests), compared=0, cpu_fallback=0, no_reachable=0,
                  executions_compared=0, mismatches=0, by_pace={})
    mismatches = []
    cpu_seconds = gpu_seconds = 0.0
    try:
        for start in range(0, len(requests), batch):
            chunk = requests[start:start + batch]
            triples = [(r["state"], r["delay"], BY_ID[r["pace"]]) for r in chunk]
            tick = time.perf_counter()
            gpu_answers = gpu.plan(triples)
            gpu_seconds += time.perf_counter() - tick
            routes = list(gpu.last_routes)
            for offset, ((state, delay, pace), answer, route) in enumerate(zip(triples, gpu_answers, routes)):
                index = start + offset
                tick = time.perf_counter()
                try:
                    reference = plan_candidates(cpu, state, delay, pace)
                except NoReachablePlacement:
                    reference = None
                cpu_seconds += time.perf_counter() - tick
                pace_counts = counts["by_pace"].setdefault(pace.id, dict(compared=0, cpu_fallback=0))
                if route != "cuda":
                    counts["cpu_fallback"] += 1
                    pace_counts["cpu_fallback"] += 1
                    continue
                counts["compared"] += 1
                pace_counts["compared"] += 1
                if reference is None or answer is None:
                    if (reference is None) != (answer is None):
                        mismatches.append(dict(index=index, field="no_reachable"))
                    else:
                        counts["no_reachable"] += 1
                    continue
                field = _same_candidate(reference, answer)
                if field is None and executions != "none" and index % stride == 0:
                    feasible = np.flatnonzero(reference[-1] != 0xFFFF)
                    if executions == "chosen":
                        feasible = feasible[:1]
                    for action in feasible:
                        a = execution_for_action(reference, int(action), pace, delay=delay)
                        b = execution_for_action(answer, int(action), pace, delay=delay)
                        counts["executions_compared"] += 1
                        if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
                            field = f"execution[{int(action)}]"
                            break
                if field is not None:
                    mismatches.append(dict(index=index, field=field))
            print(json.dumps(dict(progress=start + len(chunk), **{k: counts[k] for k in
                  ("compared", "cpu_fallback", "executions_compared")}, mismatches=len(mismatches))), flush=True)
    finally:
        gpu.close()
    counts["mismatches"] = len(mismatches)
    result = dict(schema="trainer-planner-parity-v1", sources=[str(p) for p in paths], **counts,
                  first_mismatches=mismatches[:20], cpu_seconds=round(cpu_seconds, 3),
                  cuda_seconds=round(gpu_seconds, 3), executions=executions, execution_stride=stride)
    report.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "first_mismatches"}), flush=True)
    if mismatches:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--config", type=Path, required=True)
    cap.add_argument("--out", type=Path, required=True)
    cap.add_argument("--limit", type=int, default=200000)
    cmp = sub.add_parser("compare")
    cmp.add_argument("--requests", type=Path, nargs="+", required=True)
    cmp.add_argument("--report", type=Path, required=True)
    cmp.add_argument("--batch", type=int, default=256)
    cmp.add_argument("--executions", choices=("all", "chosen", "none"), default="all",
                     help="feasible actions whose execution_for_action output is compared")
    cmp.add_argument("--execution-stride", type=int, default=1,
                     help="compare executions on every Nth request (reach arrays are always compared)")
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.config, args.out, args.limit)
    else:
        compare(args.requests, args.report, args.batch, args.executions, args.execution_stride)


if __name__ == "__main__":
    main()
