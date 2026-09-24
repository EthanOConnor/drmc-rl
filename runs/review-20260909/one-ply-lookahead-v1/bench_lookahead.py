"""Per-decision lookahead compute on real decision states, by phase and device.

usage: bench_lookahead.py PACE SEED OUT.json

Plays one champion-vs-champion argmax game on the event runner and, at every
side-0 decision, times the plain root pass and each lookahead phase at K = 4
and 6 (margin 0, followups 1 and 2) on MPS (the WebGPU-like accelerator path)
and on CPU at 1 and 4 threads. Network passes are timed after one untimed warmup
per shape bucket; each timed pass ends with the host copy of its outputs.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from drmc_rl.envs.backends.vs_frames import EventVsPool
from drmc_rl.execution.pace import resolve_pace
from drmc_rl.human.anticipation import execution_for_action
from drmc_rl.human.backend import NoReachablePlacement, plan_candidates
from drmc_rl.human.lookahead import Root, _follow_state, _kept, _settle, score_with_value
from drmc_rl.planning.native_reach import NativeReachabilityRunner
from tools.trainer_event_rollout import controller_inputs
from tools.vs_head_to_head import PlainPolicy

CH = "/Users/ethan/dev/drmario/drmc-rl/runs/review-20260909/controller-retention-mixed-v2/core-final-inference.pt"
NL = "/Users/ethan/dev/drmario/drmc-rl/runs/review-20260909/controller-arena-0c76c0e-source/native-libraries/libdrmario_pool.dylib"
pace = resolve_pace(sys.argv[1])
seed = int(sys.argv[2])
out = Path(sys.argv[3])
DELAY = max(4, pace.reaction_frames)
devices = {"mps": PlainPolicy(Path(CH), "mps", public_only=True),
           "cpu": PlainPolicy(Path(CH), "cpu", public_only=True)}
planner = NativeReachabilityRunner()
warm = set()


def timed_pass(device, threads, obs, infos):
    torch.set_num_threads(threads)
    key = (device, threads, len(infos))
    policy = devices[device]
    if key not in warm:
        score_with_value(policy, obs, infos)
        warm.add(key)
    start = time.perf_counter()
    score_with_value(policy, obs, infos)
    return (time.perf_counter() - start) * 1000


def lookahead_inputs(state, candidate, scores, k, followups):
    """Build every lookahead input for one decision, timing the host-side phases."""
    root = Root(devices["mps"], state, candidate, scores, pace, DELAY, state["public_pair_state"], 4, DELAY,
                dict(k=k, margin=0.0, max_kept=k, followups=followups))
    timing = {}
    t = time.perf_counter()
    kept = _kept(scores, root.params)
    board = bytes(root.public.sides[root.public.viewer_side].board)
    follows = []
    for action in map(int, kept):
        after, facts = _settle(board, tuple(state["pill"]), action)
        if not facts[8] and not facts[10]:
            follows.append(_follow_state(root, after))
    timing["settle_ms"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    planned = []
    for s1, view, _ in follows:
        try:
            planned.append((s1, view, plan_candidates(planner, s1, DELAY, pace)))
        except NoReachablePlacement:
            pass
    timing["plan_ms"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    first = [controller_inputs(root.actor, c, s1, pace, DELAY, 4, view, DELAY) for s1, view, c in planned]
    timing["encode1_ms"] = (time.perf_counter() - t) * 1000
    if not first:
        return timing, None, None
    obs1 = np.concatenate([o for o, _ in first])
    info1 = [i[0] for _, i in first]
    scores1, _ = score_with_value(devices["mps"], obs1, info1)
    t = time.perf_counter()
    second = []
    for (s1, view, c), row in zip(planned, scores1):
        legal = np.flatnonzero(np.isfinite(row))
        for a2 in legal[np.argsort(-row[legal], kind="stable")][:followups]:
            after, facts = _settle(bytes(view.sides[view.viewer_side].board), tuple(s1["pill"]), int(a2))
            if facts[8] or facts[10]:
                continue
            leaf, leaf_view, _ = _follow_state(root, after)
            costs = np.full(512, 0xFFFF, np.uint16)
            costs[int(a2)] = 0
            second.append(controller_inputs(root.actor, (costs,), leaf, pace, DELAY, 4, leaf_view, DELAY))
    timing["settle2_encode2_ms"] = (time.perf_counter() - t) * 1000
    obs2 = np.concatenate([o for o, _ in second]) if second else None
    info2 = [i[0] for _, i in second]
    return timing, (obs1, info1), (obs2, info2)


rows = []
with EventVsPool(1, lib_path=NL) as pool:
    pool.reset([seed], level=14)
    while True:
        progress = pool.advance(120000)
        ready = [s for s, p in enumerate(progress) if p.needs_action]
        if not ready:
            break
        for side in ready:
            state = pool.semantic(side, public_context=True)
            try:
                candidate = plan_candidates(planner, state, DELAY, pace)
            except NoReachablePlacement:
                pool.install(side)
                continue
            obs, info = controller_inputs(devices["mps"], candidate, state, pace, DELAY, 4, state["public_pair_state"], DELAY)
            scores, _ = score_with_value(devices["mps"], obs, info)
            action = int(scores[0].argmax())
            if side == 0 and np.isfinite(scores[0]).sum() >= 2:
                row = dict(legal=int(np.isfinite(scores[0]).sum()))
                for device, threads in (("mps", 1), ("cpu", 1), ("cpu", 4)):
                    row[f"root_{device}{threads}_ms"] = timed_pass(device, threads, obs, info)
                for k in (4, 6):
                    for followups in (1, 2):
                        timing, first, second = lookahead_inputs(state, candidate, scores[0], k, followups)
                        prefix = f"k{k}f{followups}_"
                        row.update({prefix + name: v for name, v in timing.items()})
                        row[prefix + "states1"] = 0 if first is None else len(first[1])
                        row[prefix + "states2"] = 0 if second is None or second[0] is None else len(second[1])
                        for device, threads in (("mps", 1), ("cpu", 1), ("cpu", 4)):
                            tag = f"{device}{threads}"
                            row[prefix + f"pass1_{tag}_ms"] = 0.0 if first is None else timed_pass(device, threads, *first)
                            row[prefix + f"pass2_{tag}_ms"] = (0.0 if second is None or second[0] is None
                                                             else timed_pass(device, threads, *second))
                rows.append(row)
            pool.install(side, execution_for_action(candidate, action, pace, delay=DELAY), delay=DELAY)
out.write_text(json.dumps(rows))
print(len(rows), "decisions timed")
