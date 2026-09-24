"""Ground-truth ranking of the kept root placements by exact counterfactual rollouts.

For each seed a champion-vs-champion argmax game is played on the event runner.
At sampled side-0 decisions the top-K root placements are scored by every
lookahead scorer (root logit, candidate WDL head, settled-bottle value, value
after the best 1 or 2 follow-ups, follow-up best logit). Then one pool replays
the game once per (decision, candidate): identical argmax play except that the
sampled decision executes that candidate, and both sides continue as the
champion. Games are deterministic, so each rollout's outcome is that
candidate's exact value under champion continuation.

usage: counterfactual_ranking.py PACE SEEDS(comma) OUT.jsonl [K] [EVERY]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from drmc_rl.envs.backends.vs_frames import EventVsPool
from drmc_rl.execution.pace import resolve_pace
from drmc_rl.human.anticipation import execution_for_action
from drmc_rl.human.backend import NoReachablePlacement, plan_candidates
from drmc_rl.human.lookahead import Root, score_with_value, select_lookahead
from drmc_rl.planning.native_reach import NativeReachabilityRunner
from tools.trainer_arena_cache import MemoPlanner
from tools.trainer_event_rollout import controller_inputs
from tools.vs_head_to_head import PlainPolicy

CH = "/Users/ethan/dev/drmario/drmc-rl/runs/review-20260909/controller-retention-mixed-v2/core-final-inference.pt"
NL = "/Users/ethan/dev/drmario/drmc-rl/runs/review-20260909/controller-arena-0c76c0e-source/native-libraries/libdrmario_pool.dylib"
pace = resolve_pace(sys.argv[1])
seeds = [int(s) for s in sys.argv[2].split(",")]
out = Path(sys.argv[3])
K = int(sys.argv[4]) if len(sys.argv) > 4 else 6
EVERY = int(sys.argv[5]) if len(sys.argv) > 5 else 5
DELAY = max(4, pace.reaction_frames)
torch.set_num_threads(1)
policy = PlainPolicy(Path(CH), "mps", public_only=True)
planner = MemoPlanner(NativeReachabilityRunner())


def plan(requests):
    result = []
    for state, delay, request_pace in requests:
        try:
            result.append(plan_candidates(planner, state, delay, request_pace))
        except NoReachablePlacement:
            result.append(None)
    return result


def candidate_wdl(obs, info):
    inputs, aux, ca, cm = policy.model_inputs(obs, info)
    with torch.inference_mode():
        _, _, extra = policy.net(*inputs, aux=aux, return_aux=True)
    wdl = extra["candidate_wdl_logits"].float().softmax(-1).cpu().numpy()[0]
    out = np.full(512, np.nan, np.float32)
    for slot in np.flatnonzero(cm[0]):
        out[ca[0, slot]] = wdl[slot, 0] - wdl[slot, 2]
    return out, extra["state_wdl_logits"].float().softmax(-1).cpu().numpy()[0].tolist()


def decide(pool, sides):
    """Argmax decisions for many parked sides in one scoring pass."""
    states, candidates, observations, infos = [], [], [], []
    for side in sides:
        state = pool.semantic(side, public_context=True)
        try:
            candidate = plan_candidates(planner, state, DELAY, pace)
        except NoReachablePlacement:
            candidate = None
        states.append(state)
        candidates.append(candidate)
        if candidate is not None:
            obs, info = controller_inputs(policy, candidate, state, pace, DELAY, 4, state["public_pair_state"], DELAY)
            observations.append(obs)
            infos.extend(info)
    scores, values = score_with_value(policy, np.concatenate(observations), infos) if infos else (None, None)
    result, row = [], 0
    for state, candidate in zip(states, candidates):
        if candidate is None:
            result.append((state, None, None, None))
        else:
            result.append((state, candidate, scores[row], float(values[row])))
            row += 1
    return result


SCORERS = {
    "value_f1": dict(mode="value", followups=1),
    "value_f2": dict(mode="value", followups=2),
    "settled_value": dict(mode="root_value", followups=1),
    "followup_logit": dict(mode="logit", followups=1),
}


def base_game(seed):
    points = []
    with EventVsPool(1, lib_path=NL) as pool:
        pool.reset([seed], level=14)
        count = 0
        while True:
            progress = pool.advance(120000)
            ready = [s for s, p in enumerate(progress) if p.needs_action]
            if not ready:
                break
            for side, (state, candidate, scores, value) in zip(ready, decide(pool, ready)):
                if candidate is None:
                    pool.install(side)
                    continue
                action = int(scores.argmax())
                if side == 0:
                    if count % EVERY == 2 and np.isfinite(scores).sum() >= 2:
                        point = dict(seed=seed, decision=count, frame=int(pool.states[0].frame), root_value=value)
                        legal = np.flatnonzero(np.isfinite(scores))
                        kept = legal[np.argsort(-scores[legal], kind="stable")][:K]
                        point["kept"] = kept.tolist()
                        point["logit"] = [float(scores[a]) for a in kept]
                        obs, info = controller_inputs(policy, candidate, state, pace, DELAY, 4,
                                                      state["public_pair_state"], DELAY)
                        wdl, state_wdl = candidate_wdl(obs, info)
                        point["candidate_wdl"] = [float(wdl[a]) for a in kept]
                        point["state_wdl"] = state_wdl
                        for name, extra in SCORERS.items():
                            block = dict(k=K, margin=0.0, max_kept=K, prior=0.0, charge_frames=0, when="always", **extra)
                            root = Root(policy, state, candidate, scores, pace, DELAY, state["public_pair_state"], 4, DELAY, block)
                            (_, diagnostics), = select_lookahead([root], plan, controller_inputs)
                            point[name] = diagnostics["q"]
                        points.append(point)
                    count += 1
                pool.install(side, execution_for_action(candidate, action, pace, delay=DELAY), delay=DELAY)
    return points


def rollouts(seed, points):
    jobs = [(p, k) for p in points for k in range(len(p["kept"]))]
    outcomes = [None] * len(jobs)
    with EventVsPool(len(jobs), lib_path=NL) as pool:
        pool.reset([seed] * len(jobs), level=14)
        counts = [0] * len(jobs)
        while True:
            progress = pool.advance(120000)
            ready = [s for s, p in enumerate(progress) if p.needs_action]
            if not ready:
                break
            for side, (state, candidate, scores, _) in zip(ready, decide(pool, ready)):
                if candidate is None:
                    pool.install(side)
                    continue
                pair, physical = divmod(side, 2)
                action = int(scores.argmax())
                if physical == 0:
                    point, k = jobs[pair]
                    if counts[pair] == point["decision"]:
                        action = point["kept"][k]
                        if not np.isfinite(scores[action]):
                            raise RuntimeError("replayed decision lost a kept candidate")
                    counts[pair] += 1
                pool.install(side, execution_for_action(candidate, action, pace, delay=DELAY), delay=DELAY)
        for pair in range(len(jobs)):
            end = pool.states[2 * pair]
            outcomes[pair] = None if not end.terminal else 1.0 if end.outcome == 1 else 0.0 if end.outcome == 2 else 0.5
    for p in points:
        p["outcomes"] = []
    for (p, _), outcome in zip(jobs, outcomes):
        p["outcomes"].append(outcome)
    return points


with out.open("a") as stream:
    for seed in seeds:
        points = rollouts(seed, base_game(seed))
        for p in points:
            stream.write(json.dumps({**p, "pace": pace.id}) + "\n")
        stream.flush()
        print(seed, len(points), flush=True)
