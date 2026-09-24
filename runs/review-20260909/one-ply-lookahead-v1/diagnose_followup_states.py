"""How far is a constructed follow-up state's value from the real next state's?

Plays champion-vs-champion argmax games on the event runner. At each side-0
decision the executed placement's settled bottle is built exactly as the
lookahead builds it; at that side's next decision the constructed value is
compared with the value of the real state (same network, same pass shape).
"""
import json, sys
from pathlib import Path
import numpy as np
from drmc_rl.envs.backends.vs_frames import EventVsPool
from drmc_rl.execution.pace import resolve_pace
from drmc_rl.human.anticipation import execution_for_action
from drmc_rl.human.backend import plan_candidates
from drmc_rl.human.lookahead import Root, score_with_value, _follow_state, _settle
from drmc_rl.planning.native_reach import NativeReachabilityRunner
from tools.vs_head_to_head import PlainPolicy
from tools.trainer_event_rollout import controller_inputs
import drmc_rl.human.lookahead as L

CH = "/Users/ethan/dev/drmario/drmc-rl/runs/review-20260909/controller-retention-mixed-v2/core-final-inference.pt"
NL = "/Users/ethan/dev/drmario/drmc-rl/runs/review-20260909/controller-arena-0c76c0e-source/native-libraries/libdrmario_pool.dylib"
pace = resolve_pace(sys.argv[1] if len(sys.argv) > 1 else "frame_perfect")
seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [915, 3040]
builders = dict(stale=L._follow_state)
if len(sys.argv) > 3:
    import importlib
    builders.update(importlib.import_module(sys.argv[3]).BUILDERS)
policy = PlainPolicy(Path(CH), "mps", public_only=True)
planner = NativeReachabilityRunner()
rows = []
for seed in seeds:
    with EventVsPool(1, lib_path=NL) as pool:
        pool.reset([seed], level=14)
        pending = {}
        while True:
            progress = pool.advance(120000)
            ready = [s for s, p in enumerate(progress) if p.needs_action]
            if not ready:
                break
            for side in ready:
                state = pool.semantic(side, public_context=True)
                delay = max(4, pace.reaction_frames)
                candidate = plan_candidates(planner, state, delay, pace)
                obs, info = controller_inputs(policy, candidate, state, pace, delay, 4, state["public_pair_state"], delay)
                scores, values = score_with_value(policy, obs, info)
                action = int(scores[0].argmax())
                if side == 0 and side in pending:
                    prev = pending.pop(side)
                    actual_board = bytes(state["public_pair_state"].sides[0].board)
                    row = dict(seed=seed, frame=int(pool.states[0].frame), elapsed=int(pool.states[0].frame) - prev["frame"],
                               v_root=prev["v"], v_actual=float(values[0]), exact_board=actual_board == prev["after"],
                               true_preview=tuple(state["preview"]), next_pill=tuple(state["pill"]))
                    for name, build in builders.items():
                        for preview_mode in ("repeat", "true"):
                            root = prev["root"]
                            s1, view, nxt = build(root, prev["after"], **({} if preview_mode == "repeat" else {"preview": tuple(state["preview"])}), **({"action": prev["action"], "facts": prev["facts"]} if name != "stale" else {}))
                            c1 = plan_candidates(planner, s1, delay, pace)
                            o1, i1 = controller_inputs(policy, c1, s1, pace, delay, 4, view, delay)
                            sc1, v1 = score_with_value(policy, o1, i1)
                            row[f"v_{name}_{preview_mode}"] = float(v1[0])
                            row[f"same_action_{name}_{preview_mode}"] = int(sc1[0].argmax()) == action
                    rows.append(row)
                if side == 0:
                    root = Root(policy, state, candidate, scores[0], pace, delay, state["public_pair_state"], 4, delay, {})
                    after, facts = _settle(bytes(state["public_pair_state"].sides[0].board), tuple(state["pill"]), action)
                    pending[side] = dict(root=root, after=after, facts=facts, action=action, v=float(values[0]), frame=int(pool.states[0].frame))
                move = execution_for_action(candidate, action, pace, delay=delay)
                pool.install(side, move, delay=delay)
out = Path(sys.argv[4] if len(sys.argv) > 4 else "/Users/ethan/dev/drmario/drmc-rl-lookahead-data/diagnose-followup.json")
out.write_text(json.dumps(rows))
exact = [r for r in rows if r["exact_board"]]
print(len(rows), "transitions,", len(exact), "with exact settled-board prediction")
keys = [k for k in rows[0] if k.startswith("v_") and k not in ("v_actual",)]
for k in keys:
    d = np.array([r[k] - r["v_actual"] for r in exact])
    c = np.corrcoef([r[k] for r in exact], [r["v_actual"] for r in exact])[0, 1]
    print(f"{k:24s} bias {d.mean():+.3f}  mae {np.abs(d).mean():.3f}  corr {c:.3f}")
for k in [k for k in rows[0] if k.startswith("same_action")]:
    print(k, np.mean([r[k] for r in exact]))
