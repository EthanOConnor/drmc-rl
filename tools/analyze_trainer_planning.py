"""Replay recorded controller games and explain anticipation's tradeoffs."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from drmc_rl.arena.experiment import dump
from drmc_rl.envs.backends.vs_frames import FrameVsPool
from drmc_rl.execution.pace import resolve_pace
from drmc_rl.human.anticipation import public_policy_inputs, score_public_inputs
from drmc_rl.human.backend import plan_candidates
from drmc_rl.planning.native_reach import NativeReachabilityRunner
from tools.vs_head_to_head import PlainPolicy


def replay_roots(path, per_game, library):
    """Recover exact public microstates from the seed and recorded inputs."""
    with gzip.open(path, "rt") as stream:
        data = json.load(stream)
    game, moves = data["game"], data["moves"]
    hits = [i for i,m in enumerate(moves) if m["side"] == game["side"]
            and m["cache"] in ("hit", "stale_opponent")]
    selected = {hits[int(i)] for i in np.linspace(0, len(hits)-1, min(per_game,len(hits)))}
    roots, previous, controllers, index = [], [None,None], [None,None], 0
    with FrameVsPool(lib_path=library) as pool:
        pool.reset([game["seed"]], level=game["level"])
        for frame in range(game["frames"]):
            while index < len(moves) and moves[index]["frame"] == frame:
                move = moves[index]
                side = move["side"]
                assert list(pool.states[side].board) == move["board"], (path, frame, "own board")
                assert list(pool.states[side^1].board) == move["opponent"], (path, frame, "opponent board")
                state = pool.states[side].semantic(pool.states[side^1])
                if index in selected:
                    source_frame, source = previous[side]
                    roots.append((state, source, {"game":path.name, "seed":game["seed"], "frame":frame,
                        "action":move["placement"]["action"], "age_frames":frame-source_frame,
                        "script_frames":len(move["controller_frames"])}))
                previous[side] = frame,state
                controllers[side] = frame+move["delay"],move["controller_frames"]
                index += 1
            buttons = []
            for controller in controllers:
                k = -1 if controller is None else frame-controller[0]
                buttons.append(controller[1][k] if controller is not None and 0 <= k < len(controller[1]) else 0)
            pool.step(buttons)
        assert index == len(moves)
        final = pool.states[game["side"]]
        score = 1 if final.outcome == 1 else 0 if final.outcome == 2 else .5
        assert score == game["score"] and final.frame == game["frames"], (path, "outcome")
    return roots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moves", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--native-library")
    parser.add_argument("--per-game", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    paths = sorted(args.moves.glob("*.json.gz"))
    roots = [root for path in paths for root in replay_roots(path,args.per_game,args.native_library)]
    if not roots:
        raise ValueError("no prepared decisions found in the supplied move files")
    policy = PlainPolicy(args.checkpoint, args.device, public_only=True)
    planner = NativeReachabilityRunner()
    pace = resolve_pace("frame_perfect")
    rows = []
    try:
        for state, source, record in roots:
            candidate = plan_candidates(planner,state,0,pace)
            record["candidates_0"] = int(candidate[-2].count)
            record["opponent_changed"] = not np.array_equal(source["opponent_board_planes"],state["opponent_board_planes"]) or source["opponent_pill"] != state["opponent_pill"]
            observations, infos = [], []
            for context in (source,state):
                obs,info = public_policy_inputs(candidate[0],context["opponent_board_planes"],candidate[2],
                    context["opponent_pill"],candidate[-1],[state["preview"]])
                observations.append(obs)
                infos.extend(info)
            scores = score_public_inputs(policy,np.concatenate(observations),infos)
            action = record["action"]
            record["old_context_choice_gap"] = float(scores[0].max()-scores[0,action])
            record["fresh_context_choice_gap"] = float(scores[1].max()-scores[1,action])
            record["fresh_context_changes_choice"] = bool(scores[1].argmax() != action)
            record["tied_best_actions"] = int(np.count_nonzero(scores[0] == scores[0].max()))
            for delay in (4,8):
                try:
                    delayed = plan_candidates(planner,state,delay,pace)
                    record[f"candidates_{delay}"] = int(delayed[-2].count)
                    record[f"chosen_reachable_{delay}"] = bool(delayed[-1][action] != 65535)
                except (ValueError,RuntimeError):
                    record[f"candidates_{delay}"] = 0
                    record[f"chosen_reachable_{delay}"] = False
            rows.append(record)
    finally:
        planner.close()
    report = {"games_replayed":len(paths), "seed_pairs":len({r["seed"] for r in rows}), "sampled_decisions":len(rows),
        "scope":"Evenly spaced prepared decisions within supplied whole games; side-swapped repetitions are correlated. Logit changes measure policy sensitivity, not competitive regret.",
        "opponent_changed_fraction":float(np.mean([r["opponent_changed"] for r in rows])),
        "fresh_context_changes_choice_fraction":float(np.mean([r["fresh_context_changes_choice"] for r in rows])),
        "old_context_max_choice_gap":max(r["old_context_choice_gap"] for r in rows),
        "tied_best_decisions":sum(r["tied_best_actions"]>1 for r in rows),
        "age_frames_median":float(np.median([r["age_frames"] for r in rows])),
        "age_frames_p95":float(np.quantile([r["age_frames"] for r in rows],.95)),
        "candidate_means":{str(d):float(np.mean([r[f"candidates_{d}"] for r in rows])) for d in (0,4,8)},
        "chosen_unreachable_fraction":{str(d):float(np.mean([not r[f"chosen_reachable_{d}"] for r in rows])) for d in (4,8)},
        "rows":rows}
    dump(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k != "rows"},indent=2))


if __name__ == "__main__":
    main()
