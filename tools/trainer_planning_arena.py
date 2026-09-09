"""Paired trainer execution tournaments using real controller frames.

Run through the trainer-planning-arena program recipe. GPU wall time is not
pretended to be Mac wall time: compute availability is explicitly charged in
console frames, using the measured Mac budget in the experiment config.
Each variant's delay is its assumed fresh-decision deadline. Real hardware
acceptance must verify that deadline; a fast offline score does not establish it.
"""
from __future__ import annotations

import argparse
import cProfile
from collections import Counter
from datetime import UTC, datetime
import gzip
import json
import math
from pathlib import Path
import socket
import sqlite3
import time
import traceback

import numpy as np
import torch

from drmc_rl.arena.store import ArenaStore
from drmc_rl.arena.experiment import dump, outcome_summary
from drmc_rl.envs.backends.vs_frames import FrameVsPool
from drmc_rl.execution.pace import resolve_pace, strategy_context
from drmc_rl.human.anticipation import (
    NextTurnPreparer, execution_for_action, own_board_only, public_policy_inputs, score_public_inputs, select_prepared,
)
from drmc_rl.human.backend import NoReachablePlacement, plan_candidates
from drmc_rl.planning.native_reach import NativeReachabilityRunner
from tools.vs_head_to_head import PlainPolicy

FPS = 60.0988


def run_batch(config, match, jobs, policy, planner, preparer, *, policies=None):
    pace = resolve_pace(match.get("pace", "frame_perfect"))
    variants = config["variants"]
    count = len(jobs)
    controllers = [None] * (count * 2)
    last_spawn = [None] * (count * 2)
    prepared = [None] * (count * 2)
    ready_at = [0] * (count * 2)
    statistics = [Counter() for _ in controllers]
    moves = [[] for _ in jobs]
    replays = [[] for _ in jobs]
    was_falling = [False] * len(controllers)
    begun = time.perf_counter()
    budget = config.get("reactive_compute_frames", 3)
    preparation_budget = config.get("preparation_compute_frames", 6)
    max_frames = config.get("max_game_frames", 60000)
    replay_stride = max(1, math.ceil(match["games"] / max(2, config.get("replay_games", 16))))
    traced = [config.get("replay_games", 16) > 0 and (job[2] // 2) % replay_stride == 0 for job in jobs]
    with FrameVsPool(count, lib_path=config.get("native_library")) as pool:
        pool.reset([job[0] for job in jobs], level=match["level"])
        for frame in range(max_frames):
            states = pool.states
            if all(states[2*p].terminal for p in range(count)):
                break
            fresh, observations, infos, policy_ids = [], [], [], []
            for side, current in enumerate(states):
                pair, physical = divmod(side, 2)
                variant = match["a"] if physical == jobs[pair][1] else match["b"]
                params = variants[variant]
                if was_falling[side] and not current.falling:
                    statistics[side]["locks"] += 1
                    if controllers[side] is None:
                        statistics[side]["unplanned_locks"] += 1
                    controllers[side] = None
                was_falling[side] = current.falling
                key = (current.spawn_id, current.pill_counter_total)
                if not current.falling or last_spawn[side] == key:
                    continue
                last_spawn[side] = key
                statistics[side]["decisions"] += 1
                state = current.semantic(states[side ^ 1])
                if params.get("own_board_only", False):
                    state = own_board_only(state)
                anticipates = (params.get("anticipation", False) and pace.reaction_frames <= 6
                               and pace.reaction_frames < int(params["delay"]))
                if anticipates and policies is not None:
                    raise ValueError("mixed-policy preparation requires separate per-actor preparers")
                selected, reason = (None, "disabled")
                if anticipates:
                    if frame >= ready_at[side]:
                        selected, reason = select_prepared(prepared[side], state,
                            strict_opponent=params.get("strict_opponent", True))
                    else:
                        reason = "not_ready"
                    statistics[side]["cache_" + reason] += 1
                delay = 0 if selected else max(int(params["delay"]), pace.reaction_frames)
                if selected:
                    controllers[side] = (frame, selected)
                    fresh.append((side, state, None, selected, delay, reason, anticipates))
                else:
                    try:
                        candidate = plan_candidates(planner, state, delay, pace)
                    except NoReachablePlacement:
                        statistics[side]["no_reachable_after_delay"] += 1
                        prepared[side] = None
                        continue
                    obs, info = public_policy_inputs(candidate[0], candidate[1], candidate[2],
                        state["opponent_pill"], candidate[-1], [state["preview"]])
                    info[0]["pace/context"] = strategy_context(pace, state, delay)
                    legal_count = int(np.count_nonzero(info[0]["placements/feasible_mask"]))
                    statistics[side]["feasible_candidates"] += legal_count
                    statistics[side]["forced_placements"] += int(legal_count == 1)
                    observations.append(obs)
                    infos.extend(info)
                    policy_ids.append(variant)
                    fresh.append((side, state, candidate, None, delay, reason, anticipates))
            learning = {}
            if infos:
                obs_batch = np.concatenate(observations)
                all_scores = np.empty((len(infos),512), np.float32)
                groups = set(policy_ids) if policies is not None else {None}
                for id in groups:
                    indices = [i for i,v in enumerate(policy_ids) if id is None or v == id]
                    actor = policy if id is None else policies[id]
                    all_scores[indices] = score_public_inputs(actor, obs_batch[indices], [infos[i] for i in indices])
                    records = getattr(actor, "learning_records", None)
                    if records is not None:
                        learning.update(zip(indices,records))
                scored = iter(enumerate(all_scores))
            else:
                scored = iter(())
            for side, state, candidate, selected, delay, reason, anticipates in fresh:
                sample = None
                if selected is None:
                    score_index, scores = next(scored)
                    selected = execution_for_action(candidate, int(scores.argmax()), pace, delay=delay)
                    sample = learning.get(score_index)
                    if sample is not None and sample["action"] != selected["placement"]["action"]:
                        raise RuntimeError("learning record differs from the executed action")
                    controllers[side] = (frame + delay, selected)
                statistics[side]["spawn_wait_frames"] += delay
                pair, physical = divmod(side, 2)
                moves[pair].append({"frame": frame, "side": physical, "delay": delay, "cache": reason,
                    "placement": selected["placement"], "controller_frames": selected["controller_frames"],
                    "board": list(states[side].board), "opponent": list(states[side ^ 1].board),
                    "pill": state["pill"], "preview": state["preview"], "speed_ups": state["speed_ups"]})
                if sample is not None:
                    moves[pair][-1]["learning"] = sample
                if anticipates:
                    prepared[side] = preparer.prepare(state, selected, pace)
                    ready_at[side] = frame + (0 if delay == 0 else budget) + preparation_budget
            buttons = [0] * len(controllers)
            for side, controller in enumerate(controllers):
                if controller is None or not states[side].falling:
                    continue
                start, move = controller
                index = frame - start
                if index < 0:
                    continue
                if index >= len(move["controller_frames"]):
                    raise RuntimeError(f"controller script exhausted while still falling: {side}, {frame}")
                expected = move["controller_states"][index]
                current = states[side]
                held = current.held_buttons
                actual = (current.x, current.y_top, current.rotation, current.speed_counter,
                    current.horizontal_velocity, 1 if held & 2 else 2 if held & 1 else 0,
                    1 if held & 128 else 2 if held & 64 else 0, current.frame_parity)
                wanted = tuple(expected[k] for k in ("x", "y", "rotation", "speed_counter",
                    "horizontal_velocity", "hold_dir", "rotation_hold", "frame_parity"))
                if actual != wanted:
                    raise RuntimeError(f"controller state mismatch at side={side} frame={frame}: {actual} != {wanted}")
                buttons[side] = move["controller_frames"][index]
                statistics[side]["validated_input_frames"] += 1
            if frame % 2 == 0:
                for pair in range(count):
                    if not traced[pair] or states[2*pair].terminal:
                        continue
                    order = [2*pair + jobs[pair][1], 2*pair + 1 - jobs[pair][1]]
                    replays[pair].append({"frame": frame,
                        "boards": [list(states[s].board) for s in order],
                        "pills": [[states[s].x, states[s].y_top, states[s].rotation,
                            *states[s].pill] if states[s].falling else None for s in order],
                        "note": "Controller-frame replay · " + " / ".join(
                            f"{'A' if i == 0 else 'B'}: {buttons[s]:02x}" for i,s in enumerate(order))})
            pool.step(buttons)
        output = []
        for pair, (seed, assignment, job_index) in enumerate(jobs):
            a, b = 2*pair+assignment, 2*pair+1-assignment
            end = pool.states[a]
            outcome = end.outcome if end.terminal else None
            score = (
                None if outcome is None else 1.0 if outcome == 1 else 0.0 if outcome == 2 else 0.5
            )
            reason = "timeout" if not end.terminal else "clear" if any(pool.states[s].event_type == 1 for s in (a,b)) else "topout"
            row = {
                "seed": seed,
                "side": assignment,
                "index": job_index,
                "score": score,
                "winner": None
                if score is None
                else "a"
                if score == 1
                else "b"
                if score == 0
                else "draw",
                "reason": reason,
                "frames": int(end.frame),
                "a_stats": dict(statistics[a]),
                "b_stats": dict(statistics[b]),
            }
            output.append((row, moves[pair], replays[pair]))
    return output, time.perf_counter() - begun


def publish(config, results, output, store):
    tournaments, aggregates = [], {}
    for match in config["schedule"]:
        rows = results.get(match["id"], [])
        tournaments.append(
            {
                **match,
                "target": match["games"],
                "played": len(rows),
                **outcome_summary(rows),
                "status": "Complete"
                if len(rows) >= match["games"]
                else "Waiting for checkpoint"
                if not match_ready(config, match)
                else "Playing"
                if match["id"] == config.get("_current_match")
                else "Queued",
            }
        )
        for row in rows:
            for label in ("a", "b"):
                aggregates.setdefault(match[label], Counter()).update(row[label+"_stats"])
    metrics = []
    for variant, stats in aggregates.items():
        decisions = stats["decisions"]
        metrics.append({"label": f"{variant} · spawn wait", "value": f"{stats['spawn_wait_frames']/max(decisions,1):.2f} frames",
            "detail": f"{decisions:,} decisions · {stats['cache_hit']:,} exact preparation hits · {stats['cache_stale_opponent']:,} older opponent contexts"})
    dump(output / "results.json", {"updated_at": datetime.now(UTC).isoformat(), "tournaments": tournaments,
        "metrics": metrics, "execution_totals": {k: dict(v) for k,v in aggregates.items()},
        "compute_model": {k: config[k] for k in ("reactive_compute_frames", "preparation_compute_frames")},
        "worker": {"host":socket.gethostname(),"status":config.get("_worker_status","Running"),
                   "current_match":config.get("_current_match"),"error":config.get("_error"),
                   "traceback":config.get("_traceback"),"device":config.get("device","cuda")},
        "entrants": [{"id":id,"name":p["name"],"ready":variant_ready(config,p)}
                     for id,p in config["variants"].items()]})
    # Publish a closed, transactionally consistent database for the viewer.
    destination = output / "arena.next.sqlite"
    with sqlite3.connect(destination) as snapshot:
        store.conn.backup(snapshot)
    destination.replace(output / "arena.sqlite")


def paired_jobs(config, match):
    """Honor an explicit held-out bank, or draw the historical random schedule."""
    if match["games"] < 2 or match["games"] % 2:
        raise ValueError("tournaments require complete side-swapped seed pairs")
    excluded = set(config.get("seed_exclusions", []))
    if "seeds" in match:
        seeds = match["seeds"]
        if (len(seeds) != match["games"]//2 or len(set(seeds)) != len(seeds)
                or any(type(s) is not int or not 1 <= s <= 65535 or s in excluded for s in seeds)):
            raise ValueError("explicit tournament seeds must be unique, valid, allowed, and complete")
    else:
        rng = np.random.default_rng(match["seed"])
        available = np.setdiff1d(np.arange(1, 65536), list(excluded))
        seeds = rng.choice(available, match["games"]//2, replace=False)
    return [(int(seed), side, 2*i+side) for i, seed in enumerate(seeds) for side in (0, 1)]


def variant_ready(config, params):
    paths = [params.get("checkpoint",config["checkpoint"])]
    if "adapter_checkpoint" in params:
        paths.append(params["adapter_checkpoint"])
    if not all(Path(path).is_file() for path in paths):
        return False
    gate = params.get("ready_when")
    if gate:
        try:
            state = json.loads(Path(gate["path"]).read_text())
        except (OSError,ValueError):
            return False
        return state.get(gate["field"]) == gate["equals"]
    return True


def match_ready(config, match):
    return all(variant_ready(config,config["variants"][match[side]]) for side in ("a","b"))


def next_live_match(config, results):
    """Give every ready matchup an initial batch before deepening coverage."""
    pending = [match for match in config["schedule"]
               if len(results.get(match["id"],[])) < match["games"] and match_ready(config,match)]
    return min(pending,key=lambda m:len(results.get(m["id"],[])),default=None)


def variant_policy(config, params, parent):
    """A frozen core override is a different player, not another parent alias."""
    device = config.get("device","cuda")
    checkpoint = params.get("checkpoint",config["checkpoint"])
    if "adapter_checkpoint" in params:
        from drmc_rl.models.policy.pace_adapter import PacePolicy
        actor = PacePolicy(checkpoint,device,adapter_path=params["adapter_checkpoint"])
    elif checkpoint != config["checkpoint"]:
        actor = PlainPolicy(Path(checkpoint),device,public_only=True)
        if actor.aux_dim and actor.aux_spec != "zero_v1_vs":
            raise ValueError("historical public opponents must have a public auxiliary-input contract")
    else:
        return parent
    if config.get("memoize",False):
        from tools.trainer_arena_cache import MemoPolicy
        actor = MemoPolicy(actor)
    return actor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.get("threads", 1))
    torch.set_num_interop_threads(1)
    if config.get("strict_fp32",False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    policy = PlainPolicy(Path(config["checkpoint"]), config.get("device", "cuda"), public_only=True)
    planner = NativeReachabilityRunner()
    if config.get("memoize", False):
        from tools.trainer_arena_cache import MemoPlanner, MemoPolicy
        policy, planner = MemoPolicy(policy), MemoPlanner(planner)
    mixed = any("adapter_checkpoint" in p or "checkpoint" in p for p in config["variants"].values())
    if mixed and any(p.get("anticipation") for p in config["variants"].values()):
        raise ValueError("mixed-policy evaluation currently requires reaction-covered computation")
    policies = {} if mixed else None
    preparer = None if policies is not None else NextTurnPreparer(policy, planner, lib_path=config.get("native_library"))
    store = ArenaStore(config["working_db"], replay_dir=output / "replays")
    for id, params in config["variants"].items():
        store.register(agent_id=id, name=params["name"], family="trainer planning", generation=1,
            checkpoint=params.get("adapter_checkpoint", params.get("checkpoint",config["checkpoint"])),
            params={"parent_checkpoint":params.get("checkpoint",config["checkpoint"]), **params}, status="active")
    records = output / "games.jsonl"
    results = {}
    if records.exists():
        for line in records.read_text().splitlines():
            row = json.loads(line)
            results.setdefault(row["comparison"], []).append(row)
    records.touch(exist_ok=True)
    publish(config, results, output, store)
    try:
        schedule = iter(config["schedule"])
        while True:
            if config.get("watch",False):
                match = next_live_match(config,results)
                if match is None:
                    complete = all(len(results.get(m["id"],[])) >= m["games"] for m in config["schedule"])
                    config.update(_worker_status="Complete" if complete else "Waiting for checkpoints",_current_match=None)
                    publish(config,results,output,store)
                    if complete:
                        break
                    time.sleep(min(60,max(5,config.get("poll_seconds",20))))
                    continue
            else:
                match = next(schedule,None)
                if match is None:
                    break
            if policies is not None:
                for id in (match["a"],match["b"]):
                    if id not in policies:
                        policies[id] = variant_policy(config,config["variants"][id],policy)
            jobs = paired_jobs(config, match)
            completed = {row["index"] for row in results.get(match["id"], [])}
            jobs = [job for job in jobs if job[2] not in completed]
            batch_size = config.get("pairs", 16)
            if batch_size < 2 or batch_size%2:
                raise ValueError("evaluation batches require complete side-swapped pairs")
            if config.get("watch",False):
                jobs = jobs[:batch_size]
            for start in range(0, len(jobs), batch_size):
                config.update(_worker_status="Playing",_current_match=match["id"])
                publish(config,results,output,store)
                profile = cProfile.Profile() if config.get("profile") else None
                if profile:
                    profile.enable()
                batch, elapsed = run_batch(config, match, jobs[start:start+batch_size], policy, planner, preparer, policies=policies)
                if profile:
                    profile.disable()
                    profile.dump_stats(output / "profile.pstats")
                censored_seeds = {r["seed"] for r, _, _ in batch if r["reason"] == "timeout"}
                for row, moves, replay in batch:
                    row.update(comparison=match["id"], level=match["level"], pace=match.get("pace", "frame_perfect"))
                    results.setdefault(match["id"], []).append(row)
                    trace = output / "moves" / f"{match['id']}-{row['index']:04d}.json.gz"
                    trace.parent.mkdir(exist_ok=True)
                    with gzip.open(trace, "wt") as stream:
                        json.dump({"game": row, "moves": moves}, stream)
                    if row["seed"] not in censored_seeds:
                        store.record(
                            match["a"],
                            match["b"],
                            seed=row["seed"],
                            side=row["side"],
                            winner=row["winner"],
                            match_len_sec=row["frames"] / FPS,
                            decisions=row["a_stats"].get("decisions", 0)
                            + row["b_stats"].get("decisions", 0),
                            terminal_reason=row["reason"],
                            replay=replay,
                            match_key=f"{match['id']}-{row['index']}",
                            level=match["level"],
                            speed_setting=2,
                            provenance={"controller_frames": True, "move_trace": str(trace.name)},
                            commit=False,
                        )
                    with records.open("a") as stream:
                        stream.write(json.dumps(row)+"\n")
                store.record_worker_sample(worker_id=f"{socket.gethostname()}-trainer-frames-{Path(config['working_db']).stem}", device=config.get("device", "cuda"),
                    threads=config.get("threads",1), batch_size=len(batch), agent_a=match["a"], agent_b=match["b"],
                    games=len(batch), simulated_frames=sum(r[0]["frames"] for r in batch),
                    decisions=sum(r[0]["a_stats"].get("decisions",0)+r[0]["b_stats"].get("decisions",0) for r in batch),
                    wall_seconds=elapsed)
                publish(config, results, output, store)
                rows = results[match["id"]]
                print(
                    json.dumps(
                        {
                            "comparison": match["id"],
                            "games": len(rows),
                            "target": match["games"],
                            **outcome_summary(rows),
                            "batch_seconds": round(elapsed, 2),
                        }
                    ),
                    flush=True,
                )
        config.update(_worker_status="Complete",_current_match=None)
        publish(config,results,output,store)
    except BaseException as error:
        config.update(_worker_status="Failed",_error=str(error),_traceback=traceback.format_exc())
        publish(config,results,output,store)
        raise
    finally:
        store.close()
        if preparer is not None:
            preparer.close()
        planner.close()


if __name__ == "__main__":
    main()
