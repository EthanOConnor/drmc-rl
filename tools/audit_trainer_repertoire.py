"""Bounded public-state repertoire and pace-input audit; never changes a policy.

Collect fresh roots from the frozen incumbent, then compare complete candidate
sets at each live pace. Cost ablation holds the feasible set fixed. It measures
policy sensitivity, not a win-probability gain. No hidden state enters inference.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner, build_vs_reset_spec
from drmc_rl.execution.pace import resolve_pace
from drmc_rl.game.observation import board_bytes_to_semantic_planes, legacy_vs_policy_boards
from drmc_rl.human.afterstate_sim import NativeAfterstateSimulator
from drmc_rl.human.repertoire import placement_geometry
from drmc_rl.planning.fast_reach import FrameState, HoldDir, Rotation, compute_speed_threshold, simulate_frame
from drmc_rl.planning.native_reach import NativeReachabilityRunner, resolve_library_path
from drmc_rl.training.envs.start_bank import StartBank
from tools.annotate_replay_events import POSE_TO_ACTION
from tools.vs_head_to_head import PlainPolicy


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def frame_state(p):
    return FrameState(int(p["x"]), int(p["y_top"]), int(p["rot"]), int(p["sc"]),
                      int(p["hv"]), HoldDir(int(p["hd"])), int(p["parity"]), Rotation(int(p["rh"])))


def macro_costs(reach, same_color=False):
    result = np.full(512, 65535, np.uint16)
    for pose in np.flatnonzero(reach.costs_u16 != 65535):
        action = int(POSE_TO_ACTION[pose])
        if action >= 0 and (not same_color or action < 256):
            result[action] = reach.costs_u16[pose]
    return result


def infer(policy, rows, costs):
    obs, infos = [], []
    for r, cost in zip(rows, costs, strict=True):
        boards = legacy_vs_policy_boards(board_bytes_to_semantic_planes(r["board"]),
            board_bytes_to_semantic_planes(r["opponent"]), r["pill"], r["opponent_pill"])
        mask = (cost != 65535).reshape(4, 16, 8)
        obs.append(np.concatenate((boards, mask.astype(np.float32))))
        infos.append({"placements/feasible_mask": mask, "placements/cost_to_lock": cost.reshape(4, 16, 8),
            "next_pill_colors": r["pill"], "preview_pill": {
                "first_color": (1, 0, 2)[r["preview"][0]], "second_color": (1, 0, 2)[r["preview"][1]]}})
    actions, masks, scores = policy.score(np.asarray(obs), infos)
    result = []
    for aa, mm, ss, cost in zip(actions, masks, scores, costs, strict=True):
        row_scores = {int(a): float(s) for a, m, s in zip(aa, mm, ss, strict=True) if m}
        if (set(row_scores) != set(np.flatnonzero(cost != 65535))
                or not np.isfinite(list(row_scores.values())).all()):
            raise ValueError("policy must score every feasible candidate with finite logits")
        result.append(row_scores)
    return result


def collect(policy, planner, games, seed):
    def solve(states):
        return np.asarray([macro_costs(planner.bfs_full(s["cols"], frame_state(s),
            speed_threshold=int(s["thr"]))) for s in states])

    runner = DrMarioVsPoolRunner(num_pairs=games, plan_solver=solve)
    rng = np.random.default_rng(seed)
    specs, targets = [], []
    for i in range(games):
        level, speed = (10, 14, 20)[i % 3], (1, 2)[(i // 3) % 2]
        value = int(rng.integers(65536))
        specs.append(build_vs_reset_spec(level=(level, level), speed_setting=(speed, speed),
            rng_override=True, rng_state=(value & 255, value >> 8), frame_counter_base=i % 2))
        targets.append((0, 4, 12, 24)[(i // 6) % 4])
    chosen, terminated_before_target = {}, set()
    decisions = np.zeros(2*games, int)
    try:
        runner.reset(None, specs)
        for _ in range(256):
            buf = runner.buffers
            indices = np.flatnonzero(buf.need_action)
            roots = []
            for side in indices:
                pair = int(side // 2)
                p = buf.plan_state[side]
                speed = (1, 2)[(pair // 3) % 2]
                ups = next(n for n in range(50) if compute_speed_threshold(speed, n) == int(p["thr"]))
                row = {"game": pair, "side": int(side % 2), "level": (10, 14, 20)[pair % 3],
                    "speed": speed, "speed_ups": ups, "threshold": int(p["thr"]),
                    "board": buf.board_bytes[side].copy(), "opponent": buf.board_bytes[side ^ 1].copy(),
                    "pill": buf.pill_colors[side].tolist(), "preview": buf.preview_colors[side].tolist(),
                    "opponent_pill": buf.pill_colors[side ^ 1].tolist(),
                    "columns": p["cols"].copy(), "spawn": frame_state(p)}
                roots.append(row)
                if side % 2 == pair % 2 and decisions[side] == targets[pair] and pair not in chosen:
                    chosen[pair] = row
                decisions[side] += 1
            if len(chosen) + len(terminated_before_target) == games:
                break
            if not roots:
                break
            scores = infer(policy, roots, [buf.cost_to_lock[i] for i in indices])
            actions = np.full(2*games, -1, np.int32)
            for side, score in zip(indices, scores, strict=True):
                actions[side] = max(score, key=score.get) if score else -1
            runner.step(actions, None, None)
            terminated_before_target.update(int(p) for p in np.flatnonzero(buf.terminated) if p not in chosen)
        return list(chosen.values()), sorted(terminated_before_target)
    finally:
        runner.close()


def collect_bank(path, planner, games, seed):
    """One synthetic held-out root per source replay; no rollout-state leakage."""
    bank = StartBank(path)
    source = np.load(path, allow_pickle=False)
    names = source["quark_names"][source["quark_idx"]]
    indices, seen = [], set()
    for index in np.random.default_rng(seed).permutation(len(bank)):
        if str(names[index]) not in seen:
            indices.append(int(index))
            seen.add(str(names[index]))
        if len(indices) == games:
            break
    if len(indices) < games:
        raise ValueError(f"bank has only {len(indices)} distinct games; requested {games}")
    runner = DrMarioVsPoolRunner(num_pairs=1, plan_solver=lambda states: np.asarray([
        macro_costs(planner.bfs_full(s["cols"], frame_state(s), speed_threshold=int(s["thr"])))
        for s in states]))
    rows = []
    try:
        for game, index in enumerate(indices):
            side = game % 2
            speed = int(source["speeds"][index, side])
            runner.reset(None, [build_vs_reset_spec(level=tuple(map(int, source["levels"][index])),
                speed_setting=tuple(map(int, source["speeds"][index])), rng_override=True,
                rng_state=(1, 1), **bank.spec_kwargs(index))])
            buf, p = runner.buffers, runner.buffers.plan_state[side]
            if buf.terminated[0] or not buf.need_action[side]:
                raise ValueError("held-out curriculum root is not actionable")
            rows.append({"game": game, "source_game": str(names[index]), "source_row": index,
                "side": side, "level": int(source["levels"][index, side]), "speed": speed,
                "speed_ups": int(bank.speed_ups[index, side]), "threshold": int(p["thr"]),
                "board": buf.board_bytes[side].copy(), "opponent": buf.board_bytes[side ^ 1].copy(),
                "pill": buf.pill_colors[side].tolist(), "preview": buf.preview_colors[side].tolist(),
                "opponent_pill": buf.pill_colors[side ^ 1].tolist(),
                "columns": p["cols"].copy(), "spawn": frame_state(p)})
        return rows, []
    finally:
        runner.close()


def audit_root(row, pace, planner, policy, simulator):
    delay = max(8, pace.reaction_frames)
    spawn = row["spawn"]
    for _ in range(delay):
        spawn = simulate_frame(row["columns"], spawn, 0, speed_threshold=row["threshold"])
        if spawn.locked:
            return {"locked_during_reaction": True}
    # Runner results borrow reusable buffers; retain the paced witness before
    # the unrestricted ablation invokes that same runner again.
    reach = deepcopy(planner.bfs_full(row["columns"], spawn, speed_threshold=row["threshold"],
                                     **pace.planner_args(delay)))
    cost = macro_costs(reach, row["pill"][0] == row["pill"][1])
    if not np.any(cost != 65535):
        return {"no_candidates": True}
    unrestricted = planner.bfs_full(row["columns"], spawn, speed_threshold=row["threshold"])
    raw_cost = macro_costs(unrestricted, row["pill"][0] == row["pill"][1])
    if np.any((cost != 65535) & (raw_cost == 65535)):
        raise ValueError("paced candidate absent from unrestricted oracle")
    ablated_cost = np.where(cost != 65535, raw_cost, 65535).astype(np.uint16)
    scores, ablated = infer(policy, [row, row], [cost, ablated_cost])
    action, cost_control = max(scores, key=scores.get), max(ablated, key=ablated.get)
    actions = np.asarray(sorted(scores), np.int32)
    geometries = [placement_geometry(row["board"], row["pill"], int(a)) for a in actions]
    canonical_to_raw = np.asarray((1, 0, 2), np.uint8)
    effects = simulator.simulate_packed(fields=row["board"][None],
        pills=canonical_to_raw[np.asarray([row["pill"]])],
        previews=canonical_to_raw[np.asarray([row["preview"]])], candidate_actions=actions[None],
        candidate_costs=cost[actions][None], candidate_count=np.asarray([len(actions)]),
        speed=np.asarray([row["speed"]]), speed_ups=np.asarray([row["speed_ups"]]))
    if effects.invalid.any():
        raise ValueError("invalid exact afterstate")
    for i, geometry in enumerate(geometries):
        cleared = int(effects.viruses_cleared[i]) + int(effects.nonviruses_cleared[i])
        if (bool(geometry["first_wave_cells"]) != bool(effects.clear_events[i])
                or geometry["first_wave_cells"] > cleared):
            raise ValueError("first-wave geometry disagrees with native clear resolution")
    slot = int(np.flatnonzero(actions == action)[0])
    # Independently replay the selected witness with the full motor envelope.
    pose = int(np.flatnonzero(POSE_TO_ACTION == action)[0])
    witness = reach.script_for_pose(pose & 7, (pose >> 3) & 15, pose >> 7)
    actual = pace.validate(row["columns"], spawn, witness,
                           speed_threshold=row["threshold"], execution_delay=delay)
    if (actual["x"], actual["y"], actual["rotation"]) != (pose & 7, (pose >> 3) & 15, pose >> 7):
        raise ValueError("paced witness locked at a different candidate pose")
    return {"candidates": len(actions), "unrestricted_candidates": int((raw_cost != 65535).sum()),
        "cost_only_choice_changed": action != cost_control, "chosen_action": action,
        "unrestricted_cost_choice": cost_control,
        "cost_clip_fraction": float(np.mean(cost[actions] >= 256)),
        "selected_motion_frames": int(cost[action]), "selected_geometry": geometries[slot],
        "horizontal_available": any(g["horizontal_lines"] for g in geometries),
        "crossing_available": any(g["crossing_cells"] for g in geometries),
        "chosen_clear_events": int(effects.clear_events[slot]),
        "max_clear_events": int(effects.clear_events.max()),
        "chosen_viruses_cleared": int(effects.viruses_cleared[slot]),
        "max_viruses_cleared": int(effects.viruses_cleared.max()),
        "candidates_detail": [{"action": int(a), "policy_logit": scores[int(a)],
            "geometry": g, "clear_events": int(effects.clear_events[i]),
            "viruses_cleared": int(effects.viruses_cleared[i])}
            for i, (a, g) in enumerate(zip(actions, geometries, strict=True))]}


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--evaluation-checkpoint", type=Path,
                    help="Score another frozen model on the incumbent's identical collected roots")
    ap.add_argument("--source-bank", type=Path, help="Score a game-disjoint synthetic curriculum bank")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--paces", nargs="+", default=["sloth", "relaxed", "normal", "frame_perfect"])
    args = ap.parse_args()
    if args.output.exists() or not 1 <= args.games <= 96:
        ap.error("use a fresh output and 1..96 games")
    torch.set_num_threads(2)
    started = time.monotonic()
    planner = NativeReachabilityRunner(max_frames=2048)
    policy = PlainPolicy(args.checkpoint, device=args.device, public_only=True)
    rows, early = (collect_bank(args.source_bank, planner, args.games, args.seed) if args.source_bank
                   else collect(policy, planner, args.games, args.seed))
    scoring_path = args.evaluation_checkpoint or args.checkpoint
    if args.evaluation_checkpoint:
        policy = PlainPolicy(scoring_path, device=args.device, public_only=True)
    print(json.dumps({"collected": len(rows), "early_terminals": early}), flush=True)
    records = []
    with NativeAfterstateSimulator(num_envs=128) as simulator:
        for row in rows:
            record = {k: row[k] for k in ("game", "side", "level", "speed", "speed_ups")}
            record.update({k: row[k] for k in ("source_game", "source_row") if k in row})
            public_root = {k: row[k].tolist() if isinstance(row[k], np.ndarray) else row[k]
                for k in ("board", "opponent", "pill", "preview", "opponent_pill", "columns", "threshold")}
            public_root["spawn"] = {k: getattr(v, "value", v) for k, v in asdict(row["spawn"]).items()}
            record["public_root"] = public_root
            record["root_sha256"] = hashlib.sha256(json.dumps(public_root, sort_keys=True).encode()).hexdigest()
            record["paces"] = {name: audit_root(row, resolve_pace(name), planner, policy, simulator)
                               for name in args.paces}
            records.append(record)
            print(json.dumps({"audited": len(records), "elapsed_seconds": round(time.monotonic()-started, 1)}), flush=True)
    summary = {}
    for name in args.paces:
        values = [r["paces"][name] for r in records]
        live = [v for v in values if "candidates" in v]
        horizontal = [v for v in live if v["horizontal_available"]]
        summary[name] = {"states": len(values), "actionable": len(live),
            "locks_during_reaction": sum(v.get("locked_during_reaction", False) for v in values),
            "mean_candidates": float(np.mean([v["candidates"] for v in live])) if live else None,
            "cost_only_choice_changes": sum(v["cost_only_choice_changed"] for v in live),
            "horizontal_available": len(horizontal),
            "horizontal_taken_when_available": sum(v["selected_geometry"]["horizontal_lines"] > 0 for v in horizontal),
            "multi_event_clear_available": sum(v["max_clear_events"] > 1 for v in live),
            "multi_event_clear_selected": sum(v["chosen_clear_events"] > 1 for v in live)}
    report = {"schema": "drmc-trainer-repertoire-audit-v2", "diagnostic_only": True,
        "checkpoint_sha256": digest(scoring_path), "collector_checkpoint_sha256": digest(args.checkpoint),
        "planner_sha256": digest(resolve_library_path()),
        "source_sha256": {p: digest(p) for p in (str(Path(__file__)), "drmc_rl/human/repertoire.py",
            "drmc_rl/execution/pace.py", "drmc_rl/models/policy/candidate_policy_g5.py")},
        "runtime": {"torch": torch.__version__, "device": args.device},
        "seed": args.seed, "requested_games": args.games, "early_terminals": early,
        "source": "synthetic curriculum bank" if args.source_bank else "fresh incumbent self-play",
        "source_bank_sha256": digest(args.source_bank) if args.source_bank else None,
        "selection": ("one root per source replay; alternating root side" if args.source_bank else
            "one root per game; levels 10/14/20, MED/HI; target decisions 0/4/12/24; alternating side"),
        "limits": [resolve_pace(n).to_dict() for n in args.paces], "summary": summary, "rows": records,
        "limitations": ["Fixed-root diagnostics, not outcomes under paced continuation.",
            "First-wave geometry and settled native event counts do not label Sweet T or Fat Log.",
            "Cost-only choice changes are sensitivity, not evidence that ablation is stronger.",
            "Early terminal games are reported, not silently replaced."],
        "elapsed_seconds": time.monotonic()-started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
