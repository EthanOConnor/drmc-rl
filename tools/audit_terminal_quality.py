"""Pilot full-candidate terminal outcomes under the deployed competitive core.

This evaluates a different teacher from the failed shallow-search release.
It produces diagnostic evidence only and cannot open the existing quality gate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from drmc_rl.game.observation import board_bytes_to_semantic_planes
from drmc_rl.human.afterstate_runtime import AfterstatePolicyRuntime
from drmc_rl.search.native_pair import state_from_payload
from drmc_rl.search.pill_belief import CHANCE_MODEL_ID, PillReserveBelief
from drmc_rl.search.public_policy import PublicPolicyContinuation
from drmc_rl.teachers.counterfactual_release import sha256_file
from drmc_rl.teachers.terminal_rollout import (
    RolloutTask, aggregate_outcomes, reserve_hypotheses, rollout_tasks,
)
from drmc_rl.teachers.v3_baseline import load_source_rows
from tools.audit_rollout_consistency import select_rows


def score_v3(runtime, rows, states):
    requests = []
    for row, state in zip(rows, states, strict=True):
        side = int(row["root_side"])
        own, opponent = state.privileged.public.sides[side], state.privileged.public.sides[1-side]
        actions = np.asarray(state.legal_actions_by_side[side], np.int64)
        requests.append(dict(
            board_planes=board_bytes_to_semantic_planes(own.board),
            opponent_board_planes=board_bytes_to_semantic_planes(opponent.board),
            opponent_state_age_frames=abs(state.privileged.public.observable_clock_delta_frames),
            pill=own.pill, preview=own.preview, candidate_actions=actions,
            candidate_costs=np.asarray(state.action_costs_by_side[side], np.float32),
            candidate_mask=np.ones(len(actions), bool), rating=runtime.condition.mean,
            speed=state.speed_setting, speed_ups=int(row.get("speed_ups", 0)),
        ))
    return runtime.score_batch(requests)


def quality_summary(records, *, seed):
    eligible = [r for r in records if all(c["wdl"] is not None for c in r["candidates"])]
    selected = defaultdict(list)
    for row in eligible:
        by_action = {c["action"]: c for c in row["candidates"]}
        utility = {a: c["wdl"][0]+.5*c["wdl"][1] for a, c in by_action.items()}
        best = max(utility.values())
        for key in ("public_action", "v3_action"):
            selected[key].append(utility[row[key]])
            selected[key+"_regret"].append(best-utility[row[key]])
    difference = np.asarray(selected["public_action"])-np.asarray(selected["v3_action"])
    interval = None
    if len(difference) > 1:
        # Selection guarantees one position per game; each state is one unit.
        rng = np.random.default_rng(seed)
        boot = difference[rng.integers(0, len(difference), (4000, len(difference)))].mean(1)
        interval = np.quantile(boot, [.025, .975]).tolist()
    return {"states": len(records), "complete_states": len(eligible),
            "candidate_count": sum(len(r["candidates"]) for r in records),
            "incomplete_candidates": sum(c["wdl"] is None for r in records for c in r["candidates"]),
            "mean": {key: float(np.mean(values)) for key, values in selected.items() if values},
            "public_minus_v3_score_ci95": interval,
            "selection_game_count": len({r["game_id"] for r in records})}


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-events", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.states, args.batch_size, args.max_events) < 1:
        parser.error("state, batch, and event counts must be positive")
    raw_path = args.output.with_suffix(".rollouts.json")
    if args.output.exists() or raw_path.exists():
        raise FileExistsError("terminal-quality outputs already exist")
    torch.set_num_threads(2)
    started = time.monotonic()
    rows = select_rows(load_source_rows(args.state_bank), args.states, args.seed,
                       stratum_fields=("level", "speed"))
    states = [state_from_payload(row) for row in rows]
    continuation = PublicPolicyContinuation(args.checkpoint, device=args.device)
    public_scores = continuation.infer_batch([(s, int(r["root_side"]))
                                              for s, r in zip(states, rows, strict=True)])
    v3 = AfterstatePolicyRuntime(args.v3_checkpoint, device=args.device, seed=args.seed)
    try:
        v3_scores = score_v3(v3, rows, states)
    finally:
        v3.close()
    tasks, records, task_groups = [], [], []
    for row, state, public, v3_score in zip(rows, states, public_scores, v3_scores, strict=True):
        side = int(row["root_side"])
        legal = state.legal_actions_by_side[side]
        hypotheses = reserve_hypotheses(PillReserveBelief.from_dict(row["reserve_belief"]))
        public_action = max(legal, key=lambda action: public[0].get(action, 1e-8))
        v3_quality = np.asarray(v3_score["competitive_score"], np.float64)
        if not np.isfinite(v3_quality).all():
            raise ValueError("non-finite V3 candidate score")
        record = {"source_id": row["id"], "game_id": row["game_id"],
                  "stratum": [row[k] for k in ("level", "speed", "tactical_stratum")],
                  "public_action": int(public_action), "v3_action": int(legal[int(v3_quality.argmax())]),
                  "public_decision_value": public[1], "reserve_hypotheses": len(hypotheses),
                  "candidates": []}
        for action, quality in zip(legal, v3_quality, strict=True):
            group = []
            for reserve, weight in hypotheses:
                task = RolloutTask(len(tasks), state, side, action, reserve, weight)
                tasks.append(task)
                group.append(task)
            candidate = {"action": action, "public_policy_probability": public[0].get(action, 0.),
                         "v3_quality": float(quality), "wdl": None}
            record["candidates"].append(candidate)
            task_groups.append((candidate, group))
        records.append(record)
    print(json.dumps({"states": len(rows), "candidates": len(task_groups),
                      "rollouts": len(tasks), "distinct_reserve_counts": [r["reserve_hypotheses"] for r in records]}), flush=True)
    result = rollout_tasks(tasks, continuation, batch_size=args.batch_size, max_events=args.max_events,
        progress=lambda n: print(f"completed {n}/{len(tasks)} rollouts in {time.monotonic()-started:.1f}s", flush=True))
    by_id = {r["id"]: r for r in result}
    for candidate, group in task_groups:
        candidate["wdl"] = aggregate_outcomes(group, [by_id[t.id] for t in group])
    repo = Path(__file__).resolve().parents[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("x") as handle:
        handle.write(json.dumps(result, sort_keys=True)+"\n")
    report = {
        "schema": "drmc-terminal-quality-pilot-v1", "diagnostic_only": True,
        "product_gates_passed": False, "summary": quality_summary(records, seed=args.seed),
        "continuation": "fixed-public-core-both-sides-argmax-v1", "chance_model": CHANCE_MODEL_ID,
        "information_scope": "public-continuation-public-posterior-privileged-transition-v1",
        "critic_used": False, "full_posterior_mass": True, "candidate_truncation": 0,
        "max_events": args.max_events, "batch_size": args.batch_size, "seed": args.seed,
        "source_sha256": sha256_file(args.state_bank), "checkpoint_sha256": sha256_file(args.checkpoint),
        "v3_checkpoint_sha256": sha256_file(args.v3_checkpoint), "rollout_sha256": sha256_file(raw_path),
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "native_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo/"vendor/drmario_native", text=True).strip(),
        "source_code_sha256": {path: sha256_file(repo/path) for path in (
            "tools/audit_terminal_quality.py", "tools/audit_rollout_consistency.py",
            "drmc_rl/teachers/terminal_rollout.py", "drmc_rl/search/public_policy.py",
            "drmc_rl/search/pill_belief.py", "drmc_rl/search/native_pair.py",
            "drmc_rl/game/observation.py", "tools/vs_head_to_head.py")},
        "elapsed_seconds": time.monotonic()-started, "records": records,
    }
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
