"""Compare corrected search predictions on exact held-out observed actions.

This diagnostic retains the old frozen calibration to isolate input/search
corrections. It does not produce a complete candidate release or promote a gate.
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from drmc_rl.eval.wdl_calibration import paired_game_bootstrap, weighted_metrics
from drmc_rl.search.joint_event import LEAF_VALUE_CONTRACT, JointEventSearch, SearchConfig, WDL
from drmc_rl.search.strong_league_memberwise import frozen_strong_league_memberwise_factory
from drmc_rl.teachers.bootstrap_comparison import load_bootstrap_bundle
from drmc_rl.teachers.counterfactual_release import sha256_file
from drmc_rl.teachers.release_analysis import load_release


def score_public_values(policy, rows, *, batch_size=32):
    """Batched decision values using exactly the public zero-aux actor inputs."""
    import torch
    from drmc_rl.game.observation import board_bytes_to_semantic_planes, legacy_vs_policy_boards
    from drmc_rl.models.policy.candidate_packing import pack_feasible_candidates
    from drmc_rl.search.native_pair import state_from_payload

    scores = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start+batch_size]
        states = [state_from_payload(row) for row in batch]
        width = max(32, max(len(s.legal_actions_by_side[int(r["root_side"])])
                            for s, r in zip(states, batch, strict=True)))
        observations, pills, previews, actions, costs, masks = [], [], [], [], [], []
        for state, row in zip(states, batch, strict=True):
            side = int(row["root_side"])
            if not state.privileged.need_action[side]:
                raise ValueError("public value audit requires an acting root side")
            own, opponent = state.privileged.public.sides[side], state.privileged.public.sides[1-side]
            legal = state.legal_actions_by_side[side]
            feasible = np.zeros(512, dtype=bool)
            cost = np.full(512, 0xFFFF, dtype=np.uint16)
            feasible[list(legal)] = True
            cost[list(legal)] = state.action_costs_by_side[side]
            if own.pill[0] == own.pill[1]:
                feasible[256:] = False
                cost[256:] = 0xFFFF
            feasible = feasible.reshape(4, 16, 8)
            packed = pack_feasible_candidates(feasible, cost.reshape(4, 16, 8),
                                               max_candidates=width, sort_by_cost=True)
            if packed.count != feasible.sum():
                raise RuntimeError("public value audit truncated a feasible candidate")
            board = legacy_vs_policy_boards(
                board_bytes_to_semantic_planes(own.board),
                board_bytes_to_semantic_planes(opponent.board), own.pill, opponent.pill)
            observations.append(np.concatenate((board, feasible.astype(np.float32))))
            pills.append(own.pill)
            previews.append(own.preview)
            actions.append(packed.actions)
            costs.append(packed.cost)
            masks.append(packed.mask)
        def tensor(values, dtype):
            return torch.as_tensor(np.asarray(values), dtype=dtype, device=policy.device)
        with torch.inference_mode():
            _logits, value = policy.net(
                tensor(observations, torch.float32), tensor(pills, torch.int64),
                tensor(previews, torch.int64), tensor(actions, torch.int32),
                tensor(costs, torch.float32), tensor(masks, torch.bool),
                aux=torch.zeros((len(batch), policy.aux_dim), device=policy.device),
            )
        scores.extend(value.reshape(-1).float().cpu().tolist())
    result = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("non-finite public decision value")
    return result


def audit_public_value(args):
    import torch
    from drmc_rl.eval.wdl_calibration import fit_parameters, outcome_game_counts, probabilities
    from drmc_rl.teachers.v3_baseline import game_set_sha256, load_source_rows
    from tools.vs_head_to_head import PlainPolicy

    torch.set_num_threads(2)
    if args.output.exists() or args.output.with_suffix(".scores.npz").exists():
        raise FileExistsError("public value audit output already exists")
    if args.calibration_bank is None:
        raise ValueError("public value audit requires an independent calibration bank")
    baseline, baseline_manifest = load_bootstrap_bundle(args.bootstrap, args.bootstrap_manifest)
    by_id = {r.source_id: r for r in baseline}
    calibration = load_source_rows(args.calibration_bank)
    evaluation = [r for r in load_source_rows(args.state_bank) if r["id"] in by_id]
    if len(evaluation) != len(baseline):
        raise ValueError("public value evaluation does not cover the frozen baseline")
    if sha256_file(args.calibration_bank) != baseline_manifest["calibration_bank_sha256"]:
        raise ValueError("public and V3 calibration banks differ")
    calibration_groups = np.asarray([r["game_id"] for r in calibration])
    evaluation_groups = np.asarray([r["game_id"] for r in evaluation])
    if set(calibration_groups) & set(evaluation_groups):
        raise ValueError("public value calibration leaks evaluation games")
    outcomes = {"win": 0, "draw": 1, "loss": 2}
    calibration_targets = np.asarray([outcomes[r["outcome"]] for r in calibration])
    targets = np.asarray([by_id[r["id"]].outcome for r in evaluation])
    for row in evaluation:
        original = by_id[row["id"]]
        if (row["game_id"] != original.game_id or outcomes[row["outcome"]] != original.outcome
                or int(row["observed_action"]) != original.observed_action):
            raise ValueError("public value source differs from held-out baseline")
    policy = PlainPolicy(args.public_checkpoint, device=args.device, public_only=True)
    policy.net.eval()
    if policy.in_channels != 20:
        raise ValueError("public value audit requires the full-pair competitive model")
    calibration_scores = score_public_values(policy, calibration)
    evaluation_scores = score_public_values(policy, evaluation)
    parameters = fit_parameters(calibration_scores, calibration_targets, calibration_groups)
    predicted = probabilities(evaluation_scores, parameters)
    baseline_wdl = np.asarray([by_id[r["id"]].baseline_wdl for r in evaluation])
    report = {
        "schema": "drmc-public-value-audit-v1", "diagnostic_only": True, "product_gates_passed": False,
        "score": "public competitive decision value before the observed action; not candidate regret",
        "checkpoint_sha256": sha256_file(args.public_checkpoint),
        "calibration_bank_sha256": sha256_file(args.calibration_bank),
        "evaluation_source_sha256": sha256_file(args.state_bank),
        "baseline_manifest_sha256": sha256_file(args.bootstrap_manifest),
        "calibration_rows": len(calibration), "calibration_games": len(set(calibration_groups)),
        "calibration_outcome_games": outcome_game_counts(calibration_targets, calibration_groups),
        "evaluation_rows": len(evaluation), "evaluation_games": len(set(evaluation_groups)),
        "evaluation_outcome_games": outcome_game_counts(targets, evaluation_groups),
        "calibration_game_set_sha256": game_set_sha256(calibration_groups),
        "evaluation_game_set_sha256": game_set_sha256(evaluation_groups),
        "game_sets_disjoint": True, "parameters": parameters.to_dict(),
        "metrics": {"public_value": weighted_metrics(predicted, targets, evaluation_groups),
                    "v3": weighted_metrics(baseline_wdl, targets, evaluation_groups)},
        "public_value_minus_v3": paired_game_bootstrap(
            predicted, baseline_wdl, targets, evaluation_groups,
            seed=args.seed, samples=args.bootstrap_samples),
        "seed": args.seed, "bootstrap_samples": args.bootstrap_samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    archive = args.output.with_suffix(".scores.npz")
    np.savez_compressed(archive, calibration_scores=calibration_scores,
        calibration_targets=calibration_targets, calibration_groups=calibration_groups,
        evaluation_scores=evaluation_scores, evaluation_targets=targets,
        evaluation_groups=evaluation_groups, source_ids=np.asarray([r["id"] for r in evaluation]),
        public_value_wdl=predicted, v3_wdl=baseline_wdl)
    repo = Path(__file__).resolve().parents[1]
    report["scores_sha256"] = sha256_file(archive)
    report["source_code_sha256"] = {path: sha256_file(repo/path) for path in (
        "tools/audit_search_predictions.py", "tools/vs_head_to_head.py",
        "drmc_rl/game/observation.py", "drmc_rl/eval/wdl_calibration.py")}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


def audit(args):
    import torch

    torch.set_num_threads(2)
    if args.output.exists() or args.output.with_suffix(".rows.jsonl.gz").exists():
        raise FileExistsError("prediction audit output already exists")
    baseline, baseline_manifest = load_bootstrap_bundle(args.bootstrap, args.bootstrap_manifest)
    baseline = {row.source_id: row for row in baseline}
    reference = load_release([args.reference_release])
    source_sha = sha256_file(args.state_bank)
    if reference.settings["input_sha256"] != source_sha:
        raise ValueError("reference release uses a different source bank")
    if reference.settings["mixture_manifest_sha256"] != sha256_file(args.mixture_manifest):
        raise ValueError("reference release uses a different frozen mixture")
    if reference.settings["wdl_calibration_sha256"] != sha256_file(args.wdl_calibration):
        raise ValueError("reference release uses a different W/D/L calibration")
    config = SearchConfig(depth_events=2, own_beam=512, opponent_beam=8, chance_beam=9)
    for key, value in asdict(config).items():
        if key in reference.settings["search"] and reference.settings["search"][key] != value:
            raise ValueError(f"reference search differs in {key}")
    if args.public_search_calibration:
        from drmc_rl.search.public_policy import public_policy_belief_factory
        ensemble, decode = public_policy_belief_factory(args)
    else:
        ensemble, decode = frozen_strong_league_memberwise_factory(args)
    records = []
    try:
        with gzip.open(args.state_bank, "rt") as handle:
            for line in handle:
                payload = json.loads(line)
                row = baseline.get(payload["id"])
                if row is None:
                    continue
                if (payload["game_id"] != row.game_id or
                        int(payload["observed_action"]) != row.observed_action):
                    raise ValueError("held-out observed action/game does not match source state")
                state = decode(payload)
                values = []
                nodes = 0
                for model in ensemble.models:
                    result = JointEventSearch(model, config).search(
                        state, root_side=int(payload["root_side"]), root_actions=[row.observed_action]
                    )
                    if result.budget_exhausted:
                        raise RuntimeError(f"search budget exhausted for {row.source_id}")
                    values.append(result.values[0])
                    nodes += result.nodes
                value = WDL.mixture(ensemble.weights, values)
                old = reference.states[row.source_id].candidates[row.observed_action]
                records.append({
                    "source_id": row.source_id, "game_id": row.game_id,
                    "observed_action": row.observed_action, "outcome": row.outcome,
                    "stratum": row.stratum, "nodes": nodes,
                    "corrected": [value.win, value.draw, value.loss],
                    "previous": [old["win"], old["draw"], old["loss"]],
                    "v3": row.baseline_wdl,
                    "member_wdl": [[v.win, v.draw, v.loss] for v in values],
                })
                if len(records) % 50 == 0:
                    print(f"scored {len(records)}/{len(baseline)} observed actions", flush=True)
        if len(records) != len(baseline):
            raise ValueError("not every held-out baseline row was scored")
        groups = np.asarray([r["game_id"] for r in records])
        targets = np.asarray([r["outcome"] for r in records])
        predictions = {name: np.asarray([r[name] for r in records])
                       for name in ("corrected", "previous", "v3")}
        metrics = {name: weighted_metrics(p, targets, groups) for name, p in predictions.items()}
        comparisons = {name: paired_game_bootstrap(
            predictions["corrected"], predictions[name], targets, groups,
            seed=args.seed, samples=args.bootstrap_samples,
        ) for name in ("previous", "v3")}
        repo = Path(__file__).resolve().parents[1]
        rows_path = args.output.with_suffix(".rows.jsonl.gz")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with rows_path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as handle:
                for row in records:
                    handle.write((json.dumps(row, sort_keys=True)+"\n").encode())
        report = {
            "schema": "drmc-search-prediction-audit-v1", "diagnostic_only": True,
            "product_gates_passed": False, "leaf_value_contract": LEAF_VALUE_CONTRACT,
            "search": asdict(config), "seed": args.seed, "bootstrap_samples": args.bootstrap_samples,
            "rows": len(records), "games": len(set(groups)),
            "teacher_ids": ensemble.ids, "teacher_weights": ensemble.weights,
            "information_scope": sorted({m.information_scope for m in ensemble.models}),
            "metrics": metrics, "corrected_minus_reference": comparisons,
            "source_sha256": source_sha,
            "mixture_manifest_sha256": sha256_file(args.mixture_manifest),
            "calibration_sha256": sha256_file(args.public_search_calibration or args.wdl_calibration),
            "reference_calibration_sha256": sha256_file(args.wdl_calibration),
            "public_checkpoint_sha256": sha256_file(args.public_checkpoint) if args.public_checkpoint else None,
            "public_calibration_sha256": sha256_file(args.public_search_calibration) if args.public_search_calibration else None,
            "baseline_manifest_sha256": sha256_file(args.bootstrap_manifest),
            "baseline_rows_sha256": sha256_file(args.bootstrap),
            "baseline_provenance": baseline_manifest,
            "reference_release_sha256": list(reference.release_sha256),
            "rows_sha256": sha256_file(rows_path), "rows_path": str(rows_path),
            "source_code_sha256": {path: sha256_file(repo/path) for path in (
                "tools/audit_search_predictions.py", "drmc_rl/search/joint_event.py",
                "drmc_rl/search/native_pair.py", "drmc_rl/search/belief_native_pair.py",
                "drmc_rl/search/strong_league.py", "drmc_rl/search/strong_league_memberwise.py",
                "drmc_rl/search/public_policy.py", "tools/vs_head_to_head.py",
                "drmc_rl/game/observation.py")},
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "native_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo/"vendor/drmario_native", text=True).strip(),
            "interpretation": (
                "Same held-out actions, search depth and beams. Public outcome-trained continuation replaces the failed legacy mixture; its independently fitted calibration is diagnostic."
                if args.public_search_calibration else
                "Same frozen members/calibration and held-out actions; corrected observation support and leaf boundaries. Incomplete candidate coverage, so diagnostic only."
            ),
        }
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({k: report[k] for k in ("rows", "games", "metrics", "corrected_minus_reference")}, indent=2))
    finally:
        for model in ensemble.models:
            model.runner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("state-bank", "mixture-manifest", "wdl-calibration", "bootstrap",
                 "bootstrap-manifest", "reference-release", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--public-checkpoint", type=Path,
                        help="screen the public competitive decision value instead of running search")
    parser.add_argument("--calibration-bank", type=Path)
    parser.add_argument("--public-search-calibration", type=Path,
                        help="use the public competitive core in search with this frozen diagnostic W/D/L link")
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    if args.public_search_calibration and not args.public_checkpoint:
        parser.error("--public-search-calibration requires --public-checkpoint")
    if args.public_checkpoint and not args.public_search_calibration:
        audit_public_value(args)
    else:
        audit(args)


if __name__ == "__main__":
    main()
