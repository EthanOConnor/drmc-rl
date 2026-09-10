"""Compare adaptive full-pair root allocation with complete queued search.

This registered diagnostic uses an explicitly uncalibrated critic. It checks
response bounds and actual work; no resulting row is a quality-training label.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from drmc_rl.arena.experiment import dump
from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner
from drmc_rl.search.adaptive_event import AdaptiveJointEventSearch
from drmc_rl.search.belief_native_pair import BeliefNativePairSearchModel
from drmc_rl.search.joint_event import SearchConfig
from drmc_rl.search.native_pair import TACTICAL_PREDICATE, state_from_payload
from drmc_rl.search.pill_belief import PillReserveBelief
from drmc_rl.search.queued_event import QueuedJointEventSearch
from drmc_rl.search.strong_league import DavidsonCalibration
from drmc_rl.teachers.counterfactual_release import sha256_file
from drmc_rl.teachers.v3_baseline import load_source_rows
from tools.benchmark_search_frontiers import MeteredContinuation


def full_utilities(result):
    # SearchResult.utilities is a display float32 array. Retain the actual
    # backed-up values for a numerical certificate against unilateral roots.
    if result.joint_utilities is not None:
        return result.joint_utilities
    return np.asarray([v.utility for v in result.values], np.float64)


def compare_adaptive_result(complete, actual):
    if complete.budget_exhausted or not complete.equilibrium_converged:
        raise RuntimeError("complete reference did not finish with a valid equilibrium")
    if set(complete.actions) != set(actual.actions):
        raise RuntimeError("adaptive search changed the complete legal inventory")
    own_indices = [complete.actions.index(a) for a in actual.actions]
    reference = full_utilities(complete)
    simultaneous = complete.joint_utilities is not None
    if simultaneous:
        if set(complete.opponent_actions) != set(actual.opponent_actions):
            raise RuntimeError("adaptive search changed the complete opponent inventory")
        reference = reference[np.ix_(own_indices,
            [complete.opponent_actions.index(a) for a in actual.opponent_actions])]
        gap = max(0., float((reference @ actual.opponent_policy).max()
                           - (actual.policy_target @ reference).min()))
    else:
        reference = reference[own_indices]
        gap = max(0., float(reference.max() - actual.policy_target @ reference))
    if reference.shape != actual.utility_lower.shape or reference.shape != actual.utility_upper.shape:
        raise RuntimeError("adaptive interval shape does not match the full reference")
    violation = float(max(0., np.max(actual.utility_lower - reference),
                          np.max(reference - actual.utility_upper)))
    comparison = dict(certified=actual.certified, stop_reason=actual.stop_reason,
        nested=actual.nested, total_allocated_joint_actions=actual.total_allocated_joint_actions,
        response_gap_bound=actual.gap_upper, interval_violation=violation,
        bounds_hold=violation <= 1e-12 and gap <= actual.gap_upper + 1e-12)
    if simultaneous:
        comparison.update(evaluated_joint_actions=int(actual.evaluated.sum()),
            total_joint_actions=int(actual.evaluated.size), full_matrix_response_gap=gap)
    else:
        comparison.update(evaluated_root_actions=int(actual.evaluated.sum()),
            total_root_actions=int(actual.evaluated.size), full_vector_regret=gap)
    return comparison


def audit(config):
    output = Path(config["output"])
    if output.exists():
        raise FileExistsError("adaptive audit identity already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(config.get("threads", 1)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    continuation = MeteredContinuation(Path(config["checkpoint"]),
        DavidsonCalibration(1., 0., -3., "uncalibrated-adaptive-mechanics-probe"),
        device=config.get("device", "cpu"))
    search_config = SearchConfig(depth_events=int(config.get("depth_events", 1)),
        tactical_extension_events=config.get("tactical_extension_events", 0),
        policy_temperature=float(config.get("policy_temperature", .25)),
        opponent_mode="mixed", max_nodes=int(config.get("max_nodes", 100000)),
        matrix_time_limit_seconds=float(config.get("matrix_time_limit_seconds", .25)))
    batches = tuple(config.get("allocation_batches", (16, 32)))
    if (not batches or len(set(batches)) != len(batches)
            or any(type(b) is not int or b < 1 for b in batches)):
        raise ValueError("allocation batches must be distinct positive integers")
    modes = tuple(config.get('allocation_modes', ('root',)))
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= {'root', 'nested'}:
        raise ValueError('allocation modes must be distinct root/nested entries')
    allocations = {('nested-'+str(b) if mode == 'nested' else str(b)): (mode == 'nested', b)
                   for mode in modes for b in batches}
    evaluation_tolerance = float(config.get("evaluation_tolerance", 1e-5))
    response_gap = float(config.get("response_gap", .02))
    rows = load_source_rows(Path(config["state_bank"]))[:int(config.get("states", 4))]
    if not rows:
        raise ValueError("adaptive audit needs real roots")
    report = dict(schema="drmc-adaptive-search-audit-v1", status="Running", records=[],
        config=config, search_config=asdict(search_config), diagnostic_only=True,
        calibrated=False, usable_for_quality_training=False, product_gates_passed=False,
        checkpoint_sha256=sha256_file(Path(config["checkpoint"])),
        source_sha256=sha256_file(Path(config["state_bank"])),
        native_sha256=sha256_file(Path(config["native_library"])))
    report["tactical_predicate"] = TACTICAL_PREDICATE if search_config.tactical_extension_events else None
    warmup = QueuedJointEventSearch(None, search_config)
    warmup._solve_payoffs(np.array([[1., -1.], [-1., 1.]]))
    report["matrix_cold_start_ms"] = warmup._matrix_solve_ms
    dump(output, report)
    try:
        variants = (["unextended"] if config.get("compare_unextended", False) else []) + ["complete", *allocations]
        for row_index, row in enumerate(rows):
            state, side = state_from_payload(row), int(row["root_side"])
            continuation.batch_sizes = []
            continuation.infer_batch([(state, side)])
            rotation = row_index % len(variants)
            order = variants[rotation:] + variants[:rotation]
            record = dict(source_id=row["id"], order=order,
                          root_boundary=state.privileged.decision_boundary.value, root_side=side,
                          public_observation_schema=state.public_observation_schema, variants={})
            report["records"].append(record)
            report["current_source_id"] = row["id"]
            results = {}
            for name in order:
                report["current_variant"] = name
                dump(output, report)
                runner = DrMarioVsPoolRunner(num_pairs=1, lib_path=config["native_library"])
                try:
                    model = BeliefNativePairSearchModel(runner, continuation=continuation,
                        belief_cache_size=int(config.get('belief_cache_size',65536)))
                    model.register_belief(state, PillReserveBelief.from_dict(row["reserve_belief"]))
                    continuation._cache.clear()
                    continuation.batch_sizes = []
                    kwargs = dict(batch_size=int(config.get("batch_size", 32)))
                    if name in ("complete", "unextended"):
                        settings = (replace(search_config, tactical_extension_events=0)
                                    if name == "unextended" else search_config)
                        search = QueuedJointEventSearch(model, settings, **kwargs)
                    else:
                        nested, allocation_batch = allocations[name]
                        search = AdaptiveJointEventSearch(model, search_config, **kwargs,
                            allocation_batch=allocation_batch, nested=nested, response_gap=response_gap,
                            evaluation_tolerance=evaluation_tolerance,
                            max_root_actions=int(config.get("max_root_actions", 512)),
                            max_joint_actions=int(config.get("max_joint_actions", 262144)))
                    start = time.monotonic()
                    result = search.search(state, root_side=side,
                        root_actions=state.legal_actions_by_side[side])
                    elapsed = time.monotonic() - start
                    results[name] = result
                    summary = dict(seconds=elapsed, nodes=result.nodes,
                        inference_rows=sum(continuation.batch_sizes),
                        inference_calls=len(continuation.batch_sizes),
                        largest_batch=max(continuation.batch_sizes, default=0),
                        matrix_solve_ms=result.matrix_solve_ms,
                        tactical_extensions=result.tactical_extensions,
                        tactical_reasons=dict(result.tactical_reasons))
                    if name in ("complete", "unextended"):
                        summary.update(actions=list(result.actions),
                            opponent_actions=list(result.opponent_actions),
                            joint_utilities=(result.joint_utilities.tolist()
                                             if result.joint_utilities is not None else None),
                            action_utilities=(full_utilities(result).tolist()
                                              if result.joint_utilities is None else None),
                            policy_target=result.policy_target.tolist(),
                            opponent_policy=list(result.opponent_policy),
                            equilibrium_gap=result.equilibrium_gap,
                            equilibrium_converged=result.equilibrium_converged,
                            budget_exhausted=result.budget_exhausted)
                    else:
                        summary.update(result.to_dict())
                    record["variants"][name] = summary
                    dump(output, report)
                finally:
                    runner.close()
            complete = results["complete"]
            if complete.budget_exhausted or not complete.equilibrium_converged:
                raise RuntimeError("complete reference did not finish with a valid equilibrium")
            record["comparisons"] = {}
            for name in allocations:
                actual = results[name]
                comparison = compare_adaptive_result(complete, actual)
                comparison["speedup"] = (record["variants"]["complete"]["seconds"]
                                         / record["variants"][name]["seconds"])
                record["comparisons"][name] = comparison
            if "unextended" in results:
                base = results["unextended"]
                if base.budget_exhausted or not base.equilibrium_converged:
                    raise RuntimeError("unextended comparison did not complete")
                if base.actions != complete.actions or base.opponent_actions != complete.opponent_actions:
                    raise RuntimeError("extension changed a root legal inventory")
                record["tactical_comparison"] = dict(
                    max_abs_utility_change=float(np.abs(full_utilities(base) - full_utilities(complete)).max()),
                    policy_total_variation=float(np.abs(base.policy_target - complete.policy_target).sum() / 2),
                    elapsed_ratio=record["variants"]["complete"]["seconds"] / record["variants"]["unextended"]["seconds"],
                    scope="Change relative to a shallower uncalibrated critic game, not evidence of better Q or strength.")
            dump(output, report)
            print(json.dumps({k:v for k,v in record.items() if k != "variants"}), flush=True)
            if not all(c["bounds_hold"] for c in record["comparisons"].values()):
                raise RuntimeError("adaptive intervals did not contain the independent complete matrix")
        report["status"] = "Complete"
    except BaseException as error:
        report.update(status="Failed", error=str(error))
        raise
    finally:
        dump(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    audit(json.loads(parser.parse_args().config.read_text()))
