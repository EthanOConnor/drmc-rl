"""Compare recursive and queued search on identical real neural/native roots.

The explicit uncalibrated link is a performance/parity probe only. It supplies
no competitive-quality evidence and cannot be used as a teacher release.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from drmc_rl.arena.experiment import dump
from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner
from drmc_rl.search.belief_native_pair import BeliefNativePairSearchModel
from drmc_rl.search.joint_event import JointEventSearch, SearchConfig, WDL
from drmc_rl.search.matrix_check import compare_matrix_games
from drmc_rl.search.native_pair import state_from_payload
from drmc_rl.search.pill_belief import PillReserveBelief
from drmc_rl.search.public_policy import PublicPolicyContinuation
from drmc_rl.search.queued_event import QueuedJointEventSearch
from drmc_rl.search.strong_league import DavidsonCalibration
from drmc_rl.teachers.counterfactual_release import sha256_file
from drmc_rl.teachers.v3_baseline import load_source_rows


class MeteredContinuation(PublicPolicyContinuation):
    def infer_batch(self, requests):
        self.batch_sizes.append(len(requests))
        return super().infer_batch(requests)


def benchmark(config):
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("search benchmark output already exists")
    torch.set_num_threads(int(config.get("threads", 2)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    continuation = MeteredContinuation(
        Path(config["checkpoint"]),
        DavidsonCalibration(1.0, 0.0, -3.0, "uncalibrated-performance-probe"),
        device=config.get("device", "cpu"),
    )
    rows = load_source_rows(Path(config["state_bank"]))[: int(config.get("states", 2))]
    variant_order = tuple(config.get("variant_order", ("recursive", "queued")))
    if sorted(variant_order) != ["queued", "recursive"]:
        raise ValueError("variant_order must include recursive and queued exactly once")
    comparison_contract = config.get("comparison_contract", "strict-vector-v1")
    if comparison_contract not in ("strict-vector-v1", "mixed-game-certificate-v1"):
        raise ValueError("unknown search comparison contract")
    if comparison_contract == "mixed-game-certificate-v1" and config.get("opponent_mode") != "mixed":
        raise ValueError("matrix certificates require mixed search")
    search_config = SearchConfig(
        depth_events=int(config.get("depth_events", 2)),
        own_beam=int(config.get("own_beam", 2)),
        opponent_beam=int(config.get("opponent_beam", 2)),
        max_nodes=int(config.get("max_nodes", 10000)),
        opponent_mode=config.get("opponent_mode", "expectation"),
        matrix_iterations=int(config.get("matrix_iterations", 2048)),
        matrix_temperature=float(config.get("matrix_temperature", .001)),
        matrix_gap_tolerance=float(config.get("matrix_gap_tolerance", .02)),
        matrix_solver=config.get("matrix_solver", "linear_program"),
        matrix_time_limit_seconds=float(config.get("matrix_time_limit_seconds", .25)),
    )
    report = dict(
        schema="drmc-frontier-benchmark-v1",
        status="Running",
        records=[],
        checkpoint_sha256=sha256_file(Path(config["checkpoint"])),
        source_sha256=sha256_file(Path(config["state_bank"])),
        diagnostic_only=True,
        calibrated=False,
        product_gates_passed=False,
        search_config=asdict(search_config),
        comparison_contract=comparison_contract,
        native_library_sha256=(sha256_file(Path(config["native_library"]))
                               if config.get("native_library") else None),
    )
    if search_config.opponent_mode == "mixed":
        # Initialize the solver outside the order-sensitive native timings;
        # retain cold import/initialization cost as a separate measurement.
        warmup = JointEventSearch(None, search_config)
        warmup._solve_matrix([[WDL.terminal(1), WDL.terminal(-1)],
                              [WDL.terminal(-1), WDL.terminal(1)]])
        report["matrix_cold_start_ms"] = warmup._matrix_solve_ms
    dump(output, report)
    try:
        for row_index, row in enumerate(rows):
            root = state_from_payload(row)
            side = int(row["root_side"])
            continuation.batch_sizes = []
            continuation.infer_batch([(root, side)])  # kernel warmup, outside timings
            results = {}
            order = variant_order[::-1] if config.get("alternate_order") and row_index % 2 else variant_order
            record = dict(source_id=row["id"], variants={}, order=list(order))
            for name in order:
                runner = DrMarioVsPoolRunner(num_pairs=1, lib_path=config.get("native_library"))
                try:
                    model = BeliefNativePairSearchModel(runner, continuation=continuation)
                    model.register_belief(root, PillReserveBelief.from_dict(row["reserve_belief"]))
                    continuation._cache.clear()
                    continuation.batch_sizes = []
                    search = (
                        JointEventSearch(model, search_config)
                        if name == "recursive"
                        else QueuedJointEventSearch(
                            model, search_config, batch_size=int(config.get("batch_size", 32))
                        )
                    )
                    start = time.monotonic()
                    result = search.search(
                        root, root_side=side, root_actions=root.legal_actions_by_side[side]
                    )
                    elapsed = time.monotonic() - start
                    if result.budget_exhausted:
                        raise RuntimeError("benchmark exhausted its exact comparison budget")
                    results[name] = result
                    record["variants"][name] = dict(
                        seconds=elapsed,
                        nodes=result.nodes,
                        inference_calls=len(continuation.batch_sizes),
                        inference_rows=sum(continuation.batch_sizes),
                        largest_batch=max(continuation.batch_sizes, default=0),
                        actions=list(result.actions),
                        utilities=result.utilities.tolist(),
                        best_action=result.best_action,
                        policy_target=result.policy_target.tolist(),
                        opponent_actions=list(result.opponent_actions),
                        opponent_policy=list(result.opponent_policy),
                        matrix_games=result.matrix_games,
                        matrix_solve_ms=result.matrix_solve_ms,
                        equilibrium_gap=result.equilibrium_gap,
                        equilibrium_converged=result.equilibrium_converged,
                        matrix_failures=list(result.matrix_failures),
                        public_observation_schema=root.public_observation_schema,
                        boundary=model.boundary(root).value,
                        joint_utilities=(None if result.joint_utilities is None
                                         else result.joint_utilities.tolist()),
                    )
                finally:
                    runner.close()
            a, b = results["recursive"], results["queued"]
            record["maximum_utility_error"] = float(np.max(np.abs(a.utilities - b.utilities)))
            record["parity"] = (
                a.actions == b.actions
                and a.best_action == b.best_action
                and np.allclose(a.utilities, b.utilities, rtol=0, atol=1e-5)
                and np.allclose(a.policy_target, b.policy_target, rtol=0, atol=1e-5)
                and a.opponent_actions == b.opponent_actions
                and np.allclose(a.opponent_policy, b.opponent_policy, rtol=0, atol=1e-5)
                and a.equilibrium_converged == b.equilibrium_converged
                and (a.joint_utilities is None and b.joint_utilities is None
                     or a.joint_utilities is not None and b.joint_utilities is not None
                     and np.allclose(a.joint_utilities, b.joint_utilities, rtol=0, atol=1e-5))
            )
            record["converged"] = a.equilibrium_converged and b.equilibrium_converged
            record["strict_vector_parity"] = record["parity"]
            if a.joint_utilities is not None and b.joint_utilities is not None:
                record["matrix_comparison"] = compare_matrix_games(
                    a, b, gap_tolerance=search_config.matrix_gap_tolerance)
                if comparison_contract == "mixed-game-certificate-v1":
                    record["parity"] = record["matrix_comparison"]["equivalent"]
            record["speedup"] = (
                record["variants"]["recursive"]["seconds"] / record["variants"]["queued"]["seconds"]
            )
            report["records"].append(record)
            dump(output, report)
            print(json.dumps({k: v for k, v in record.items() if k != "variants"}), flush=True)
            if not record["parity"]:
                raise RuntimeError("queued search differs from the recursive reference")
            if not record["converged"]:
                raise RuntimeError("mixed matrix did not converge; comparison retained")
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
    benchmark(json.loads(parser.parse_args().config.read_text()))
