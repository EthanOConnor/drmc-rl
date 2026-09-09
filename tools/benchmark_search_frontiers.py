"""Compare recursive and queued search on identical real neural/native roots.

The explicit uncalibrated link is a performance/parity probe only. It supplies
no competitive-quality evidence and cannot be used as a teacher release.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from drmc_rl.arena.experiment import dump
from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner
from drmc_rl.search.belief_native_pair import BeliefNativePairSearchModel
from drmc_rl.search.joint_event import JointEventSearch, SearchConfig
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
    search_config = SearchConfig(
        depth_events=int(config.get("depth_events", 2)),
        own_beam=int(config.get("own_beam", 2)),
        opponent_beam=int(config.get("opponent_beam", 2)),
        max_nodes=int(config.get("max_nodes", 10000)),
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
    )
    dump(output, report)
    try:
        for row in rows:
            root = state_from_payload(row)
            side = int(row["root_side"])
            continuation.batch_sizes = []
            continuation.infer_batch([(root, side)])  # kernel warmup, outside timings
            results = {}
            record = dict(source_id=row["id"], variants={})
            for name in ("recursive", "queued"):
                runner = DrMarioVsPoolRunner(num_pairs=1)
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
                    )
                finally:
                    runner.close()
            a, b = results["recursive"], results["queued"]
            record["maximum_utility_error"] = float(np.max(np.abs(a.utilities - b.utilities)))
            record["parity"] = (
                a.actions == b.actions
                and a.best_action == b.best_action
                and np.allclose(a.utilities, b.utilities, rtol=0, atol=1e-5)
            )
            record["speedup"] = (
                record["variants"]["recursive"]["seconds"] / record["variants"]["queued"]["seconds"]
            )
            report["records"].append(record)
            dump(output, report)
            print(json.dumps({k: v for k, v in record.items() if k != "variants"}), flush=True)
            if not record["parity"]:
                raise RuntimeError("queued search differs from the recursive reference")
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
