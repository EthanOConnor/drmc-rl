"""Collect fresh frozen-policy games and separate the actual PPO loss gradients.

This is a finite training diagnostic, not additional training or a tournament.
Every batch uses the real public controller inputs, complete natural returns,
unchanged behavior probabilities and complete-collection loss normalization.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import time

import numpy as np
import torch

from drmc_rl.arena.experiment import dump
from drmc_rl.execution.pace import BY_ID
from drmc_rl.models.policy.controller_core import ControllerCorePolicy, write_public_replay
from drmc_rl.models.policy.pace_adapter import PacePolicy
from drmc_rl.teachers.counterfactual_release import sha256_file
from drmc_rl.training.episodic_objective import objective_contract
from drmc_rl.training.gradient_diagnostics import loss_gradient_geometry
from drmc_rl.training.utils.checkpoint_io import load_checkpoint
from tools.trainer_event_rollout import ParallelPlanning, run_event_batch
from tools.train_pace_strategy import (
    _policy_snapshot, _training_batch, prepare_training_records,
    terminal_samples, training_loss_terms, weighted_training_terms,
)
from tools.vs_head_to_head import PlainPolicy


def audit_collection(actor, records, config, *, seed, activity=None):
    actor.finish_collection(records)
    inverse, center, scale = prepare_training_records(records, config)
    _, agreement = _policy_snapshot(actor, records, config.get("minibatch", 128), activity=activity)
    before = actor._parameter_versions()
    sampling_rng = actor.rng.get_state().clone()
    cpu_rng = torch.get_rng_state().clone()
    if any(p.grad is not None for p in actor.net.parameters()):
        raise ValueError("gradient audit requires untouched parameter grad buffers")
    order = np.random.default_rng(seed).permutation(len(records))
    size = int(config.get("minibatch", 128))
    batches = int(config.get("gradient_batches", 2))
    if size < 1 or batches < 1:
        raise ValueError("positive minibatch and gradient batch counts are required")
    measurements = []
    for start in range(0, min(len(order), size * batches), size):
        indices = order[start:start + size]
        features, data = _training_batch(actor, [records[i] for i in indices])
        _, terms = training_loss_terms(actor, features, data, config)
        values = {k: float(v.detach()) for k, v in terms.items()}
        geometry = loss_gradient_geometry(actor.net.named_parameters(), weighted_training_terms(terms, config))
        measurements.append(dict(rows=indices.tolist(), losses=values, geometry=geometry))
        if activity:
            activity("measuring_gradients", batches=len(measurements), target=batches)
    if (actor._parameter_versions() != before
            or any(p.grad is not None for p in actor.net.parameters())
            or not torch.equal(sampling_rng, actor.rng.get_state())
            or not torch.equal(cpu_rng, torch.get_rng_state())):
        raise RuntimeError("diagnostic changed weights, gradient buffers or sampling RNG")
    return dict(decisions=len(records), completed_learning_games=int(round(inverse.sum())),
        mean_episode_decisions=float(len(records) / inverse.sum()),
        advantage_center=center, advantage_scale=scale, collection_audit=agreement,
        measurements=measurements, optimizer_updates=0)


def audit(config):
    output = Path(config["output"])
    if output.exists():
        raise FileExistsError("inspect an existing gradient study before any recovery")
    torch.set_num_threads(int(config.get("threads", 1)))
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    payload = load_checkpoint(Path(config["checkpoint"]), map_location="cpu")
    training = payload["training_config"]
    # The diagnostic inherits the actual loss coefficients, not renamed defaults.
    loss_config = dict(training)
    loss_config.update(minibatch=int(config.get("minibatch", training.get("minibatch", 128))),
                       gradient_batches=int(config.get("gradient_batches", 2)))
    loss_config["objective"] = objective_contract(training)
    seeds = list(config["seeds"])
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or not 1 <= s <= 65535 for s in seeds)
            or set(seeds) & set(training["holdout_seeds"])):
        raise ValueError("use unique valid training-eligible seeds, never reserved tournament seeds")
    paces = list(config["paces"])
    if len(set(paces)) != len(paces) or any(p not in BY_ID for p in paces):
        raise ValueError("paces must be distinct declared execution profiles")
    counts = {p: int(config.get("games_per_pace", {}).get(p, config["games_per_pace_default"])) for p in paces}
    if any(n < 4 or n % 4 or n // 4 > len(seeds) for n in counts.values()):
        raise ValueError("each pace needs complete side pairs against both opponents")
    device = config.get("device", "cpu")
    actor = ControllerCorePolicy(config["parent"], device, resume=config["checkpoint"],
                                 training=True, seed=int(config["seed"]))
    opponents = {
        "parent": PlainPolicy(Path(config["parent"]), device, public_only=True),
        "pace_corrected": PacePolicy(config["parent"], device,
                                     adapter_path=config["adapter_checkpoint"]),
    }
    rollout_config = dict(config,
        variants={name: {"delay": 4} for name in ("learner", *opponents)},
        replay_games=0, mixed_core_actor=None, strict_fp32=True)
    report = dict(schema="drmc-controller-gradient-audit-v1", status="Running", phase="starting",
        config=config, loss_contract=loss_config["objective"],
        checkpoint_sha256=sha256_file(Path(config["checkpoint"])),
        parent_sha256=sha256_file(Path(config["parent"])),
        adapter_sha256=sha256_file(Path(config["adapter_checkpoint"])),
        optimizer_updates=0, training_frames=0, diagnostic_frames=0, diagnostic_games=0,
        diagnostic_decisions=0, conditions=[], product_gates_passed=False,
        scope="Fresh stochastic natural-terminal diagnostic at one frozen policy. Gradients use actual coefficients and complete-collection normalization. Separate minibatches are measured at unchanged weights, not sequential Adam updates. Gradient conflict is not proof of a strength-loss cause.")
    output.mkdir(parents=True)
    last_write = 0.0

    def activity(phase, **work):
        nonlocal last_write
        changed = report["phase"] != phase
        report.update(phase=phase, activity=work)
        if changed or time.monotonic() - last_write > 5:
            report["updated_at"] = datetime.now(UTC).isoformat()
            dump(output / "progress.json", report)
            last_write = time.monotonic()

    planner = ParallelPlanning(int(config.get("planner_workers", 1)))
    try:
        for index, pace in enumerate(paces):
            report["current_pace"] = pace
            batches = []
            for opponent_id, opponent in opponents.items():
                jobs = [(s, side, 2*i+side) for i, s in enumerate(seeds[:counts[pace]//4]) for side in (0, 1)]
                match = dict(id=f"gradient-{pace}-{opponent_id}", a="learner", b=opponent_id,
                             games=len(jobs), level=14, pace=pace)
                for start in range(0, len(jobs), int(config.get("rollout_games", 32))):
                    part, _ = run_event_batch(rollout_config, match,
                        jobs[start:start + int(config.get("rollout_games", 32))], None, planner, None,
                        policies={"learner": actor, opponent_id: opponent},
                        activity=lambda work: activity("collecting", opponent=opponent_id, **work))
                    for row, _, _ in part:
                        row["opponent"] = opponent_id
                    batches.extend(part)
            records = terminal_samples(batches)
            games = [row for row, _, _ in batches]
            write_public_replay(output/f"{pace}.npz", records, games, update=0, pace=pace, level=14)
            with (output/f"{pace}-games.jsonl").open("w") as handle:
                for game in games:
                    handle.write(json.dumps(game)+"\n")
            result = audit_collection(actor, records, loss_config,
                                      seed=int(config["seed"])+index, activity=activity)
            result.update(pace=pace, games=len(games), frames=sum(g["frames"] for g in games),
                          censored=sum(g["reason"] == "timeout" for g in games))
            report["conditions"].append(result)
            report["diagnostic_frames"] += result["frames"]
            report["diagnostic_games"] += len(games)
            report["diagnostic_decisions"] += len(records)
            activity("condition_complete", conditions=len(report["conditions"]), target=len(paces))
            del records, batches
        report["status"] = "Complete"
        activity("complete", conditions=len(report["conditions"]), target=len(paces))
    except BaseException as error:
        report.update(status="Failed", error=str(error))
        activity("failed")
        raise
    finally:
        planner.close()
        dump(output/"progress.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    audit(json.loads(parser.parse_args().config.read_text()))
