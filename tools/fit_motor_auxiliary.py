"""Fit exact effect/future-access auxiliaries while bounding policy drift.

The shared core receives supervised gradients; outcomes are not replaced with
motif rewards. A separate broad public replay anchor excludes validation seeds
and annotated decisions. Validation predictions and policy drift are reported
after accepted epochs and never enter gradients or the rollback criterion.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from drmc_rl.arena.experiment import dump
from drmc_rl.models.policy.controller_core import CORE_SCHEMA
from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA
from drmc_rl.training.motor_supervision import (
    assign_game_weights, cache_reference, evaluate_motor, forward_motor, load_bank,
    make_motor_batch, motor_loss, policy_kl, upgrade_motor_model,
)
from drmc_rl.training.utils.checkpoint_io import load_checkpoint
from tools.build_motor_opportunity_bank import split_for_seed


def load_anchors(config, annotated):
    source = Path(config["anchor_replay_directory"])
    bank_config = json.loads((Path(config["bank"]) / "config.json").read_text())
    excluded = set(map(int, bank_config["holdout_seeds"]))
    annotated_ids = {(r["record"]["source_sha256"], r["record"]["source_row"]) for r in annotated}
    watermark = json.loads((source.parent / "training.json").read_text())["updates"]
    rng = np.random.default_rng(config["seed"])
    rows, seen_games = [], set()
    paths = sorted(source.glob("update-*.npz"))
    rng.shuffle(paths)
    wanted = int(config.get("anchor_rows", 4096))
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"]))
            if metadata["schema"] != "drmc-public-controller-replay-v2" or metadata["update"] > watermark:
                continue
            data = {key: payload[key] for key in payload.files}
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        # At most one anchor per source game per scan keeps broad policy
        # preservation from being dominated by a few long trajectories.
        for i in rng.permutation(len(data["game_seed"])):
            seed, side = int(data["game_seed"][i]), int(data["learner_port"][i])
            game_id = (digest, seed, side)
            if (seed in excluded or split_for_seed(seed, bank_config["seed"]) != "train"
                    or (digest, int(i)) in annotated_ids or game_id in seen_games):
                continue
            lo, hi = map(int, data["offsets"][i:i + 2])
            row = {key: data[key][i].copy() for key in (
                "observation", "pill", "preview", "public_context", "controller_geometry",
            )}
            row.update(actions=data["actions"][lo:hi].copy(), costs=data["costs"][lo:hi].copy(),
                       game_id=game_id, source_row=int(i))
            rows.append(row)
            seen_games.add(game_id)
            if len(rows) >= wanted:
                break
        if len(rows) >= wanted:
            break
    if len(rows) < int(config.get("minimum_anchor_games", 256)):
        raise ValueError("insufficient independent public anchor games; collect broader replay")
    assign_game_weights(rows)
    return rows


def run(config):
    torch.set_num_threads(int(config.get("threads", 1)))
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = config.get("device", "cpu")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "config.json", config)
    started = time.perf_counter()
    progress = dict(schema="drmc-motor-auxiliary-study-v1", status="Running", phase="loading", epoch=0,
                    target_epochs=int(config.get("epochs", 20)), accepted_examples=0,
                    examples_processed=0)

    def report(phase, **work):
        progress.update(phase=phase, **work, wall_seconds=time.perf_counter() - started,
                        updated_at=datetime.now(UTC).isoformat())
        dump(output / "progress.json", progress)

    try:
        report("loading")
        train, validation = load_bank(config["bank"])
        anchors = load_anchors(config, train + validation)
        parent = load_checkpoint(Path(config["checkpoint"]), map_location=device)
        with Path(config["checkpoint"]).open("rb") as stream:
            parent_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        net, cfg = upgrade_motor_model(parent, device=device)
        batch_size = int(config.get("batch_size", 16))
        if batch_size < 1 or progress["target_epochs"] < 1:
            raise ValueError("positive minibatch and epoch budgets required")
        progress.update(train_roots=len(train), validation_roots=len(validation),
                        anchor_games=len(anchors), trainable_parameters=sum(p.numel() for p in net.parameters()))
        report("reference_policy")
        for rows in (train, validation, anchors):
            cache_reference(net, rows, batch_size=batch_size, device=device)
        baseline = {name: evaluate_motor(net, rows, batch_size=batch_size, device=device,
                                        targets=name != "anchor")
                    for name, rows in (("train", train), ("validation", validation), ("anchor", anchors))}
        dump(output / "baseline.json", baseline)
        optimizer = torch.optim.AdamW(net.parameters(), lr=float(config.get("lr", 3e-6)), weight_decay=0.)
        rng = np.random.default_rng(config["seed"])
        history = []
        max_kl = float(config.get("max_policy_kl", .01))
        if max_kl <= 0:
            raise ValueError("policy preservation cap must be positive")
        for epoch in range(1, progress["target_epochs"] + 1):
            order = rng.permutation(len(train))
            anchor_order = np.resize(rng.permutation(len(anchors)), len(train))
            previous_model, previous_optimizer = deepcopy(net.state_dict()), deepcopy(optimizer.state_dict())
            for attempt in range(1 + int(config.get("max_backtracks", 3))):
                next_report = 0.
                for start in range(0, len(train), batch_size):
                    part = [train[i] for i in order[start:start + batch_size]]
                    reference_part = [anchors[i] for i in anchor_order[start:start + batch_size]]
                    batch = make_motor_batch(part, device=device)
                    anchor = make_motor_batch(reference_part, device=device, targets=False)
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = motor_loss(forward_motor(net, batch), batch,
                                         anchor_weight=float(config.get("anchor_weight", 10.)))
                    logits, _ = forward_motor(net, anchor, auxiliary=False)
                    loss = loss + float(config.get("anchor_weight", 10.)) * (
                        policy_kl(logits, anchor) * anchor["weights"]).mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("nonfinite motor auxiliary loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    progress["examples_processed"] += len(part)
                    if time.monotonic() >= next_report:
                        report("fitting", epoch=epoch, attempt=attempt,
                               step=min(start + batch_size, len(train)), steps=len(train))
                        next_report = time.monotonic() + 5
                report("checking_policy", epoch=epoch, attempt=attempt)
                train_metrics = evaluate_motor(net, train, batch_size=batch_size, device=device)
                anchor_metrics = evaluate_motor(net, anchors, batch_size=batch_size, device=device, targets=False)
                if max(train_metrics["anchor_kl"], anchor_metrics["anchor_kl"]) <= max_kl:
                    break
                net.load_state_dict(previous_model, strict=True)
                optimizer.load_state_dict(previous_optimizer)
                for group in optimizer.param_groups:
                    group["lr"] *= .5 ** (attempt + 1)
            else:
                raise RuntimeError("motor auxiliary epoch could not satisfy the policy preservation cap")
            # Holdout never supplies gradients, reference anchoring examples,
            # or the decision to retry this optimizer update.
            report("validation", epoch=epoch)
            metrics = dict(epoch=epoch, train=train_metrics, anchor=anchor_metrics,
                           validation=evaluate_motor(net, validation, batch_size=batch_size, device=device),
                           backtracks=attempt, lr=optimizer.param_groups[0]["lr"])
            history.append(metrics)
            progress["accepted_examples"] += len(train)
            report("saving", epoch=epoch, metrics=metrics)
            path = output / "core-latest.pt"
            temporary = path.with_suffix(".pt.next")
            torch.save(dict(schema=CORE_SCHEMA, cfg=cfg,
                            state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
                            observation_schema=PUBLIC_CONTEXT_SCHEMA, calibrated=False, diagnostic_only=True,
                            parent_sha256=parent_sha,
                            fit_config=config, fit_progress=progress), temporary)
            temporary.replace(path)
            dump(output / "fit.json", dict(baseline=baseline, epochs=history))
            del previous_model, previous_optimizer
        progress["status"] = "Complete"
        report("saving_final")
        temporary = output / "core-final.pt.next"
        shutil.copyfile(output / "core-latest.pt", temporary)
        temporary.replace(output / "core-final.pt")
        report("complete")
    except BaseException as error:
        progress.update(status="Failed", error=str(error))
        report("failed")
        raise
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.config.read_text())), indent=2))


if __name__ == "__main__":
    main()
