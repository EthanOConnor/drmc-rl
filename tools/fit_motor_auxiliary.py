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
    make_motor_batch, motor_loss, policy_kl, upgrade_motor_model, initialize_motor_priors,
    cache_motor_features, cached_motor_batch, forward_cached_motor, evaluate_cached_motor,
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


def core_digest(net):
    digest = hashlib.sha256()
    for key, value in net.state_dict().items():
        if not key.startswith("motor_auxiliary."):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def warmup_heads(net, train, validation, config, report, progress, output):
    """Fixed-budget head fitting; no core updates or holdout-driven selection."""
    epochs = int(config.get("head_epochs", 0))
    if epochs <= 0:
        return
    device = config.get("device", "cpu")
    head_device = config.get("head_device", "cpu")
    batch_size = int(config.get("head_batch_size", config.get("batch_size", 16)))
    before = core_digest(net)
    requires_grad = {name: value.requires_grad for name, value in net.named_parameters()}
    for name, value in net.named_parameters():
        value.requires_grad_(name.startswith("motor_auxiliary."))
    for name, rows in (("train", train), ("validation", validation)):
        cache_motor_features(net, rows, batch_size=int(config.get("batch_size", 16)), device=device,
                             activity=lambda count: report("head_features", split=name,
                                                           feature_roots=count, feature_target=len(rows)))
    head = net.motor_auxiliary.to(head_device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(config.get("head_lr", 3e-4)), weight_decay=0.)
    rng = np.random.default_rng(config["seed"])
    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train))
        next_report = 0.
        for start in range(0, len(train), batch_size):
            part = [train[i] for i in order[start:start + batch_size]]
            batch = cached_motor_batch(part, device=head_device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = motor_loss(forward_cached_motor(head, batch), batch, anchor_weight=0.)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite frozen-core motor head loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            progress["head_examples_processed"] += len(part)
            if time.monotonic() >= next_report:
                report("head_fitting", head_epoch=epoch, head_step=min(start+batch_size, len(train)))
                next_report = time.monotonic() + 5
        metrics = dict(epoch=epoch,
            train=evaluate_cached_motor(head, train, batch_size=batch_size, device=head_device),
            validation=evaluate_cached_motor(head, validation, batch_size=batch_size, device=head_device))
        history.append(metrics)
        progress["head_accepted_examples"] += len(train)
        report("head_validation", head_epoch=epoch, head_metrics=metrics)
        dump(output / "head-fit.json", dict(epochs=history, core_sha256=before,
             contract="Fixed final epoch; cached public features; zero core optimizer updates"))
    head.to(device)
    if core_digest(net) != before:
        raise RuntimeError("head warmup changed a competitive-core parameter or buffer")
    for name, value in net.named_parameters():
        value.requires_grad_(requires_grad[name])
    # No stale frozen representation survives into shared-core optimization.
    for row in train + validation:
        for key in ("frozen_motor_features", "frozen_motor_geometry", "cached_motor_targets", "known_effects"):
            del row[key]
    progress["head_core_unchanged"] = True


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
                    examples_processed=0, head_examples_processed=0, head_accepted_examples=0,
                    target_head_epochs=int(config.get("head_epochs", 0)))

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
        initialization = config.get("head_initialization", "legacy_random")
        if initialization == "training_cell_prior":
            if any(key.startswith("motor_auxiliary.") for key in parent["state_dict"]):
                raise ValueError("training priors cannot silently reset existing learned motor heads")
            dump(output / "head-initialization.json", initialize_motor_priors(net, train))
        elif initialization != "legacy_random":
            raise ValueError("unknown motor head initialization")
        batch_size = int(config.get("batch_size", 16))
        if (batch_size < 1 or progress["target_epochs"] < 1 or progress["target_head_epochs"] < 0
                or int(config.get("head_batch_size", batch_size)) < 1):
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
        warmup_heads(net, train, validation, config, report, progress, output)
        if progress["target_head_epochs"]:
            dump(output / "post-head-warmup.json", {
                name: evaluate_motor(net, rows, batch_size=batch_size, device=device,
                                     targets=name != "anchor")
                for name, rows in (("train", train), ("validation", validation), ("anchor", anchors))})
        core_lr = float(config.get("lr", 3e-6))
        head_lr = float(config.get("joint_head_lr", core_lr))
        if min(core_lr, head_lr, float(config.get("head_lr", 3e-4))) <= 0:
            raise ValueError("motor learning rates must be positive")
        optimizer = torch.optim.AdamW([
            dict(params=[p for name, p in net.named_parameters() if not name.startswith("motor_auxiliary.")], lr=core_lr),
            dict(params=list(net.motor_auxiliary.parameters()), lr=head_lr),
        ], weight_decay=0.)
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
                           backtracks=attempt, lr=optimizer.param_groups[0]["lr"],
                           head_lr=optimizer.param_groups[1]["lr"])
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
        # Compare with an independently defined training-only pace/cell
        # predictor. These final descriptive scores never control fitting.
        from collections import defaultdict
        from tools.audit_motor_auxiliary import opportunity_priors, prior_metrics

        report("condition_validation")
        priors = opportunity_priors(train)
        conditions = []
        for split, rows in (("train", train), ("validation", validation)):
            groups = defaultdict(list)
            for row in rows:
                groups[row["record"]["level"], row["record"]["pace"]].append(dict(row))
            for (level, pace), group in sorted(groups.items()):
                assign_game_weights(group)
                metrics = evaluate_motor(net, group, batch_size=batch_size, device=device)
                prior = prior_metrics(group, priors, pace) if all((pace, n) in priors for n in ("reach", "clear")) else None
                conditions.append(dict(split=split, level=level, pace=pace, roots=len(group),
                    reset_seeds=len({r["record"]["game_seed"] for r in group}),
                    fitted=metrics, training_prior=prior))
        dump(output / "condition-metrics.json", dict(conditions=conditions,
             scope="Fixed-final development measurements; fresh seed confirmation remains required"))
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
