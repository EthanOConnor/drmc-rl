"""Fit diagnostic quality heads with disjoint policy anchors and bounded evaluation."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from drmc_rl.arena.experiment import dump
from drmc_rl.teachers.counterfactual_release import sha256_file
from drmc_rl.teachers.v3_baseline import load_source_rows
from drmc_rl.training.quality_checkpoints import QualityCheckpoints
from drmc_rl.training.quality_supervision import (
    assert_disjoint_sources,
    forward,
    join_quality_rows,
    layer_diagnostics,
    make_batch,
    policy_kl,
    policy_rows,
    quality_loss,
    source_group,
    split_games,
    upgrade_public_model,
)
from drmc_rl.training.utils.checkpoint_io import load_checkpoint


def device_batch(data, index, device):
    # Weights were assigned over the entire split. Never renormalize them
    # within a minibatch: a long game's many roots share its fixed total weight.
    return {key: value[index].to(device) for key, value in data.items()}


@torch.no_grad()
def cache_reference(net, data, *, batch_size, device, report=lambda **kw: None):
    net.eval()
    reference = []
    for start in range(0, len(data["mask"]), batch_size):
        batch = device_batch(data, slice(start, start + batch_size), device)
        logp = forward(net, batch, return_aux=False)[0].float().log_softmax(-1)
        reference.append(logp.cpu())
        report(cached_rows=min(start + batch_size, len(data["mask"])))
    data["reference_logp"] = torch.cat(reference)


@torch.no_grad()
def evaluate(
    net,
    data,
    rows,
    *,
    batch_size,
    device,
    targets=True,
    representation_mask=None,
    predictions=None,
    report=lambda **kw: None,
):
    """Add per-root sufficient statistics before normalizing whole-game means.

    Ranking ties are averaged across the complete greedy set. Informative-only
    ratios use the full split denominator, never an average of batch ratios.
    No evaluation tensor or candidate representation grows on the accelerator
    with the total number of source games.
    """
    net.eval()
    totals, weight, representations = {}, 0.0, []
    informative_roots, informative_groups = 0, set()
    handle = None
    if predictions is not None:
        temporary = predictions.with_suffix(predictions.suffix + ".tmp")
        handle = temporary.open("w")
    try:
        for start in range(0, len(rows), batch_size):
            index = slice(start, start + batch_size)
            batch = device_batch(data, index, device)
            output = forward(net, batch, return_aux=targets)
            if targets:
                _, values = quality_loss(
                    output,
                    batch,
                    anchor_logp=batch["reference_logp"],
                    row_metrics=True,
                    measure_ranking=True,
                )
            else:
                values = {"anchor_kl": policy_kl(output[0], batch["reference_logp"], batch["mask"])}
            names = list(values)
            # One device transfer also avoids synchronizing each metric separately.
            measured = torch.stack([values[k] for k in names]).double().cpu()
            weights = data["weights"][index].double()
            weight += float(weights.sum())
            for key, value in zip(names, measured):
                totals[key] = totals.get(key, 0.0) + float((value * weights).sum())
            if targets:
                informative = measured[names.index("informative_fraction")].bool().tolist()
                informative_roots += sum(informative)
                informative_groups.update(
                    source_group(r) for r, keep in zip(rows[index], informative) if keep
                )
                if representation_mask is not None:
                    chosen = representation_mask[index].to(device)
                    representations.append(output[2]["candidate_context"][chosen].cpu())
            if handle is not None:
                probabilities = output[0].float().softmax(-1).cpu()
                wdl = (
                    output[2]["candidate_wdl_logits"].float().softmax(-1).cpu() if targets else None
                )
                for i, row in enumerate(rows[index]):
                    n = int(data["mask"][start + i].sum())
                    record = dict(
                        source_id=row["source_id"],
                        game_id=row["game_id"],
                        reset_seed=row.get("reset_seed"),
                        actions=data["actions"][start + i, :n].tolist(),
                        probability=probabilities[i, :n].tolist(),
                        metrics={k: float(measured[j, i]) for j, k in enumerate(names)},
                    )
                    if wdl is not None:
                        record["candidate_wdl"] = wdl[i, :n].tolist()
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
            report(evaluated_rows=min(start + batch_size, len(rows)))
    finally:
        if handle is not None:
            handle.close()
    if predictions is not None:
        temporary.replace(predictions)
    metrics = {key: value / weight for key, value in totals.items()}
    if targets:
        fraction = metrics["informative_fraction"]
        for name in ("rank_accuracy", "greedy_regret", "greedy_gain"):
            numerator = metrics.pop("informative_" + name + "_numerator")
            metrics["informative_" + name] = numerator / fraction if fraction else None
        metrics.update(
            informative_roots=informative_roots, informative_games=len(informative_groups)
        )
    return metrics, torch.cat(representations) if representations else None


def cpu_snapshot(value):
    """Rollback copies use host memory; optimizer loads must not alias them."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(v) for v in value)
    return deepcopy(value)


def _finite(metrics):
    return all(value is None or np.isfinite(value) for value in metrics.values())


def fit(config):
    device = config.get("device", "cuda")
    phase = config.get("phase", "auxiliary")
    if phase not in ("auxiliary", "policy_improvement"):
        raise ValueError("quality phase is explicitly supervised, never PPO")
    if not config.get("anchor_bank"):
        raise ValueError(
            "quality fitting requires a separate anchor_bank of independent public games"
        )
    if config.get("checkpoint_only") and not config.get("checkpoint_directory"):
        raise ValueError("checkpoint_only requires a checkpoint_directory")
    kl_limit = float(config.get("max_policy_kl", 0.02))
    epochs = int(config.get("epochs", 10))
    batch_size = int(config.get("batch_size", 16))
    evaluation_batch_size = int(config.get("evaluation_batch_size", batch_size))
    minimum_anchor_games = int(config.get("minimum_anchor_games", 256))
    representation_limit = int(config.get("representation_samples", 4096))
    if (
        min(epochs, batch_size, evaluation_batch_size, minimum_anchor_games, representation_limit)
        < 1
        or not np.isfinite(kl_limit)
        or kl_limit <= 0
    ):
        raise ValueError("invalid quality training budget")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "progress.json").exists():
        raise FileExistsError("quality fitting requires a fresh experiment identity")
    dump(output / "config.json", config)
    started, last_report = time.monotonic(), 0.0
    progress = dict(
        schema="drmc-paired-quality-fit-v2",
        status="Running",
        phase="loading",
        fitting_phase=phase,
        mode=config["mode"],
        epochs=[],
        accepted_examples=0,
        examples_processed=0,
        optimizer_steps=0,
        accepted_optimizer_steps=0,
        candidate_truncation=0,
        calibrated=False,
        product_gates_passed=False,
    )

    def report(force=False, **updates):
        nonlocal last_report
        progress.update(updates)
        if force or time.monotonic() - last_report >= 5:
            last_report = time.monotonic()
            progress.update(
                updated_at=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=last_report - started,
            )
            dump(output / "progress.json", progress)

    report(force=True)
    try:
        torch.set_num_threads(int(config.get("threads", 2)))
        seed = int(config["seed"])
        split_seed = int(config.get("split_seed", seed))
        torch.manual_seed(seed)
        # Construction consumes different RNG amounts across architectures.
        sampler = torch.Generator(device="cpu").manual_seed(seed)
        anchor_sampler = torch.Generator(device="cpu").manual_seed(seed + 1)
        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        validity_path = Path(config["targets"]).parent / "label-validity.json"
        if validity_path.exists():
            validity = json.loads(validity_path.read_text())
            if validity.get("eligible_for_quality_training") is not True:
                raise ValueError(
                    "quarantined terminal labels: "
                    + validity.get("reason", "unresolved validity audit")
                )
        sources = load_source_rows(Path(config["state_bank"]))
        anchor_sources = load_source_rows(Path(config["anchor_bank"]))
        # Read confirmation metadata only to verify isolation. Its observations
        # and outcomes never enter model construction, reference caches or fits.
        confirmation = (
            load_source_rows(Path(config["confirmation_bank"]))
            if config.get("confirmation_bank")
            else []
        )
        assert_disjoint_sources(sources, anchor_sources, confirmation)
        rows = join_quality_rows(sources, load_source_rows(Path(config["targets"])))
        train, validation = split_games(
            rows, seed=split_seed, validation_fraction=config.get("validation_fraction", 0.25)
        )
        assert_disjoint_sources(train, validation)
        anchors = policy_rows(anchor_sources)
        if len({source_group(r) for r in anchors}) < minimum_anchor_games:
            raise ValueError("insufficient independent anchor games; collect broader public replay")
        net, cfg = upgrade_public_model(
            load_checkpoint(Path(config["checkpoint"]), map_location="cpu"),
            mode=config["mode"],
            device="cpu" if config.get("encoder_growth") else device,
        )
        growth = None
        if config.get("encoder_growth"):
            from drmc_rl.training.model_growth import grow_bottle_encoder

            requested = config["encoder_growth"]
            net, cfg, growth = grow_bottle_encoder(
                dict(cfg=cfg, state_dict=net.state_dict()),
                channels=requested["channels"],
                blocks=requested["blocks"],
                seed=seed,
                device=device,
            )
        schema = cfg.get("smdp_ppo", cfg)["aux_spec"]
        datasets = {
            name: make_batch(part, schema=schema, device="cpu", targets=name != "anchor")
            for name, part in (("train", train), ("validation", validation), ("anchor", anchors))
        }
        progress.update(
            source_sha256=sha256_file(Path(config["state_bank"])),
            target_sha256=sha256_file(Path(config["targets"])),
            parent_sha256=sha256_file(Path(config["checkpoint"])),
            anchor_bank_sha256=sha256_file(Path(config["anchor_bank"])),
            confirmation_sha256=sha256_file(Path(config["confirmation_bank"]))
            if confirmation
            else None,
            train_games=sorted({r["game_id"] for r in train}),
            validation_games=sorted({r["game_id"] for r in validation}),
            anchor_games=sorted({r["game_id"] for r in anchors}),
            train_states=len(train),
            validation_states=len(validation),
            anchor_states=len(anchors),
            training_parameters=sum(p.numel() for p in net.parameters()),
            encoder_growth=growth,
            split_seed=split_seed,
            policy_kl_reference="post-migration-initial-policy",
            policy_kl_measured_splits=["train", "anchor", "validation"],
            policy_kl_rollback_splits=["train", "anchor"],
            loss_weighting="equal-reset-seed-groups-with-game-id-fallback",
            evaluation_batch_size=evaluation_batch_size,
        )
        for name, data in datasets.items():
            report(force=True, phase="reference", current_split=name, cached_rows=0)
            cache_reference(
                net, data, batch_size=evaluation_batch_size, device=device, report=report
            )
        data, heldout, broad = (datasets[name] for name in ("train", "validation", "anchor"))
        # A fixed, bounded sample of training candidate representations is only
        # for layer diagnostics; neither the loss nor the frontier is sampled.
        representation_mask = torch.zeros_like(data["mask"])
        valid_indices = torch.nonzero(data["mask"].flatten()).flatten()
        diagnostics_rng = torch.Generator().manual_seed(seed + 2)
        selected = valid_indices[
            torch.randperm(len(valid_indices), generator=diagnostics_rng)[:representation_limit]
        ]
        representation_mask.view(-1)[selected] = True
        report(force=True, phase="validation", current_split="validation", evaluated_rows=0)
        initial, _ = evaluate(
            net,
            heldout,
            validation,
            batch_size=evaluation_batch_size,
            device=device,
            report=report,
            predictions=output / "initial-validation-predictions.jsonl",
        )
        progress["initial_validation"] = initial
        if not _finite(initial):
            raise FloatingPointError("non-finite initial quality metrics")
        checkpoints = (
            QualityCheckpoints(
                config["checkpoint_directory"],
                interval_seconds=config.get("checkpoint_interval_seconds", 30.0),
            )
            if config.get("checkpoint_directory")
            else None
        )

        def checkpoint_payload():
            return dict(
                cfg=cfg,
                state_dict=cpu_snapshot(net.state_dict()),
                training_contract=dict(progress, status="Checkpoint", phase="validated"),
                observation_schema=schema,
                calibrated=False,
                diagnostic_only=True,
            )

        if checkpoints:
            checkpoints.save(checkpoint_payload, progress, force=True)
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=float(config.get("lr", 1e-5)), weight_decay=0.0
        )
        policy_weight = 0.0 if phase == "auxiliary" else float(config.get("policy_weight", 1.0))
        anchor_weight = float(config.get("anchor_weight", 1.0))
        for epoch in range(epochs):
            model_state, optim_state = (
                cpu_snapshot(net.state_dict()),
                cpu_snapshot(optimizer.state_dict()),
            )
            permutation = torch.randperm(len(train), generator=sampler)
            reference_order = torch.randperm(len(anchors), generator=anchor_sampler)
            reference_order = reference_order.repeat(
                (len(train) + len(anchors) - 1) // len(anchors)
            )[: len(train)]
            accepted = False
            for attempt in range(5):
                net.load_state_dict(model_state)
                optimizer.load_state_dict(deepcopy(optim_state))
                for group in optimizer.param_groups:
                    group["lr"] *= 0.5**attempt
                net.train()
                report(
                    force=True,
                    phase="fitting",
                    epoch=epoch + 1,
                    attempt=attempt + 1,
                    current_split="train",
                    fitted_rows=0,
                )
                for start in range(0, len(train), batch_size):
                    batch = device_batch(data, permutation[start : start + batch_size], device)
                    loss, _ = quality_loss(
                        forward(net, batch),
                        batch,
                        anchor_logp=batch["reference_logp"],
                        policy_weight=policy_weight,
                        anchor_weight=anchor_weight,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite supervised quality loss")
                    # Backpropagate separately so two complete forward graphs
                    # do not coexist for a larger teacher.
                    loss.backward()
                    reference = device_batch(
                        broad, reference_order[start : start + batch_size], device
                    )
                    logits = forward(net, reference, return_aux=False)[0]
                    preservation = (
                        anchor_weight
                        * (
                            policy_kl(logits, reference["reference_logp"], reference["mask"])
                            * reference["weights"]
                        ).mean()
                    )
                    if not torch.isfinite(preservation):
                        raise FloatingPointError("non-finite broad policy preservation loss")
                    preservation.backward()
                    if start + batch_size >= len(train):
                        parameters = [
                            (name, p) for name, p in net.named_parameters() if p.grad is not None
                        ]
                        norms = (
                            torch.stack([p.grad.detach().norm() for _, p in parameters])
                            .cpu()
                            .tolist()
                        )
                        gradient_norms = {
                            name: value for (name, _), value in zip(parameters, norms)
                        }
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    progress["examples_processed"] += len(batch["mask"])
                    progress["optimizer_steps"] += 1
                    report(fitted_rows=min(start + batch_size, len(train)))
                report(force=True, phase="policy_check", current_split="train", evaluated_rows=0)
                metrics, representation = evaluate(
                    net,
                    data,
                    train,
                    batch_size=evaluation_batch_size,
                    device=device,
                    report=report,
                    representation_mask=representation_mask,
                )
                report(force=True, current_split="anchor", evaluated_rows=0)
                anchor_metrics, _ = evaluate(
                    net,
                    broad,
                    anchors,
                    batch_size=evaluation_batch_size,
                    device=device,
                    targets=False,
                    report=report,
                )
                if not _finite(metrics) or not _finite(anchor_metrics):
                    raise FloatingPointError("non-finite full-split training/anchor metrics")
                if max(metrics["anchor_kl"], anchor_metrics["anchor_kl"]) <= kl_limit:
                    accepted = True
                    break
            if not accepted:
                net.load_state_dict(model_state)
                optimizer.load_state_dict(deepcopy(optim_state))
                progress["stop_reason"] = "training or independent-anchor policy KL budget reached"
                progress["rejected_policy_check"] = dict(train=metrics, anchor=anchor_metrics)
                break
            # Holdout cannot supply gradients, anchoring examples, an early-stop
            # condition, a rollback rule or the next learning rate.
            report(force=True, phase="validation", current_split="validation", evaluated_rows=0)
            validation_metrics, _ = evaluate(
                net,
                heldout,
                validation,
                batch_size=evaluation_batch_size,
                device=device,
                report=report,
            )
            if not _finite(validation_metrics):
                raise FloatingPointError("non-finite held-out quality metrics")
            record = dict(
                epoch=epoch + 1,
                attempts=attempt + 1,
                train=metrics,
                anchor=anchor_metrics,
                validation=validation_metrics,
                elapsed_seconds=time.monotonic() - started,
                **layer_diagnostics(net, model_state, representation, gradient_norms),
            )
            progress["accepted_examples"] += len(train)
            progress["accepted_optimizer_steps"] += (len(train) + batch_size - 1) // batch_size
            progress["epochs"].append(record)
            if checkpoints:
                checkpoints.save(checkpoint_payload, progress)
            report(force=True)
            print(json.dumps({k: v for k, v in record.items() if k != "layers"}), flush=True)
        report(force=True, phase="final_evaluation", current_split="validation", evaluated_rows=0)
        final, _ = evaluate(
            net,
            heldout,
            validation,
            batch_size=evaluation_batch_size,
            device=device,
            report=report,
            predictions=output / "validation-predictions.jsonl",
        )
        progress["final_validation"] = final
        if checkpoints:
            checkpoints.save(checkpoint_payload, progress, force=True)
        report(force=True, phase="saving")
        contract = dict(progress, status="Complete", phase="complete")
        if not config.get("checkpoint_only"):
            torch.save(
                dict(
                    cfg=cfg,
                    state_dict=cpu_snapshot(net.state_dict()),
                    training_contract=contract,
                    observation_schema=schema,
                    calibrated=False,
                    diagnostic_only=True,
                ),
                output / "diagnostic.pt.tmp",
            )
            (output / "diagnostic.pt.tmp").replace(output / "diagnostic.pt")
        report(force=True, status="Complete", phase="complete")
    except BaseException as error:
        report(force=True, status="Failed", error=str(error))
        raise
    finally:
        report(force=True)
    return progress


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    fit(json.loads(parser.parse_args().config.read_text()))
