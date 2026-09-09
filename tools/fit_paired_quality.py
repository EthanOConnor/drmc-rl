"""Fit diagnostic G5 critic/context ablations from frozen paired terminal panels."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch

from drmc_rl.arena.experiment import dump
from drmc_rl.teachers.counterfactual_release import sha256_file
from drmc_rl.teachers.v3_baseline import load_source_rows
from drmc_rl.training.quality_supervision import (
    forward,
    join_quality_rows,
    layer_diagnostics,
    make_batch,
    quality_loss,
    split_games,
    upgrade_public_model,
)
from drmc_rl.training.utils.checkpoint_io import load_checkpoint


def fit(config):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "progress.json").exists():
        raise FileExistsError("quality fitting requires a fresh experiment identity")
    device = config.get("device", "cuda")
    torch.set_num_threads(int(config.get("threads", 2)))
    seed = int(config["seed"])
    torch.manual_seed(seed)
    # Architecture construction consumes different amounts of RNG. Keep the
    # data order identical across all four modes and across CPU/CUDA devices.
    sampler = torch.Generator(device="cpu").manual_seed(seed)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    targets = [
        json.loads(line)
        for line in Path(config["targets"]).read_text().splitlines()
        if line.strip()
    ]
    rows = join_quality_rows(load_source_rows(Path(config["state_bank"])), targets)
    train, validation = split_games(
        rows, seed=seed, validation_fraction=config.get("validation_fraction", 0.25)
    )
    net, cfg = upgrade_public_model(
        load_checkpoint(Path(config["checkpoint"]), map_location=device),
        mode=config["mode"],
        device=device,
    )
    schema = cfg.get("smdp_ppo", cfg)["aux_spec"]
    data = make_batch(train, schema=schema, device=device)
    heldout = make_batch(validation, schema=schema, device=device)
    with torch.no_grad():
        anchor = forward(net, data)[0].float().log_softmax(-1).detach()
        _, initial = quality_loss(forward(net, heldout), heldout)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=float(config.get("lr", 1e-5)), weight_decay=0.0
    )
    phase = config.get("phase", "auxiliary")
    if phase not in ("auxiliary", "policy_improvement"):
        raise ValueError("quality phase is explicitly supervised, never PPO")
    policy_weight = 0.0 if phase == "auxiliary" else float(config.get("policy_weight", 1.0))
    kl_limit = float(config.get("max_policy_kl", 0.02))
    epochs = int(config.get("epochs", 10))
    batch_size = int(config.get("batch_size", 16))
    if min(epochs, batch_size) < 1 or not np.isfinite(kl_limit) or kl_limit <= 0:
        raise ValueError("invalid quality training budget")
    progress = dict(
        schema="drmc-paired-quality-fit-v1",
        status="Running",
        phase=phase,
        mode=config["mode"],
        source_sha256=sha256_file(Path(config["state_bank"])),
        target_sha256=sha256_file(Path(config["targets"])),
        parent_sha256=sha256_file(Path(config["checkpoint"])),
        train_games=sorted({r["game_id"] for r in train}),
        validation_games=sorted({r["game_id"] for r in validation}),
        train_states=len(train),
        validation_states=len(validation),
        candidate_truncation=0,
        calibrated=False,
        product_gates_passed=False,
        initial_validation={k: float(v) for k, v in initial.items()},
        epochs=[],
    )
    dump(output / "progress.json", progress)
    started = time.monotonic()
    try:
        for epoch in range(epochs):
            before = {name: p.detach().clone() for name, p in net.named_parameters()}
            model_state = deepcopy(net.state_dict())
            optim_state = deepcopy(optimizer.state_dict())
            permutation = torch.randperm(len(train), generator=sampler).to(device)
            accepted = False
            for attempt in range(5):
                net.load_state_dict(model_state)
                optimizer.load_state_dict(optim_state)
                for group in optimizer.param_groups:
                    group["lr"] *= 0.5**attempt
                net.train()
                for start in range(0, len(train), batch_size):
                    index = permutation[start : start + batch_size]
                    batch = {k: v[index] for k, v in data.items()}
                    loss, _ = quality_loss(
                        forward(net, batch),
                        batch,
                        anchor_logp=anchor[index],
                        policy_weight=policy_weight,
                        anchor_weight=float(config.get("anchor_weight", 1.0)),
                    )
                    optimizer.zero_grad(set_to_none=True)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite supervised quality loss")
                    loss.backward()
                    gradient_norms = {
                        name: float(p.grad.detach().norm())
                        for name, p in net.named_parameters()
                        if p.grad is not None
                    }
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                net.eval()
                with torch.no_grad():
                    output_batch = forward(net, data)
                    _, metrics = quality_loss(
                        output_batch, data, anchor_logp=anchor, policy_weight=policy_weight
                    )
                if (
                    all(np.isfinite(float(value)) for value in metrics.values())
                    and float(metrics["anchor_kl"]) <= kl_limit
                ):
                    accepted = True
                    break
            if not accepted:
                net.load_state_dict(model_state)
                optimizer.load_state_dict(optim_state)
                progress["stop_reason"] = "full-dataset policy KL budget reached"
                break
            with torch.no_grad():
                _, validation_metrics = quality_loss(forward(net, heldout), heldout)
            if not all(np.isfinite(float(value)) for value in validation_metrics.values()):
                raise FloatingPointError("non-finite held-out quality metrics")
            report = dict(
                epoch=epoch + 1,
                attempts=attempt + 1,
                train={k: float(v) for k, v in metrics.items()},
                validation={k: float(v) for k, v in validation_metrics.items()},
                elapsed_seconds=time.monotonic() - started,
                **layer_diagnostics(
                    net, before, output_batch[2]["candidate_context"][data["mask"]], gradient_norms
                ),
            )
            progress["epochs"].append(report)
            dump(output / "progress.json", progress)
            print(json.dumps({k: v for k, v in report.items() if k != "layers"}), flush=True)
        progress.update(status="Complete", elapsed_seconds=time.monotonic() - started)
        torch.save(
            dict(
                cfg=cfg,
                state_dict=net.state_dict(),
                training_contract=progress,
                observation_schema=schema,
                calibrated=False,
                diagnostic_only=True,
            ),
            output / "diagnostic.pt",
        )
    except BaseException as error:
        progress.update(status="Failed", error=str(error))
        raise
    finally:
        progress["elapsed_seconds"] = time.monotonic() - started
        dump(output / "progress.json", progress)
    return progress


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    fit(json.loads(parser.parse_args().config.read_text()))
