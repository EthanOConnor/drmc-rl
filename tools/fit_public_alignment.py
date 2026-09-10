"""Fit the full public-input core to frozen parent behavior on training replay.

Whole reset seeds are held out for descriptive validation. The fixed final
epoch is retained; validation never selects weights or learning rates. This
is policy migration, not outcome learning or quality distillation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA
from drmc_rl.models.policy.controller_core import CORE_SCHEMA
from drmc_rl.training.public_alignment import alignment_kl, alignment_target, legacy_teacher_inputs
from drmc_rl.training.quality_supervision import upgrade_public_model
from drmc_rl.training.utils.checkpoint_io import load_checkpoint
from tools.vs_head_to_head import PlainPolicy


def _write(path, data):
    temporary = path.with_suffix(path.suffix + ".next")
    temporary.write_text(json.dumps({**data, "updated_at": datetime.now(timezone.utc).isoformat()},
                                    indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_rows(paths, *, roots_per_seed):
    rows, identities = [], []
    for path in map(Path, paths):
        with np.load(path, allow_pickle=False) as replay:
            metadata = json.loads(str(replay["metadata"]))
            if (metadata["schema"] not in {"drmc-public-controller-replay-v1", "drmc-public-controller-replay-v2"}
                    or metadata["observation_schema"] != PUBLIC_CONTEXT_SCHEMA):
                raise ValueError("alignment requires recorded public controller training inputs")
            for seed in sorted(set(replay["game_seed"].tolist())):
                indices = np.flatnonzero(replay["game_seed"] == seed)
                indices = indices[np.linspace(0, len(indices)-1, min(roots_per_seed, len(indices)), dtype=int)]
                for index in indices:
                    lo, hi = map(int, replay["offsets"][index:index+2])
                    actions = replay["actions"][lo:hi].astype(np.int64)
                    costs = replay["costs"][lo:hi].astype(np.float32)
                    obs = replay["observation"][index].astype(np.float32)
                    context = replay["public_context"][index].astype(np.float32)
                    pill = replay["pill"][index].astype(np.int64)
                    if (not len(actions) or len(set(actions.tolist())) != len(actions)
                            or ((actions < 0) | (actions >= 512)).any()
                            or obs.shape != (20, 16, 8) or not np.isfinite(obs).all()
                            or not np.isfinite(context).all() or (costs >= 65534).any()):
                        raise ValueError("invalid complete public controller frontier")
                    if set(np.flatnonzero(obs[16:].reshape(512))) != set(actions.tolist()):
                        raise ValueError("stored public frontier differs from the model observation")
                    np.testing.assert_array_equal(context[:6].reshape(2, 3).argmax(-1), pill)
                    rows.append(dict(seed=int(seed), source=path.name, source_row=int(index),
                        observation=obs, pill=pill, preview=replay["preview"][index].astype(np.int64),
                        actions=actions, costs=costs, context=context))
            identities.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), metadata=metadata))
    return rows, identities


def batch(rows, device):
    width = max(32, max(len(r["actions"]) for r in rows))
    if str(device).startswith("mps"):
        width = 1 << (width-1).bit_length()
    actions = np.full((len(rows), width), -1, np.int64)
    costs = np.zeros((len(rows), width), np.float32)
    mask = np.zeros((len(rows), width), bool)
    target = np.zeros((len(rows), width), np.float32)
    for i, row in enumerate(rows):
        n = len(row["actions"])
        actions[i, :n], costs[i, :n], mask[i, :n] = row["actions"], row["costs"], True
        if "target" in row:
            target[i, :n] = row["target"]
    inputs = tuple(torch.as_tensor(array, device=device) for array in (
        np.stack([r["observation"] for r in rows]), np.stack([r["pill"] for r in rows]),
        np.stack([r["preview"] for r in rows]), actions, costs, mask,
    ))
    context = torch.as_tensor(np.stack([r["context"] for r in rows]), device=device)
    return inputs, context, torch.as_tensor(target, device=device)


def evaluate(net, rows, *, device, batch_size):
    net.eval()
    records = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start+batch_size]
            inputs, context, target = batch(selected, device)
            logits, _ = net(*inputs, aux=context)
            kl = alignment_kl(logits, target, inputs[-1]).cpu().numpy()
            agreement = (logits.argmax(-1) == target.argmax(-1)).float().cpu().numpy()
            for row, loss, same in zip(selected, kl, agreement, strict=True):
                records.append(dict(seed=row["seed"], kl=float(loss), agreement=float(same)))
    groups = sorted(set(r["seed"] for r in records))
    return {key: float(np.mean([np.mean([r[key] for r in records if r["seed"] == seed]) for seed in groups]))
            for key in ("kl", "agreement")}


def run(config):
    started = time.monotonic()
    output = Path(config["output"])
    if output.exists() and any(output.iterdir()):
        raise ValueError("alignment fit requires a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    if config.get("source_scope") != "training-only-controller-replay":
        raise ValueError("alignment must use an explicitly training-only replay source")
    seed, epochs = int(config.get("seed", 81729)), int(config.get("epochs", 8))
    size, cap = int(config.get("batch_size", 32)), int(config.get("roots_per_seed_per_shard", 32))
    fraction = float(config.get("validation_fraction", .2))
    if epochs < 1 or size < 1 or cap < 1 or not 0 < fraction < 1:
        raise ValueError("invalid alignment exposure or grouped split")
    torch.set_num_threads(int(config.get("threads", 1)))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = config.get("device", "cpu")
    parent_path = Path(config["parent"])
    parent = PlainPolicy(parent_path, device=device, public_only=True)
    if parent.aux_spec != "zero_v1_vs" or parent.in_channels != 20:
        raise ValueError("alignment teacher must be the frozen public zero-aux parent")
    net, cfg = upgrade_public_model(load_checkpoint(parent_path, map_location="cpu"), mode="context", device=device)
    rows, identities = load_rows(config["replays"], roots_per_seed=cap)
    unsupported, usable = 0, []
    progress = dict(status="Running", phase="building_targets", source_rows=len(rows), processed_rows=0)
    _write(output/"progress.json", progress)
    last_write = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(rows), size):
            selected = rows[start:start+size]
            inputs, context, _ = batch(selected, device)
            obs, mask = legacy_teacher_inputs(inputs[0], inputs[1], context, inputs[3], inputs[-1])
            logits, _ = parent.net(obs, *inputs[1:5], mask, aux=torch.zeros((len(selected), parent.aux_dim), device=device))
            target, supported = alignment_target(logits, mask, inputs[-1], support_epsilon=float(config.get("support_epsilon", 1e-4)))
            target, supported = target.cpu().numpy(), supported.cpu().numpy()
            for row, probability, known in zip(selected, target, supported, strict=True):
                if known:
                    row["target"] = probability[:len(row["actions"])].copy()
                    usable.append(row)
                else:
                    unsupported += 1
            if time.monotonic()-last_write >= 5:
                progress["processed_rows"] = start+len(selected)
                _write(output/"progress.json", progress)
                last_write = time.monotonic()
    del parent
    groups = sorted(set(r["seed"] for r in usable))
    if len(groups) < 4:
        raise ValueError("alignment needs at least four distinct reset seeds")
    shuffled = rng.permutation(groups)
    validation_seeds = set(shuffled[:max(1, min(len(groups)-1, round(fraction*len(groups))))].tolist())
    training = [r for r in usable if r["seed"] not in validation_seeds]
    validation = [r for r in usable if r["seed"] in validation_seeds]
    counts = Counter(r["seed"] for r in training)
    optimizer = torch.optim.AdamW(net.parameters(), lr=float(config.get("learning_rate", 1e-5)), weight_decay=0)
    report = dict(status="Running", phase="initial_evaluation", schema="drmc-public-input-alignment-v1", config=config,
                  parent_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(), sources=identities,
                  training_seeds=len(counts), validation_seeds=sorted(validation_seeds),
                  training_rows=len(training), validation_rows=len(validation), unsupported_rows=unsupported,
                  root_presentations=0, console_frames_trained=0, epochs=[])
    report["initial"] = dict(training=evaluate(net, training, device=device, batch_size=size),
                             validation=evaluate(net, validation, device=device, batch_size=size))
    _write(output/"progress.json", report)
    for epoch in range(epochs):
        net.train()
        report.update(phase="optimizing", current_epoch=epoch+1)
        order = rng.permutation(len(training))
        for start in range(0, len(order), size):
            selected = [training[i] for i in order[start:start+size]]
            inputs, context, target = batch(selected, device)
            logits, _ = net(*inputs, aux=context)
            weights = torch.as_tensor([len(training)/(len(counts)*counts[r["seed"]]) for r in selected], device=device)
            loss = (alignment_kl(logits, target, inputs[-1]) * weights).mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite alignment loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            report["root_presentations"] += len(selected)
            if time.monotonic()-last_write >= 5:
                report["last_batch_loss"] = float(loss.detach())
                _write(output/"progress.json", report)
                last_write = time.monotonic()
        metrics = dict(epoch=epoch+1, training=evaluate(net, training, device=device, batch_size=size),
                       validation=evaluate(net, validation, device=device, batch_size=size))
        report["epochs"].append(metrics)
        report["elapsed_seconds"] = time.monotonic()-started
        _write(output/"progress.json", report)
        print(json.dumps(metrics), flush=True)
    net.eval()
    checkpoint = dict(schema=CORE_SCHEMA, cfg=cfg,
        state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},
        parent_sha256=report["parent_sha256"], observation_schema=PUBLIC_CONTEXT_SCHEMA,
        regularization_reference="fixed-post-alignment-policy", diagnostic_only=True, calibrated=False,
        alignment=dict(schema=report["schema"], epochs=epochs, root_presentations=report["root_presentations"],
                       console_frames_trained=0, source_scope=config["source_scope"], support_epsilon=float(config.get("support_epsilon",1e-4))))
    temporary = output/"core-final.pt.next"
    torch.save(checkpoint, temporary)
    temporary.replace(output/"core-final.pt")
    report.update(status="Complete", phase="complete", checkpoint_sha256=hashlib.sha256((output/"core-final.pt").read_bytes()).hexdigest(),
                  selection="fixed final epoch; validation never selects weights, epochs, or learning rate")
    _write(output/"progress.json", report)
    _write(output/"assessment.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    try:
        run(config)
    except BaseException as exc:
        output = Path(config["output"])
        progress = output/"progress.json"
        if progress.exists():
            report = json.loads(progress.read_text())
            report.update(status="Failed", error={"kind":type(exc).__name__, "message":str(exc)})
            _write(progress, report)
        raise


if __name__ == "__main__":
    main()
