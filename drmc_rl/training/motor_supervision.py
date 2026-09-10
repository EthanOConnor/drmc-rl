"""Supervised deterministic effects/access learning with policy preservation."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA
from drmc_rl.human.motor_opportunity import OPPORTUNITY_CONDITION, OPPORTUNITY_SCHEMA, UNREACHABLE
from drmc_rl.human.search import semantic_planes_to_nes_board
from drmc_rl.models.policy.effect_tokens import EFFECT_TOKEN_NAMES, build_effect_tokens
from drmc_rl.models.policy.motor_auxiliary import MOTOR_AUXILIARY_SCHEMA


def upgrade_motor_model(checkpoint, *, device):
    from tools.eval_policy import _build_net_from_cfg

    cfg = deepcopy(checkpoint["cfg"])
    sp = cfg.get("smdp_ppo", cfg)
    if sp.get("candidate_architecture") != "g5" or sp.get("aux_spec") != PUBLIC_CONTEXT_SCHEMA:
        raise ValueError("motor fitting requires the live public-context G5 core")
    old_schema = sp.get("candidate_motor_auxiliary")
    if old_schema not in (None, MOTOR_AUXILIARY_SCHEMA):
        raise ValueError("motor auxiliary checkpoint schema changed")
    sp["candidate_motor_auxiliary"] = MOTOR_AUXILIARY_SCHEMA
    net, _, _ = _build_net_from_cfg(cfg, 20, device)
    weights = checkpoint.get("ema_state_dict") or checkpoint["state_dict"]
    missing, unexpected = net.load_state_dict(weights, strict=False)
    if unexpected or any(not key.startswith("motor_auxiliary.") for key in missing) or (old_schema and missing):
        raise ValueError("motor migration changed an existing competitive-core parameter")
    return net.eval(), cfg


def load_bank(path):
    directory = Path(path)
    progress = json.loads((directory / "progress.json").read_text())
    if progress["status"] != "Complete":
        raise ValueError("fit requires a completed immutable opportunity bank")
    records = [json.loads(line) for line in (directory / "roots.jsonl").read_text().splitlines()]
    if len(records) != progress["roots"] or len({r["id"] for r in records}) != len(records):
        raise ValueError("opportunity bank identity/count mismatch")
    seen_seeds, rows = {}, []
    for record in records:
        if record["schema"] != OPPORTUNITY_SCHEMA or record["condition"] != OPPORTUNITY_CONDITION:
            raise ValueError("opportunity bank mixed incompatible geometry conditions")
        seed = record["game_seed"]
        if seen_seeds.setdefault(seed, record["split"]) != record["split"]:
            raise ValueError("a reset seed crosses motor train/validation splits")
        with np.load(directory / record["path"], allow_pickle=False) as payload:
            row = {key: payload[key] for key in payload.files if key != "metadata"}
        row["record"] = record
        row["game_id"] = (record["source_sha256"], seed, record["learner_port"])
        rows.append(row)
    split = [[r for r in rows if r["record"]["split"] == name] for name in ("train", "validation")]
    if not all(split):
        raise ValueError("motor fitting requires independent training and validation games")
    for group in split:
        assign_game_weights(group)
    return tuple(split)


def assign_game_weights(rows):
    counts = Counter(r["game_id"] for r in rows)
    for row in rows:
        row["weight"] = len(rows) / (len(counts) * counts[row["game_id"]])


def make_motor_batch(rows, *, device, targets=True):
    width = max(32, max(len(r["actions"]) for r in rows))
    count = len(rows)
    actions = np.full((count, width), -1, np.int64)
    costs = np.zeros((count, width), np.float32)
    mask = np.zeros((count, width), bool)
    for i, row in enumerate(rows):
        n = len(row["actions"])
        actions[i, :n] = row["actions"]
        costs[i, :n] = row.get("root_costs", row.get("costs"))
        mask[i, :n] = True
    values = (
        np.stack([r["observation"] for r in rows]).astype(np.float32),
        np.stack([r["pill"] for r in rows]).astype(np.int64),
        np.stack([r["preview"] for r in rows]).astype(np.int64), actions, costs, mask,
        np.stack([r["public_context"] for r in rows]),
    )
    result = dict(inputs=tuple(torch.as_tensor(x, device=device) for x in values),
                  weights=torch.as_tensor([r.get("weight", 1.) for r in rows], device=device),
                  mask=torch.as_tensor(mask, device=device))
    geometry = np.stack([r["controller_geometry"] for r in rows]).astype(np.float32)
    geometry[:, 2] = geometry[:, 2].astype(np.int32) & 15  # Known placements until the next speed-up.
    geometry /= np.asarray((2, 49, 9, 60, 60, 7, 15, 3, 81, 15, 2, 2, 1), np.float32)
    result["geometry"] = torch.as_tensor(geometry, device=device)
    if all("reference_logp" in r for r in rows):
        logs = np.full((count, width), -1e9, np.float32)
        for i, row in enumerate(rows):
            logs[i, :len(row["actions"])] = row["reference_logp"]
        result["reference_logp"] = torch.as_tensor(logs, device=device)
    if not targets:
        return result
    fields = np.full((count, width, 128), 255, np.uint8)
    root_terminal = np.zeros((count, width), np.uint8)
    effects = {key: np.zeros((count, width), np.float32) for key in (
        "viruses_cleared", "nonviruses_cleared", "clear_events",
    )}
    reachable = np.zeros((count, width, 2, 128), bool)
    clearable = np.zeros_like(reachable)
    cost_target = np.zeros((count, width, 2, 128), np.float32)
    for i, row in enumerate(rows):
        n = len(row["actions"])
        fields[i, :n], root_terminal[i, :n] = row["after_fields"], row["root_terminal"]
        for key in effects:
            effects[key][i, :n] = row["root_" + key]
        reachable[i, :n] = row["reachable_cells"] != UNREACHABLE
        clearable[i, :n] = row["clearable_cells"] != UNREACHABLE
        costs_i = np.where(reachable[i, :n], row["reachable_cells"], 0)
        cost_target[i, :n] = np.log1p(costs_i.astype(np.float32)) / np.log1p(600.)
    roots = np.stack([semantic_planes_to_nes_board(r["observation"][:8]) for r in rows])
    tokens = build_effect_tokens(roots, fields, mask, terminal_reason=root_terminal, **effects)
    # Neither incoming attack state nor teacher uncertainty is observed by this
    # own-board labeler. Their default zeros are not supervised as measurements.
    known_effects = torch.tensor([n not in ("attack", "uncertainty") for n in EFFECT_TOKEN_NAMES], device=device)
    result.update(effect_target=tokens.to(device), known_effects=known_effects,
                  future_mask=torch.as_tensor(mask & (root_terminal == 0), device=device),
                  reach_target=torch.as_tensor(reachable, device=device),
                  clear_target=torch.as_tensor(clearable, device=device),
                  cost_target=torch.as_tensor(cost_target, device=device))
    return result


def forward_motor(net, batch, *, auxiliary=True):
    features = batch["inputs"]
    return net(*features[:6], aux=features[6], return_aux=auxiliary,
               motor_geometry=batch["geometry"] if auxiliary else None)


def policy_kl(logits, batch):
    reference = batch["reference_logp"]
    current = logits.float().log_softmax(-1)
    return (reference.exp() * (reference - current)).sum(-1).clamp_min(0)


def motor_loss(output, batch, *, anchor_weight=10.):
    logits, _, extra = output
    mask, future = batch["mask"], batch["future_mask"]

    def root_mean(value, valid):
        return (value * valid).sum(-1) / valid.sum(-1).clamp_min(1)

    def mean(value):
        return (value * batch["weights"]).mean()

    known = batch["known_effects"]
    effects = root_mean(F.smooth_l1_loss(
        extra["effect_predictions"][..., known].float(), batch["effect_target"][..., known],
        reduction="none").mean(-1), mask)
    losses, metrics = {}, {"effect_error": mean(effects)}
    for name in ("reach", "clear"):
        prediction = extra[f"motor_{name}_logits"].float()
        target = batch[f"{name}_target"].float()
        losses[name] = root_mean(F.binary_cross_entropy_with_logits(
            prediction, target, reduction="none").mean((-1, -2)), future)
        metrics[f"{name}_brier"] = mean(root_mean((prediction.sigmoid() - target).square().mean((-1, -2)), future))
        metrics[f"{name}_ce"] = mean(losses[name])
    cost_mask = batch["reach_target"] & future[..., None, None]
    cost = F.smooth_l1_loss(extra["motor_log_cost"].float(), batch["cost_target"], reduction="none")
    cost = (cost * cost_mask).sum((-1, -2)) / cost_mask.sum((-1, -2)).clamp_min(1)
    cost = root_mean(cost, future & cost_mask.any(-1).any(-1))
    kl = policy_kl(logits, batch)
    metrics.update(cost_error=mean(cost), anchor_kl=mean(kl))
    loss = effects + losses["reach"] + losses["clear"] + .25 * cost + anchor_weight * kl
    return mean(loss), metrics


def cache_reference(net, rows, *, batch_size, device):
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            batch = make_motor_batch(part, device=device, targets=False)
            logits, _ = forward_motor(net, batch, auxiliary=False)
            logs = logits.float().log_softmax(-1).cpu().numpy()
            for i, row in enumerate(part):
                row["reference_logp"] = logs[i, :len(row["actions"])].copy()


def evaluate_motor(net, rows, *, batch_size, device, targets=True):
    totals = Counter()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            batch = make_motor_batch(part, device=device, targets=targets)
            result = forward_motor(net, batch, auxiliary=targets)
            if targets:
                _, metrics = motor_loss(result, batch)
            else:
                metrics = {"anchor_kl": (policy_kl(result[0], batch) * batch["weights"]).mean()}
            for key, value in metrics.items():
                totals[key] += float(value) * len(part) / len(rows)
    return dict(totals)


def initialize_motor_priors(net, rows, *, probability_floor=1e-4):
    """Start new heads at training-only cell prevalence, never random 50% maps.

    This is a global cell/parity baseline. Pace and candidate deviations must
    still be learned from the actual public candidate and controller inputs.
    Zero output weights preserve the competitive core and its policy exactly.
    """
    if not rows or any(r["record"]["split"] != "train" for r in rows):
        raise ValueError("motor initialization accepts training roots only")
    if not 0 < probability_floor < .5:
        raise ValueError("motor probability floor must lie inside (0, .5)")
    sums = {name: torch.zeros(2, 128) for name in ("reach", "clear", "cost", "cost_mass")}
    effect_sum = torch.zeros(len(EFFECT_TOKEN_NAMES))
    mass = effect_mass = 0.
    for row in rows:
        batch = make_motor_batch([row], device="cpu")
        weight = float(row["weight"])
        mask, future = batch["mask"][0], batch["future_mask"][0]
        effect_sum += weight * batch["effect_target"][0, mask].mean(0)
        effect_mass += weight
        if not future.any():
            continue
        mass += weight
        for name in ("reach", "clear"):
            sums[name] += weight * batch[name + "_target"][0, future].float().mean(0)
        reachable = batch["reach_target"][0, future].float()
        sums["cost"] += weight * (batch["cost_target"][0, future] * reachable).mean(0)
        sums["cost_mass"] += weight * reachable.mean(0)
    if mass <= 0:
        raise ValueError("motor initialization needs nonterminal training candidates")
    prior = torch.stack([torch.logit((sums[name] / mass).clamp(probability_floor, 1-probability_floor))
                         for name in ("reach", "clear")] +
                        [sums["cost"] / sums["cost_mass"].clamp_min(1e-12)])
    head = net.motor_auxiliary
    with torch.no_grad():
        head.opportunity.weight.zero_()
        head.opportunity.bias.copy_(prior.flatten())
        head.effects.weight.zero_()
        head.effects.bias.copy_(effect_sum / effect_mass)
    return dict(schema="drmc-motor-training-cell-initialization-v1", roots=len(rows),
                probability_floor=probability_floor, weighting="equal game, then root and candidate",
                opportunity_bias=prior.flatten().tolist(), effect_bias=(effect_sum / effect_mass).tolist())


def cache_motor_features(net, rows, *, batch_size, device, activity=None):
    """Cache lossless features only while the entire competitive core is frozen.

    Targets live in a separate cache entry. The feature forward receives actual
    public inputs and no afterstate, clear map, future reach target or outcome.
    """
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            batch = make_motor_batch(part, device=device, targets=False)
            inputs = batch["inputs"]
            logits, _, extra = net(*inputs[:6], aux=inputs[6], return_aux=True)
            features = extra["candidate_context"].float().cpu().numpy()
            logp = logits.float().log_softmax(-1).cpu().numpy()
            targets = make_motor_batch(part, device="cpu")
            for i, row in enumerate(part):
                n = len(row["actions"])
                row["reference_logp"] = logp[i, :n].copy()
                row["frozen_motor_features"] = features[i, :n].copy()
                row["frozen_motor_geometry"] = targets["geometry"][i].numpy().copy()
                row["cached_motor_targets"] = {
                    key: targets[key][i, :n].numpy().copy()
                    for key in ("effect_target", "future_mask", "reach_target", "clear_target", "cost_target")}
                row["known_effects"] = targets["known_effects"].numpy().copy()
            if activity is not None:
                activity(min(start + batch_size, len(rows)))


def cached_motor_batch(rows, *, device):
    """Pack complete frozen candidate frontiers for head-only optimization."""
    width = max(len(row["actions"]) for row in rows)
    count = len(rows)
    mask = np.zeros((count, width), bool)
    features = np.zeros((count, width, rows[0]["frozen_motor_features"].shape[-1]), np.float32)
    reference = np.full((count, width), -1e9, np.float32)
    targets = {key: np.zeros((count, width, *value.shape[1:]), value.dtype)
               for key, value in rows[0]["cached_motor_targets"].items()}
    for i, row in enumerate(rows):
        n = len(row["actions"])
        mask[i, :n] = True
        features[i, :n] = row["frozen_motor_features"]
        reference[i, :n] = row["reference_logp"]
        for key in targets:
            targets[key][i, :n] = row["cached_motor_targets"][key]
    arrays = dict(features=features, geometry=np.stack([r["frozen_motor_geometry"] for r in rows]),
                  reference_logp=reference, mask=mask,
                  weights=np.asarray([r["weight"] for r in rows], np.float32),
                  known_effects=rows[0]["known_effects"], **targets)
    return {key: torch.as_tensor(value, device=device) for key, value in arrays.items()}


def forward_cached_motor(head, batch):
    return batch["reference_logp"], None, head(batch["features"], batch["geometry"])


def evaluate_cached_motor(head, rows, *, batch_size, device):
    totals = Counter()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            batch = cached_motor_batch(part, device=device)
            _, metrics = motor_loss(forward_cached_motor(head, batch), batch, anchor_weight=0.)
            for key, value in metrics.items():
                totals[key] += float(value) * len(part) / len(rows)
    return dict(totals)
