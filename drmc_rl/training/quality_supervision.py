"""Supervised paired-outcome learning, separate from on-policy PPO.

Targets describe a frozen continuation panel, never optimal play. Every row
retains the entire legal frontier. Losses average within a state and then give
each source game equal total weight. Censored panels cannot silently become
complete-case policy-improvement data.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA, context_from_info
from drmc_rl.search.native_pair import CAUSAL_PUBLIC_SCHEMA, state_from_payload
from drmc_rl.search.public_policy import policy_request


def join_quality_rows(sources, targets):
    source_by_id = {row["id"]: row for row in sources}
    if len(source_by_id) != len(sources):
        raise ValueError("duplicate source identity")
    result, seen, contracts = [], set(), set()
    for target in targets:
        source_id = target["source_id"]
        if source_id in seen:
            raise ValueError(
                "duplicate quality target; repeated deterministic labels add no evidence"
            )
        seen.add(source_id)
        source = source_by_id[source_id]
        state = state_from_payload(source)
        side = int(source["root_side"])
        legal = state.legal_actions_by_side[side]
        if (
            target["schema"] != "drmc-paired-terminal-quality-v1"
            or state.public_observation_schema != CAUSAL_PUBLIC_SCHEMA
        ):
            raise ValueError(
                "quality fitting requires paired terminal targets and causal public observations"
            )
        if target["game_id"] != source["game_id"] or target["root_side"] != side:
            raise ValueError("target/source identity mismatch")
        if (
            tuple(target["actions"]) != legal
            or tuple(c["action"] for c in target["candidates"]) != legal
        ):
            raise ValueError("quality fitting requires the full ordered feasible frontier")
        if not target["posterior_enumerated"] or target["candidate_truncation"]:
            raise ValueError("unsupported or truncated counterfactual labels")
        if target["policy_target"] is None or any(
            c["wdl"] is None or c["unknown_mass"] for c in target["candidates"]
        ):
            raise ValueError("censored candidate panels are unknown, not supervised policy targets")
        import json

        contracts.add(
            json.dumps([target["member_sha256"], target["continuations"]], sort_keys=True)
        )
        wdl = np.asarray([c["wdl"] for c in target["candidates"]], np.float32)
        prior = np.asarray(target["reference_prior"], np.float32)
        improved = np.asarray(target["policy_target"]["probability"], np.float32)
        if (
            wdl.shape != (len(legal), 3)
            or not np.isfinite(wdl).all()
            or (wdl < 0).any()
            or not np.allclose(wdl.sum(-1), 1)
        ):
            raise ValueError("invalid terminal WDL distribution")
        for probabilities in (prior, improved):
            if (
                probabilities.shape != (len(legal),)
                or not np.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or not np.isclose(probabilities.sum(), 1)
            ):
                raise ValueError("invalid supported policy distribution")
        if (prior <= 0).any():
            raise ValueError("reference policy must support every feasible candidate")
        # A sharp improved target can underflow to zero in float32. The full
        # WDL/ranking inventory remains present even for zero policy mass.
        result.append(
            dict(
                state=state,
                side=side,
                game_id=source["game_id"],
                source_id=source_id,
                wdl=wdl,
                prior=prior,
                improved=improved,
                incumbent=legal.index(target["incumbent"]),
            )
        )
    if len(contracts) != 1:
        raise ValueError("reanalysis versions or continuation panels cannot be pooled silently")
    return result


def split_games(rows, *, seed, validation_fraction=0.25):
    games = sorted({row["game_id"] for row in rows})
    if len(games) < 2 or not 0 < validation_fraction < 1:
        raise ValueError("quality validation requires at least two independent source games")
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    n = max(1, min(len(games) - 1, round(len(games) * validation_fraction)))
    validation = set(games[:n])
    return (
        [r for r in rows if r["game_id"] not in validation],
        [r for r in rows if r["game_id"] in validation],
    )


def make_batch(rows, *, schema, device):
    if schema not in (PUBLIC_CONTEXT_SCHEMA, "zero_v1_vs"):
        raise ValueError("quality actor must use an explicit public schema")
    # Pack directly from the complete inventory, including equivalent poses.
    # Legacy observations retain their historical bond masking, but labels and
    # head comparisons never drop a feasible candidate.
    requests = [
        policy_request(
            r["state"].privileged.public,
            r["side"],
            r["state"].legal_actions_by_side[r["side"]],
            r["state"].action_costs_by_side[r["side"]],
            context_schema=schema,
        )
        for r in rows
    ]
    width = max(len(r["prior"]) for r in rows)
    batch = len(rows)
    actions = np.zeros((batch, width), np.int64)
    costs = np.zeros((batch, width), np.float32)
    mask = np.zeros((batch, width), bool)
    wdl = np.zeros((batch, width, 3), np.float32)
    prior = np.zeros((batch, width), np.float32)
    improved = np.zeros_like(prior)
    for i, row in enumerate(rows):
        n = len(row["prior"])
        side = row["side"]
        state = row["state"]
        actions[i, :n] = state.legal_actions_by_side[side]
        costs[i, :n] = state.action_costs_by_side[side]
        mask[i, :n] = True
        wdl[i, :n] = row["wdl"]
        prior[i, :n] = row["prior"]
        improved[i, :n] = row["improved"]
    counts = Counter(row["game_id"] for row in rows)
    weights = np.asarray([1 / counts[row["game_id"]] for row in rows], np.float32)
    weights /= weights.mean()
    values = dict(
        obs=np.stack([r[0] for r in requests]),
        actions=actions,
        costs=costs,
        mask=mask,
        pills=np.asarray([r["state"].privileged.public.sides[r["side"]].pill for r in rows]),
        previews=np.asarray([r["state"].privileged.public.sides[r["side"]].preview for r in rows]),
        aux=np.stack([context_from_info(r[1]) for r in requests])
        if schema == PUBLIC_CONTEXT_SCHEMA
        else np.zeros((batch, 72), np.float32),
        wdl=wdl,
        prior=prior,
        improved=improved,
        weights=weights,
        incumbent=np.asarray([r["incumbent"] for r in rows]),
    )
    return {key: torch.as_tensor(value, device=device) for key, value in values.items()}


def forward(net, batch):
    return net(
        batch["obs"],
        batch["pills"],
        batch["previews"],
        batch["actions"],
        batch["costs"],
        batch["mask"],
        aux=batch["aux"],
        return_aux=True,
    )


def quality_loss(
    output, batch, *, anchor_logp=None, policy_weight=0.0, anchor_weight=1.0, gap_weight=0.25
):
    logits, value, extra = output
    valid = batch["mask"]
    counts = valid.sum(-1).clamp_min(1)
    candidate_logp = extra["candidate_wdl_logits"].float().log_softmax(-1)
    candidate_probability = candidate_logp.exp()
    state_logp = extra["state_wdl_logits"].float().log_softmax(-1)
    state_target = (batch["wdl"] * batch["prior"].unsqueeze(-1)).sum(1)
    candidate_ce = (-(candidate_logp * batch["wdl"]).sum(-1) * valid).sum(-1) / counts
    state_ce = -(state_logp * state_target).sum(-1)
    target_utility = batch["wdl"][..., 0] - batch["wdl"][..., 2]
    predicted_utility = candidate_probability[..., 0] - candidate_probability[..., 2]
    index = batch["incumbent"].unsqueeze(-1)
    target_gap = target_utility - target_utility.gather(1, index)
    predicted_gap = predicted_utility - predicted_utility.gather(1, index)
    gap = ((predicted_gap - target_gap).square() * valid).sum(-1) / counts
    unequal = valid & (target_gap.abs() > 1e-5)
    ranking = (F.softplus(-target_gap.sign() * predicted_gap / 0.1) * unequal).sum(
        -1
    ) / unequal.sum(-1).clamp_min(1)
    logp = logits.float().log_softmax(-1)
    policy_ce = -(batch["improved"] * logp).sum(-1)
    anchor_kl = (
        torch.zeros_like(policy_ce)
        if anchor_logp is None
        else (anchor_logp.exp() * (anchor_logp - logp)).sum(-1)
    )
    expected_utility = state_target[:, 0] - state_target[:, 2]
    value_mse = (value.flatten() - expected_utility).square()
    loss = (
        state_ce
        + candidate_ce
        + 0.25 * value_mse
        + gap_weight * (gap + 0.1 * ranking)
        + policy_weight * policy_ce
        + anchor_weight * anchor_kl
    )
    mean = lambda x: (x * batch["weights"]).mean()
    metrics = {
        k: mean(v)
        for k, v in dict(
            state_ce=state_ce,
            candidate_ce=candidate_ce,
            gap_mse=gap,
            ranking_loss=ranking,
            policy_ce=policy_ce,
            anchor_kl=anchor_kl,
            value_mse=value_mse,
            state_brier=(state_logp.exp() - state_target).square().sum(-1),
            candidate_brier=((candidate_probability - batch["wdl"]).square().sum(-1) * valid).sum(
                -1
            )
            / counts,
        ).items()
    }
    return mean(loss), metrics


def upgrade_public_model(checkpoint, *, mode, device):
    """Migrate the frozen core once, then preserve a fitted model across phases."""
    from tools.eval_policy import _build_net_from_cfg

    cfg = deepcopy(checkpoint["cfg"])
    sp = cfg.get("smdp_ppo", cfg)
    if mode not in ("baseline", "critic", "context", "combined"):
        raise ValueError("unknown quality architecture ablation")
    expected_schema = PUBLIC_CONTEXT_SCHEMA if mode in ("context", "combined") else "zero_v1_vs"
    expected_critic = "candidate_attention" if mode in ("critic", "combined") else "global"
    old = checkpoint.get("ema_state_dict") or checkpoint["state_dict"]
    # Every supervised phase uses the causal source-bank contract. A later
    # rollout must not silently return this checkpoint to warped observations.
    cfg.setdefault("env", {})["public_observations"] = True
    if sp.get("candidate_terminal_wdl") or sp.get("candidate_wdl"):
        if (
            sp.get("candidate_architecture") != "g5"
            or not sp.get("candidate_terminal_wdl")
            or not sp.get("candidate_wdl")
            or sp.get("aux_spec") != expected_schema
            or sp.get("candidate_critic_context", "global") != expected_critic
        ):
            raise ValueError("a fitted quality model must retain its architecture and schema")
        net, _, _ = _build_net_from_cfg(cfg, 20, device)
        net.load_state_dict(old, strict=True)
        return net, cfg
    if sp.get("candidate_architecture") != "g5" or sp.get("aux_spec") != "zero_v1_vs":
        raise ValueError("quality warm start requires the frozen public G5 zero-aux core")
    sp.update(
        candidate_terminal_wdl=True,
        candidate_wdl=True,
        candidate_critic_context=expected_critic,
    )
    if mode in ("context", "combined"):
        sp["aux_spec"] = PUBLIC_CONTEXT_SCHEMA
    net, _, _ = _build_net_from_cfg(cfg, 20, device)
    state = net.state_dict()
    allowed = (
        "value_query.",
        "value_projection.",
        "state_wdl_head.",
        "candidate_wdl_head.",
        "side_condition.",
    )
    for key, value in state.items():
        if key in old and old[key].shape == value.shape:
            value.copy_(old[key])
        elif key == "condition.0.weight" and mode in ("context", "combined"):
            width = old[key].shape[1] - 72
            value.zero_()
            value[:, :width].copy_(old[key][:, :width])
        elif not key.startswith(allowed):
            raise ValueError(f"unexpected warm-start shape or missing tensor: {key}")
    if set(old) - set(state):
        raise ValueError("warm start would discard unknown learned tensors")
    if mode in ("context", "combined"):
        width = old["condition.0.weight"].shape[1] - 72
        state["side_condition.0.weight"].zero_()
        state["side_condition.0.weight"][:, :width].copy_(old["condition.0.weight"][:, :width])
        for suffix in ("0.bias", "2.weight", "2.bias"):
            state["side_condition." + suffix].copy_(old["condition." + suffix])
    net.load_state_dict(state, strict=True)
    if mode in ("critic", "combined"):
        # The shared new critic starts identically in both ablations despite
        # the context model constructing additional modules before it.
        generator = torch.Generator(device="cpu").manual_seed(torch.initial_seed())
        with torch.no_grad():
            for parameter in net.value_query.parameters():
                initial = torch.empty(parameter.shape, dtype=parameter.dtype, device="cpu")
                if initial.ndim > 1:
                    torch.nn.init.xavier_uniform_(initial, generator=generator)
                else:
                    initial.zero_()
                parameter.copy_(initial)
    # Each architecture starts with the same neutral outcome heads. Otherwise
    # adding a module changes RNG consumption and confounds the first ablation
    # with different arbitrary WDL logits. Do not perturb the caller's RNG.
    with torch.no_grad():
        for head in (net.state_wdl_head, net.candidate_wdl_head):
            head.weight.zero_()
            head.bias.zero_()
    return net, cfg


def layer_diagnostics(net, before, representation, gradient_norms=None):
    groups = {}
    for name, param in net.named_parameters():
        layer = name.rsplit(".", 1)[0]
        row = groups.setdefault(
            layer, dict(weight_squared=0.0, update_squared=0.0, gradient_squared=0.0)
        )
        row["weight_squared"] += float(before[name].float().square().sum())
        row["update_squared"] += float((param.detach() - before[name]).float().square().sum())
        if gradient_norms is not None:
            row["gradient_squared"] += gradient_norms.get(name, 0.0) ** 2
        elif param.grad is not None:
            row["gradient_squared"] += float(param.grad.float().square().sum())
    result = {
        name: dict(
            gradient_norm=row["gradient_squared"] ** 0.5,
            weight_norm=row["weight_squared"] ** 0.5,
            update_norm=row["update_squared"] ** 0.5,
            update_to_weight=(row["update_squared"] / row["weight_squared"]) ** 0.5
            if row["weight_squared"]
            else None,
        )
        for name, row in groups.items()
    }
    centered = representation.detach().float().flatten(0, -2)
    centered = centered - centered.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered.cpu())
    mass = singular.square()
    mass /= mass.sum().clamp_min(1e-30)
    rank = (
        float(torch.exp(-(mass * mass.clamp_min(1e-30).log()).sum()))
        if singular.square().sum() > 0
        else 0.0
    )
    return dict(
        layers=result,
        gradient_scope="last-minibatch-before-clipping"
        if gradient_norms is not None
        else "current-gradients",
        representation_effective_rank=rank,
        representation_samples=len(centered),
    )
