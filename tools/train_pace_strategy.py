"""Natural full-game PPO for the pace adapter or the live public controller core.

Uses exact controller execution, optionally batched at causal decisions.
The adapter retains frozen features; the full core retains exact public model
inputs and trains its board/context representations. Every transition receives
its natural terminal W/D/L return (gamma=1). The actor sums decision credit per game; critic and
regularization reductions are independently declared in checkpoint metadata.
No shaped reward, search label, hidden opponent field or time-limit draw trains
the policy. Run through the trainer-pace-strategy program recipe.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
from copy import deepcopy
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from drmc_rl.arena.experiment import dump
from drmc_rl.arena.store import ArenaStore
from drmc_rl.execution.pace import BY_ID
from drmc_rl.models.policy.pace_adapter import PacePolicy
from drmc_rl.planning.native_reach import NativeReachabilityRunner
from drmc_rl.training.episodic_objective import (
    categorical_kl,
    clipped_surrogate,
    normalize_advantages,
    normalized_weights,
    objective_contract,
    validate_resume_objective,
)
from drmc_rl.training.public_league import PublicOpponentPool
from tools.trainer_arena_cache import MemoPlanner
from tools.trainer_planning_arena import run_batch
from tools.vs_head_to_head import PlainPolicy


class TrainingActivity:
    """Report real work, keeping completed-update counters unchanged."""
    def __init__(self, path, progress):
        self.path, self.progress = path, progress
        self.next_write = 0.0

    def __call__(self, phase, **work):
        activity = dict(phase=phase, **work)
        if activity == self.progress.get("activity"):
            return  # A timer alone must not make a stalled worker look healthy.
        changed_phase = self.progress.get("phase") != phase
        self.progress.update(phase=phase, activity=activity)
        now = time.monotonic()
        if changed_phase or now >= self.next_write:
            self.progress["updated_at"] = datetime.now(UTC).isoformat()
            dump(self.path, self.progress)
            self.next_write = now + 5


def terminal_samples(batch):
    """Exclude censored games; retain length metadata without choosing a loss."""
    samples = []
    for game_id, (row, moves, _) in enumerate(batch):
        if row["reason"] == "timeout":
            continue
        decisions = [m["learning"] for m in moves if "learning" in m]
        for sample in decisions:
            samples.append(
                {
                    **sample,
                    "return": 2 * row["score"] - 1,
                    "game_id": game_id,
                    "episode_length": len(decisions),
                    "weight": 1 / max(1, len(decisions)),
                }
            )
    return samples


def restore_game_journal(path, committed_update):
    """Keep only complete games covered by the restored optimizer checkpoint."""
    path = Path(path)
    if not path.exists():
        return
    temporary = path.with_suffix(path.suffix+".next")
    with path.open() as source, temporary.open("w") as target:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if line.endswith("\n"):
                    raise
                continue  # An interrupted final write, never a completed row.
            if row["update"] <= committed_update:
                target.write(json.dumps(row)+"\n")
    temporary.replace(path)


def add_game_totals(stats, row):
    """Separate simulated time, attempted spawns and actual learning decisions."""
    stats["games"] = stats.get("games",0)+1
    stats["frames"] = stats.get("frames",0)+row["frames"]
    for key in ("decisions", "no_reachable_after_delay", "forced_placements", "feasible_candidates", "validated_input_frames"):
        stats[key] = stats.get(key,0)+row["a_stats"].get(key,0)
    timeout = row["reason"] == "timeout"
    stats["timeouts"] = stats.get("timeouts",0)+int(timeout)
    if not timeout:
        key = {1.0:"wins",0.0:"losses",.5:"draws"}[row["score"]]
        stats[key] = stats.get(key,0)+1
        learned = row["a_stats"].get("decisions",0)-row["a_stats"].get("no_reachable_after_delay",0)
        stats["learning_decisions"] = stats.get("learning_decisions",0)+learned


def training_target_met(progress, config):
    if config.get("target_frames") is None:
        return progress["updates"] >= config["updates"]
    return (progress["frames"] >= config["target_frames"] and
            progress.get("decisions", 0) >= config.get("target_decisions", 0) and
            all(progress["paces"].get(p,{}).get("learning_decisions",0) >= config.get("minimum_decisions_per_pace",0)
                for p in config["paces"]))


def tensor_batch(records, device):
    n, k = len(records), max(len(r["base_logits"]) for r in records)
    width = records[0]["candidate"].shape[-1]
    c = np.zeros((n,k,width), np.float32)
    logits = np.full((n,k), -1e9, np.float32)
    mask = np.zeros((n,k), bool)
    for i,r in enumerate(records):
        count = len(r["base_logits"])
        c[i,:count], logits[i,:count], mask[i,:count] = r["candidate"], r["base_logits"], True
    arrays = (c, np.stack([r["context"] for r in records]), np.stack([r["motor"] for r in records]),
        logits, np.asarray([r["base_value"] for r in records], np.float32), mask)
    features = tuple(torch.as_tensor(a, device=device) for a in arrays)
    keys = ("slot", "old_logprob", "old_value", "return", "weight", "advantage")
    keys += tuple(
        key + "_weight"
        for key in ("actor", "value", "entropy", "parent_kl")
        if key + "_weight" in records[0]
    )
    extras = {
        key: torch.as_tensor(
            [r[key] for r in records],
            device=device,
            dtype=torch.long if key == "slot" else torch.float32,
        )
        for key in keys
    }
    return features, extras


def _training_module(actor):
    return actor.training_module if hasattr(actor, "training_module") else actor.adapter


def _training_batch(actor, rows):
    if hasattr(actor, "training_batch"):
        return actor.training_batch(rows)
    return tensor_batch(rows, actor.device)


def _training_forward(actor, features):
    if hasattr(actor, "training_forward"):
        return actor.training_forward(features)
    return actor.adapter(*features)


@torch.no_grad()
def _policy_snapshot(actor, records, size, *, reference=None, activity=None):
    """Exact categorical KL on all decisions, including unchosen moves."""
    distributions, divergences, errors = [], [], []
    agreement = {}
    for start in range(0, len(records), size):
        rows = records[start : start + size]
        features, data = _training_batch(actor, rows)
        logits, values = _training_forward(actor, features)
        logs = logits.log_softmax(-1)
        errors.extend((values - data["return"]).square().cpu().tolist())
        if reference is None:
            chosen = logs.gather(1, data["slot"][:, None]).squeeze(1)
            if "behavior_logp" in rows[0]:
                # GPU reductions vary slightly with batch/candidate padding.
                # Preserve the ACTUAL behavior distribution for PPO/KL and
                # audit total probability mass, not one rare sampled logp.
                current = logs.cpu().numpy()
                old = np.full_like(current, -1e9)
                for i, row in enumerate(rows):
                    old[i, :len(row["behavior_logp"])] = row["behavior_logp"]
                    if abs(float(old[i, row["slot"]]) - row["old_logprob"]) > 1e-6:
                        raise RuntimeError("stored behavior likelihood differs from the sampled action")
                    distributions.append(torch.from_numpy(row["behavior_logp"].copy()))
                variations = np.abs(np.exp(old.astype(np.float64)) - np.exp(current.astype(np.float64))).sum(-1) / 2
                tv = float(np.max(variations))
                error = float(np.max(np.abs(old - current)))
                agreement["collection_max_total_variation"] = max(agreement.get("collection_max_total_variation", 0), tv)
                agreement["collection_max_logp_error"] = max(agreement.get("collection_max_logp_error", 0), error)
                if not np.isfinite(tv) or not np.isfinite(error):
                    raise RuntimeError(f"collection distribution differs from frozen update policy: total variation={tv:.9g}, max logp error={error:.9g}")
                # Weight-version checks establish an unchanged collection
                # network; this audit catches input/distribution corruption.
                # FP32 GPU reductions vary with shape and kernel. Bound total
                # probability mass at 0.01% and likelihood ratio error at 0.1%,
                # far below a PPO clipping interval. Always retain ACTUAL
                # behavior logs. Recheck shape outliers before rejecting them.
                for i in np.flatnonzero((variations > 1e-4) | (np.abs(old-current).max(-1) > 1e-3)):
                    if len(rows) == 1:
                        if not hasattr(actor, "precise_behavior_logp"):
                            raise RuntimeError(f"collection distribution differs from frozen update policy: total variation={variations[i]:.9g}, max logp error={error:.9g}")
                        # Two FP32 evaluations can fall on opposite sides of
                        # an accurate result. Adjudicate against an independent
                        # FP64 forward of the SAME inputs/weights, with the
                        # unchanged bounds. Never replace collection logs.
                        precise = np.asarray(actor.precise_behavior_logp(rows[i]), dtype=np.float64)
                        behavior = rows[i]["behavior_logp"].astype(np.float64)
                        if precise.shape != behavior.shape:
                            raise RuntimeError("precise collection audit changed the candidate inventory")
                        precise_tv = float(np.abs(np.exp(behavior)-np.exp(precise)).sum()/2)
                        precise_error = float(np.abs(behavior-precise).max())
                        if (not np.isfinite(precise_tv) or not np.isfinite(precise_error)
                                or precise_tv > 1e-4 or precise_error > 1e-3):
                            raise RuntimeError(f"collection distribution differs from frozen update policy after FP64 audit: total variation={precise_tv:.9g}, max logp error={precise_error:.9g}; FP32 total variation={variations[i]:.9g}")
                        agreement["collection_precision_rechecks"] = agreement.get("collection_precision_rechecks", 0) + 1
                        agreement["collection_max_precise_total_variation"] = max(
                            agreement.get("collection_max_precise_total_variation", 0), precise_tv)
                        agreement["collection_max_precise_logp_error"] = max(
                            agreement.get("collection_max_precise_logp_error", 0), precise_error)
                        continue
                    _, recheck = _policy_snapshot(actor, [rows[i]], 1)
                    agreement["collection_rechecked_decisions"] = agreement.get("collection_rechecked_decisions", 0) + 1
                    agreement["collection_max_rechecked_total_variation"] = max(
                        agreement.get("collection_max_rechecked_total_variation", 0),
                        recheck["collection_max_total_variation"],
                    )
                    agreement["collection_precision_rechecks"] = agreement.get("collection_precision_rechecks", 0) + recheck.get("collection_precision_rechecks", 0)
                    for key in ("collection_max_precise_total_variation", "collection_max_precise_logp_error"):
                        if key in recheck:
                            agreement[key] = max(agreement.get(key, 0), recheck[key])
            else:
                if not torch.allclose(chosen, data["old_logprob"], atol=3e-5, rtol=0):
                    difference = float((chosen - data["old_logprob"]).abs().max())
                    raise RuntimeError(f"collection likelihood differs from the frozen update policy: max logp error={difference:.9g}")
                host_logs = logs.cpu()
                distributions.extend(host_logs[i, : len(row["base_logits"])].clone()
                                     for i, row in enumerate(rows))
        else:
            old = np.full(tuple(logs.shape), -1e9, np.float32)
            for i, row in enumerate(rows):
                old[i, : len(row["base_logits"])]=reference[start + i].numpy()
            divergences.extend(categorical_kl(torch.as_tensor(old, device=logs.device), logs).cpu().tolist())
        if activity:
            activity("auditing_collection" if reference is None else "checking_update",
                     checked=min(start + size, len(records)), total=len(records))
    if reference is None:
        return distributions, agreement
    kl = float(np.mean(divergences))
    return (max(0.0, kl) if np.isfinite(kl) else kl), float(np.mean(errors))


def update_adapter(actor, optimizer, records, config, seed, *, activity=None):
    if not records:
        raise RuntimeError("no natural-terminal learner decisions; cannot train")
    if hasattr(actor, "finish_collection"):
        actor.finish_collection(records)
    contract = objective_contract(config)
    inverse_lengths = np.asarray([r["weight"] for r in records])
    advantages, center, scale = normalize_advantages(
        [r["return"] - r["old_value"] for r in records],
        inverse_lengths,
        contract["advantage_normalization"],
    )
    reductions = {
        key: normalized_weights(inverse_lengths, contract[key])
        for key in ("actor", "value", "entropy", "parent_kl")
    }
    for i, record in enumerate(records):
        record["advantage"] = float(advantages[i])
        record.update({key + "_weight": float(weights[i]) for key, weights in reductions.items()})
    size = config.get("minibatch", 128)
    old_distributions, collection_agreement = _policy_snapshot(actor, records, size, activity=activity)
    rng, totals = np.random.default_rng(seed), defaultdict(list)
    max_kl = float(config.get("max_update_kl", 0.06))
    if not np.isfinite(max_kl) or max_kl <= 0:
        raise ValueError("max_update_kl must be finite and positive")
    accepted_kl, rejected, steps, stopped = 0.0, 0, 0, False
    first_step_kl = None
    value_mse = float(np.mean([(r["old_value"] - r["return"]) ** 2 for r in records]))
    initial_mse = value_mse
    # Epoch guards avoid quadratic work from a full-batch check per minibatch.
    # Rejected updates restore BOTH weights and Adam moments before retrying.
    for _epoch in range(config.get("epochs", 2)):
        indices = rng.permutation(len(records))
        saved_model = deepcopy(_training_module(actor).state_dict())
        saved_optimizer = deepcopy(optimizer.state_dict())
        rates = [group["lr"] for group in optimizer.param_groups]
        for attempt in range(config.get("kl_backtracks", 4) + 1):
            attempt_totals = defaultdict(list)
            attempt_steps = 0
            candidate_first_kl = None
            for start in range(0, len(indices), size):
                rows = [records[i] for i in indices[start : start + size]]
                features, data = _training_batch(actor, rows)
                logits, values = _training_forward(actor, features)
                log_probs = logits.log_softmax(-1)
                chosen = log_probs.gather(1, data["slot"][:, None]).squeeze(1)
                policy_loss = clipped_surrogate(
                    chosen - data["old_logprob"],
                    data["advantage"],
                    data["actor_weight"],
                    config.get("clip", 0.15),
                )
                value_loss = (
                    data["value_weight"]
                    * F.smooth_l1_loss(values, data["return"], reduction="none")
                ).mean()
                entropy = -(data["entropy_weight"] * (log_probs.exp() * log_probs).sum(-1)).mean()
                base_kl = (
                    data["parent_kl_weight"]
                    * categorical_kl(
                        data["parent_logp"] if "parent_logp" in data else features[3].log_softmax(-1),
                        log_probs,
                    )
                ).mean()
                loss = (
                    policy_loss
                    + config.get("value_coefficient", 0.5) * value_loss
                    - config.get("entropy", 0.003) * entropy
                    + config.get("parent_kl", 0.02) * base_kl
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite pace training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(_training_module(actor).parameters(), 0.7)
                if not torch.isfinite(norm):
                    raise RuntimeError("non-finite pace adapter gradient")
                optimizer.step()
                attempt_steps += 1
                if activity:
                    activity("optimizing", epoch=_epoch+1, epochs=config.get("epochs", 2),
                             step=attempt_steps, steps=(len(indices)+size-1)//size,
                             attempt=attempt+1)
                if steps == 0 and attempt_steps == 1:
                    with torch.no_grad():
                        after = _training_forward(actor, features)[0].log_softmax(-1)
                        candidate_first_kl = categorical_kl(log_probs.detach(), after).mean().item()
                for key, value in dict(
                    policy_loss=policy_loss.item(),
                    value_loss=value_loss.item(),
                    entropy=entropy.item(),
                    parent_kl=base_kl.item(),
                    gradient_norm=norm.item(),
                ).items():
                    attempt_totals[key].append((value, len(rows)))
            measured_kl, measured_mse = _policy_snapshot(
                actor, records, size, reference=old_distributions, activity=activity
            )
            if np.isfinite(measured_kl) and measured_kl <= max_kl:
                accepted_kl, value_mse = measured_kl, measured_mse
                for key, values in attempt_totals.items():
                    totals[key].extend(values)
                if first_step_kl is None:
                    first_step_kl = candidate_first_kl
                steps += attempt_steps
                break
            rejected += 1
            _training_module(actor).load_state_dict(saved_model)
            optimizer.load_state_dict(saved_optimizer)
            if attempt == config.get("kl_backtracks", 4):
                stopped = True
                break
            for group, rate in zip(optimizer.param_groups, rates, strict=True):
                group["lr"] = rate * 0.5 ** (attempt + 1)
        if stopped:
            break
    return {
        key: float(sum(v * n for v, n in values) / sum(n for _, n in values))
        for key, values in totals.items()
    } | collection_agreement | dict(
        update_kl=accepted_kl,
        first_step_kl=first_step_kl or 0.0,
        value_mse=value_mse,
        initial_value_mse=initial_mse,
        advantage_center=center,
        advantage_scale=scale,
        optimizer_steps=steps,
        kl_backtracks=rejected,
        early_kl_stop=stopped,
        completed_learning_games=int(round(inverse_lengths.sum())),
        mean_episode_decisions=float(len(records) / inverse_lengths.sum()),
        max_episode_decisions=float(1 / inverse_lengths.min()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config["objective"] = objective_contract(config)
    if config.get("resume") and config.get("init_adapter"):
        raise ValueError("resume and init_adapter are mutually exclusive")
    output = Path(config["output"])
    output.mkdir(parents=True,exist_ok=True)
    if not config.get("resume") and (output / "training.json").exists():
        raise ValueError("a new training run requires a new output directory")
    torch.set_num_threads(config.get("threads",1))
    torch.set_num_interop_threads(1)
    if config.get("strict_fp32",False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config["seed"])
    full_core = config.get("training_model", "pace_adapter") == "public_core"
    if full_core:
        from drmc_rl.models.policy.controller_core import ControllerCorePolicy
        if config.get("init_adapter"):
            raise ValueError("full-core learning takes core weights; a frozen-feature adapter is not a core")
        actor = ControllerCorePolicy(config["checkpoint"], config["device"],
                                     resume=config.get("resume"), seed=config["seed"])
    else:
        if config.get("training_model", "pace_adapter") != "pace_adapter":
            raise ValueError("training_model must be pace_adapter or public_core")
        actor = PacePolicy(
            config["checkpoint"], config["device"], training=True,
            adapter_path=config.get("resume") or config.get("init_adapter"), seed=config["seed"],
        )
    checkpoint_prefix = "core" if full_core else "adapter"
    parent = PlainPolicy(Path(config["checkpoint"]),config["device"],public_only=True)
    opponents = PublicOpponentPool(
        config.get("opponent_pool"), parent, config["checkpoint"], config["device"]
    )
    opponent_identities = (
        opponents.identities()
        if config.get("opponent_pool")
        else {"parent": {"checkpoint_sha256": actor.parent_sha256, "adapter_sha256": None}}
    )
    rollout = run_batch
    if config.get("rollout_backend", "frames") == "events":
        from tools.trainer_event_rollout import ParallelPlanning, run_event_batch
        planner = ParallelPlanning(config.get("planner_workers",4))
        rollout = run_event_batch
        config["mixed_core_actor"] = None if full_core else "learner"
    else:
        if config.get("rollout_backend", "frames") != "frames":
            raise ValueError("rollout_backend must be frames or events")
        planner = MemoPlanner(NativeReachabilityRunner())
    module = _training_module(actor)
    optimizer = torch.optim.AdamW(module.parameters(),lr=config.get("lr",2e-4),weight_decay=.001)
    initial = {k:v.detach().clone() for k,v in module.state_dict().items()}
    start_update = 0
    progress = {
        "status": "Running",
        "updates": 0,
        "target_updates": config["updates"],
        "games": 0,
        "frames": 0,
        "decisions": 0,
        "trainable_parameters": sum(p.numel() for p in module.parameters()),
        "training_model": config.get("training_model", "pace_adapter"),
        "parent_sha256": actor.parent_sha256,
        "paces": {},
        "checkpoints": [],
        "objective": config["objective"],
        "initial_adapter": config.get("init_adapter"),
    }
    progress["opponent_identities"] = opponent_identities
    if config.get("resume"):
        previous = torch.load(config["resume"],map_location=config["device"],weights_only=True)
        validate_resume_objective(previous.get("training_config", {}), config)
        if previous.get("training_config", {}).get("training_model", "pace_adapter") != config.get("training_model", "pace_adapter"):
            raise ValueError("resume changed the trainable model contract")
        if "training_config" in previous:
            for key in ("checkpoint", "seed", "holdout_seeds", "paces"):
                if previous["training_config"][key] != config[key]:
                    raise ValueError(f"resume changed the training/evaluation contract: {key}")
            if previous["training_config"].get("opponent_pool") != config.get("opponent_pool"):
                raise ValueError("resume changed the frozen opponent population")
            if (
                config.get("opponent_pool")
                and previous["progress"].get("opponent_identities") != opponent_identities
            ):
                raise ValueError(
                    "resume changed frozen opponent bytes or lacks their identity; initialize a fresh run"
                )
        start_update = previous["update"]
        optimizer.load_state_dict(previous["optimizer"])
        actor.rng.set_state(previous["sampling_rng"].cpu())
        progress.update(previous["progress"])
        progress.update(status="Running",target_updates=config["updates"])
        if not (output/"training-games.jsonl").exists() and config.get("resume_journal"):
            shutil.copyfile(config["resume_journal"],output/"training-games.jsonl")
        restore_game_journal(output/"training-games.jsonl", start_update)
        if (output/"training-games.jsonl").exists():
            progress["paces"] = {}
            with (output/"training-games.jsonl").open() as journal:
                for line in journal:
                    row = json.loads(line)
                    add_game_totals(progress["paces"].setdefault(row["pace"],{}),row)
            if sum(s.get("learning_decisions",0) for s in progress["paces"].values()) != progress["decisions"]:
                raise RuntimeError("restored game journal differs from checkpoint learning total")
        elif config.get("minimum_decisions_per_pace",0):
            raise ValueError("per-pace training budgets require the resume game journal")
    progress.update(
        target_frames=config.get("target_frames"),
        target_decisions=config.get("target_decisions"),
        opponent_identities=opponent_identities,
        objective=config["objective"],
        minimum_decisions_per_pace=config.get("minimum_decisions_per_pace", 0),
        games_per_pace=config.get("games_per_pace", {}),
    )
    available = np.setdiff1d(np.arange(1,65536),config["holdout_seeds"])
    config["variants"] = {id: {"delay": 4} for id in ("learner", *opponents.names)}
    config["replay_games"] = 0
    store = ArenaStore(config["working_db"],replay_dir=output/"replays")
    store.conn.commit()
    import sqlite3
    with closing(sqlite3.connect(output/"arena.sqlite")) as snapshot:
        store.conn.backup(snapshot)
        snapshot.execute("PRAGMA journal_mode=DELETE")
    store.close()
    dump(output/"results.json",{"updated_at":datetime.now(UTC).isoformat(),"tournaments":[]})
    if full_core and not config.get("resume"):
        actor.save(output / "core-initial.pt", update=0, training_config=config)
    started = time.perf_counter()
    activity = TrainingActivity(output / "training.json", progress)
    try:
        for update in range(start_update+1,config["updates"]+1):
            if training_target_met(progress,config):
                break
            pace = config["paces"][(update-1)%len(config["paces"])]
            if pace not in BY_ID or (not full_core and pace in ("super_human","frame_perfect")):
                raise ValueError("the frozen-feature pilot trains Sloth through Top Humans only")
            rng = np.random.default_rng(config["seed"]+update)
            opponent_id = opponents.choose(rng)
            opponent = opponents.load(opponent_id)
            if config.get("rollout_backend") == "events":
                config["mixed_core_actor"] = (
                    "learner" if not full_core and opponent is parent and opponent_id == "parent" else None
                )
            level = 20 if pace != "sloth" and rng.random()<config.get("level20_fraction",.15) else 14
            count = config.get("games_per_pace",{}).get(pace,config.get("games_per_update",16))
            if count < 2 or count%2:
                raise ValueError("training batches require complete paired seeds")
            seeds = rng.choice(available,count//2,replace=False)
            jobs = [(int(seed),side,2*i+side) for i,seed in enumerate(seeds) for side in (0,1)]
            match = {
                "id": f"train-{update}",
                "a": "learner",
                "b": opponent_id,
                "games": count,
                "pace": pace,
                "level": level,
            }
            progress.update(
                current_pace=pace,
                current_level=level,
                current_opponent=opponent_id,
                collecting_update=update,
                collecting_games=0,
                collecting_target=count,
                collecting_decisions=0,
                collecting_frames=0,
                phase="collecting",
                activity=None,
                updated_at=datetime.now(UTC).isoformat(),
            )
            dump(output/"training.json",progress)
            update_started = time.perf_counter()
            batch, elapsed, breakdown = [], 0.0, defaultdict(float)
            chunk_size = config.get("rollout_games",count)
            if chunk_size < 2 or chunk_size%2:
                raise ValueError("rollout chunks require complete paired seeds")
            for start in range(0,len(jobs),chunk_size):
                metrics = {}
                completed_games = len(batch)
                completed_frames = sum(row["frames"] for row, _, _ in batch)
                completed_requests = sum(row["a_stats"].get("decisions", 0) for row, _, _ in batch)

                def collection_activity(work):
                    activity("collecting", games=completed_games+work["games"], target=count,
                             frames=completed_frames+work["frames"],
                             decision_requests=completed_requests+work["decision_requests"])

                part, seconds = rollout(
                    config,
                    match,
                    jobs[start : start + chunk_size],
                    None,
                    planner,
                    None,
                    policies={"learner": actor, opponent_id: opponent},
                    activity=collection_activity,
                    **({"metrics": metrics} if rollout is not run_batch else {}),
                )
                batch.extend(part)
                elapsed += seconds
                for key,value in metrics.items():
                    breakdown[key] += value
                progress.update(collecting_games=len(batch),collecting_target=count,
                    updated_at=datetime.now(UTC).isoformat())
                dump(output/"training.json",progress)
            records = terminal_samples(batch)
            rows = [r for r,_,_ in batch]
            if full_core and config.get("public_replay", False):
                activity("saving_replay", decisions=len(records))
                from drmc_rl.models.policy.controller_core import write_public_replay
                # Preserve complete natural experience even if the subsequent
                # optimizer audit fails. Resume atomically replaces a repeated
                # uncommitted update shard together with its new behavior data.
                write_public_replay(output / "public-replay" / f"update-{update:05d}.npz",
                                    records, rows, update=update, pace=pace, level=level)
            progress.update(phase="optimizing", collecting_decisions=len(records),
                            collecting_frames=sum(row["frames"] for row, _, _ in batch),
                            updated_at=datetime.now(UTC).isoformat())
            dump(output/"training.json", progress)
            optimizing = time.perf_counter()
            losses = update_adapter(actor,optimizer,records,config,config["seed"]+update,activity=activity)
            breakdown["optimizer_seconds"] = time.perf_counter()-optimizing
            stats = progress["paces"].setdefault(pace,{})
            activity("saving_checkpoint", update=update)
            journaling = time.perf_counter()
            with (output/"training-games.jsonl").open("a") as stream:
                for row in rows:
                    add_game_totals(stats,row)
                    add_game_totals(
                        progress.setdefault("opponents", {}).setdefault(opponent_id, {}), row
                    )
                    stream.write(
                        json.dumps(
                            {
                                **row,
                                "update": update,
                                "pace": pace,
                                "level": level,
                                "opponent": opponent_id,
                            }
                        )
                        + "\n"
                    )
            breakdown["journal_seconds"] = time.perf_counter()-journaling
            progress.update(updates=update, games=progress["games"]+len(rows),
                frames=progress["frames"]+sum(r["frames"] for r in rows),
                decisions=progress["decisions"]+len(records), losses=losses,
                batch_seconds=elapsed, wall_seconds=time.perf_counter()-started,
                max_parameter_change=max((v-initial[k]).abs().max().item() for k,v in module.state_dict().items()),
                updated_at=datetime.now(UTC).isoformat())
            frames = sum(r["frames"] for r in rows)
            update_seconds = time.perf_counter()-update_started
            progress["throughput"] = dict(backend=config.get("rollout_backend","frames"),
                async_planning=config.get("async_planning",False), strict_fp32=config.get("strict_fp32",False),
                frames_per_second=frames/update_seconds,
                rollout_frames_per_second=frames/elapsed,
                learning_decisions_per_second=len(records)/update_seconds,
                breakdown=dict(breakdown))
            checkpoint = output/f"{checkpoint_prefix}-u{update:03d}.pt"
            progress["checkpoints"].append(checkpoint.name)
            actor.save(checkpoint,update=update,optimizer=optimizer.state_dict(),sampling_rng=actor.rng.get_state(),progress=progress,training_config=config)
            keep = config.get("checkpoint_keep_last")
            if keep is not None:
                if type(keep) is not int or keep < 1:
                    raise ValueError("checkpoint_keep_last must be a positive integer")
                # Only superseded update checkpoints from this fresh run are
                # removed; milestone and final artifacts are retained.
                for old_name in progress["checkpoints"][:-keep]:
                    (output / old_name).unlink(missing_ok=True)
                progress["checkpoints"] = progress["checkpoints"][-keep:]
            for milestone in config.get("milestone_frames",[]):
                path = output/f"{checkpoint_prefix}-f{milestone:09d}.pt"
                if progress["frames"] >= milestone and not path.exists():
                    actor.save(path,update=update,progress=progress,training_config=config)
            progress.update(phase="between_updates", activity=None,
                            updated_at=datetime.now(UTC).isoformat())
            dump(output/"training.json",progress)
            print(json.dumps({k:progress[k] for k in ("updates","games","frames","decisions","current_pace","batch_seconds","throughput","losses")}),flush=True)
            del records, batch
        if not training_target_met(progress,config):
            raise RuntimeError("update safety limit reached before the frame and per-pace learning targets")
        progress.update(status="Training complete",final_checkpoint=f"{checkpoint_prefix}-final.pt",updated_at=datetime.now(UTC).isoformat())
        actor.save(output/f"{checkpoint_prefix}-final.pt",update=progress["updates"],optimizer=optimizer.state_dict(),
            sampling_rng=actor.rng.get_state(),progress=progress,training_config=config)
    except BaseException as error:
        progress.update(status="Failed",error=str(error),traceback=traceback.format_exc(),
                        updated_at=datetime.now(UTC).isoformat())
        raise
    finally:
        dump(output/"training.json",progress)
        planner.close()


if __name__ == "__main__":
    main()
