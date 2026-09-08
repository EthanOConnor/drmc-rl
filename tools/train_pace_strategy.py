"""Bounded full-game PPO for a motor-conditioned residual on the public core.

Uses exact controller execution, optionally batched at causal decisions.
Frozen features make the small adapter cheap to update. Every transition receives its natural terminal
W/D/L return (gamma=1); games receive equal weight regardless of duration.
No shaped reward, search label, hidden opponent field or time-limit draw trains
the policy. Run through the trainer-pace-strategy program recipe.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from drmc_rl.arena.experiment import dump
from drmc_rl.arena.store import ArenaStore
from drmc_rl.execution.pace import BY_ID
from drmc_rl.models.policy.pace_adapter import PacePolicy
from drmc_rl.planning.native_reach import NativeReachabilityRunner
from tools.trainer_arena_cache import MemoPlanner
from tools.trainer_planning_arena import run_batch
from tools.vs_head_to_head import PlainPolicy


def terminal_samples(batch):
    """Exclude incomplete games and preserve whole-game weighting."""
    samples = []
    for row, moves, _ in batch:
        if row["reason"] == "timeout":
            continue
        decisions = [m["learning"] for m in moves if "learning" in m]
        for sample in decisions:
            samples.append({**sample, "return": 2*row["score"]-1,
                            "weight": 1/max(1,len(decisions))})
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
    extras = {key:torch.as_tensor([r[key] for r in records],device=device,
        dtype=torch.long if key=="slot" else torch.float32)
        for key in ("slot","old_logprob","old_value","return","weight","advantage")}
    return features, extras


def update_adapter(actor, optimizer, records, config, seed):
    if not records:
        raise RuntimeError("no natural-terminal learner decisions; cannot train")
    weights = np.asarray([r["weight"] for r in records])
    advantages = np.asarray([r["return"]-r["old_value"] for r in records])
    mean = np.average(advantages,weights=weights)
    scale = np.sqrt(np.average((advantages-mean)**2,weights=weights)+1e-8)
    for record, advantage in zip(records,(advantages-mean)/scale):
        record["advantage"] = float(advantage)
    weight_scale = len(records)/weights.sum()
    rng, totals = np.random.default_rng(seed), defaultdict(list)
    for _ in range(config.get("epochs",2)):
        indices = rng.permutation(len(records))
        size = config.get("minibatch",128)
        for start in range(0,len(indices),size):
            rows = [records[i] for i in indices[start:start+size]]
            features, data = tensor_batch(rows,actor.device)
            logits, values = actor.adapter(*features)
            log_probs = logits.log_softmax(-1)
            chosen = log_probs.gather(1,data["slot"][:,None]).squeeze(1)
            log_ratio = chosen-data["old_logprob"]
            ratio = log_ratio.exp()
            clip = config.get("clip",.15)
            weights = data["weight"]*weight_scale
            policy_loss = -(weights*torch.minimum(ratio*data["advantage"],
                ratio.clamp(1-clip,1+clip)*data["advantage"])).mean()
            value_loss = (weights*F.smooth_l1_loss(values,data["return"],reduction="none")).mean()
            probabilities = log_probs.exp()
            entropy = -(weights*(probabilities*log_probs).sum(-1)).mean()
            base_log = features[3].log_softmax(-1)
            base_kl = (weights*(base_log.exp()*(base_log-log_probs)).sum(-1)).mean()
            loss = policy_loss + .5*value_loss - config.get("entropy",.003)*entropy + config.get("parent_kl",.02)*base_kl
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite pace training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(actor.adapter.parameters(), .7)
            if not torch.isfinite(norm):
                raise RuntimeError("non-finite pace adapter gradient")
            optimizer.step()
            approx_kl = (weights*((ratio-1)-log_ratio)).mean().item()
            for key,value in dict(policy_loss=policy_loss.item(),value_loss=value_loss.item(),
                entropy=entropy.item(),parent_kl=base_kl.item(),approx_kl=approx_kl,gradient_norm=norm.item()).items():
                totals[key].append(value)
            if approx_kl > config.get("max_update_kl",.06):
                return {k:float(np.mean(v)) for k,v in totals.items()} | {"early_kl_stop":True}
    return {k:float(np.mean(v)) for k,v in totals.items()} | {"early_kl_stop":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config["output"])
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(config.get("threads",1))
    torch.set_num_interop_threads(1)
    if config.get("strict_fp32",False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config["seed"])
    actor = PacePolicy(config["checkpoint"],config["device"],training=True,
                       adapter_path=config.get("resume"),seed=config["seed"])
    parent = PlainPolicy(Path(config["checkpoint"]),config["device"],public_only=True)
    rollout = run_batch
    if config.get("rollout_backend", "frames") == "events":
        from tools.trainer_event_rollout import ParallelPlanning, run_event_batch
        planner = ParallelPlanning(config.get("planner_workers",4))
        rollout = run_event_batch
        config["mixed_core_actor"] = "learner"
    else:
        if config.get("rollout_backend", "frames") != "frames":
            raise ValueError("rollout_backend must be frames or events")
        planner = MemoPlanner(NativeReachabilityRunner())
    optimizer = torch.optim.AdamW(actor.adapter.parameters(),lr=config.get("lr",2e-4),weight_decay=.001)
    initial = {k:v.detach().clone() for k,v in actor.adapter.state_dict().items()}
    start_update = 0
    progress = {"status":"Running", "updates":0, "target_updates":config["updates"],
                "games":0, "frames":0, "decisions":0, "trainable_parameters":sum(p.numel() for p in actor.adapter.parameters()),
                "parent_sha256":actor.parent_sha256,"paces":{},"checkpoints":[]}
    if config.get("resume"):
        previous = torch.load(config["resume"],map_location=config["device"],weights_only=True)
        if "training_config" in previous:
            for key in ("checkpoint", "seed", "holdout_seeds", "paces"):
                if previous["training_config"][key] != config[key]:
                    raise ValueError(f"resume changed the training/evaluation contract: {key}")
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
    progress.update(target_frames=config.get("target_frames"),
        minimum_decisions_per_pace=config.get("minimum_decisions_per_pace",0),
        games_per_pace=config.get("games_per_pace",{}))
    available = np.setdiff1d(np.arange(1,65536),config["holdout_seeds"])
    config["variants"] = {"learner":{"delay":4},"parent":{"delay":4}}
    config["replay_games"] = 0
    store = ArenaStore(config["working_db"],replay_dir=output/"replays")
    store.conn.commit()
    import sqlite3
    with sqlite3.connect(output/"arena.sqlite") as snapshot:
        store.conn.backup(snapshot)
    store.close()
    dump(output/"results.json",{"updated_at":datetime.now(UTC).isoformat(),"tournaments":[]})
    started = time.perf_counter()
    try:
        for update in range(start_update+1,config["updates"]+1):
            if training_target_met(progress,config):
                break
            pace = config["paces"][(update-1)%len(config["paces"])]
            if pace not in BY_ID or pace in ("super_human","frame_perfect"):
                raise ValueError("this isolated pilot trains Sloth through Top Humans only")
            rng = np.random.default_rng(config["seed"]+update)
            level = 20 if pace != "sloth" and rng.random()<config.get("level20_fraction",.15) else 14
            count = config.get("games_per_pace",{}).get(pace,config.get("games_per_update",16))
            if count < 2 or count%2:
                raise ValueError("training batches require complete paired seeds")
            seeds = rng.choice(available,count//2,replace=False)
            jobs = [(int(seed),side,2*i+side) for i,seed in enumerate(seeds) for side in (0,1)]
            match = {"id":f"train-{update}","a":"learner","b":"parent","games":count,"pace":pace,"level":level}
            progress.update(current_pace=pace,current_level=level,collecting_update=update,
                collecting_games=0,collecting_target=count,updated_at=datetime.now(UTC).isoformat())
            dump(output/"training.json",progress)
            update_started = time.perf_counter()
            batch, elapsed, breakdown = [], 0.0, defaultdict(float)
            chunk_size = config.get("rollout_games",count)
            if chunk_size < 2 or chunk_size%2:
                raise ValueError("rollout chunks require complete paired seeds")
            for start in range(0,len(jobs),chunk_size):
                metrics = {}
                part, seconds = rollout(config,match,jobs[start:start+chunk_size],None,planner,None,
                    policies={"learner":actor,"parent":parent},
                    **({"metrics":metrics} if rollout is not run_batch else {}))
                batch.extend(part)
                elapsed += seconds
                for key,value in metrics.items():
                    breakdown[key] += value
                progress.update(collecting_games=len(batch),collecting_target=count,
                    updated_at=datetime.now(UTC).isoformat())
                dump(output/"training.json",progress)
            records = terminal_samples(batch)
            optimizing = time.perf_counter()
            losses = update_adapter(actor,optimizer,records,config,config["seed"]+update)
            breakdown["optimizer_seconds"] = time.perf_counter()-optimizing
            rows = [r for r,_,_ in batch]
            stats = progress["paces"].setdefault(pace,{})
            journaling = time.perf_counter()
            with (output/"training-games.jsonl").open("a") as stream:
                for row in rows:
                    add_game_totals(stats,row)
                    stream.write(json.dumps({**row,"update":update,"pace":pace,"level":level})+"\n")
            breakdown["journal_seconds"] = time.perf_counter()-journaling
            progress.update(updates=update, games=progress["games"]+len(rows),
                frames=progress["frames"]+sum(r["frames"] for r in rows),
                decisions=progress["decisions"]+len(records), losses=losses,
                batch_seconds=elapsed, wall_seconds=time.perf_counter()-started,
                max_parameter_change=max((v-initial[k]).abs().max().item() for k,v in actor.adapter.state_dict().items()),
                updated_at=datetime.now(UTC).isoformat())
            frames = sum(r["frames"] for r in rows)
            update_seconds = time.perf_counter()-update_started
            progress["throughput"] = dict(backend=config.get("rollout_backend","frames"),
                async_planning=config.get("async_planning",False), strict_fp32=config.get("strict_fp32",False),
                frames_per_second=frames/update_seconds,
                rollout_frames_per_second=frames/elapsed,
                learning_decisions_per_second=len(records)/update_seconds,
                breakdown=dict(breakdown))
            checkpoint = output/f"adapter-u{update:03d}.pt"
            progress["checkpoints"].append(checkpoint.name)
            actor.save(checkpoint,update=update,optimizer=optimizer.state_dict(),sampling_rng=actor.rng.get_state(),progress=progress,training_config=config)
            for milestone in config.get("milestone_frames",[]):
                path = output/f"adapter-f{milestone:09d}.pt"
                if progress["frames"] >= milestone and not path.exists():
                    actor.save(path,update=update,progress=progress,training_config=config)
            dump(output/"training.json",progress)
            print(json.dumps({k:progress[k] for k in ("updates","games","frames","decisions","current_pace","batch_seconds","throughput","losses")}),flush=True)
            del records, batch
        if not training_target_met(progress,config):
            raise RuntimeError("update safety limit reached before the frame and per-pace learning targets")
        progress.update(status="Training complete",final_checkpoint="adapter-final.pt",updated_at=datetime.now(UTC).isoformat())
        actor.save(output/"adapter-final.pt",update=progress["updates"],optimizer=optimizer.state_dict(),
            sampling_rng=actor.rng.get_state(),progress=progress,training_config=config)
    except BaseException as error:
        progress.update(status="Failed",error=str(error),updated_at=datetime.now(UTC).isoformat())
        raise
    finally:
        dump(output/"training.json",progress)
        planner.close()


if __name__ == "__main__":
    main()
