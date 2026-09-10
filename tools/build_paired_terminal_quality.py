"""Generate complete terminal panels with bounded batching across source roots.

Each completed root is committed atomically with its full rollout inventory.
Resume reuses complete roots under the identical source/model/panel contract;
unfinished roots are recomputed, and censored outcomes remain unknown.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from drmc_rl.arena.experiment import dump
from drmc_rl.envs.backends.drmario_pool import resolve_library_path
from drmc_rl.search.native_pair import state_from_payload
from drmc_rl.search.pill_belief import CHANCE_MODEL_ID, PillReserveBelief
from drmc_rl.search.public_policy import PublicPolicyContinuation
from drmc_rl.teachers.counterfactual_release import sha256_file
from drmc_rl.teachers.paired_terminal import ContinuationPair, aggregate_panel, build_panel
from drmc_rl.teachers.terminal_rollout import reserve_hypotheses, rollout_tasks
from drmc_rl.teachers.v3_baseline import load_source_rows
from tools.audit_rollout_consistency import select_rows


def root_path(output, source_id):
    return output / "roots" / (hashlib.sha256(source_id.encode()).hexdigest() + ".json")


def export_completed(output, rows, completed):
    target = output / "targets.jsonl.next"
    raw = output / "rollouts.jsonl.next"
    with target.open("w") as targets, raw.open("w") as rollouts:
        for row in rows:
            if row["id"] not in completed:
                continue
            item = json.loads(root_path(output, row["id"]).read_text())
            targets.write(json.dumps(item["target"]) + "\n")
            rollouts.write(json.dumps(item["rollouts"]) + "\n")
    target.replace(output / "targets.jsonl")
    raw.replace(output / "rollouts.jsonl")


def run(config):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    configuration = output / "contract.json"
    if not configuration.exists() and (output / "targets.jsonl").exists():
        raise FileExistsError("legacy terminal output cannot resume as a batched-root job")
    torch.set_num_threads(config.get("threads", 2))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    identities = {id: sha256_file(Path(path)) for id, path in config["members"].items()}
    if len(set(identities.values())) != len(identities):
        raise ValueError("duplicate frozen checkpoints cannot supply continuation diversity")
    panel = [ContinuationPair(**item) for item in config["continuations"]]
    if any(c.actor not in identities or c.opponent not in identities for c in panel):
        raise ValueError("continuation pair names an unregistered public member")
    rows = load_source_rows(Path(config["state_bank"]))
    for key, field in (("levels", "level"), ("speeds", "speed")):
        if config.get(key):
            rows = [r for r in rows if r[field] in config[key]]
    rows = select_rows(
        rows, config.get("states", 8), config["seed"],
        stratum_fields=tuple(config.get("stratum_fields", ("level", "speed"))),
        allowed_policies=("frozen-strong-league-mixture-argmax", "frozen-public-core-argmax"),
    )
    root_batch_size = int(config.get("root_batch_size", 1))
    if root_batch_size < 1:
        raise ValueError("root batch size must be positive")
    contract = dict(schema="drmc-paired-terminal-job-v2", config=config,
                    source_sha256=sha256_file(Path(config["state_bank"])),
                    member_sha256=identities, native_sha256=sha256_file(resolve_library_path()),
                    source_ids=[r["id"] for r in rows])
    if configuration.exists() and json.loads(configuration.read_text()) != contract:
        raise ValueError("resume changed terminal source, panel, model or execution configuration")
    dump(configuration, contract)
    (output / "roots").mkdir(exist_ok=True)
    completed, completed_rollouts, censored_rollouts = set(), 0, 0
    for row in rows:
        path = root_path(output, row["id"])
        if path.exists():
            saved = json.loads(path.read_text())
            target, raw = saved["target"], saved["rollouts"]
            if (target["source_id"] != row["id"] or raw["source_id"] != row["id"]
                    or target["member_sha256"] != identities):
                raise ValueError("committed root identity does not match the frozen job")
            # Check durable coverage rather than trusting a previous status flag.
            rebuilt = aggregate_panel(target["actions"], target["incumbent"], raw["inventory"],
                                      raw["results"], target["reference_prior"],
                                      kl_budget=config.get("target_kl", .02),
                                      sensitivity_penalty=config.get("sensitivity_penalty", 1.))
            if any(rebuilt[k] != target[k] for k in rebuilt):
                raise ValueError("committed terminal root target differs from its full inventory")
            completed.add(row["id"])
            completed_rollouts += len(raw["results"])
            censored_rollouts += sum(r["outcome"] is None for r in raw["results"])
    previous = json.loads((output / "progress.json").read_text()) if (output / "progress.json").exists() else {}
    elapsed_before = float(previous.get("elapsed_seconds", 0))
    progress = dict(
        schema="drmc-paired-terminal-job-v2", status="Running", states=len(completed), target_states=len(rows),
        rollouts=completed_rollouts, censored_rollouts=censored_rollouts,
        member_sha256=identities, chance_model=CHANCE_MODEL_ID, execution="native-smdp-v1",
        root_batch_size=root_batch_size, native_workers=int(config.get("native_workers", 1)),
        reserve_execution=config.get("reserve_execution", "boundary"),
        device=config.get("device", "cuda"), strict_fp32=True, product_gates_passed=False,
        source_sha256=contract["source_sha256"], batches=list(previous.get("batches", [])),
    )
    started = time.monotonic()

    def report(**values):
        progress.update(**values, elapsed_seconds=elapsed_before + time.monotonic()-started,
                        updated_at=datetime.now(UTC).isoformat())
        dump(output / "progress.json", progress)

    report(phase="loading")
    export_completed(output, rows, completed)
    if len(completed) == len(rows):
        report(status="Complete", phase="complete")
        return progress
    remaining = [row for row in rows if row["id"] not in completed]
    try:
        actors = {id: PublicPolicyContinuation(Path(path), device=config.get("device", "cuda"),
                                              cache_size=config.get("policy_cache_size", 0))
                  for id, path in config["members"].items()}
        reference = actors[config["reference"]]
        for start in range(0, len(remaining), root_batch_size):
            current = remaining[start:start+root_batch_size]
            states = [state_from_payload(row) for row in current]
            predictions = reference.infer_batch([(state, int(row["root_side"]))
                                                 for state, row in zip(states, current, strict=True)])
            roots, tasks, destinations = {}, [], {}
            for row, state, (probability, _) in zip(current, states, predictions, strict=True):
                side = int(row["root_side"])
                legal = state.legal_actions_by_side[side]
                prior = np.asarray([max(config.get("prior_floor", 1e-8), probability.get(a, 0.))
                                    for a in legal], dtype=np.float64)
                prior /= prior.sum()
                source_tasks, inventory = build_panel(
                    state, side, reserve_hypotheses(PillReserveBelief.from_dict(row["reserve_belief"])), panel)
                root = dict(row=row, side=side, legal=legal, prior=prior.tolist(),
                            incumbent=legal[int(prior.argmax())], inventory=inventory, results=[])
                roots[row["id"]] = root
                for task in source_tasks:
                    global_id = len(tasks)
                    destinations[global_id] = row["id"], task.id
                    tasks.append(replace(task, id=global_id))
            report(phase="rolling_out", current_source_ids=[r["id"] for r in current],
                   current_candidates=sum(len(r["legal"]) for r in roots.values()),
                   current_rollouts=len(tasks), current_rollouts_complete=0)

            def receive(result):
                source_id, local_id = destinations[result["id"]]
                root = roots[source_id]
                root["results"].append({**result, "id": local_id})
                if len(root["results"]) != len(root["inventory"]):
                    return
                target = aggregate_panel(root["legal"], root["incumbent"], root["inventory"],
                    root["results"], root["prior"], kl_budget=config.get("target_kl", .02),
                    sensitivity_penalty=config.get("sensitivity_penalty", 1.))
                target.update(source_id=source_id, game_id=root["row"]["game_id"], root_side=root["side"],
                    prior_floor=config.get("prior_floor", 1e-8), member_sha256=identities,
                    continuations=[dict(actor=c.actor, opponent=c.opponent, weight=c.weight,
                                        execution=c.execution) for c in panel])
                raw = dict(source_id=source_id, inventory=root["inventory"],
                           results=sorted(root["results"], key=lambda r: r["id"]))
                dump(root_path(output, source_id), dict(target=target, rollouts=raw))
                completed.add(source_id)
                report(states=len(completed), rollouts=progress["rollouts"]+len(raw["results"]),
                       censored_rollouts=progress["censored_rollouts"]+
                           sum(r["outcome"] is None for r in raw["results"]))

            measured = {}
            rollout_tasks(tasks, actors, batch_size=config.get("batch_size", 32),
                          max_events=config.get("max_events", 4096),
                          native_workers=config.get("native_workers", 1),
                          reserve_execution=config.get("reserve_execution", "boundary"),
                          progress=lambda count: report(current_rollouts_complete=count),
                          on_result=receive, metrics=measured)
            if any(row["id"] not in completed for row in current):
                raise RuntimeError("terminal scheduler returned without completing its root inventories")
            report(last_batch=measured, batches=progress["batches"]+
                   [dict(source_ids=[r["id"] for r in current], **measured)])
            export_completed(output, rows, completed)
            print(json.dumps(progress), flush=True)
        report(status="Complete", phase="complete")
    except BaseException as error:
        report(status="Failed", error=str(error))
        raise
    finally:
        export_completed(output, rows, completed)
    return progress


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
