"""Generate full-root terminal targets under a declared public policy panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from drmc_rl.arena.experiment import dump
from drmc_rl.search.native_pair import state_from_payload
from drmc_rl.search.pill_belief import CHANCE_MODEL_ID, PillReserveBelief
from drmc_rl.search.public_policy import PublicPolicyContinuation
from drmc_rl.teachers.counterfactual_release import sha256_file
from drmc_rl.teachers.paired_terminal import ContinuationPair, aggregate_panel, build_panel
from drmc_rl.teachers.terminal_rollout import reserve_hypotheses, rollout_tasks
from drmc_rl.teachers.v3_baseline import load_source_rows
from tools.audit_rollout_consistency import select_rows


def run(config):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "targets.jsonl").exists():
        raise FileExistsError("paired terminal teacher requires a fresh output identity")
    torch.set_num_threads(config.get("threads", 2))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    identities = {id: sha256_file(Path(path)) for id, path in config["members"].items()}
    if len(set(identities.values())) != len(identities):
        raise ValueError(
            "duplicate frozen checkpoints cannot supply independent continuation diversity"
        )
    panel = [ContinuationPair(**item) for item in config["continuations"]]
    if any(c.actor not in identities or c.opponent not in identities for c in panel):
        raise ValueError("continuation pair names an unregistered public member")
    actors = {
        id: PublicPolicyContinuation(Path(path), device=config.get("device", "cuda"))
        for id, path in config["members"].items()
    }
    reference = actors[config["reference"]]
    rows = load_source_rows(Path(config["state_bank"]))
    if config.get("levels"):
        rows = [r for r in rows if r["level"] in config["levels"]]
    if config.get("speeds"):
        rows = [r for r in rows if r["speed"] in config["speeds"]]
    rows = select_rows(
        rows,
        config.get("states", 8),
        config["seed"],
        stratum_fields=("level", "speed"),
        allowed_policies=("frozen-strong-league-mixture-argmax", "frozen-public-core-argmax"),
    )
    progress = dict(
        status="Running",
        states=0,
        target_states=len(rows),
        rollouts=0,
        member_sha256=identities,
        chance_model=CHANCE_MODEL_ID,
        execution="native-smdp-v1",
        native_workers=int(config.get("native_workers", 1)),
        device=config.get("device", "cuda"),
        strict_fp32=True,
        product_gates_passed=False,
        source_sha256=sha256_file(Path(config["state_bank"])),
    )
    started = time.monotonic()
    dump(output / "progress.json", progress)
    try:
        with (
            (output / "targets.jsonl").open("x") as stream,
            (output / "rollouts.jsonl").open("x") as raw,
        ):
            for row in rows:
                state = state_from_payload(row)
                side = int(row["root_side"])
                legal = state.legal_actions_by_side[side]
                probability, _ = reference.infer_batch([(state, side)])[0]
                # Explicit support floor allows equivalent old deduplicated
                # rotations to remain in the new complete-root inventory.
                prior = np.asarray(
                    [max(config.get("prior_floor", 1e-8), probability.get(a, 0.0)) for a in legal]
                )
                prior /= prior.sum()
                incumbent = legal[int(prior.argmax())]
                hypotheses = reserve_hypotheses(PillReserveBelief.from_dict(row["reserve_belief"]))
                tasks, inventory = build_panel(state, side, hypotheses, panel)
                progress.update(
                    current_source_id=row["id"],
                    current_candidates=len(legal),
                    current_rollouts=len(tasks),
                    current_rollouts_complete=0,
                )
                dump(output / "progress.json", progress)

                def rollout_progress(count):
                    progress.update(
                        current_rollouts_complete=count, elapsed_seconds=time.monotonic() - started
                    )
                    dump(output / "progress.json", progress)

                result = rollout_tasks(
                    tasks,
                    actors,
                    batch_size=config.get("batch_size", 32),
                    max_events=config.get("max_events", 4096),
                    progress=rollout_progress,
                    native_workers=config.get("native_workers", 1),
                )
                target = aggregate_panel(
                    legal,
                    incumbent,
                    inventory,
                    result,
                    prior.tolist(),
                    kl_budget=config.get("target_kl", 0.02),
                    sensitivity_penalty=config.get("sensitivity_penalty", 1.0),
                )
                target.update(
                    source_id=row["id"],
                    game_id=row["game_id"],
                    root_side=side,
                    prior_floor=config.get("prior_floor", 1e-8),
                    member_sha256=identities,
                    continuations=[
                        dict(
                            actor=c.actor,
                            opponent=c.opponent,
                            weight=c.weight,
                            execution=c.execution,
                        )
                        for c in panel
                    ],
                )
                stream.write(json.dumps(target) + "\n")
                stream.flush()
                raw.write(
                    json.dumps(dict(source_id=row["id"], inventory=inventory, results=result))
                    + "\n"
                )
                raw.flush()
                progress.update(
                    states=progress["states"] + 1,
                    rollouts=progress["rollouts"] + len(result),
                    current_rollouts_complete=len(result),
                    elapsed_seconds=time.monotonic() - started,
                )
                dump(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        progress["status"] = "Complete"
    except BaseException as error:
        progress.update(status="Failed", error=str(error))
        raise
    finally:
        dump(output / "progress.json", progress)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
