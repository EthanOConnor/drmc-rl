"""Summarize exposure and realized updates without treating training as an arena.

The input lists arms with a label, training directory and stdout log. Repeated
training seeds are counted explicitly; side-swapped games are not described as
independent observations. This remains usable while a study is running.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from drmc_rl.arena.experiment import dump, outcome_summary


def distribution(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return None
    if not np.isfinite(array).all():
        raise ValueError("non-finite study diagnostic")
    return dict(
        count=len(array),
        mean=float(array.mean()),
        median=float(np.median(array)),
        p10=float(np.quantile(array, 0.1)),
        p90=float(np.quantile(array, 0.9)),
        minimum=float(array.min()),
        maximum=float(array.max()),
    )


def summarize_arm(rows, updates, progress):
    conditions = defaultdict(list)
    for row in rows:
        if row["update"] <= progress["updates"]:
            conditions[(row["level"], row["pace"])].append(row)
    committed = {r["updates"]: r for r in updates if r["updates"] <= progress["updates"]}
    diagnostics = defaultdict(list)
    for update in sorted(committed.values(), key=lambda r: r["updates"]):
        diagnostics[update["current_pace"]].append(update["losses"])
    reports = []
    for (level, pace), games in sorted(conditions.items()):
        natural = [r for r in games if r["reason"] != "timeout" and r.get("score") is not None]
        seeds = {r["seed"] for r in games}
        side_seeds = {(r["seed"], r["side"]) for r in games}
        pairs = defaultdict(set)
        for row in games:
            pairs[(row["update"], row["seed"])].add(row["side"])
        reports.append(
            dict(
                level=level,
                pace=pace,
                **outcome_summary(games, include_interval=False),
                score=float(np.mean([r["score"] for r in games]))
                if len(natural) == len(games)
                else None,
                collected_games=len(games),
                natural_games=len(natural),
                distinct_seeds=len(seeds),
                distinct_side_seed_games=len(side_seeds),
                repeated_side_seed_experiences=len(games) - len(side_seeds),
                complete_pairs_collected=sum(sides == {0, 1} for sides in pairs.values()),
                simulated_frames=sum(r["frames"] for r in games),
                natural_game_frames=distribution([r["frames"] for r in natural]),
                natural_learning_decisions=distribution(
                    [
                        r["a_stats"]["decisions"] - r["a_stats"].get("no_reachable_after_delay", 0)
                        for r in natural
                    ]
                ),
            )
        )
    measured = {}
    for pace, losses in diagnostics.items():
        measured[pace] = dict(
            updates=len(losses),
            first_update=losses[0],
            last_update=losses[-1],
            update_kl=distribution([r["update_kl"] for r in losses]),
            first_minibatch_kl=distribution([r["first_step_kl"] for r in losses]),
            critic_decision_mse=distribution([r["value_mse"] for r in losses]),
            kl_backtracks=sum(r["kl_backtracks"] for r in losses),
        )
    return dict(
        status=progress["status"],
        frames=progress["frames"],
        updates=progress["updates"],
        learning_decisions=progress["decisions"],
        conditions=reports,
        update_diagnostics=measured,
        reported_training_updates=len(committed),
        training_outcomes_are_promotion_evidence=False,
    )


def report(config):
    arms = []
    for arm in config["arms"]:
        directory = Path(arm["directory"])
        progress = json.loads((directory / "training.json").read_text())
        # Only newline-terminated journal records belong to a closed write.
        journal = (directory / "training-games.jsonl").read_text().splitlines(keepends=True)
        rows = [json.loads(line) for line in journal if line.endswith("\n")]
        updates = []
        for line in Path(arm["log"]).read_text().splitlines():
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "updates" in row and "losses" in row:
                updates.append(row)
        arms.append(dict(label=arm["label"], **summarize_arm(rows, updates, progress)))
    return dict(
        schema="drmc-objective-study-diagnostics-v1",
        arms=arms,
        notes=[
            "Realized update KL must be compared, not just the configured cap.",
            "Training policies change over time; these are descriptive diagnostics, not confidence intervals.",
            "Distinct NES seeds and repeated side-swapped experiences are separate counts.",
            "Per-pace update diagnostics can include both levels; game outcomes keep levels separate.",
            "Historical log field independent_games counted completed learning games, not independent seed clusters.",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dump(args.output, report(json.loads(args.config.read_text())))
