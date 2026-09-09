"""Run paired objective ablations through registered training/evaluation recipes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys

from drmc_rl.arena.experiment import dump
from drmc_rl.training.episodic_objective import objective_contract


def validate_arms(paths):
    if len(paths) != 2:
        raise ValueError("objective comparison requires exactly two training arms")
    configs = [json.loads(Path(path).read_text()) for path in paths]
    # Only the actor reduction and artifact destinations differ. In particular,
    # both arms share initialization, seed exclusions, normalization and KL cap.
    ignored = {"objective", "output", "working_db"}
    contracts = [{k: v for k, v in config.items() if k not in ignored} for config in configs]
    if contracts[0] != contracts[1] or any(config.get("resume") for config in configs):
        raise ValueError("objective arms must have identical fresh-run training contracts")
    objectives = [objective_contract(config) for config in configs]
    if {o["actor"] for o in objectives} != {"episode_mean", "decision_mean"}:
        raise ValueError("compare the historical and episodic actor reductions")
    if {k: v for k, v in objectives[0].items() if k != "actor"} != {
        k: v for k, v in objectives[1].items() if k != "actor"
    }:
        raise ValueError("only the actor reduction may differ")
    if len({c["output"] for c in configs}) != 2:
        raise ValueError("objective arms require separate outputs")
    return configs


def run_study(config):
    configs = validate_arms(config["training_configs"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    logs = Path(config.get("log_directory", output))
    logs.mkdir(parents=True, exist_ok=True)
    progress = dict(
        status="Training",
        completed_arms=0,
        total_arms=2,
        training_outputs=[c["output"] for c in configs],
        evaluations_complete=0,
    )

    def publish():
        dump(output / "pipeline.json", progress | {"updated_at": datetime.now(UTC).isoformat()})

    def launch(recipe, binding, path, name):
        with (logs / (name + ".log")).open("a") as stream:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.program",
                    "launch",
                    recipe,
                    "--set",
                    f"{binding}={path}",
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    publish()
    try:
        for index, path in enumerate(config["training_configs"]):
            progress["current_actor_reduction"] = objective_contract(configs[index])["actor"]
            publish()
            launch("trainer-pace-strategy", "trainer_pace_config", path, f"training-{index}")
            progress["completed_arms"] += 1
            publish()
        progress["status"] = "Held-out evaluation"
        publish()
        with ThreadPoolExecutor(max_workers=config.get("evaluation_workers", 3)) as workers:
            pending = [
                workers.submit(
                    launch,
                    "trainer-planning-arena",
                    "trainer_arena_config",
                    path,
                    f"evaluation-{index}",
                )
                for index, path in enumerate(config.get("evaluation_configs", []))
            ]
            for future in as_completed(pending):
                future.result()
                progress["evaluations_complete"] += 1
                publish()
        progress["status"] = "Complete"
    except BaseException as error:
        progress.update(status="Failed", error=str(error))
        raise
    finally:
        publish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run_study(json.loads(parser.parse_args().config.read_text()))
