"""Run a bounded pace-training continuation, then its fixed held-out schedule."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys

from drmc_rl.arena.experiment import dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    config = json.loads(parser.parse_args().config.read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    progress = {"status":"Training", "evaluations_complete":0,
                "evaluations_total":len(config["evaluation_configs"])}

    def publish():
        dump(output/"pipeline.json", {**progress,"updated_at":datetime.now(UTC).isoformat()})

    def launch(recipe, binding, path, log):
        with log.open("w") as stream:
            return subprocess.run([sys.executable,"-m","tools.program","launch",recipe,
                "--set",f"{binding}={path}"],stdout=stream,stderr=subprocess.STDOUT).returncode

    publish()
    try:
        if launch("trainer-pace-strategy", "trainer_pace_config", config["training_config"], output/"study-training.log"):
            raise RuntimeError("training failed; see study-training.log")
        progress["status"] = "Held-out evaluation"
        publish()
        failures = []
        with ThreadPoolExecutor(max_workers=config.get("evaluation_workers",3)) as workers:
            pending = {workers.submit(launch,"trainer-planning-arena","trainer_arena_config",path,
                        output/f"study-evaluation-{i}.log"):i for i,path in enumerate(config["evaluation_configs"])}
            for future in as_completed(pending):
                if future.result():
                    failures.append(pending[future])
                progress["evaluations_complete"] += 1
                progress["failed_evaluations"] = failures
                publish()
        if failures:
            raise RuntimeError(f"evaluation workers failed: {failures}; see study-evaluation logs")
        progress["status"] = "Complete · results ready for review"
    except BaseException as error:
        progress.update(status="Failed",error=str(error))
        raise
    finally:
        publish()


if __name__ == "__main__":
    main()
