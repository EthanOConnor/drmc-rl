"""Run a bounded pace-training continuation, then its fixed held-out schedule."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
import traceback

from drmc_rl.arena.experiment import dump


def run_study(config):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    # SSHFS reconnects invalidate open handles even when new file opens work.
    # Keep child stdout/stderr on a local filesystem when artifacts overflow.
    log_directory = Path(config.get("log_directory", output))
    log_directory.mkdir(parents=True, exist_ok=True)
    progress = {"status":"Training", "evaluations_complete":0,
                "evaluations_total":len(config["evaluation_configs"]),
                "log_directory":str(log_directory)}

    def publish():
        dump(output/"pipeline.json", {**progress,"updated_at":datetime.now(UTC).isoformat()})

    def launch(recipe, binding, path, log):
        with log.open("a") as stream:
            stream.write(f"\n{datetime.now(UTC).isoformat()} · starting {recipe}\n")
            stream.flush()
            return subprocess.run([sys.executable,"-m","tools.program","launch",recipe,
                "--set",f"{binding}={path}"],stdout=stream,stderr=subprocess.STDOUT).returncode

    publish()
    try:
        training_log = log_directory/"study-training.log"
        if launch("trainer-pace-strategy", "trainer_pace_config", config["training_config"], training_log):
            raise RuntimeError(f"training failed; see {training_log}")
        progress["status"] = "Held-out evaluation"
        publish()
        failures = []
        with ThreadPoolExecutor(max_workers=config.get("evaluation_workers",3)) as workers:
            pending = {workers.submit(launch,"trainer-planning-arena","trainer_arena_config",path,
                        log_directory/f"study-evaluation-{i}.log"):i for i,path in enumerate(config["evaluation_configs"])}
            for future in as_completed(pending):
                if future.result():
                    failures.append(pending[future])
                progress["evaluations_complete"] += 1
                progress["failed_evaluations"] = failures
                publish()
        if failures:
            raise RuntimeError(f"evaluation workers failed: {failures}; see {log_directory}/study-evaluation logs")
        progress["status"] = "Complete · results ready for review"
    except BaseException as error:
        progress.update(status="Failed",error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        publish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run_study(json.loads(parser.parse_args().config.read_text()))


if __name__ == "__main__":
    main()
