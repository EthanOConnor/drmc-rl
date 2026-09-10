"""Run one quality fit inside a monitored, exclusive CUDA wall-time allowance.

The child owns the GPU. An independent deadline terminates it even during an
optimizer step or a stalled progress write. Only a complete, fully checked
checkpoint published before the cutoff can be exported. Actual elapsed cost,
unused time, cutoff overrun and checkpoint age are reported separately.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from drmc_rl.arena.experiment import dump
from drmc_rl.training.quality_checkpoints import eligible_checkpoint


def cuda_pids(gpu_uuid):
    completed = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=2,
    )
    pids = set()
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 2 or not fields[1].isdigit():
            raise RuntimeError("cannot verify CUDA process ownership")
        if int(fields[1]) <= 0:
            raise RuntimeError("invalid CUDA process identity")
        if fields[0] == gpu_uuid:
            pids.add(int(fields[1]))
    return pids


def child_command(path):
    return [sys.executable, "-m", "tools.fit_paired_quality", "--config", str(path)]


def run(config):
    seconds = float(config["allocation_seconds"])
    interval = float(config.get("checkpoint_interval_seconds", 30))
    gpu_uuid = config["gpu_uuid"]
    if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(interval) or interval < 0:
        raise ValueError("invalid quality allocation or checkpoint interval")
    if (
        not isinstance(gpu_uuid, str)
        or re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", gpu_uuid)
        is None
    ):
        raise ValueError("a physical NVIDIA GPU UUID is required")
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("a budgeted quality fit requires a fresh identity")
    dump(output / "config.json", config)
    progress = dict(
        schema="drmc-budgeted-quality-fit-v1",
        status="Running",
        phase="checking_gpu",
        allocation_seconds=seconds,
        gpu_uuid=gpu_uuid,
        observed_child_cuda_context=False,
        calibrated=False,
        product_gates_passed=False,
        ownership_monitor="NVIDIA compute-process query before launch and approximately once per second",
        checkpoint_selection="latest fully checked snapshot stored before cutoff",
    )

    def report(**updates):
        progress.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
        dump(output / "progress.json", progress)

    report()
    try:
        occupied = cuda_pids(gpu_uuid)
        if occupied:
            raise RuntimeError(f"GPU already has compute processes: {sorted(occupied)}")
        # Fast local snapshots avoid synchronous network checkpoint writes in
        # the GPU allowance. The selected artifact is copied after child exit.
        with tempfile.TemporaryDirectory(
            prefix="quality-fit-", dir=config["scratch_root"], delete=False
        ) as scratch:
            checkpoint_dir = Path(scratch) / "checkpoints"
            report(checkpoint_scratch=str(checkpoint_dir))
            fit_config = deepcopy(config["fit_config"])
            fit_config.update(
                output=str(output / "fit"),
                device="cuda:0",
                checkpoint_directory=str(checkpoint_dir),
                checkpoint_interval_seconds=interval,
                checkpoint_only=True,
            )
            fit_path = output / "fit-config.json"
            dump(fit_path, fit_config)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid)
            expired = threading.Event()
            stopped = threading.Event()
            process = None
            watchdog = None
            ownership_error = None
            started_ns = time.monotonic_ns()
            deadline_ns = started_ns + int(seconds * 1e9)
            with (output / "fit.log").open("w") as log:
                try:
                    process = subprocess.Popen(
                        child_command(fit_path), env=env, stdout=log, stderr=subprocess.STDOUT
                    )

                    def enforce_deadline():
                        if stopped.wait(max(0, (deadline_ns - time.monotonic_ns()) / 1e9)):
                            return
                        if process.poll() is not None:
                            return
                        expired.set()
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=2)

                    watchdog = threading.Thread(target=enforce_deadline, daemon=True)
                    watchdog.start()
                    report(
                        phase="fitting",
                        child_pid=process.pid,
                        allocation_started_monotonic_ns=started_ns,
                    )
                    checked_at, reported_at = 0.0, 0.0
                    while process.poll() is None:
                        now = time.monotonic()
                        if now - checked_at >= 1:
                            try:
                                occupied = cuda_pids(gpu_uuid)
                            except Exception as error:
                                ownership_error = f"CUDA ownership monitoring failed: {error}"
                                break
                            progress["observed_child_cuda_context"] |= process.pid in occupied
                            foreign = occupied - {process.pid}
                            if foreign:
                                ownership_error = (
                                    f"GPU allocation was shared with processes {sorted(foreign)}"
                                )
                                break
                            checked_at = now
                        if now - reported_at >= 5:
                            report(
                                allocation_elapsed_seconds=(time.monotonic_ns() - started_ns) / 1e9
                            )
                            reported_at = now
                        stopped.wait(0.1)
                finally:
                    stopped.set()
                    if process is not None and process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=2)
                    if watchdog is not None:
                        watchdog.join(timeout=5)
            ended_ns = time.monotonic_ns()
            elapsed = (ended_ns - started_ns) / 1e9
            report(
                phase="exporting",
                child_exit_code=process.returncode,
                stop_reason="allocation_exhausted" if expired.is_set() else "fit_exited",
                allocation_elapsed_seconds=elapsed,
                unused_allocation_seconds=max(0, seconds - elapsed),
                cutoff_overrun_seconds=max(0, elapsed - seconds),
            )
            fit_progress_path = output / "fit" / "progress.json"
            if fit_progress_path.exists():
                observed = json.loads(fit_progress_path.read_text())
                report(
                    last_reported_fit_work={
                        key: observed.get(key)
                        for key in (
                            "updated_at",
                            "phase",
                            "accepted_examples",
                            "accepted_optimizer_steps",
                            "examples_processed",
                            "optimizer_steps",
                        )
                    }
                )
                if process.returncode != 0:
                    observed.update(
                        status="Stopped" if expired.is_set() and not ownership_error else "Failed",
                        stop_reason=progress["stop_reason"],
                        controlled_by="budgeted-quality-fit",
                    )
                    dump(fit_progress_path, observed)
            selected = eligible_checkpoint(checkpoint_dir, deadline_ns=deadline_ns)
            if selected is not None:
                source = checkpoint_dir / selected["filename"]
                temporary = output / "diagnostic.pt.tmp"
                shutil.copyfile(source, temporary)
                temporary.replace(output / "diagnostic.pt")
                with (output / "diagnostic.pt").open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                report(
                    selected_checkpoint=selected,
                    checkpoint_sha256=digest,
                    checkpoint_age_at_stop_seconds=max(
                        0, (ended_ns - selected["validated_monotonic_ns"]) / 1e9
                    ),
                )
            else:
                raise RuntimeError("allocation ended before a fully checked checkpoint was stored")
            if ownership_error:
                raise RuntimeError(ownership_error)
            if process.returncode != 0 and not expired.is_set():
                raise RuntimeError(f"quality fitter failed with exit code {process.returncode}")
            if not progress["observed_child_cuda_context"]:
                raise RuntimeError(
                    "no child CUDA context was observed; GPU comparison is unverified"
                )
            report(status="Complete", phase="complete", eligible_for_allocation_comparison=True)
            try:
                shutil.rmtree(scratch)
            except OSError as error:
                report(scratch_cleanup_error=str(error))
    except BaseException as error:
        report(
            status="Failed",
            phase="failed",
            error=str(error),
            eligible_for_allocation_comparison=False,
        )
        raise
    return progress


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    run(json.loads(parser.parse_args().config.read_text()))
