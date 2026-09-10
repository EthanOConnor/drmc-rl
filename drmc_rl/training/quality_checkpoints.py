"""Atomic, bounded snapshots of fully checked supervised quality models."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

from drmc_rl.arena.experiment import dump


class QualityCheckpoints:
    def __init__(self, directory, *, interval_seconds=30.0):
        self.directory = Path(directory)
        self.interval_seconds = float(interval_seconds)
        if not math.isfinite(self.interval_seconds) or self.interval_seconds < 0:
            raise ValueError("checkpoint interval must be finite and nonnegative")
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise FileExistsError("quality checkpoints require a fresh directory")
        self.records = []
        self.last_saved = None

    def save(self, payload, progress, *, force=False):
        """Call only after policy checks and descriptive validation complete.

        The payload callback copies tensors to CPU only when a save is due.
        Immutable names keep the old index valid if termination interrupts a
        write. Two retained versions permit selection at a strict time cutoff.
        """
        epoch = len(progress["epochs"])
        now = time.monotonic()
        if self.records and self.records[-1]["epoch"] == epoch:
            return self.records[-1]
        if (
            not force
            and self.last_saved is not None
            and now - self.last_saved < self.interval_seconds
        ):
            return None
        import torch

        validated_ns = time.monotonic_ns()
        name = f"checkpoint-{epoch:06d}.pt"
        path = self.directory / name
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload(), temporary)
        temporary.replace(path)
        record = dict(
            filename=name,
            epoch=epoch,
            validated_monotonic_ns=validated_ns,
            stored_monotonic_ns=time.monotonic_ns(),
            accepted_examples=progress["accepted_examples"],
            accepted_optimizer_steps=progress["accepted_optimizer_steps"],
            examples_processed=progress["examples_processed"],
            optimizer_steps=progress["optimizer_steps"],
        )
        records = [*self.records, record][-2:]
        dump(
            self.directory / "index.json",
            dict(schema="drmc-quality-checkpoints-v1", checkpoints=records),
        )
        self.records = records
        self.last_saved = time.monotonic()
        retained = {row["filename"] for row in records}
        for old in self.directory.glob("checkpoint-*.pt"):
            if old.name not in retained:
                old.unlink()
        return record


def eligible_checkpoint(directory, *, deadline_ns):
    """Read only a fully published snapshot completed inside this allocation."""
    directory = Path(directory)
    index = directory / "index.json"
    if not index.exists():
        return None
    document = json.loads(index.read_text())
    if document.get("schema") != "drmc-quality-checkpoints-v1":
        raise ValueError("unsupported quality checkpoint index")
    eligible = []
    for record in document["checkpoints"]:
        filename = record["filename"]
        if (
            Path(filename).name != filename
            or not filename.startswith("checkpoint-")
            or not filename.endswith(".pt")
        ):
            raise ValueError("invalid quality checkpoint filename")
        if not 0 < record["validated_monotonic_ns"] <= record["stored_monotonic_ns"]:
            raise ValueError("invalid checkpoint timing")
        if record["stored_monotonic_ns"] > deadline_ns:
            continue
        path = directory / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("published quality checkpoint is missing or empty")
        eligible.append(record)
    return max(eligible, key=lambda row: row["stored_monotonic_ns"]) if eligible else None
