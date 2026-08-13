from __future__ import annotations

import gzip
import json
from types import SimpleNamespace

import numpy as np

from drmc_rl.training.diagnostics.logger import DiagLogger
from drmc_rl.training.diagnostics.video import VideoWriter


def test_video_writer(tmp_path) -> None:
    writer = VideoWriter(tmp_path)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    for _ in range(3):
        writer.add_frame("rollout", frame)
    path = writer.finalize("rollout", 12)
    assert path.exists()
    assert path.stat().st_size > 0


def test_scalar_batch_preserves_jsonl_rows_and_batches_remote_log(tmp_path) -> None:
    logger = DiagLogger(SimpleNamespace(logdir=tmp_path, viz=[]))

    class _Run:
        def __init__(self) -> None:
            self.calls: list[dict[str, float | int]] = []

        def log(self, values) -> None:
            self.calls.append(values)

        def finish(self) -> None:
            pass

    run = _Run()
    logger._wandb_run = run
    logger.log_scalars({"loss": 1.5, "reward": 2}, step=7)
    logger.close()

    assert run.calls == [{"loss": 1.5, "reward": 2.0, "global_step": 7}]
    with gzip.open(tmp_path / "metrics.jsonl.gz", "rt", encoding="utf-8") as fp:
        rows = [json.loads(line) for line in fp]
    assert rows == [
        {"step": 7, "type": "scalar", "name": "loss", "value": 1.5},
        {"step": 7, "type": "scalar", "name": "reward", "value": 2.0},
    ]
