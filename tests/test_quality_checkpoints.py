import json
import time

import pytest
import torch

from drmc_rl.training.quality_checkpoints import QualityCheckpoints, eligible_checkpoint


def progress(epoch):
    return dict(
        epochs=[{}] * epoch,
        accepted_examples=epoch * 8,
        accepted_optimizer_steps=epoch,
        examples_processed=epoch * 8,
        optimizer_steps=epoch,
    )


def test_cutoff_ignores_partial_and_late_writes_and_bounds_retention(tmp_path):
    writer = QualityCheckpoints(tmp_path / "checkpoints", interval_seconds=0)
    writer.save(lambda: {"weight": torch.tensor(0)}, progress(0))
    first = writer.save(lambda: {"weight": torch.tensor(1)}, progress(1))
    cutoff = time.monotonic_ns()
    (writer.directory / "checkpoint-000002.pt.tmp").write_bytes(b"interrupted")
    assert eligible_checkpoint(writer.directory, deadline_ns=cutoff) == first
    writer.save(lambda: {"weight": torch.tensor(2)}, progress(2))
    assert eligible_checkpoint(writer.directory, deadline_ns=cutoff) == first
    assert len(list(writer.directory.glob("*.pt"))) == 2
    selected = eligible_checkpoint(writer.directory, deadline_ns=time.monotonic_ns())
    assert selected["epoch"] == 2
    assert torch.load(writer.directory / selected["filename"], weights_only=True)["weight"] == 2
    assert eligible_checkpoint(writer.directory, deadline_ns=1) is None


def test_interval_does_not_copy_tensors_and_final_snapshot_is_forced(tmp_path):
    writer = QualityCheckpoints(tmp_path / "checkpoints", interval_seconds=3600)
    called = []

    def payload():
        called.append(1)
        return {"weight": torch.tensor(len(called))}

    first = writer.save(payload, progress(0), force=True)
    assert writer.save(payload, progress(1)) is None and len(called) == 1
    second = writer.save(payload, progress(1), force=True)
    assert len(called) == 2 and second["epoch"] == 1
    assert writer.save(payload, progress(1), force=True) == second and len(called) == 2
    assert first["filename"] != second["filename"]
    with pytest.raises(FileExistsError):
        QualityCheckpoints(writer.directory)


def test_published_snapshot_cannot_escape_the_directory(tmp_path):
    (tmp_path / "index.json").write_text(
        json.dumps(
            {
                "schema": "drmc-quality-checkpoints-v1",
                "checkpoints": [{"filename": "../checkpoint-stolen.pt"}],
            }
        )
    )
    with pytest.raises(ValueError, match="filename"):
        eligible_checkpoint(tmp_path, deadline_ns=time.monotonic_ns())
