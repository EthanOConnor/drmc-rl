from __future__ import annotations

from argparse import Namespace

import torch

from tools.distill_v5_from_afterstate import _checkpoint_payload, _outcome_value_loss


def test_outcome_value_loss_is_fp32_and_differentiable_under_autocast() -> None:
    value = torch.tensor([[-0.5], [0.5]], dtype=torch.float32, requires_grad=True)
    won = torch.tensor([0.0, 1.0])
    row_weight = torch.tensor([1.0, 2.0])

    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = _outcome_value_loss(value, won, row_weight)

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()


def test_diagnostic_checkpoint_payload_records_partial_epoch_provenance() -> None:
    student = torch.nn.Linear(2, 1)
    args = Namespace(
        teacher="teacher.pt.gz",
        dataset="dataset",
        afterstates="afterstates",
        temperature=1.5,
    )

    payload = _checkpoint_payload(
        student=student,
        cfg={"smdp_ppo": {"candidate_architecture": "g5"}},
        args=args,
        epoch=2,
        optimizer_steps=1234,
        metrics={"train_loss": 0.5},
        epoch_complete=False,
    )

    assert payload["schema"] == "drmc-v5-v3-distill-v1"
    assert payload["distillation"]["epoch"] == 2
    assert payload["distillation"]["optimizer_steps"] == 1234
    assert payload["distillation"]["epoch_complete"] is False
    assert payload["distillation"]["metrics"] == {"train_loss": 0.5}


def test_checkpoint_payload_can_preserve_optimizer_and_scheduler_state() -> None:
    student = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)
    args = Namespace(
        teacher="teacher.pt.gz",
        dataset="dataset",
        afterstates="afterstates",
        temperature=1.5,
    )

    payload = _checkpoint_payload(
        student=student,
        cfg={"smdp_ppo": {"candidate_architecture": "g5"}},
        args=args,
        epoch=4,
        optimizer_steps=199065,
        metrics={"train_loss": 0.5},
        epoch_complete=False,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    assert payload["optimizer_state_dict"] == optimizer.state_dict()
    assert payload["scheduler_state_dict"] == scheduler.state_dict()
