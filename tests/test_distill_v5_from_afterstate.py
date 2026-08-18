from __future__ import annotations

import torch

from tools.distill_v5_from_afterstate import _outcome_value_loss


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
