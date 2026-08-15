from __future__ import annotations

import pytest
import torch

from drmc_rl.training.distillation import (
    masked_distillation_loss,
    masked_teacher_distribution,
)


def test_masked_teacher_distribution_excludes_padding() -> None:
    logits = torch.tensor([[1.0, 2.0, 1000.0], [3.0, -1.0, 0.0]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    probability = masked_teacher_distribution(logits, mask)
    assert probability[0, 2] == 0.0
    assert probability[1].tolist() == [1.0, 0.0, 0.0]
    assert probability.sum(dim=1).tolist() == pytest.approx([1.0, 1.0])


def test_matching_student_minimizes_masked_distillation_loss() -> None:
    teacher = torch.tensor([[3.0, 1.0, -4.0], [0.0, 2.0, 1.0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    matching = masked_distillation_loss(teacher, teacher, mask)
    reversed_logits = teacher.flip(dims=(1,))
    reversed_loss = masked_distillation_loss(reversed_logits, teacher, mask)
    assert matching < reversed_loss


def test_distillation_rejects_empty_candidate_row() -> None:
    with pytest.raises(ValueError, match="at least one"):
        masked_teacher_distribution(torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))
