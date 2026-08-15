"""Losses shared by offline competitive-policy distillation tools."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_teacher_distribution(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return a finite teacher distribution over each row's legal candidates."""

    if logits.shape != mask.shape:
        raise ValueError(f"logits/mask shape mismatch: {logits.shape} != {mask.shape}")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    valid = mask.bool()
    if (~valid).all(dim=1).any():
        raise ValueError("every row must contain at least one legal candidate")
    masked = (logits.float() / float(temperature)).masked_fill(~valid, -torch.inf)
    return masked.softmax(dim=-1)


def masked_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float = 1.0,
    row_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Temperature-scaled cross entropy, reduced once per decision row."""

    if student_logits.shape != teacher_logits.shape or student_logits.shape != mask.shape:
        raise ValueError("student, teacher, and mask must have identical shapes")
    target = masked_teacher_distribution(teacher_logits, mask, temperature=temperature)
    student = (student_logits.float() / float(temperature)).masked_fill(~mask.bool(), -torch.inf)
    log_probability = F.log_softmax(student, dim=-1).masked_fill(~mask.bool(), 0.0)
    per_row = -(target * log_probability).sum(dim=-1)
    if row_weight is None:
        return per_row.mean() * float(temperature) ** 2
    weight = row_weight.float()
    if weight.shape != per_row.shape:
        raise ValueError(f"row_weight shape mismatch: {weight.shape} != {per_row.shape}")
    return (per_row * weight).sum() / weight.sum().clamp_min(1e-8) * float(temperature) ** 2


__all__ = ["masked_distillation_loss", "masked_teacher_distribution"]
