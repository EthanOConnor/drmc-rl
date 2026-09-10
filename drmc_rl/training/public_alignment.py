"""Behavior-preserving targets for the full public controller input contract.

This teaches the frozen parent's decisions, not candidate quality or optimal
play. The student keeps all public features and legal controller witnesses.
"""

from __future__ import annotations

import torch

from drmc_rl.game.public_context import SIDE_FEATURE_DIM


def legacy_teacher_inputs(observation, pill, context, actions, mask):
    """Reconstruct the frozen actor's input from actual full public features."""
    opponent_pill = context[:, SIDE_FEATURE_DIM:SIDE_FEATURE_DIM + 6].reshape(-1, 2, 3).argmax(-1)
    own_same = pill[:, 0] == pill[:, 1]
    opponent_same = opponent_pill[:, 0] == opponent_pill[:, 1]
    boards = observation.clone()
    boards[:, 6:8] *= (~own_same)[:, None, None, None]
    boards[:, 14:16] *= (~opponent_same)[:, None, None, None]
    legacy_mask = mask & (~own_same[:, None] | (actions < 256))
    # Retain the exact complete inventory as the student input. Only the
    # frozen teacher's view has its historical orientation/bond filtering.
    feasible = torch.zeros((len(pill), 512), dtype=boards.dtype, device=boards.device)
    feasible.scatter_add_(1, actions.clamp_min(0).long(), legacy_mask.to(boards.dtype))
    boards[:, 16:20] = feasible.reshape(-1, 4, 16, 8)
    return boards, legacy_mask


def alignment_target(logits, teacher_mask, student_mask, *, support_epsilon=1e-4):
    """Keep parent probabilities with explicit small support for every action.

    A root with no historically supported action has no teacher target. The
    caller excludes it from fitting instead of inventing a quality label.
    """
    if not 0 < support_epsilon < 1:
        raise ValueError("alignment support must be strictly between zero and one")
    if torch.any(teacher_mask & ~student_mask) or not bool(student_mask.any(-1).all()):
        raise ValueError("teacher actions must be a subset of the nonempty full frontier")
    supported = teacher_mask.any(-1)
    probability = logits.float().masked_fill(~teacher_mask, -1e9).softmax(-1)
    probability = probability.masked_fill(~teacher_mask, 0)
    uniform = student_mask.float() / student_mask.sum(-1, keepdim=True)
    probability = (1 - support_epsilon) * probability + support_epsilon * uniform
    probability = torch.where(supported[:, None], probability, uniform)
    return probability, supported


def alignment_kl(logits, target, mask):
    logp = logits.float().masked_fill(~mask, -1e9).log_softmax(-1)
    return (target * (target.clamp_min(1e-30).log() - logp)).masked_fill(~mask, 0).sum(-1)
