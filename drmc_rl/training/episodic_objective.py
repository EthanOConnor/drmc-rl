"""Explicit reductions for natural-terminal, undiscounted policy learning.

An episode contributes a SUM of score-function terms. A global decision mean
rescales that estimator by one positive batch constant; multiplying each term
by its own episode's inverse length changes the objective. Critic fitting and
regularization may deliberately use a different sampling measure.
"""

from __future__ import annotations

import numpy as np
import torch

REDUCTIONS = ("decision_mean", "episode_mean")
DEFAULT_OBJECTIVE = {
    "actor": "decision_mean",
    "value": "episode_mean",
    "entropy": "episode_mean",
    "parent_kl": "episode_mean",
    "advantage_normalization": "episode_center_scale",
}


def objective_contract(config, *, historical=False):
    defaults = dict(DEFAULT_OBJECTIVE)
    if historical:
        defaults["actor"] = "episode_mean"
    supplied = config.get("objective", {})
    if set(supplied) - set(defaults):
        raise ValueError("unknown episodic objective field")
    result = defaults | supplied
    if any(result[key] not in REDUCTIONS for key in ("actor", "value", "entropy", "parent_kl")):
        raise ValueError("unknown episodic loss reduction")
    if result["advantage_normalization"] not in (
        "none",
        "decision_scale",
        "decision_center_scale",
        "episode_center_scale",
    ):
        raise ValueError("unknown advantage normalization")
    return result


def normalized_weights(inverse_lengths, reduction):
    """Unit-mean weights computed ONCE over the complete collection batch."""
    weights = np.asarray(inverse_lengths, dtype=np.float64)
    if weights.size == 0 or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("episode inverse lengths must be finite and positive")
    if reduction == "decision_mean":
        return np.ones_like(weights)
    if reduction == "episode_mean":
        return weights / weights.mean()
    raise ValueError(f"unknown reduction {reduction!r}")


def normalize_advantages(advantages, inverse_lengths, mode):
    advantage = np.asarray(advantages, dtype=np.float64)
    if not np.isfinite(advantage).all():
        raise ValueError("non-finite advantage")
    if mode == "none":
        return advantage, 0.0, 1.0
    reduction = "episode_mean" if mode == "episode_center_scale" else "decision_mean"
    if mode not in ("episode_center_scale", "decision_center_scale", "decision_scale"):
        raise ValueError("unknown advantage normalization")
    weights = normalized_weights(inverse_lengths, reduction)
    mean = np.average(advantage, weights=weights)
    scale = np.sqrt(np.average((advantage - mean) ** 2, weights=weights) + 1e-8)
    center = 0.0 if mode == "decision_scale" else mean
    return (advantage - center) / scale, float(center), float(scale)


def clipped_surrogate(log_ratio, advantage, weights, clip):
    ratio = log_ratio.exp()
    return -(
        weights * torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    ).mean()


def categorical_kl(old_log_probs, new_log_probs):
    return (old_log_probs.exp() * (old_log_probs - new_log_probs)).sum(-1)


def validate_resume_objective(previous_config, config):
    """Missing metadata denotes the historical algorithm, never a new default."""
    previous = objective_contract(previous_config, historical=True)
    current = objective_contract(config)
    if previous != current:
        raise ValueError("resume changed the episodic objective; use init_adapter in a new run")
    return current
