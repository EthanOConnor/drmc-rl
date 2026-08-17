"""Group-balanced Davidson W/D/L calibration with paired uncertainty.

Rows sampled from a long game must not outweigh rows sampled from a short game.
Every game therefore has equal total fitting and metric weight. Cross-fitting
uses whole games, distributes naturally drawn games across folds when possible,
and paired uncertainty is bootstrapped by game rather than decision row.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

OUTCOME_NAMES = ("win", "draw", "loss")


@dataclass(frozen=True, slots=True)
class DavidsonParameters:
    slope: float
    bias: float
    draw_logit: float

    def __post_init__(self) -> None:
        values = np.asarray((self.slope, self.bias, self.draw_logit), dtype=np.float64)
        if not np.isfinite(values).all() or self.slope <= 0:
            raise ValueError("Davidson parameters must be finite with positive slope")

    def to_dict(self) -> dict[str, float]:
        return {
            "slope": float(self.slope),
            "bias": float(self.bias),
            "draw_logit": float(self.draw_logit),
        }


def probabilities(scores: np.ndarray, parameters: DavidsonParameters) -> np.ndarray:
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(score).all():
        raise ValueError("calibration scores must be finite")
    strength = np.clip(parameters.slope * score + parameters.bias, -30.0, 30.0)
    logits = np.stack(
        (strength, np.full_like(strength, parameters.draw_logit), -strength), axis=1
    )
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(np.clip(logits, -60.0, 0.0))
    return result / result.sum(axis=1, keepdims=True)


def group_balanced_weights(groups: np.ndarray) -> np.ndarray:
    value = np.asarray(groups).reshape(-1)
    if value.size == 0:
        raise ValueError("groups cannot be empty")
    _unique, inverse, counts = np.unique(value, return_inverse=True, return_counts=True)
    weight = 1.0 / counts[inverse].astype(np.float64)
    weight *= len(weight) / weight.sum()
    return weight


def _validate(
    scores: np.ndarray, outcomes: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    outcome = np.asarray(outcomes, dtype=np.int64).reshape(-1)
    group = np.asarray(groups).reshape(-1)
    if not (len(score) == len(outcome) == len(group)) or len(score) < 3:
        raise ValueError("scores, outcomes, and groups must have matching nontrivial length")
    if not np.isfinite(score).all() or not np.isin(outcome, (0, 1, 2)).all():
        raise ValueError("calibration data contains invalid scores or outcomes")
    if len(np.unique(group)) < 2:
        raise ValueError("calibration requires at least two independent game groups")
    return score, outcome, group


def outcome_game_counts(outcomes: np.ndarray, groups: np.ndarray) -> dict[str, int]:
    outcome = np.asarray(outcomes, dtype=np.int64).reshape(-1)
    group = np.asarray(groups).reshape(-1)
    if len(outcome) != len(group):
        raise ValueError("outcomes and groups must have matching length")
    unique = np.unique(group)
    return {
        name: int(sum(bool(np.any(outcome[group == item] == target)) for item in unique))
        for target, name in enumerate(OUTCOME_NAMES)
    }


def fit_parameters(
    scores: np.ndarray,
    outcomes: np.ndarray,
    groups: np.ndarray,
    *,
    max_iter: int = 250,
) -> DavidsonParameters:
    """Fit a positive-slope Davidson link with equal total weight per game."""

    import torch

    score, outcome, group = _validate(scores, outcomes, groups)
    weight = group_balanced_weights(group)
    x = torch.from_numpy(score).double()
    y = torch.from_numpy(outcome).long()
    w = torch.from_numpy(weight).double()
    raw_slope = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    bias = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    draw = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [raw_slope, bias, draw],
        max_iter=int(max_iter),
        tolerance_grad=1e-10,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        slope = torch.nn.functional.softplus(raw_slope) + 1e-6
        strength = slope * x + bias
        logits = torch.stack((strength, draw.expand_as(strength), -strength), dim=1)
        per_row = torch.nn.functional.cross_entropy(logits, y, reduction="none")
        loss = (per_row * w).sum() / w.sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return DavidsonParameters(
        slope=float(torch.nn.functional.softplus(raw_slope).detach() + 1e-6),
        bias=float(bias.detach()),
        draw_logit=float(draw.detach()),
    )


def metric_contributions(
    probability: np.ndarray, outcomes: np.ndarray
) -> dict[str, np.ndarray]:
    target = np.asarray(outcomes, dtype=np.int64).reshape(-1)
    prediction = np.asarray(probability, dtype=np.float64)
    if prediction.shape != (len(target), 3):
        raise ValueError("probability must have shape [rows,3]")
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(prediction.sum(axis=1), 1.0, atol=2e-6):
        raise ValueError("probability rows must sum to one")
    one_hot = np.eye(3, dtype=np.float64)[target]
    return {
        "brier": np.sum(np.square(prediction - one_hot), axis=1),
        "log_loss": -np.log(
            np.maximum(prediction[np.arange(len(target)), target], 1e-12)
        ),
    }


def weighted_metrics(
    probability: np.ndarray,
    outcomes: np.ndarray,
    groups: np.ndarray,
    *,
    ece_bins: int = 10,
) -> dict[str, float]:
    target = np.asarray(outcomes, dtype=np.int64).reshape(-1)
    weight = group_balanced_weights(np.asarray(groups))
    contributions = metric_contributions(probability, target)
    result = {
        name: float(np.average(values, weights=weight))
        for name, values in contributions.items()
    }
    confidence = np.asarray(probability).max(axis=1)
    correct = np.asarray(probability).argmax(axis=1) == target
    ece = 0.0
    edges = np.linspace(0.0, 1.0, int(ece_bins) + 1)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (confidence >= low) & (
            confidence <= high if index == len(edges) - 2 else confidence < high
        )
        if not mask.any():
            continue
        bin_weight = float(weight[mask].sum() / weight.sum())
        accuracy = float(np.average(correct[mask].astype(np.float64), weights=weight[mask]))
        mean_confidence = float(np.average(confidence[mask], weights=weight[mask]))
        ece += bin_weight * abs(accuracy - mean_confidence)
    result[f"ece_{ece_bins}"] = float(ece)
    return result


def _stable_group_order(groups: np.ndarray, *, seed: int) -> list[object]:
    return sorted(
        np.unique(groups).tolist(),
        key=lambda item: hashlib.sha256(f"{int(seed)}:{item}".encode()).digest(),
    )


def _fold_assignment(
    outcomes: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    folds: int,
) -> dict[object, int]:
    """Deterministically spread draw and decisive games across folds."""

    outcome = np.asarray(outcomes, dtype=np.int64).reshape(-1)
    group = np.asarray(groups).reshape(-1)
    unique = np.unique(group)
    fold_count = min(max(2, int(folds)), len(unique))
    categories: dict[str, list[object]] = {"draw": [], "decisive": []}
    for item in unique:
        category = "draw" if bool(np.any(outcome[group == item] == 1)) else "decisive"
        categories[category].append(item)
    assignment: dict[object, int] = {}
    for offset, category in enumerate(("draw", "decisive")):
        ordered = _stable_group_order(np.asarray(categories[category]), seed=seed + offset)
        for index, item in enumerate(ordered):
            assignment[item] = index % fold_count
    return assignment


def cross_fitted_predictions(
    scores: np.ndarray,
    outcomes: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    folds: int = 5,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    score, outcome, group = _validate(scores, outcomes, groups)
    assignment = _fold_assignment(outcome, group, seed=seed, folds=folds)
    fold_count = max(assignment.values()) + 1
    prediction = np.full((len(score), 3), np.nan, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for fold in range(fold_count):
        validation = np.asarray([assignment[item] == fold for item in group])
        if not validation.any():
            raise RuntimeError(f"cross-fitting fold {fold} contains no validation games")
        train = ~validation
        if len(np.unique(group[train])) < 2:
            raise ValueError("cross-fitting fold leaves fewer than two training games")
        fitted = fit_parameters(score[train], outcome[train], group[train])
        prediction[validation] = probabilities(score[validation], fitted)
        records.append(
            {
                "fold": fold,
                "train_rows": int(train.sum()),
                "validation_rows": int(validation.sum()),
                "train_games": int(len(np.unique(group[train]))),
                "validation_games": int(len(np.unique(group[validation]))),
                "train_outcome_games": outcome_game_counts(outcome[train], group[train]),
                "validation_outcome_games": outcome_game_counts(
                    outcome[validation], group[validation]
                ),
                "parameters": fitted.to_dict(),
            }
        )
    if not np.isfinite(prediction).all():
        missing = int((~np.isfinite(prediction).all(axis=1)).sum())
        raise RuntimeError(f"cross-fitting left {missing} rows without predictions")
    return prediction, records


def paired_game_bootstrap(
    calibrated: np.ndarray,
    baseline: np.ndarray,
    outcomes: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    samples: int = 2000,
) -> dict[str, dict[str, float]]:
    """Bootstrap calibrated-minus-baseline metric deltas by whole game."""

    target = np.asarray(outcomes, dtype=np.int64).reshape(-1)
    group = np.asarray(groups).reshape(-1)
    unique = np.unique(group)
    if len(unique) < 2 or samples < 1:
        raise ValueError("paired bootstrap requires multiple games and positive samples")
    calibrated_loss = metric_contributions(calibrated, target)
    baseline_loss = metric_contributions(baseline, target)
    group_delta: dict[str, np.ndarray] = {}
    for name in calibrated_loss:
        group_delta[name] = np.asarray(
            [
                float(
                    np.mean(
                        calibrated_loss[name][group == item]
                        - baseline_loss[name][group == item]
                    )
                )
                for item in unique
            ],
            dtype=np.float64,
        )
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(unique), size=(int(samples), len(unique)))
    result: dict[str, dict[str, float]] = {}
    for name, delta in group_delta.items():
        bootstrap = delta[indices].mean(axis=1)
        result[name] = {
            "delta": float(delta.mean()),
            "ci95_low": float(np.quantile(bootstrap, 0.025)),
            "ci95_high": float(np.quantile(bootstrap, 0.975)),
            "probability_improves": float(np.mean(bootstrap < 0.0)),
        }
    return result


def calibration_report(
    scores: np.ndarray,
    outcomes: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    folds: int = 5,
    bootstrap_samples: int = 2000,
    baseline: DavidsonParameters = DavidsonParameters(1.0, 0.0, -3.0),
) -> dict[str, Any]:
    score, outcome, group = _validate(scores, outcomes, groups)
    crossfit, fold_records = cross_fitted_predictions(
        score, outcome, group, seed=seed, folds=folds
    )
    baseline_probability = probabilities(score, baseline)
    final = fit_parameters(score, outcome, group)
    unique_groups = np.unique(group)
    outcome_games = outcome_game_counts(outcome, group)
    draw_games = outcome_games["draw"]
    return {
        "schema": "drmc-grouped-davidson-calibration-v3",
        "parameters": final.to_dict(),
        "baseline_parameters": baseline.to_dict(),
        "rows": int(len(score)),
        "games": int(len(unique_groups)),
        "outcome_games": outcome_games,
        "natural_draw_games": draw_games,
        "draw_identifiable": draw_games > 0,
        "weighting": "equal-total-weight-per-game",
        "crossfit": {
            "folds": fold_records,
            "fold_count": len(fold_records),
            "all_training_folds_draw_identifiable": all(
                int(record["train_outcome_games"]["draw"]) > 0
                for record in fold_records
            ),
            "baseline": weighted_metrics(baseline_probability, outcome, group),
            "calibrated": weighted_metrics(crossfit, outcome, group),
            "paired_game_bootstrap": paired_game_bootstrap(
                crossfit,
                baseline_probability,
                outcome,
                group,
                seed=seed + 1,
                samples=bootstrap_samples,
            ),
        },
    }


__all__ = [
    "DavidsonParameters",
    "calibration_report",
    "cross_fitted_predictions",
    "fit_parameters",
    "group_balanced_weights",
    "metric_contributions",
    "outcome_game_counts",
    "paired_game_bootstrap",
    "probabilities",
    "weighted_metrics",
]
