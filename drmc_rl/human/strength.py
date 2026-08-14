"""Human strength control from conditional empirical action-regret tails."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


REGRET_CALIBRATION_SCHEMA = "drmc-regret-calibration-v2"
DEFAULT_QUANTILES = np.asarray((0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.97, 0.99))


def _decreasing_isotonic(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators fit constrained to non-increasing."""

    y = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    blocks: list[list[float | int]] = []
    for i, (value, weight) in enumerate(zip(y, w, strict=True)):
        blocks.append([i, i + 1, float(value), float(max(weight, 1e-12))])
        while len(blocks) >= 2 and float(blocks[-2][2]) < float(blocks[-1][2]):
            right = blocks.pop()
            left = blocks.pop()
            total = float(left[3]) + float(right[3])
            mean = (float(left[2]) * float(left[3]) + float(right[2]) * float(right[3])) / total
            blocks.append([int(left[0]), int(right[1]), mean, total])
    out = np.empty_like(y)
    for start, stop, mean, _weight in blocks:
        out[int(start) : int(stop)] = float(mean)
    return out


def quality_opportunity(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Rating-independent scale of how much candidate quality varies per state."""

    quality = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(mask, dtype=np.bool_)
    if quality.shape != valid.shape or quality.ndim != 2:
        raise ValueError("scores and mask must have matching [batch,candidate] shapes")
    count = valid.sum(axis=1).clip(min=1)
    safe = np.where(valid, quality, 0.0)
    mean = safe.sum(axis=1) / count
    variance = (np.where(valid, (quality - mean[:, None]) ** 2, 0.0).sum(axis=1) / count)
    return np.sqrt(np.maximum(variance, 0.0))


@dataclass(frozen=True, slots=True)
class RegretCalibration:
    """Empirical regret distributions by human rating and decision opportunity.

    Strength differences often live in rare mistakes rather than the median
    placement.  The full regret quantile curve preserves those tails.  A
    rating-independent candidate-quality spread conditions the curve so the
    same player can be reliable in routine states and fallible in consequential
    ones without globally degrading ordinary placements.
    """

    rating_edges: np.ndarray
    opportunity_edges: np.ndarray
    quantile_levels: np.ndarray
    regret_quantiles: np.ndarray
    counts: np.ndarray

    @classmethod
    def fit(
        cls,
        ratings: np.ndarray,
        regrets: np.ndarray,
        opportunities: np.ndarray,
        *,
        rating_bins: int = 12,
        opportunity_bins: int = 4,
        quantile_levels: np.ndarray = DEFAULT_QUANTILES,
    ) -> "RegretCalibration":
        rating = np.asarray(ratings, dtype=np.float64)
        regret = np.asarray(regrets, dtype=np.float64)
        opportunity = np.asarray(opportunities, dtype=np.float64)
        valid = (
            np.isfinite(rating)
            & np.isfinite(regret)
            & np.isfinite(opportunity)
            & (regret >= 0)
            & (opportunity >= 0)
        )
        rating, regret, opportunity = rating[valid], regret[valid], opportunity[valid]
        if len(rating) < max(2 * rating_bins * opportunity_bins, 40):
            raise ValueError("not enough decisions to calibrate conditional regret")
        rating_edges = np.unique(
            np.quantile(rating, np.linspace(0.0, 1.0, int(rating_bins) + 1))
        )
        opportunity_edges = np.unique(
            np.quantile(opportunity, np.linspace(0.0, 1.0, int(opportunity_bins) + 1))
        )
        if len(opportunity_edges) == 1:
            value = float(opportunity_edges[0])
            epsilon = max(abs(value) * 1e-6, 1e-9)
            opportunity_edges = np.asarray((value - epsilon, value + epsilon))
        if len(rating_edges) < 3:
            raise ValueError("rating or opportunity range is too narrow to calibrate regret")
        levels = np.asarray(quantile_levels, dtype=np.float64)
        if levels.ndim != 1 or len(levels) < 3 or np.any(np.diff(levels) <= 0):
            raise ValueError("quantile levels must be a strictly increasing vector")
        if levels[0] <= 0 or levels[-1] >= 1:
            raise ValueError("quantile levels must lie strictly between zero and one")

        r_index = np.clip(
            np.searchsorted(rating_edges, rating, side="right") - 1,
            0,
            len(rating_edges) - 2,
        )
        o_index = np.clip(
            np.searchsorted(opportunity_edges, opportunity, side="right") - 1,
            0,
            len(opportunity_edges) - 2,
        )
        shape = (len(rating_edges) - 1, len(opportunity_edges) - 1)
        counts = np.zeros(shape, dtype=np.int64)
        curves = np.empty((*shape, len(levels)), dtype=np.float64)
        global_curve = np.quantile(regret, levels)
        for r in range(shape[0]):
            rating_values = regret[r_index == r]
            rating_curve = np.quantile(rating_values, levels) if len(rating_values) else global_curve
            for o in range(shape[1]):
                values = regret[(r_index == r) & (o_index == o)]
                counts[r, o] = len(values)
                curves[r, o] = np.quantile(values, levels) if len(values) >= 20 else rating_curve

        # Every point in the error distribution, including the rare tail, must
        # be non-increasing with rating. This is the intended strength prior;
        # it does not alter the rating-independent competitive value model.
        for o in range(shape[1]):
            for q in range(len(levels)):
                curves[:, o, q] = _decreasing_isotonic(curves[:, o, q], counts[:, o])
        curves = np.maximum.accumulate(curves, axis=2)
        return cls(rating_edges, opportunity_edges, levels, curves, counts)

    @property
    def rating_centers(self) -> np.ndarray:
        return (self.rating_edges[:-1] + self.rating_edges[1:]) * 0.5

    @property
    def opportunity_centers(self) -> np.ndarray:
        return (self.opportunity_edges[:-1] + self.opportunity_edges[1:]) * 0.5

    def curve(self, rating: float, opportunity: float) -> np.ndarray:
        rating_value = float(np.clip(rating, self.rating_edges[0], self.rating_edges[-1]))
        opportunity_value = float(
            np.clip(opportunity, self.opportunity_edges[0], self.opportunity_edges[-1])
        )
        by_opportunity = np.stack(
            [
                np.interp(rating_value, self.rating_centers, self.regret_quantiles[:, o, q])
                for o in range(len(self.opportunity_centers))
                for q in range(len(self.quantile_levels))
            ]
        ).reshape(len(self.opportunity_centers), len(self.quantile_levels))
        return np.asarray(
            [
                np.interp(opportunity_value, self.opportunity_centers, by_opportunity[:, q])
                for q in range(len(self.quantile_levels))
            ],
            dtype=np.float64,
        )

    def parameters(self, rating: float, opportunity: float) -> tuple[float, float]:
        curve = self.curve(rating, opportunity)
        median = float(np.interp(0.5, self.quantile_levels, curve))
        q25 = float(np.interp(0.25, self.quantile_levels, curve))
        q75 = float(np.interp(0.75, self.quantile_levels, curve))
        log_tolerance = max((np.log1p(q75) - np.log1p(q25)) * 0.35, 0.025)
        return median, float(log_tolerance)

    def sample(self, rating: float, opportunity: float, rng: np.random.Generator) -> float:
        curve = self.curve(rating, opportunity)
        quantile = float(rng.uniform(self.quantile_levels[0], self.quantile_levels[-1]))
        return float(max(np.interp(quantile, self.quantile_levels, curve), 0.0))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REGRET_CALIBRATION_SCHEMA,
            "rating_edges": self.rating_edges.tolist(),
            "opportunity_edges": self.opportunity_edges.tolist(),
            "quantile_levels": self.quantile_levels.tolist(),
            "regret_quantiles": self.regret_quantiles.tolist(),
            "counts": self.counts.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RegretCalibration":
        if payload.get("schema") != REGRET_CALIBRATION_SCHEMA:
            raise ValueError(
                f"regret calibration must be {REGRET_CALIBRATION_SCHEMA}; "
                "recalibrate this checkpoint with tools.recalibrate_afterstate_strength"
            )
        return cls(
            np.asarray(payload["rating_edges"], dtype=np.float64),
            np.asarray(payload["opportunity_edges"], dtype=np.float64),
            np.asarray(payload["quantile_levels"], dtype=np.float64),
            np.asarray(payload["regret_quantiles"], dtype=np.float64),
            np.asarray(payload["counts"], dtype=np.int64),
        )


class RegretStrengthController:
    """Select human-plausible actions from the calibrated conditional tail."""

    def __init__(self, calibration: RegretCalibration, *, seed: int = 0) -> None:
        self.calibration = calibration
        self.rng = np.random.default_rng(int(seed))

    def choose(
        self,
        competitive_scores: np.ndarray,
        human_logits: np.ndarray,
        candidate_mask: np.ndarray,
        *,
        rating: float,
        deterministic: bool = False,
    ) -> tuple[int, dict[str, float | int]]:
        valid = np.flatnonzero(np.asarray(candidate_mask, dtype=np.bool_))
        if valid.size == 0:
            raise ValueError("cannot choose without a valid candidate")
        quality = np.asarray(competitive_scores, dtype=np.float64)[valid]
        style = np.asarray(human_logits, dtype=np.float64)[valid]
        best = float(np.max(quality))
        regret = np.maximum(best - quality, 0.0)
        opportunity = float(np.std(quality))
        if deterministic:
            target, tolerance = self.calibration.parameters(rating, opportunity)
        else:
            target = self.calibration.sample(rating, opportunity, self.rng)
            _median, tolerance = self.calibration.parameters(rating, opportunity)
        distance = np.abs(np.log1p(regret) - np.log1p(target))
        closest = float(np.min(distance))
        plausible = distance <= closest + tolerance
        style = np.where(plausible, style, -np.inf)
        local = int(np.argmax(style))
        return int(valid[local]), {
            "target_regret": float(target),
            "chosen_regret": float(regret[local]),
            "opportunity": opportunity,
            "best_candidate_slot": int(valid[int(np.argmax(quality))]),
        }


__all__ = [
    "REGRET_CALIBRATION_SCHEMA",
    "RegretCalibration",
    "RegretStrengthController",
    "quality_opportunity",
]
