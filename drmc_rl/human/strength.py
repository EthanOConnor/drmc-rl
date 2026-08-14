"""Human strength control through calibrated action regret."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _decreasing_isotonic(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators fit constrained to non-increasing."""

    y = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    blocks: list[list[float | int]] = []
    for i, (value, weight) in enumerate(zip(y, w)):
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


@dataclass(frozen=True, slots=True)
class RegretCalibration:
    """Monotone empirical regret distribution across human rating."""

    rating_edges: np.ndarray
    median_regret: np.ndarray
    log_regret_std: np.ndarray
    counts: np.ndarray

    @classmethod
    def fit(
        cls,
        ratings: np.ndarray,
        regrets: np.ndarray,
        *,
        bins: int = 12,
    ) -> "RegretCalibration":
        rating = np.asarray(ratings, dtype=np.float64)
        regret = np.asarray(regrets, dtype=np.float64)
        valid = np.isfinite(rating) & np.isfinite(regret) & (regret >= 0)
        rating, regret = rating[valid], regret[valid]
        if len(rating) < max(2 * bins, 20):
            raise ValueError("not enough decisions to calibrate regret")
        edges = np.unique(np.quantile(rating, np.linspace(0.0, 1.0, bins + 1)))
        if len(edges) < 3:
            raise ValueError("rating range is too narrow to calibrate regret")
        index = np.clip(np.searchsorted(edges, rating, side="right") - 1, 0, len(edges) - 2)
        medians = np.empty(len(edges) - 1, dtype=np.float64)
        spreads = np.empty_like(medians)
        counts = np.bincount(index, minlength=len(medians)).astype(np.int64)
        for i in range(len(medians)):
            values = regret[index == i]
            logs = np.log1p(values)
            medians[i] = float(np.median(values))
            spreads[i] = float(max(np.std(logs), 0.05))
        medians = _decreasing_isotonic(medians, counts)
        return cls(edges, medians, spreads, counts)

    @property
    def rating_centers(self) -> np.ndarray:
        return (self.rating_edges[:-1] + self.rating_edges[1:]) * 0.5

    def parameters(self, rating: float) -> tuple[float, float]:
        value = float(np.clip(rating, self.rating_edges[0], self.rating_edges[-1]))
        median = float(np.interp(value, self.rating_centers, self.median_regret))
        spread = float(np.interp(value, self.rating_centers, self.log_regret_std))
        return median, spread

    def sample(self, rating: float, rng: np.random.Generator) -> float:
        median, spread = self.parameters(rating)
        return float(max(np.expm1(rng.normal(np.log1p(median), spread)), 0.0))

    def to_dict(self) -> dict[str, list[float] | list[int]]:
        return {
            "rating_edges": self.rating_edges.tolist(),
            "median_regret": self.median_regret.tolist(),
            "log_regret_std": self.log_regret_std.tolist(),
            "counts": self.counts.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RegretCalibration":
        return cls(
            np.asarray(payload["rating_edges"], dtype=np.float64),
            np.asarray(payload["median_regret"], dtype=np.float64),
            np.asarray(payload["log_regret_std"], dtype=np.float64),
            np.asarray(payload["counts"], dtype=np.int64),
        )


class RegretStrengthController:
    """Select human-plausible actions near a rating-calibrated regret target."""

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
        if deterministic:
            target, spread = self.calibration.parameters(rating)
        else:
            target = self.calibration.sample(rating, self.rng)
            _median, spread = self.calibration.parameters(rating)
        tolerance = max(float(spread), 0.05)
        style = style - np.max(style)
        objective = style - 0.5 * ((np.log1p(regret) - np.log1p(target)) / tolerance) ** 2
        local = int(np.argmax(objective))
        return int(valid[local]), {
            "target_regret": float(target),
            "chosen_regret": float(regret[local]),
            "best_candidate_slot": int(valid[int(np.argmax(quality))]),
        }


__all__ = ["RegretCalibration", "RegretStrengthController"]
