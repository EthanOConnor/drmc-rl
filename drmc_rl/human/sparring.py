"""Slow Bayesian skill adaptation for non-rubber-banding sparring blocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SkillPosterior:
    mean: float
    variance: float
    logistic_scale: float = 240.0
    process_variance: float = 16.0

    @property
    def standard_deviation(self) -> float:
        return float(np.sqrt(max(self.variance, 1e-9)))

    def update(self, *, opponent_rating: float, score: float, weight: float = 1.0) -> None:
        """Laplace/EKF update for score in {0, .5, 1}."""

        if not 0.0 <= float(score) <= 1.0:
            raise ValueError("score must be in [0,1]")
        scale = max(float(self.logistic_scale), 1e-6)
        delta = (self.mean - float(opponent_rating)) / scale
        probability = float(1.0 / (1.0 + np.exp(-np.clip(delta, -30.0, 30.0))))
        prior_variance = max(self.variance + self.process_variance, 1e-6)
        information = max(float(weight), 0.0) * probability * (1.0 - probability) / (scale * scale)
        posterior_variance = 1.0 / (1.0 / prior_variance + information)
        gradient = max(float(weight), 0.0) * (float(score) - probability) / scale
        self.mean = float(self.mean + posterior_variance * gradient)
        self.variance = float(posterior_variance)


@dataclass(slots=True)
class AdaptiveSparringController:
    posterior: SkillPosterior
    target_score: float = 0.5
    max_rating_change_per_block: float = 75.0
    minimum_rating: float = 600.0
    maximum_rating: float = 3000.0
    current_target: float | None = None

    def next_target(self) -> float:
        if not 0.05 <= self.target_score <= 0.95:
            raise ValueError("target_score must lie in [0.05,0.95]")
        logit = float(np.log(self.target_score) - np.log1p(-self.target_score))
        desired = self.posterior.mean - self.posterior.logistic_scale * logit
        desired = float(np.clip(desired, self.minimum_rating, self.maximum_rating))
        if self.current_target is None:
            self.current_target = desired
        else:
            delta = np.clip(
                desired - self.current_target,
                -self.max_rating_change_per_block,
                self.max_rating_change_per_block,
            )
            self.current_target = float(self.current_target + delta)
        return self.current_target

    def complete_block(
        self,
        *,
        opponent_rating: float,
        wins: int,
        draws: int,
        losses: int,
    ) -> float:
        games = int(wins) + int(draws) + int(losses)
        if games <= 0:
            raise ValueError("sparring block must contain at least one game")
        score = (int(wins) + 0.5 * int(draws)) / games
        self.posterior.update(opponent_rating=opponent_rating, score=score, weight=games)
        return self.next_target()


__all__ = ["AdaptiveSparringController", "SkillPosterior"]
