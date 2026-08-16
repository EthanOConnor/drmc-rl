"""Predeclared release-gate calculations for the three player products."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class CompetitiveReleaseEvidence:
    wins: int
    draws: int
    losses: int
    games: int
    win_probability_mean: float
    win_probability_lower: float
    side_win_gap: float
    worst_active_payoff: float | None
    passed: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExecutionReleaseEvidence:
    scripts: int
    profile_violations: int
    replay_divergences: int
    deadline_misses: int
    deadline_miss_fraction: float
    passed: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainerReleaseEvidence:
    requested_levels: tuple[float, ...]
    achieved_strength: tuple[float, ...]
    adjacent_inversions: int
    matched_score: float
    style_strength_leakage: float
    passed: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def beta_win_interval(
    wins: int,
    nonwins: int,
    *,
    credibility: float = 0.95,
    samples: int = 200_000,
    seed: int = 731,
) -> tuple[float, float, float]:
    """Jeffreys-posterior mean and interval for strict win probability.

    Draws count as non-wins for a superhuman *win-probability* claim. The fixed
    seed makes evidence reproducible without a scipy dependency.
    """

    if min(int(wins), int(nonwins)) < 0:
        raise ValueError("match counts must be non-negative")
    if not 0.0 < credibility < 1.0:
        raise ValueError("credibility must lie in (0,1)")
    rng = np.random.default_rng(int(seed))
    posterior = rng.beta(
        int(wins) + 0.5,
        int(nonwins) + 0.5,
        size=max(10_000, int(samples)),
    )
    tail = (1.0 - float(credibility)) * 0.5
    return (
        float(posterior.mean()),
        float(np.quantile(posterior, tail)),
        float(np.quantile(posterior, 1.0 - tail)),
    )


def competitive_release_gate(
    *,
    wins: int,
    draws: int,
    losses: int,
    wins_as_p1: int,
    games_as_p1: int,
    wins_as_p2: int,
    games_as_p2: int,
    minimum_games: int = 200,
    required_lower_win_probability: float = 0.5,
    maximum_side_win_gap: float = 0.10,
    active_payoffs: Mapping[str, float] | None = None,
    minimum_active_payoff: float = -0.20,
) -> CompetitiveReleaseEvidence:
    games = int(wins) + int(draws) + int(losses)
    if games <= 0:
        raise ValueError("competitive release evidence requires games")
    mean, lower, _upper = beta_win_interval(int(wins), int(draws) + int(losses))
    p1 = int(wins_as_p1) / max(int(games_as_p1), 1)
    p2 = int(wins_as_p2) / max(int(games_as_p2), 1)
    side_gap = abs(p1 - p2)
    worst = None if not active_payoffs else float(min(active_payoffs.values()))
    failures: list[str] = []
    if games < int(minimum_games):
        failures.append("insufficient_games")
    if lower <= float(required_lower_win_probability):
        failures.append("win_lower_bound")
    if min(int(games_as_p1), int(games_as_p2)) <= 0:
        failures.append("missing_side")
    elif side_gap > float(maximum_side_win_gap):
        failures.append("side_gap")
    if worst is not None and worst < float(minimum_active_payoff):
        failures.append("catastrophic_active_matchup")
    return CompetitiveReleaseEvidence(
        wins=int(wins),
        draws=int(draws),
        losses=int(losses),
        games=games,
        win_probability_mean=mean,
        win_probability_lower=lower,
        side_win_gap=float(side_gap),
        worst_active_payoff=worst,
        passed=not failures,
        failures=tuple(failures),
    )


def execution_release_gate(
    *,
    scripts: int,
    profile_violations: int,
    replay_divergences: int,
    deadline_misses: int,
    maximum_deadline_miss_fraction: float = 0.001,
) -> ExecutionReleaseEvidence:
    total = int(scripts)
    if total <= 0:
        raise ValueError("execution release evidence requires scripts")
    miss_fraction = int(deadline_misses) / total
    failures: list[str] = []
    if int(profile_violations):
        failures.append("profile_violation")
    if int(replay_divergences):
        failures.append("replay_divergence")
    if miss_fraction > float(maximum_deadline_miss_fraction):
        failures.append("deadline_miss_rate")
    return ExecutionReleaseEvidence(
        scripts=total,
        profile_violations=int(profile_violations),
        replay_divergences=int(replay_divergences),
        deadline_misses=int(deadline_misses),
        deadline_miss_fraction=float(miss_fraction),
        passed=not failures,
        failures=tuple(failures),
    )


def trainer_release_gate(
    requested_levels: Sequence[float],
    achieved_strength: Sequence[float],
    *,
    matched_wins: int,
    matched_draws: int,
    matched_losses: int,
    maximum_adjacent_inversions: int = 0,
    matched_score_tolerance: float = 0.08,
    style_strength_leakage: float = 0.0,
    maximum_style_strength_leakage: float = 75.0,
) -> TrainerReleaseEvidence:
    requested = np.asarray(requested_levels, dtype=np.float64).reshape(-1)
    achieved = np.asarray(achieved_strength, dtype=np.float64).reshape(-1)
    if len(requested) < 2 or len(requested) != len(achieved):
        raise ValueError("trainer gate requires matching requested/achieved ladders")
    order = np.argsort(requested, kind="stable")
    ordered = achieved[order]
    inversions = int((np.diff(ordered) < 0).sum())
    games = int(matched_wins) + int(matched_draws) + int(matched_losses)
    if games <= 0:
        raise ValueError("trainer gate requires matched-rating games")
    score = (int(matched_wins) + 0.5 * int(matched_draws)) / games
    failures: list[str] = []
    if inversions > int(maximum_adjacent_inversions):
        failures.append("nonmonotone_strength")
    if abs(score - 0.5) > float(matched_score_tolerance):
        failures.append("matched_rating_miscalibration")
    if abs(float(style_strength_leakage)) > float(maximum_style_strength_leakage):
        failures.append("style_strength_leakage")
    return TrainerReleaseEvidence(
        requested_levels=tuple(map(float, requested)),
        achieved_strength=tuple(map(float, achieved)),
        adjacent_inversions=inversions,
        matched_score=float(score),
        style_strength_leakage=float(style_strength_leakage),
        passed=not failures,
        failures=tuple(failures),
    )


__all__ = [
    "CompetitiveReleaseEvidence",
    "ExecutionReleaseEvidence",
    "TrainerReleaseEvidence",
    "beta_win_interval",
    "competitive_release_gate",
    "execution_release_gate",
    "trainer_release_gate",
]
