"""PSRO-lite meta-strategy and exploitability tools for arena payoff matrices."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class MetaStrategyResult:
    row_strategy: np.ndarray
    column_strategy: np.ndarray
    population_strategy: np.ndarray
    game_value: float
    saddle_gap: float
    row_best_response: int
    column_best_response: int
    row_best_response_value: float
    column_best_response_value: float
    iterations: int
    temperature: float

    def to_dict(self, agents: Sequence[str] | None = None) -> dict[str, object]:
        payload = asdict(self)
        for key in ("row_strategy", "column_strategy", "population_strategy"):
            payload[key] = payload[key].tolist()  # type: ignore[index,union-attr]
        if agents is not None:
            if len(agents) != len(self.population_strategy):
                raise ValueError("agent names must match population strategy")
            payload["agents"] = list(agents)
            payload["mixture"] = {
                str(agent): float(weight)
                for agent, weight in zip(agents, self.population_strategy, strict=True)
            }
        payload["schema"] = "drmc-psro-meta-strategy-v1"
        return payload


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(scores, dtype=np.float64) / max(float(temperature), 1e-9)
    scaled -= scaled.max()
    weights = np.exp(np.clip(scaled, -60.0, 0.0))
    return weights / weights.sum()


def _floor_distribution(probability: np.ndarray, floor: float) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    if floor < 0 or floor * len(p) >= 1:
        raise ValueError("floor must be non-negative and leave positive free mass")
    if floor == 0:
        return p / p.sum()
    p = (1.0 - floor * len(p)) * (p / p.sum()) + floor
    return p / p.sum()


def solve_entropy_regularized_zero_sum(
    payoff: np.ndarray,
    *,
    iterations: int = 20000,
    temperature: float = 0.05,
    floor: float = 0.002,
    burn_in_fraction: float = 0.2,
) -> MetaStrategyResult:
    """Solve a finite zero-sum empirical game by averaged multiplicative weights.

    Rows maximize the supplied payoff; columns minimize it. The returned
    saddle gap is the unregularized best-response gap, so a small value remains
    interpretable even when temperature/floor keep the training mixture broad.
    """

    matrix = np.asarray(payoff, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("payoff must be a non-empty matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("payoff contains non-finite values")
    if iterations < 10:
        raise ValueError("iterations must be at least 10")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    rows, columns = matrix.shape
    row_score = np.zeros(rows, dtype=np.float64)
    col_score = np.zeros(columns, dtype=np.float64)
    row_average = np.zeros(rows, dtype=np.float64)
    col_average = np.zeros(columns, dtype=np.float64)
    burn_in = int(np.clip(round(iterations * burn_in_fraction), 0, iterations - 1))
    averaged = 0
    for step in range(1, int(iterations) + 1):
        row = _floor_distribution(_softmax(row_score, temperature), floor)
        column = _floor_distribution(_softmax(col_score, temperature), floor)
        eta = 1.0 / np.sqrt(float(step))
        row_score += eta * (matrix @ column)
        col_score += eta * (-(matrix.T @ row))
        if step > burn_in:
            row_average += row
            col_average += column
            averaged += 1
    row_strategy = _floor_distribution(row_average / max(averaged, 1), floor)
    column_strategy = _floor_distribution(col_average / max(averaged, 1), floor)
    row_values = matrix @ column_strategy
    column_values = row_strategy @ matrix
    row_br = int(np.argmax(row_values))
    col_br = int(np.argmin(column_values))
    upper = float(row_values[row_br])
    lower = float(column_values[col_br])
    value = float(row_strategy @ matrix @ column_strategy)
    if rows == columns:
        population = 0.5 * (row_strategy + column_strategy)
        population = _floor_distribution(population, floor)
    else:
        population = row_strategy.copy()
    return MetaStrategyResult(
        row_strategy=row_strategy.astype(np.float32),
        column_strategy=column_strategy.astype(np.float32),
        population_strategy=population.astype(np.float32),
        game_value=value,
        saddle_gap=max(0.0, upper - lower),
        row_best_response=row_br,
        column_best_response=col_br,
        row_best_response_value=upper,
        column_best_response_value=lower,
        iterations=int(iterations),
        temperature=float(temperature),
    )


def antisymmetrize_pairwise_payoff(payoff: np.ndarray) -> np.ndarray:
    """Return the closest antisymmetric matrix and zero the diagonal."""

    matrix = np.asarray(payoff, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("pairwise population payoff must be square")
    result = 0.5 * (matrix - matrix.T)
    np.fill_diagonal(result, 0.0)
    return result


def active_regression_gaps(
    payoff: np.ndarray,
    candidate: int,
    active: Sequence[int] | None = None,
) -> dict[int, float]:
    matrix = np.asarray(payoff, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("payoff must be square")
    if not 0 <= int(candidate) < matrix.shape[0]:
        raise IndexError(candidate)
    opponents = range(matrix.shape[0]) if active is None else (int(item) for item in active)
    return {
        opponent: float(matrix[int(candidate), opponent])
        for opponent in opponents
        if opponent != int(candidate)
    }


def write_result(
    result: MetaStrategyResult,
    path: str | Path,
    *,
    agents: Sequence[str] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(agents), indent=2, sort_keys=True) + "\n")


__all__ = [
    "MetaStrategyResult",
    "active_regression_gaps",
    "antisymmetrize_pairwise_payoff",
    "solve_entropy_regularized_zero_sum",
    "write_result",
]
