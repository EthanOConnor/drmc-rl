import numpy as np

from drmc_rl.arena.meta_strategy import (
    antisymmetrize_pairwise_payoff,
    solve_entropy_regularized_zero_sum,
)


def test_rps_meta_strategy_is_nearly_uniform() -> None:
    payoff = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    result = solve_entropy_regularized_zero_sum(payoff, iterations=5000, temperature=0.1, floor=0.001)
    assert np.allclose(result.population_strategy, np.full(3, 1 / 3), atol=0.05)
    assert result.saddle_gap < 0.1


def test_antisymmetrize_removes_side_noise() -> None:
    matrix = np.array([[0.1, 0.8], [-0.5, -0.2]])
    result = antisymmetrize_pairwise_payoff(matrix)
    assert np.allclose(result, -result.T)
    assert np.allclose(np.diag(result), 0)
