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


def test_entropy_temperature_changes_the_solution_not_just_convergence_speed():
    # With one column, the exact regularized response is softmax(payoff/tau).
    matrix = np.array([[1.0], [-1.0]])
    result = solve_entropy_regularized_zero_sum(matrix, iterations=1500, temperature=1.0, floor=0.0)
    expected = np.exp([1.0, -1.0])
    expected /= expected.sum()
    np.testing.assert_allclose(result.row_strategy, expected, atol=1e-6)
    assert result.regularized_gap < 1e-8
    sharper = solve_entropy_regularized_zero_sum(
        matrix, iterations=1500, temperature=0.2, floor=0.0
    )
    assert sharper.row_strategy[0] > 0.99


def test_probability_floor_is_not_applied_twice_or_accumulated_each_iteration():
    result = solve_entropy_regularized_zero_sum(
        np.array([[2.0], [-2.0]]), iterations=1500, temperature=0.1, floor=0.1
    )
    np.testing.assert_allclose(result.row_strategy, [0.9, 0.1], atol=1e-6)
