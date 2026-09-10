"""Compare simultaneous-game solutions under bounded neural rounding error."""

import numpy as np


def compare_matrix_games(left, right, *, payoff_tolerance=1e-5, gap_tolerance=.02):
    """Check game certificates, without assuming a unique equilibrium policy.

    For |A-B|_infinity <= epsilon, either fixed strategy pair's best-response
    gap can increase by at most 2*epsilon under the other payoff matrix.
    This numerical statement says nothing about the learned critic's accuracy.
    """
    a, b = (np.asarray(result.joint_utilities, np.float64) for result in (left, right))
    if (left.actions != right.actions or left.opponent_actions != right.opponent_actions
            or a.ndim != 2 or a.shape != b.shape or not a.size
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("matrix comparisons require aligned complete finite payoffs")

    def strategies(result):
        p = np.asarray(result.policy_target, np.float64)
        q = np.asarray(result.opponent_policy, np.float64)
        if (p.shape != (a.shape[0],) or q.shape != (a.shape[1],)
                or not np.isfinite(p).all() or not np.isfinite(q).all()
                or (p < 0).any() or (q < 0).any()
                or not np.isclose(p.sum(), 1) or not np.isclose(q.sum(), 1)):
            raise ValueError("matrix comparison needs normalized strategies")
        return p / p.sum(), q / q.sum()

    p, q = strategies(left)
    r, s = strategies(right)

    def gap(matrix, row, column):
        return max(0.0, float((matrix @ column).max() - (row @ matrix).min()))

    epsilon = float(np.abs(a - b).max())
    gap_a, gap_b = gap(a, p, q), gap(b, r, s)
    cross_a, cross_b = gap(b, p, q), gap(a, r, s)
    roundoff = 1e-12
    bound_holds = (cross_a <= gap_a + 2 * epsilon + roundoff
                   and cross_b <= gap_b + 2 * epsilon + roundoff)
    return dict(
        contract="mixed-game-certificate-v1", maximum_payoff_error=epsilon,
        own_policy_total_variation=float(np.abs(p - r).sum() / 2),
        opponent_policy_total_variation=float(np.abs(q - s).sum() / 2),
        left_gap=gap_a, right_gap=gap_b,
        left_strategy_gap_on_right=cross_a, right_strategy_gap_on_left=cross_b,
        rounding_gap_bound=2 * epsilon, bound_holds=bound_holds,
        equivalent=(not left.budget_exhausted and not right.budget_exhausted
                    and left.equilibrium_converged and right.equilibrium_converged
                    and gap_a <= gap_tolerance and gap_b <= gap_tolerance
                    and epsilon <= payoff_tolerance and bound_holds),
        scope="Numerical game equivalence only; not critic calibration or playing strength.")
