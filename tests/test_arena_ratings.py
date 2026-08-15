import numpy as np

from drmc_rl.arena.ratings import (
    DavidsonPosterior,
    PairCounts,
    PosteriorSamples,
    RatingConfig,
    fit_bayesian_ratings,
    matchup_information_matrix,
    sequential_update,
)


def test_posterior_analytic_gradient_matches_finite_difference() -> None:
    model = DavidsonPosterior(
        4,
        [PairCounts(0, 1, 7, 2, 4), PairCounts(1, 2, 3, 5, 9),
         PairCounts(2, 3, 8, 1, 6)],
        [None, 0, None, 2],
    )
    rng = np.random.default_rng(2)
    position = model.initial_position() + rng.normal(0.0, 0.2, model.dimension)
    _, analytic = model.log_density_and_grad(position)
    numeric = np.empty_like(analytic)
    epsilon = 1e-6
    for i in range(model.dimension):
        offset = np.zeros(model.dimension)
        offset[i] = epsilon
        plus = model.log_density_and_grad(position + offset)[0]
        minus = model.log_density_and_grad(position - offset)[0]
        numeric[i] = (plus - minus) / (2.0 * epsilon)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=2e-6)


def test_bayesian_fit_recovers_order_and_keeps_draw_tendency_out_of_skill() -> None:
    # Agent 1 draws far more often, but the decisive W/L ratios still place the
    # three agents in the known order. Davidson draw propensity absorbs the
    # survival/horizon behavior instead of awarding it half-win skill credit.
    pairs = [
        PairCounts(0, 1, 70, 30, 20),
        PairCounts(1, 2, 65, 90, 25),
        PairCounts(0, 2, 85, 5, 10),
    ]
    fit = fit_bayesian_ratings(
        3, pairs, [None, 0, 1],
        config=RatingConfig(
            chains=2, warmup=160, samples=300, require_convergence=False, seed=11
        ),
    )
    means = [rating.mean for rating in fit.agents]
    assert means[0] > means[1] > means[2]
    assert fit.agents[1].draw_propensity > fit.agents[0].draw_propensity
    assert fit.agents[1].draw_propensity > fit.agents[2].draw_propensity
    assert fit.diagnostics["side_effect"] == "fixed_zero"


def test_lineage_prior_centers_an_unplayed_child_on_parent() -> None:
    fit = fit_bayesian_ratings(
        3,
        [PairCounts(0, 1, 70, 5, 30)],
        [None, None, 0],
        config=RatingConfig(
            chains=2, warmup=160, samples=300, require_convergence=False, seed=19
        ),
    )
    parent, child = fit.agents[0], fit.agents[2]
    assert abs(parent.mean - child.mean) < 30
    assert child.sd > parent.sd


def test_sequential_update_reweights_persisted_full_posterior() -> None:
    base = fit_bayesian_ratings(
        2,
        [PairCounts(0, 1, 50, 10, 50)],
        [None, None],
        config=RatingConfig(
            chains=2, warmup=120, samples=240, require_convergence=False, seed=29
        ),
    )
    restored = PosteriorSamples.decode(base.samples.encode())
    updated = sequential_update(
        restored, [PairCounts(0, 1, 12, 1, 2)], base_diagnostics=base.diagnostics
    )
    assert updated.agents[0].mean > base.agents[0].mean
    assert updated.agents[1].mean < base.agents[1].mean
    assert updated.diagnostics["method"] == "sequential_importance"
    assert 0 < updated.diagnostics["importance_ess"] <= len(restored.log_weights)


def test_information_gain_prefers_an_uncertain_matchup() -> None:
    draws = 1_000
    rng = np.random.default_rng(41)
    samples = PosteriorSamples(
        skills=np.column_stack((
            np.zeros(draws), rng.normal(0.0, 1.0, draws), np.full(draws, -5.0),
        )),
        draw_tendencies=np.zeros((draws, 3)),
        draw_intercepts=np.full(draws, -2.0),
        lineage_scales=np.ones(draws),
        draw_scales=np.ones(draws),
        log_weights=np.zeros(draws),
        parents=np.full(3, -1, dtype=np.int32),
    )
    information = matchup_information_matrix(samples)
    assert information[0, 1] > information[0, 2] * 100
    np.testing.assert_allclose(information, information.T)
    np.testing.assert_array_equal(np.diag(information), 0.0)
