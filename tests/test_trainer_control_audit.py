import numpy as np

from drmc_rl.human.strength import RegretCalibration, RegretStrengthController
from tools.audit_trainer_control import measure, measure_legacy, summarize


def test_ordered_decoder_removes_style_tolerance_reversal():
    calibration = RegretCalibration(
        rating_edges=np.asarray([0., 1600., 3200.]),
        opportunity_edges=np.asarray([0., 1.]),
        quantile_levels=np.asarray([.25, .5, .75]),
        regret_quantiles=np.asarray([[[.099, .1, .11]], [[0., .099, .105]]]),
        counts=np.full((2, 1), 100),
    )
    quality = np.asarray([[1., .9, .865]])
    style = np.asarray([[[0., 1., 2.], [0., 1., 2.]]])
    mask = np.ones_like(quality, dtype=bool)
    old = measure_legacy(calibration, [800., 2400.], quality, style, mask, quantiles=3)
    old_report = summarize(old, [800., 2400.])
    assert old_report["target_regret"]["maximum_inversion"] == 0
    assert old_report["fixed_style_regret"]["maximum_inversion"] > .03
    values = measure(calibration, [800., 2400.], quality, style, mask, quantiles=3)
    controller = RegretStrengthController(calibration)
    for r, rating in enumerate([800., 2400.]):
        _, actual = controller.choose(quality[0], style[0, r], mask[0],
                                      rating=rating, deterministic=True)
        assert values["chosen_regret"][0, r, 1] == actual["chosen_regret"]
    report = summarize(values, [800., 2400.])
    assert report["target_regret"]["maximum_inversion"] == 0
    assert report["fixed_style_regret"]["maximum_inversion"] == 0


def test_ordered_regret_stays_monotone_across_quality_gaps_and_style_preferences():
    rng = np.random.default_rng(16)
    ratings = np.repeat(np.linspace(800, 2400, 6), 200)
    regrets = rng.exponential(5 - ratings / 600)
    calibration = RegretCalibration.fit(ratings, regrets, np.ones(len(ratings)),
                                        rating_bins=5, opportunity_bins=1)
    controller = RegretStrengthController(calibration, reference_rating=1800)
    for _ in range(12):
        quality = rng.uniform(-8, 3, 32)
        quality[0] += 15  # a large gap must not produce an overlapping envelope
        style = rng.normal(size=32)
        mask = rng.random(32) > .1
        for quantile in (.01, .1, .5, .75, .9, .99):
            selected = [controller.choose(quality, style, mask, rating=rating,
                                          quantile=quantile)[1]["chosen_regret"]
                        for rating in np.linspace(800, 2400, 21)]
            assert np.all(np.diff(selected) <= 1e-10)
