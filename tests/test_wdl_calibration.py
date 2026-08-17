from __future__ import annotations

import numpy as np
import pytest

from drmc_rl.eval.wdl_calibration import (
    DavidsonParameters,
    calibration_report,
    group_balanced_weights,
    probabilities,
)


def test_group_balanced_weights_give_each_game_equal_mass() -> None:
    groups = np.asarray([0, 0, 0, 1, 2, 2])
    weights = group_balanced_weights(groups)
    masses = [float(weights[groups == group].sum()) for group in range(3)]
    assert masses == pytest.approx([2.0, 2.0, 2.0])


def test_grouped_crossfit_recovers_monotone_link_and_improves_baseline() -> None:
    rng = np.random.default_rng(19)
    true = DavidsonParameters(slope=3.0, bias=0.35, draw_logit=-0.8)
    scores: list[float] = []
    outcomes: list[int] = []
    groups: list[int] = []
    for game in range(80):
        center = float(rng.uniform(-1.25, 1.25))
        rows = 2 + game % 7
        game_scores = center + rng.normal(0.0, 0.12, size=rows)
        game_probability = probabilities(np.asarray([center]), true)[0]
        outcome = int(rng.choice(3, p=game_probability))
        scores.extend(float(item) for item in game_scores)
        outcomes.extend([outcome] * rows)
        groups.extend([game] * rows)

    report = calibration_report(
        np.asarray(scores),
        np.asarray(outcomes),
        np.asarray(groups),
        seed=23,
        folds=5,
        bootstrap_samples=300,
        baseline=DavidsonParameters(0.25, 0.0, -3.0),
    )
    assert report["schema"] == "drmc-grouped-davidson-calibration-v2"
    assert report["parameters"]["slope"] > 0
    assert report["games"] == 80
    assert sum(fold["validation_games"] for fold in report["crossfit"]["folds"]) == 80
    assert (
        report["crossfit"]["calibrated"]["log_loss"]
        < report["crossfit"]["baseline"]["log_loss"]
    )
    interval = report["crossfit"]["paired_game_bootstrap"]["log_loss"]
    assert interval["ci95_low"] <= interval["delta"] <= interval["ci95_high"]
