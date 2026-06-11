"""Tests for tools/skill_grade.py self-play fixed-point grading."""

from __future__ import annotations

import numpy as np
import pytest

from tools import skill_grade


def _toy_model(opp_slope: float, intercept: float = 1000.0) -> dict:
    """Version-2 model whose prediction is intercept + opp_slope * opp_whr.

    All standardization is identity (mu=0, sd=1) and every coefficient except
    the opp_whr base term is zero, so with zero metric features the prediction
    is linear in opp_whr and the fixed point is intercept / (1 - opp_slope).
    """
    feats = skill_grade.FIT_FEATURES
    n_expanded = len(feats) + len(feats) * (len(feats) + 1) // 2
    coef = np.zeros(n_expanded)
    coef[feats.index(skill_grade.OPP_FEATURE)] = opp_slope
    return {
        "version": 2,
        "base_features": list(feats),
        "mu": [0.0] * n_expanded,
        "sd": [1.0] * n_expanded,
        "coef": coef.tolist(),
        "intercept": intercept,
        "y_mean": 1500.0,
        "resid_std": 300.0,
    }


X_ZERO = np.zeros((3, len(skill_grade.BASE_FEATURES)))


def test_self_play_converges_contracting() -> None:
    # pred = 1000 + 0.5 * r  ->  fixed point r* = 2000
    rating, converged, iters = skill_grade.self_play_rating(_toy_model(0.5), X_ZERO)
    assert converged
    assert iters <= 20
    assert abs(rating - 2000.0) <= 1.0


def test_self_play_damping_tames_oscillation() -> None:
    # pred = 1000 - 1.5 * r: plain iteration diverges (|slope| > 1), but the
    # 0.5 step damping on sign-flipping deltas must converge to r* = 400.
    rating, converged, _iters = skill_grade.self_play_rating(_toy_model(-1.5), X_ZERO)
    assert converged
    assert abs(rating - 400.0) <= 2.0


def test_self_play_v1_model_passthrough() -> None:
    # Version-1 model (no version field, 6 base features): mean prediction.
    feats = skill_grade.BASE_FEATURES
    n_expanded = len(feats) + len(feats) * (len(feats) + 1) // 2
    model = {
        "base_features": list(feats),
        "mu": [0.0] * n_expanded,
        "sd": [1.0] * n_expanded,
        "coef": [0.0] * n_expanded,
        "intercept": 1234.0,
        "resid_std": 300.0,
    }
    rating, converged, iters = skill_grade.self_play_rating(model, X_ZERO)
    assert converged
    assert iters == 1
    assert rating == 1234.0


def test_predict_rating_requires_opp_for_v2() -> None:
    with pytest.raises(ValueError):
        skill_grade.predict_rating(_toy_model(0.5), X_ZERO)
