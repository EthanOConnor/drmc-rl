import numpy as np
import pytest

from tools.audit_motor_auxiliary import summarize_seed_metrics, validate_seeds


def test_confirmation_rejects_any_training_or_anchor_seed_overlap():
    validate_seeds([11, 12], dict(holdout_seeds=[11, 12]),
                   dict(holdout_seeds=[11, 12]), [dict(game_seed=13)])
    with pytest.raises(ValueError, match='excluded'):
        validate_seeds([11], dict(holdout_seeds=[11]),
                       dict(holdout_seeds=[12]), [])
    with pytest.raises(ValueError, match='excluded'):
        validate_seeds([11], dict(holdout_seeds=[11]),
                       dict(holdout_seeds=[11]), [dict(game_seed=11)])
    with pytest.raises(ValueError, match='distinct'):
        validate_seeds([11, 11], dict(holdout_seeds=[11]), dict(holdout_seeds=[11]), [])


def test_confirmation_intervals_are_paired_by_reset_seed():
    records = [dict(seed=i, initial={'reach_brier': .4 + .02*i},
                    fitted={'reach_brier': .2 + .02*i}, prior={'reach_brier': .3 + .02*i})
               for i in range(16)]
    result = summarize_seed_metrics(records, bootstrap_seed=71)
    assert result['independent_seeds'] == 16
    metric = result['metrics']['reach_brier']
    np.testing.assert_allclose(metric['change_ci95'], [-.2, -.2])
    np.testing.assert_allclose(metric['change_from_prior_ci95'], [-.1, -.1])
    with pytest.raises(ValueError, match='aggregate'):
        summarize_seed_metrics(records + records, bootstrap_seed=71)
