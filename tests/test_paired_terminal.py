import numpy as np
import pytest

from drmc_rl.teachers.paired_terminal import (
    ContinuationPair,
    aggregate_panel,
    conservative_policy_target,
)


def panel():
    inventory = []
    results = []
    # The incumbent is consistently a draw. Alternative A wins under one
    # continuation and loses under the other: pure continuation sensitivity.
    for action in (10, 20):
        for member in (0, 1):
            for reserve in ("a", "b"):
                i = len(inventory)
                inventory.append(
                    dict(id=i, action=action, continuation=member, reserve=reserve, weight=0.25)
                )
                results.append(
                    dict(id=i, weight=0.25, outcome=3 if action == 10 else 1 if member == 0 else 2)
                )
    return inventory, results


def test_panel_separates_chance_and_continuation_sensitivity_and_pairs_incumbent():
    inventory, results = panel()
    report = aggregate_panel((10, 20), 10, inventory, results, [0.8, 0.2])
    ref, alternative = report["candidates"]
    assert ref["paired_gap"] == 0 and ref["paired_std"] == 0
    assert alternative["wdl"] == [0.5, 0.0, 0.5]
    assert alternative["chance_variance"] == 0
    assert alternative["continuation_sensitivity"] == 1
    assert alternative["paired_std"] == 1
    assert report["sampling_standard_error"] is None
    np.testing.assert_allclose(report["policy_target"]["probability"], [0.8, 0.2])


def test_censored_candidate_stays_unknown_and_suppresses_improvement_target():
    inventory, results = panel()
    results[-1]["outcome"] = None
    report = aggregate_panel((10, 20), 10, inventory, results, [0.8, 0.2])
    assert report["candidates"][0]["wdl"] == [0.0, 1.0, 0.0]
    assert report["candidates"][1]["wdl"] is None
    assert report["candidates"][1]["unknown_mass"] == 0.25
    assert report["policy_target"] is None


def test_missing_or_duplicate_deterministic_rollouts_cannot_add_evidence():
    inventory, results = panel()
    with pytest.raises(ValueError, match="coverage"):
        aggregate_panel((10, 20), 10, inventory, results[:-1], [0.8, 0.2])
    inventory[-1]["reserve"] = "a"
    with pytest.raises(ValueError, match="duplicate deterministic"):
        aggregate_panel((10, 20), 10, inventory, results, [0.8, 0.2])
    with pytest.raises(ValueError, match="native SMDP"):
        ContinuationPair("a", "b", 1.0, execution="sloth")


def test_conservative_target_preserves_reference_ratios_and_respects_kl_budget():
    result = conservative_policy_target(
        [0.1, 0.3, 0.6], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0], kl_budget=0.005
    )
    p = result["probability"]
    assert p[0] / p[1] == pytest.approx(1 / 3)
    assert p[0] > 0.1 and p[1] > 0.3 and p[2] < 0.6
    assert result["kl_to_reference"] <= 0.005 + 1e-12
    unchanged = conservative_policy_target([0.1, 0.9], [0.1, 0.0], [1.0, 0.0])
    np.testing.assert_allclose(unchanged["probability"], [0.1, 0.9])
