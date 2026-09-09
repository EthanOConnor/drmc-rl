from tools.report_pace_objective_study import summarize_arm


def test_training_exposure_counts_repeats_and_censoring_without_claiming_independence():
    rows = [
        dict(
            update=update,
            level=14,
            pace="normal",
            seed=10,
            side=side,
            score=1.0,
            reason="clear",
            frames=100,
            a_stats=dict(decisions=4, no_reachable_after_delay=1),
        )
        for update in (1, 2)
        for side in (0, 1)
    ]
    rows[-1].update(score=None, reason="timeout")
    result = summarize_arm(rows, [], dict(status="Running", updates=2, frames=400, decisions=9))
    condition = result["conditions"][0]
    assert condition["collected_games"] == 4 and condition["natural_games"] == 3
    assert condition["distinct_seeds"] == 1 and condition["distinct_side_seed_games"] == 2
    assert condition["repeated_side_seed_experiences"] == 2
    assert condition["complete_pairs_collected"] == 2
    assert condition["score"] is None
    assert result["training_outcomes_are_promotion_evidence"] is False
