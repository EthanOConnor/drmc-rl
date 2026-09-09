from drmc_rl.arena.experiment import outcome_summary, relative_ratings, score_interval


def test_capped_games_are_unknown_and_exclude_the_affected_rating_edge():
    rows = [
        dict(seed=i // 2, side=i % 2, comparison="test", score=1.0, reason="clear")
        for i in range(64)
    ]
    assert score_interval(rows) is not None
    # Include historical timeout-as-draw encoding as well as the corrected None.
    for encoded_score in (0.5, None):
        rows[-1].update(reason="timeout", score=encoded_score)
        summary = outcome_summary(rows)
        assert summary["wins"] == 63 and summary["draws"] == 0 and summary["censored"] == 1
        assert summary["score_bounds"] == [63 / 64, 1.0]
        assert summary["score_ci"] is None
        assert (
            relative_ratings(
                {"test": dict(id="test", a="parent", b="new", level=14)},
                dict(enumerate(rows)),
                anchor="parent",
            )
            == []
        )
