import pytest

from drmc_rl.arena.paired_confirmation import confirm, lower_bound


def fixture(n=1000):
    plan = dict(
        baseline="core",
        candidates=["style"],
        opponents=["frozen"],
        conditions=[dict(level=14, speed=2, pace="normal")],
        confirmation_seeds=list(range(n)),
        score_margin=0.02,
        alpha=0.05,
        seed_design="independent",
    )
    records = [
        dict(
            agent=agent,
            opponent="frozen",
            level=14,
            speed=2,
            pace="normal",
            seed=seed,
            side=side,
            score=0.5,
        )
        for seed in range(n)
        for agent in ("core", "style")
        for side in (0, 1)
    ]
    return records, plan


def test_zero_variance_small_samples_cannot_prove_noninferiority():
    records, plan = fixture(8)
    report = confirm(records, plan)
    assert not report["noninferior"]
    assert report["comparisons"][0]["lower_bound"] < -0.02
    assert confirm(*fixture())["noninferior"]


def test_seed_pairing_censoring_and_declared_family_are_preserved():
    records, plan = fixture()
    records[0]["score"] = None
    result = confirm(records, plan)
    assert result["comparisons"][0]["censored_games"] == 1
    assert result["comparisons"][0]["score_difference"] is None
    assert not result["noninferior"]
    records, plan = fixture()
    with pytest.raises(ValueError, match="duplicate"):
        confirm(records + [records[0]], plan)
    assert lower_bound([0.0] * 1000, alpha=0.005, seed_design="independent") < lower_bound(
        [0.0] * 1000, alpha=0.05, seed_design="independent"
    )
    assert lower_bound([0.0] * 1000, alpha=0.05, seed_design="uniform_without_replacement") < -0.02


def test_real_losses_are_not_hidden_by_many_identical_decisions_or_ports():
    records, plan = fixture()
    for row in records:
        if row["agent"] == "style":
            row["score"] = 0.0
    result = confirm(records, plan)
    assert not result["noninferior"]
    assert result["comparisons"][0]["score_difference"] == -0.5
    assert result["comparisons"][0]["paired_seeds"] == 1000
