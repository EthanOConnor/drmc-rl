from drmc_rl.eval.release_gates import (
    competitive_release_gate,
    execution_release_gate,
    trainer_release_gate,
)


def test_release_gates_are_explicit_and_strict() -> None:
    competitive = competitive_release_gate(
        wins=180,
        draws=0,
        losses=20,
        wins_as_p1=89,
        games_as_p1=100,
        wins_as_p2=91,
        games_as_p2=100,
        active_payoffs={"exploiter": 0.1},
    )
    assert competitive.passed
    execution = execution_release_gate(
        scripts=10_000,
        profile_violations=0,
        replay_divergences=0,
        deadline_misses=1,
    )
    assert execution.passed
    trainer = trainer_release_gate(
        [1000, 1500, 2000],
        [1010, 1490, 2010],
        matched_wins=48,
        matched_draws=4,
        matched_losses=48,
        style_strength_leakage=20,
    )
    assert trainer.passed
