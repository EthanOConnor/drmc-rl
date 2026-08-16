from drmc_rl.human.sparring import AdaptiveSparringController, SkillPosterior


def test_sparring_target_moves_slowly_after_strong_block() -> None:
    controller = AdaptiveSparringController(
        SkillPosterior(mean=1500, variance=200**2),
        max_rating_change_per_block=50,
    )
    first = controller.next_target()
    second = controller.complete_block(opponent_rating=1500, wins=8, draws=0, losses=2)
    assert second >= first
    assert second - first <= 50
