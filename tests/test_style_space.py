import numpy as np

from drmc_rl.human.style import StyleSpace


def test_style_space_residualizes_rating_and_recovers_player_axis() -> None:
    rng = np.random.default_rng(3)
    rows = 600
    players = np.repeat(np.arange(6), rows // 6)
    ratings = 1000 + 200 * players + rng.normal(0, 20, rows)
    player_style = np.array([-2, -1, 0, 0.5, 1, 2])[players]
    features = np.column_stack(
        (
            ratings / 1000 + 0.4 * player_style + rng.normal(0, 0.05, rows),
            -0.3 * player_style + rng.normal(0, 0.05, rows),
        )
    )
    space = StyleSpace.fit(features, ratings, players, dimensions=1, min_decisions_per_player=50)
    assert space.dimensions == 1
    assert len(space.player_ids) == 6
    assert np.std(space.player_embeddings[:, 0]) > 0.1
