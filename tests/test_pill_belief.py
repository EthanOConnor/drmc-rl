from __future__ import annotations

import numpy as np
import pytest

from drmc_rl.search.pill_belief import (
    PillReserveBelief,
    canonical_pair_to_pill_id,
    pill_id_to_canonical_pair,
    reserve_for_seed,
)


def test_reserve_generation_matches_native_golden_seed() -> None:
    assert reserve_for_seed(0x89, 0x88)[:10].tolist() == [3, 5, 8, 3, 0, 1, 0, 7, 2, 0]


def test_ordered_pair_mapping_round_trips_canonical_colors() -> None:
    for left in range(3):
        for right in range(3):
            pair = (left, right)
            assert pill_id_to_canonical_pair(canonical_pair_to_pill_id(pair)) == pair


def test_unconditional_next_reveal_is_not_uniform() -> None:
    probability = PillReserveBelief().probabilities(0)
    assert probability.sum() == pytest.approx(1.0)
    assert float(np.max(np.abs(probability - 1.0 / 9.0))) > 0.01
    assert probability[[0, 3, 6]].mean() > probability[[1, 2, 4, 5, 7, 8]].mean()


def test_observation_changes_posterior_predictive_distribution() -> None:
    prior = PillReserveBelief()
    posterior = prior.condition(0, 3)
    assert posterior.seed_count < prior.seed_count
    assert not np.allclose(posterior.probabilities(1), prior.probabilities(1))
    assert posterior.probabilities(0)[3] == pytest.approx(1.0)


def test_visible_pills_condition_the_two_previous_reserve_entries() -> None:
    reserve = reserve_for_seed(0x89, 0x88)
    belief = PillReserveBelief().condition_visible(
        reserve_counter=2,
        falling_colors=pill_id_to_canonical_pair(int(reserve[0])),
        preview_colors=pill_id_to_canonical_pair(int(reserve[1])),
    )
    assert belief.probabilities(0)[int(reserve[0])] == pytest.approx(1.0)
    assert belief.probabilities(1)[int(reserve[1])] == pytest.approx(1.0)


def test_impossible_or_contradictory_observations_are_rejected() -> None:
    with pytest.raises(ValueError, match="contradictory"):
        PillReserveBelief(((0, 1), (0, 2)))
    impossible = tuple((index, 0) for index in range(16))
    try:
        PillReserveBelief(impossible)
    except ValueError as error:
        assert "impossible" in str(error)
