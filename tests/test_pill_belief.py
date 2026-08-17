from __future__ import annotations

import numpy as np
import pytest

from drmc_rl.search.pill_belief import (
    CHANCE_MODEL_ID,
    PillReserveBelief,
    canonical_pair_to_pill_id,
    pill_id_to_canonical_pair,
    reserve_for_seed,
    reserve_table,
)


def test_reserve_generation_matches_native_golden_seed() -> None:
    assert reserve_for_seed(0x89, 0x88)[:10].tolist() == [
        3,
        5,
        8,
        3,
        0,
        1,
        0,
        7,
        2,
        0,
    ]


def test_vectorized_reserve_table_matches_scalar_generator() -> None:
    rng = np.random.default_rng(20260817)
    table = reserve_table()
    for first, second in rng.integers(0, 256, size=(64, 2)):
        index = int(first) * 256 + int(second)
        assert np.array_equal(
            table[index], reserve_for_seed(int(first), int(second))
        )


def test_ordered_pair_mapping_round_trips_canonical_colors() -> None:
    for left in range(3):
        for right in range(3):
            pair = (left, right)
            assert pill_id_to_canonical_pair(
                canonical_pair_to_pill_id(pair)
            ) == pair


def test_unconditional_next_reveal_is_not_uniform() -> None:
    probability = PillReserveBelief().probabilities(0)
    assert probability.sum() == pytest.approx(1.0)
    assert float(np.max(np.abs(probability - 1.0 / 9.0))) > 0.01
    assert probability[[0, 3, 6]].mean() > probability[
        [1, 2, 4, 5, 7, 8]
    ].mean()


def test_observation_changes_posterior_predictive_distribution() -> None:
    prior = PillReserveBelief()
    posterior = prior.condition(0, 3)
    assert posterior.seed_count < prior.seed_count
    assert not np.allclose(
        posterior.probabilities(1), prior.probabilities(1)
    )
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

    reserve = reserve_for_seed(0x89, 0x88)
    belief = PillReserveBelief()
    for index, observed in enumerate(reserve):
        probability = belief.probabilities(index)
        impossible = np.flatnonzero(probability == 0.0)
        if impossible.size:
            with pytest.raises(ValueError, match="impossible"):
                belief.condition(index, int(impossible[0]))
            break
        belief = belief.condition(index, int(observed))
    else:  # pragma: no cover - a zero-support alternative appears quickly
        raise AssertionError("failed to find an impossible posterior reveal")


def test_serialized_belief_validates_count_hash_and_chance_model() -> None:
    belief = PillReserveBelief().condition(0, 3).condition(1, 5)
    payload = belief.to_dict()
    assert PillReserveBelief.from_dict(payload) == belief

    wrong_count = dict(payload)
    wrong_count["seed_count"] = int(payload["seed_count"]) + 1
    with pytest.raises(ValueError, match="seed_count"):
        PillReserveBelief.from_dict(wrong_count)

    wrong_hash = dict(payload)
    wrong_hash["stable_hash"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        PillReserveBelief.from_dict(wrong_hash)

    wrong_model = dict(payload)
    wrong_model["chance_model"] = "independent-uniform-ordered-pair-v0"
    with pytest.raises(ValueError, match="chance model"):
        PillReserveBelief.from_dict(wrong_model)
    assert payload["chance_model"] == CHANCE_MODEL_ID
