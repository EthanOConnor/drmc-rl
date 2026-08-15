from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace


torch = pytest.importorskip("torch")


def test_legacy_rating_bands_preserve_deployed_boundaries() -> None:
    from tools.train_legacy_policy import band_masks

    masks = band_masks(np.asarray([900.0, 1599.9, 1600.0, 1999.9, 2000.0, 2400.0]))
    np.testing.assert_array_equal(masks["lt1600"], [True, True, False, False, False, False])
    np.testing.assert_array_equal(
        masks["1600to2000"], [False, False, True, True, False, False]
    )
    np.testing.assert_array_equal(masks["gt2000"], [False, False, False, False, True, True])


def test_legacy_inputs_are_own_board_and_candidate_geometry_only() -> None:
    from tools.train_human_policy import KMAX
    from tools.train_legacy_policy import legacy_batch_inputs

    field = np.full((1, 128), 0xFF, dtype=np.uint8)
    field[0, -1] = 0xD1
    arrays = {
        "field": field,
        "pill": np.asarray([[0, 1]], dtype=np.uint8),
        "preview": np.asarray([[2, 0]], dtype=np.uint8),
        "candidate_actions": np.full((1, KMAX), -1, dtype=np.int16),
        "candidate_costs": np.zeros((1, KMAX), dtype=np.uint16),
        "candidate_count": np.asarray([2], dtype=np.uint8),
        "chosen_slot": np.asarray([1], dtype=np.uint8),
    }
    arrays["candidate_actions"][0, :2] = (0, 129)
    observations, _pill, _preview, _actions, _costs, mask, chosen = legacy_batch_inputs(
        arrays, np.asarray([0])
    )
    assert observations.shape == (1, 12, 16, 8)
    assert observations[0, 3, 15, 7] == 1.0
    assert observations[0, 8, 0, 0] == 1.0
    assert observations[0, 9, 0, 1] == 1.0
    assert mask[0, :3].tolist() == [True, True, False]
    assert chosen.tolist() == [1]


def test_legacy_small_config_matches_original_capacity() -> None:
    from tools.train_legacy_policy import legacy_policy_config

    config = legacy_policy_config("small")["smdp_ppo"]
    assert config["candidate_d_model"] == 96
    assert config["encoder_blocks"] == 2
    assert config["candidate_transformer_layers"] == 2
    assert config["candidate_hidden_dim"] == 192
    assert config["candidate_board_channels"] == 8


def test_legacy_training_smoke_writes_all_three_bands(tmp_path) -> None:
    from drmc_rl.training.utils.checkpoint_io import load_checkpoint
    from tools.train_human_policy import KMAX
    from tools.train_legacy_policy import train

    ratings = np.asarray([1000] * 4 + [1800] * 4 + [2200] * 4, dtype=np.float32)
    count = len(ratings)
    fields = np.full((count, 128), 0xFF, dtype=np.uint8)
    fields[:, -1] = 0xD1
    actions = np.full((count, KMAX), -1, dtype=np.int16)
    actions[:, :2] = (0, 129)
    costs = np.zeros((count, KMAX), dtype=np.uint16)
    costs[:, :2] = (5, 7)
    split = np.zeros(count, dtype=np.uint8)
    split[3::4] = 1
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    np.savez(
        dataset / "shard.npz",
        field=fields,
        pill=np.tile(np.asarray([[0, 1]], dtype=np.uint8), (count, 1)),
        preview=np.tile(np.asarray([[1, 2]], dtype=np.uint8), (count, 1)),
        candidate_actions=actions,
        candidate_costs=costs,
        candidate_count=np.full(count, 2, dtype=np.uint8),
        chosen_slot=np.arange(count, dtype=np.uint8) % 2,
        rating=ratings,
        player_key=np.arange(count, dtype=np.uint64) // 2,
        split=split,
        player_fold=np.ones(count, dtype=np.uint8),
        time_split=np.zeros(count, dtype=np.uint8),
    )
    output = tmp_path / "output"
    train(
        SimpleNamespace(
            dataset=dataset,
            output=output,
            device="cpu",
            capacity="small",
            epochs=1,
            batch_size=3,
            lr=2e-4,
            seed=1,
            log_every=0,
            validation_rows_per_shard=8,
            max_shards=None,
        )
    )
    for band in ("lt1600", "1600to2000", "gt2000"):
        checkpoint = load_checkpoint(output / f"bc_full_{band}.pt.gz", map_location="cpu")
        assert checkpoint["bc_meta"]["band"] == band
        assert checkpoint["bc_meta"]["training_rows"] == 3
