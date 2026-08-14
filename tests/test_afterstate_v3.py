from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")


def test_competitive_quality_is_independent_of_human_condition() -> None:
    from drmc_rl.human.afterstate_model import AfterstatePolicyNet

    torch.manual_seed(1)
    net = AfterstatePolicyNet(
        condition_dim=4, d_model=32, bottle_blocks=1, candidate_layers=1, heads=4
    ).eval()
    batch, candidates = 2, 5
    fields = torch.full((batch, candidates, 128), 0xFF, dtype=torch.uint8)
    fields[:, :, -1] = 0xD0
    root = torch.full((batch, 128), 0xFF, dtype=torch.uint8)
    opponent = torch.full((batch, 128), 0xFF, dtype=torch.uint8)
    pill = torch.tensor([[0, 1], [1, 2]])
    preview = torch.tensor([[2, 0], [0, 1]])
    actions = torch.arange(candidates).repeat(batch, 1)
    costs = torch.full((batch, candidates), 30.0)
    mask = torch.ones((batch, candidates), dtype=torch.bool)
    with torch.inference_mode():
        low = net(
            fields, root, opponent, pill, preview, actions, costs, mask, torch.zeros(batch, 4)
        )
        high = net(
            fields, root, opponent, pill, preview, actions, costs, mask, torch.ones(batch, 4)
        )
    torch.testing.assert_close(low["competitive_score"], high["competitive_score"])
    assert not torch.equal(low["human_logits"], high["human_logits"])


def test_afterstate_policy_masks_padding_and_has_tactical_heads() -> None:
    from drmc_rl.human.afterstate_model import AfterstatePolicyNet

    net = AfterstatePolicyNet(
        condition_dim=3, d_model=32, bottle_blocks=0, candidate_layers=1, heads=4
    ).eval()
    mask = torch.tensor([[True, True, False]])
    out = net(
        torch.full((1, 3, 128), 0xFF, dtype=torch.uint8),
        torch.full((1, 128), 0xFF, dtype=torch.uint8),
        torch.full((1, 128), 0xFF, dtype=torch.uint8),
        torch.tensor([[0, 1]]),
        torch.tensor([[1, 2]]),
        torch.tensor([[20, 21, -1]]),
        torch.tensor([[10.0, 11.0, 0.0]]),
        mask,
        torch.zeros((1, 3)),
    )
    assert set(out) == {
        "competitive_score",
        "human_logits",
        "outcome_logit",
        "clear_logit",
        "topout_logit",
        "virus_delta",
        "attack",
    }
    assert out["competitive_score"][0, 2] < -1e8
    assert out["virus_delta"][0, 2] == 0


def test_regret_calibration_is_monotone_and_controls_choices() -> None:
    from drmc_rl.human.strength import RegretCalibration, RegretStrengthController

    rng = np.random.default_rng(4)
    ratings = np.repeat(np.linspace(800, 2400, 9), 100)
    regrets = np.maximum(3.0 - (ratings - 800) / 800 + rng.normal(0, 0.2, len(ratings)), 0)
    calibration = RegretCalibration.fit(ratings, regrets, bins=8)
    assert np.all(np.diff(calibration.median_regret) <= 1e-12)
    controller = RegretStrengthController(calibration, seed=2)
    quality = np.asarray([10.0, 9.0, 7.0])
    style = np.zeros(3)
    low, low_info = controller.choose(
        quality, style, np.ones(3, bool), rating=800, deterministic=True
    )
    high, high_info = controller.choose(
        quality, style, np.ones(3, bool), rating=2400, deterministic=True
    )
    assert low_info["chosen_regret"] >= high_info["chosen_regret"]
    assert low != high


def test_sparse_afterstates_round_trip_exactly() -> None:
    from drmc_rl.human.afterstate_sim import decode_sparse_deltas, encode_sparse_deltas

    roots = np.full((2, 128), 0xFF, dtype=np.uint8)
    roots[1, -1] = 0xD1
    counts = np.asarray([2, 3])
    fields = np.repeat(roots, counts, axis=0)
    fields[0, 120:122] = (0x80, 0x90)
    fields[1, 3] = 0xD2
    fields[2, 127] = 0xFF
    fields[3, 64:68] = 0xD0
    offsets, cells, values = encode_sparse_deltas(roots, counts, fields)
    np.testing.assert_array_equal(
        decode_sparse_deltas(roots[0], 0, 2, offsets, cells, values), fields[:2]
    )
    np.testing.assert_array_equal(
        decode_sparse_deltas(roots[1], 2, 5, offsets, cells, values), fields[2:]
    )
