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
    opportunities = np.tile(np.linspace(0.1, 2.0, 100), 9)
    calibration = RegretCalibration.fit(
        ratings, regrets, opportunities, rating_bins=8, opportunity_bins=3
    )
    assert np.all(np.diff(calibration.regret_quantiles, axis=0) <= 1e-12)
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


def test_regret_calibration_preserves_skill_difference_in_error_tail() -> None:
    from drmc_rl.human.strength import RegretCalibration, RegretStrengthController

    ratings = np.repeat((900.0, 2300.0), 1000)
    # Typical choices are identical. The separation is the low-rated players'
    # occasional consequential error, which a median-only control loses.
    low = np.concatenate((np.full(800, 0.1), np.full(200, 2.0)))
    high = np.concatenate((np.full(800, 0.1), np.full(200, 0.5)))
    calibration = RegretCalibration.fit(
        ratings,
        np.concatenate((low, high)),
        np.ones(2000),
        rating_bins=2,
        opportunity_bins=1,
    )
    low_controller = RegretStrengthController(calibration, seed=11)
    high_controller = RegretStrengthController(calibration, seed=11)
    quality = np.asarray([10.0, 9.5, 8.0])
    mask = np.ones(3, bool)
    style = np.zeros(3)
    low_regrets = []
    high_regrets = []
    for _ in range(500):
        low_slot, _ = low_controller.choose(quality, style, mask, rating=900)
        high_slot, _ = high_controller.choose(quality, style, mask, rating=2300)
        low_regrets.append(quality.max() - quality[low_slot])
        high_regrets.append(quality.max() - quality[high_slot])
    assert np.mean(low_regrets) > np.mean(high_regrets) + 0.15


def test_v3_runtime_style_choice_does_not_use_quality() -> None:
    from drmc_rl.human.afterstate_runtime import AfterstatePolicyRuntime

    runtime = object.__new__(AfterstatePolicyRuntime)
    runtime.rng = np.random.default_rng(2)
    logits = np.asarray([-5.0, 8.0, 0.0])
    assert runtime.choose_style(logits, np.ones(3, bool), temperature=0) == 1
    assert runtime.choose_quality(-logits, np.ones(3, bool)) == 0


def test_v3_runtime_batches_variable_candidate_widths_losslessly() -> None:
    from drmc_rl.human.afterstate_model import AfterstatePolicyNet
    from drmc_rl.human.afterstate_runtime import AfterstatePolicyRuntime
    from drmc_rl.human.afterstate_sim import AfterstateBatch
    from drmc_rl.human.conditioning import HumanSkillCondition

    class Simulator:
        def __init__(self) -> None:
            self.batch_sizes = []

        def simulate_packed(self, **kwargs):
            counts = np.asarray(kwargs["candidate_count"], dtype=np.int64)
            fields = np.repeat(np.asarray(kwargs["fields"], dtype=np.uint8), counts, axis=0)
            total = int(counts.sum())
            self.batch_sizes.append(len(counts))
            return AfterstateBatch(
                fields=fields,
                terminal_reason=np.zeros(total, dtype=np.uint8),
                invalid=np.zeros(total, dtype=np.bool_),
                tau_frames=np.ones(total, dtype=np.uint32),
                viruses_remaining=np.zeros(total, dtype=np.uint16),
                viruses_cleared=np.zeros(total, dtype=np.uint16),
                nonviruses_cleared=np.zeros(total, dtype=np.uint16),
                clear_events=np.zeros(total, dtype=np.uint16),
            )

    torch.manual_seed(8)
    runtime = object.__new__(AfterstatePolicyRuntime)
    runtime.device = torch.device("cpu")
    runtime.condition = HumanSkillCondition.fit(np.asarray([1000.0, 2000.0]))
    runtime.policy = AfterstatePolicyNet(
        condition_dim=40, d_model=32, bottle_blocks=0, candidate_layers=1, heads=4
    ).eval()
    runtime.simulator = Simulator()
    requests = []
    for width, rating in ((3, 1200.0), (5, 1800.0)):
        requests.append(
            {
                "board_planes": np.zeros((8, 16, 8), dtype=np.float32),
                "opponent_board_planes": np.zeros((8, 16, 8), dtype=np.float32),
                "opponent_state_age_frames": 0,
                "pill": np.asarray([0, 1]),
                "preview": np.asarray([1, 2]),
                "candidate_actions": np.arange(width, dtype=np.int64),
                "candidate_costs": np.arange(10, 10 + width, dtype=np.float32),
                "candidate_mask": np.ones(width, dtype=np.bool_),
                "rating": rating,
                "speed": 2,
                "speed_ups": 0,
            }
        )

    batched = runtime.score_batch(requests)
    assert runtime.simulator.batch_sizes == [2]
    scalar = [runtime.score(**request) for request in requests]
    assert runtime.simulator.batch_sizes == [2, 1, 1]
    for width, batch_result, scalar_result in zip((3, 5), batched, scalar):
        assert batch_result["competitive_score"].shape == (width,)
        for key in (
            "competitive_score",
            "human_logits",
            "outcome_logit",
            "clear_logit",
            "topout_logit",
            "virus_delta",
            "attack",
        ):
            np.testing.assert_allclose(batch_result[key], scalar_result[key], rtol=1e-5, atol=1e-6)


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


def test_afterstate_auxiliary_losses_reduce_once_per_decision() -> None:
    from tools.train_afterstate_policy import _masked_row_mean, _weighted_mean

    candidate_losses = torch.tensor([[1.0, 3.0, 100.0], [5.0, 100.0, 100.0]])
    valid = torch.tensor([[True, True, False], [True, False, False]])
    row_losses = _masked_row_mean(candidate_losses, valid)
    torch.testing.assert_close(row_losses, torch.tensor([2.0, 5.0]))
    torch.testing.assert_close(
        _weighted_mean(row_losses, torch.tensor([1.0, 3.0])), torch.tensor(4.25)
    )


def test_afterstate_statistics_balance_players_across_shards(tmp_path) -> None:
    from tools.train_afterstate_policy import _fit_training_statistics
    from tools.train_human_policy import _sample_weights

    common = {
        "split": np.zeros(4, dtype=np.uint8),
        "player_fold": np.ones(4, dtype=np.uint8),
        "time_split": np.zeros(4, dtype=np.uint8),
    }
    np.savez(
        tmp_path / "a.npz",
        **common,
        rating=np.asarray([1000, 1000, 1000, 2000], dtype=np.float32),
        player_key=np.asarray([1, 1, 1, 2], dtype=np.uint64),
    )
    np.savez(
        tmp_path / "b.npz",
        **common,
        rating=np.asarray([1000, 1000, 2000, 2000], dtype=np.float32),
        player_key=np.asarray([1, 1, 2, 3], dtype=np.uint64),
    )
    statistics = _fit_training_statistics(sorted(tmp_path.glob("*.npz")))
    assert statistics.rows == 8
    assert statistics.player_counts == {1: 5, 2: 2, 3: 1}
    weights = _sample_weights(
        np.asarray([1000, 1000, 1000], dtype=np.float32),
        player_keys=np.asarray([1, 2, 3], dtype=np.uint64),
        player_counts=statistics.player_counts,
        rating_edges=statistics.rating_edges,
        rating_counts=statistics.rating_counts,
    )
    assert weights[0] < weights[1] < weights[2]


def test_afterstate_precision_requires_native_bf16() -> None:
    from types import SimpleNamespace

    from tools.train_afterstate_policy import _use_bf16

    pascal = SimpleNamespace(
        cuda=SimpleNamespace(
            is_bf16_supported=lambda: True,
            get_device_capability=lambda _device: (6, 1),
        )
    )
    ampere = SimpleNamespace(
        cuda=SimpleNamespace(
            is_bf16_supported=lambda: True,
            get_device_capability=lambda _device: (8, 6),
        )
    )

    assert not _use_bf16(pascal, device="cuda", precision="auto")
    assert _use_bf16(ampere, device="cuda", precision="auto")
    assert not _use_bf16(ampere, device="cuda", precision="fp32")
    with pytest.raises(ValueError, match="native bf16"):
        _use_bf16(pascal, device="cuda", precision="bf16")


def test_afterstate_shard_cache_keeps_sources_and_rotates_targets(tmp_path) -> None:
    from tools.train_afterstate_policy import _cache_afterstate_shard, _cache_source_shards

    dataset = tmp_path / "remote-dataset"
    afterstates = tmp_path / "remote-afterstates"
    cache = tmp_path / "cache"
    dataset.mkdir()
    afterstates.mkdir()
    sources = []
    for stem in ("a", "b"):
        source = dataset / f"{stem}.npz"
        source.write_bytes(f"source-{stem}".encode())
        sources.append(source)
        for suffix in (
            ".delta_offsets.npy",
            ".delta_cells.bin",
            ".delta_values.bin",
            ".targets.npz",
        ):
            (afterstates / f"{stem}{suffix}").write_bytes(f"{stem}{suffix}".encode())

    cached_sources = _cache_source_shards(sources, cache)
    assert [path.read_bytes() for path in cached_sources] == [b"source-a", b"source-b"]
    first = _cache_afterstate_shard(cached_sources[0], afterstates, cache)
    assert first.name == "a"
    assert (first / "a.targets.npz").is_file()
    second = _cache_afterstate_shard(cached_sources[1], afterstates, cache)
    assert second.name == "b"
    assert not first.exists()
    assert (second / "b.targets.npz").is_file()
