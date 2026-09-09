from types import SimpleNamespace

import numpy as np
import pytest

from drmc_rl.training.public_league import PublicOpponentPool, empirical_mixture


def test_default_parent_pool_preserves_rng_and_reuses_the_frozen_core():
    parent = SimpleNamespace(aux_spec="zero_v1_vs", in_channels=20)
    pool = PublicOpponentPool(None, parent, "parent.pt", "cpu")
    rng = np.random.default_rng(1)
    untouched = np.random.default_rng(1)
    assert pool.choose(rng) == "parent" and pool.load("parent") is parent
    assert rng.random() == untouched.random()


def test_population_payoff_uses_complete_unique_pairs_and_rejects_unknown_edges():
    match = dict(id="a-b", a="a", b="b", level=14, pace="normal")
    rows = [
        dict(comparison="a-b", seed=i // 2, side=i % 2, score=1.0, reason="clear")
        for i in range(128)
    ]
    result = empirical_mixture(["a", "b"], [match], rows, level=14, pace="normal")
    assert result["paired_counts"] == [[0, 64], [64, 0]]
    assert result["payoff"][0][1] == pytest.approx(64 / 65)
    assert result["mixture"]["a"] > result["mixture"]["b"]
    with pytest.raises(ValueError, match="insufficient"):
        empirical_mixture(["a", "b"], [match], rows, level=14, pace="normal", speed=0)
    with pytest.raises(ValueError, match="insufficient"):
        empirical_mixture(["a", "b", "c"], [match], rows, level=14, pace="normal")
    rows[-1]["reason"] = "timeout"
    with pytest.raises(ValueError, match="censored"):
        empirical_mixture(["a", "b"], [match], rows, level=14, pace="normal")


def test_frozen_population_identity_tracks_checkpoint_and_adapter_bytes(tmp_path):
    checkpoint = tmp_path / "parent.pt"
    adapter = tmp_path / "adapter.pt"
    checkpoint.write_bytes(b"frozen parent")
    adapter.write_bytes(b"first adapter")
    pool = PublicOpponentPool(
        [
            dict(
                id="adapter",
                weight=1.0,
                checkpoint=str(checkpoint),
                adapter_checkpoint=str(adapter),
            )
        ],
        None,
        checkpoint,
        "cpu",
    )
    original = pool.identities()
    adapter.write_bytes(b"different adapter")
    changed = pool.identities()
    assert original["adapter"]["checkpoint_sha256"] == changed["adapter"]["checkpoint_sha256"]
    assert original["adapter"]["adapter_sha256"] != changed["adapter"]["adapter_sha256"]
