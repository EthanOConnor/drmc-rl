from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tools.bench_policy import (  # noqa: E402
    _batch_tensors,
    _bench_candidate,
    _bench_heatmap,
    _synthetic_batch,
)


def test_bench_policy_synthetic_smoke() -> None:
    obs, infos = _synthetic_batch(batch_size=2, seed=0, feasible_count=16)
    device = torch.device("cpu")
    tensors = _batch_tensors(obs, infos, device=device, candidate_max=16)

    heat = _bench_heatmap(tensors, device=device, encoder_blocks=0, repeats=1, warmup=0)
    cand = _bench_candidate(
        tensors,
        device=device,
        board_encoder="cnn",
        candidate_max=16,
        patch_kernel=3,
        repeats=1,
        warmup=0,
    )

    assert heat["params"] > 0
    assert cand["params"] > 0
    assert heat["forward_ms_mean"] >= 0.0
    assert cand["forward_ms_mean"] >= 0.0
    assert cand["feasible_count_mean"] == 16.0
    assert cand["candidate_count_mean"] == 16.0
    assert cand["candidate_truncation_frac"] == 0.0
