"""Batched aux_v1 must be output-identical to the per-env builder."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from envs.specs import ram_to_state as ram_specs  # noqa: F401  (layout import)
from training.algo.ppo_smdp import SMDPPPOAdapter


class _StubEnv:
    num_envs = 4


def _make_adapter():
    adapter = SMDPPPOAdapter.__new__(SMDPPPOAdapter)
    adapter.aux_spec = "v1"
    adapter.aux_dim = 57
    return adapter


def test_build_aux_batch_matches_per_env():
    adapter = _make_adapter()
    rng = np.random.default_rng(0)
    B, C = 6, 12
    obs = (rng.random((B, C, 16, 8)) < 0.2).astype(np.float32)
    infos = []
    for i in range(B):
        infos.append(
            {
                "pill/speed_setting": int(rng.integers(0, 3)),
                "curriculum/env_level": int(rng.integers(-15, 21)),
                "task/frames_used": int(rng.integers(0, 4000)),
                "placements/options": int(rng.integers(0, 200)),
                "drm/viruses_initial": int(rng.integers(1, 20)),
                "viruses_remaining": int(rng.integers(0, 10)),
            }
        )
    # env 0: match-mode task; env 1: missing keys entirely
    infos[0]["task_mode"] = "matches"
    infos[0]["matches_completed"] = 3
    infos[0]["match_target"] = 8
    infos[1] = {}

    batched = adapter._build_aux_batch(obs, infos)
    for i in range(B):
        single = adapter._build_aux_v1(obs[i], infos[i])
        np.testing.assert_allclose(batched[i], single, atol=1e-6, err_msg=f"env {i}")
