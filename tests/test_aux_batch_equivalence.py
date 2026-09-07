"""Batched aux_v1 must be output-identical to the per-env builder."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from drmc_rl.game.specs import ram_to_state as ram_specs  # noqa: F401  (layout import)
from drmc_rl.training.algo.ppo_smdp import SMDPPPOAdapter


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


def test_public_zero_aux_never_reads_private_context():
    from tools.eval_policy import _make_aux_builder
    from drmc_rl.training.envs.drmario_vs_vec import DrMarioVsPoolVecEnv

    class PrivateInfo(dict):
        def get(self, *args, **kwargs):
            raise AssertionError("public policy read private context")

    adapter = _make_aux_builder(72, aux_spec="zero_v1_vs")
    private = PrivateInfo()
    obs = np.full((2, 20, 16, 8), np.nan, dtype=np.float32)
    np.testing.assert_array_equal(adapter._build_aux_batch(obs, [private] * 2), np.zeros((2, 72)))
    np.testing.assert_array_equal(adapter._build_aux(obs[0], private), np.zeros(72))
    np.testing.assert_array_equal(adapter._build_aux_v1(obs[0], private), np.zeros(72))
    # No native runner or hidden buffers exist on this instance: the direct
    # learner/opponent path must return before consulting any of them.
    env = DrMarioVsPoolVecEnv.__new__(DrMarioVsPoolVecEnv)
    np.testing.assert_array_equal(env._build_direct_aux(np.array([0, 3]), "zero_v1_vs"), np.zeros((2, 72)))


def test_aux_builder_rejects_incompatible_checkpoint_contract():
    from tools.eval_policy import _make_aux_builder

    with pytest.raises(ValueError, match="does not match width"):
        _make_aux_builder(57, aux_spec="zero_v1_vs")
