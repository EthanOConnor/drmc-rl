from __future__ import annotations

import numpy as np
import pytest

import envs.specs.ram_to_state as ram_specs
from envs.backends.drmario_pool import is_library_present
from training.envs.dr_mario_vec import VecEnvConfig, make_vec_env


@pytest.mark.skipif(
    not is_library_present(),
    reason="cpp-pool library missing (build with: python -m tools.build_drmario_pool)",
)
def test_cpp_pool_reset_step_smoke() -> None:
    prev_repr = ram_specs.get_state_representation()
    cfg = VecEnvConfig(
        id="DrMarioPlacementEnv-v0",
        obs_mode="state",
        num_envs=2,
        frame_stack=1,
        render=False,
        randomize_rng=True,
        backend="cpp-pool",
        state_repr="bitplane_bottle_mask",
        level=10,
        vectorization="sync",
        emit_raw_ram=False,
    )
    env = make_vec_env(cfg)
    try:
        obs, infos = env.reset(seed=0)
        assert isinstance(infos, (list, tuple))
        assert len(infos) == cfg.num_envs
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (cfg.num_envs, 8, 16, 8)

        actions = []
        for info in infos:
            mask = np.asarray(info.get("placements/feasible_mask"), dtype=np.uint8)
            assert mask.shape == (4, 16, 8)
            idxs = np.flatnonzero(mask.reshape(-1))
            assert idxs.size > 0
            actions.append(int(idxs[0]))
        actions_arr = np.asarray(actions, dtype=np.int32)

        obs2, rewards, terminated, truncated, infos2 = env.step(actions_arr)
        assert isinstance(obs2, np.ndarray) and obs2.shape == obs.shape
        assert np.asarray(rewards).shape == (cfg.num_envs,)
        assert np.asarray(terminated).shape == (cfg.num_envs,)
        assert np.asarray(truncated).shape == (cfg.num_envs,)
        assert isinstance(infos2, (list, tuple)) and len(infos2) == cfg.num_envs
        for info in infos2:
            tau = int(info.get("placements/tau", 1))
            assert tau >= 1
            assert isinstance(info.get("next_pill_colors"), np.ndarray)
            assert "preview_pill" in info

        # Invalid action should not advance and should be surfaced in info.
        bad = np.full((cfg.num_envs,), 512, dtype=np.int32)
        _obs3, rewards3, term3, trunc3, infos3 = env.step(bad)
        assert np.allclose(np.asarray(rewards3, dtype=np.float32), 0.0)
        assert not bool(np.any(np.asarray(term3)))
        assert not bool(np.any(np.asarray(trunc3)))
        for info in infos3:
            assert int(info.get("placements/invalid_action")) == 512
    finally:
        if hasattr(env, "close"):
            env.close()
        ram_specs.set_state_representation(prev_repr)
