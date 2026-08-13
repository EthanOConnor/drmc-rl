from __future__ import annotations

import numpy as np
import pytest

import drmc_rl.game.specs.ram_to_state as ram_specs
from drmc_rl.envs.backends.drmario_pool import is_library_present
from drmc_rl.training.envs.dr_mario_vec import VecEnvConfig, make_vec_env


def _expected_connection_edges(board: np.ndarray) -> np.ndarray:
    board_arr = np.asarray(board, dtype=np.uint8).reshape(16, 8)
    type_hi = board_arr & 0xF0
    edges = np.zeros((4, 16, 8), dtype=np.float32)
    edges[0] = (type_hi == ram_specs.T_BOTTOM).astype(np.float32)  # connected_up
    edges[1] = (type_hi == ram_specs.T_TOP).astype(np.float32)  # connected_down
    edges[2] = (type_hi == ram_specs.T_RIGHT).astype(np.float32)  # connected_left
    edges[3] = (type_hi == ram_specs.T_LEFT).astype(np.float32)  # connected_right
    return edges


@pytest.mark.skipif(
    not is_library_present(),
    reason="cpp-pool library missing (build with: python -m tools.build_drmario_pool)",
)
@pytest.mark.parametrize(
    ("state_repr", "channels"),
    [("bitplane_bottle_mask", 8), ("bitplane_bottle_conn_mask", 12)],
)
def test_cpp_pool_reset_step_smoke(state_repr: str, channels: int) -> None:
    prev_repr = ram_specs.get_state_representation()
    cfg = VecEnvConfig(
        id="DrMarioPlacementEnv-v0",
        obs_mode="state",
        num_envs=2,
        frame_stack=1,
        render=False,
        randomize_rng=True,
        backend="cpp-pool",
        state_repr=state_repr,
        level=10,
        vectorization="sync",
        emit_raw_ram=(state_repr == "bitplane_bottle_conn_mask"),
    )
    env = make_vec_env(cfg)
    try:
        obs, infos = env.reset(seed=0)
        assert isinstance(infos, (list, tuple))
        assert len(infos) == cfg.num_envs
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (cfg.num_envs, channels, 16, 8)

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
        if state_repr == "bitplane_bottle_conn_mask":
            for env_i, info in enumerate(infos2):
                board = np.asarray(info["board"], dtype=np.uint8)
                expected_edges = _expected_connection_edges(board)
                np.testing.assert_array_equal(obs2[env_i, 4:8], expected_edges)

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
