from __future__ import annotations

import numpy as np
import pytest

from envs.backends.drmario_pool import is_library_present

torch = pytest.importorskip("torch")

from models.policy.search_policy import (  # noqa: E402
    _BLOCK_ALIVE,
    _BLOCK_INVALID,
    _BLOCK_TERMINAL,
    SearchPolicy,
    backup_block_values,
)

pytestmark = pytest.mark.skipif(
    not is_library_present(),
    reason="cpp-pool library missing (build with: python -m tools.build_drmario_pool)",
)


# --------------------------------------------------------------- pure backup
def test_backup_max_over_leaves_deterministic_ordering() -> None:
    status = np.array([_BLOCK_ALIVE, _BLOCK_ALIVE, _BLOCK_ALIVE])
    depth1 = np.array([0.5, 0.9, 0.1])
    leaves = [[0.2, 0.7, 0.3], [0.6, 0.65], [1.4, -2.0]]
    out = backup_block_values(status, depth1, leaves)
    assert out.tolist() == [0.7, 0.65, 1.4]
    assert int(np.argmax(out)) == 2


def test_backup_discounted_reward_augmented() -> None:
    status = np.array([_BLOCK_ALIVE, _BLOCK_ALIVE])
    depth1 = np.array([0.0, 0.0])
    leaves = [[1.0, 2.0], [4.0]]
    r1 = np.array([0.5, 0.25])
    disc1 = np.array([0.9, 0.5])
    out = backup_block_values(status, depth1, leaves, r1=r1, disc1=disc1)
    assert out[0] == pytest.approx(0.5 + 0.9 * 2.0)
    assert out[1] == pytest.approx(0.25 + 0.5 * 4.0)


def test_backup_keeps_depth1_value_when_no_leaves() -> None:
    status = np.array([_BLOCK_ALIVE, _BLOCK_ALIVE])
    depth1 = np.array([0.5, 0.9])
    out = backup_block_values(status, depth1, [[], [0.1]])
    assert out.tolist() == [0.5, 0.1]


def test_backup_terminal_and_invalid_blocks() -> None:
    status = np.array([_BLOCK_TERMINAL, _BLOCK_INVALID, _BLOCK_ALIVE])
    depth1 = np.array([8.0, -np.inf, 0.2])
    # Leaves listed for non-alive blocks must be ignored.
    out = backup_block_values(status, depth1, [[99.0], [99.0], [0.4]])
    assert out[0] == 8.0
    assert out[1] == -np.inf
    assert out[2] == 0.4
    assert int(np.argmax(out)) == 0


# --------------------------------------------------------------- fast aux v1
def test_aux_v1_fast_matches_adapter_shim() -> None:
    import envs.specs.ram_to_state as ram_specs
    from models.policy.search_policy import aux_v1_batch_fast
    from tools.eval_policy import _make_aux_builder

    ram_specs.set_state_representation("bitplane_bottle_conn_mask")
    shim = _make_aux_builder(57)
    rng = np.random.default_rng(0)
    B = 7
    obs = (rng.random((B, 12, 16, 8)) > 0.6).astype(np.float32)
    # Viruses must be a subset of colored cells (as in real obs).
    obs[:, 3] *= obs[:, :3].max(axis=1)
    infos = [
        {
            "pill/speed_setting": int(rng.integers(0, 3)),
            "speed_setting": 2,
            "level": int(rng.integers(-16, 22)),
            "task_mode": "viruses",
            "task/frames_used": int(rng.integers(0, 20000)),
            "drm/viruses_initial": int(rng.integers(0, 80)),
            "viruses_remaining": int(rng.integers(0, 80)),
            "placements/options": int(rng.integers(0, 200)),
        }
        for _ in range(B)
    ]
    fast = aux_v1_batch_fast(obs, infos)
    ref = shim._build_aux_batch(obs, infos)
    np.testing.assert_allclose(fast, ref, atol=1e-6)


# ------------------------------------------------------------- decide() paths
def _tiny_search(**kw) -> SearchPolicy:
    from models.policy.candidate_policy import CandidatePlacementPolicyNet

    torch.manual_seed(0)
    net = CandidatePlacementPolicyNet(
        in_channels=12,
        board_channels=8,
        board_encoder="cnn",
        encoder_blocks=0,
        d_model=32,
        pill_embed_dim=32,
        pill_embed_type="ordered_pair",
        aux_dim=0,
        pos_embed_dim=8,
        cost_embed_dim=8,
        cand_hidden_dim=32,
        patch_kernel=3,
    )
    return SearchPolicy.from_net(net, candidate_max=64, num_sim_envs=16, **kw)


def _root_state(sp: SearchPolicy):
    board = np.full((16, 8), 0xFF, dtype=np.uint8)
    board[12:, 1] = 0xD0
    board[13:, 5] = 0xD1
    board = board.reshape(-1)
    from envs.backends.drmario_pool import build_reset_spec

    spec = build_reset_spec(
        level=5,
        speed_setting=2,
        rng_state=(7, 9),
        rng_override=True,
        checkpoint_enabled=True,
        checkpoint_board=board,
        checkpoint_falling_colors=(0, 1),
        checkpoint_preview_colors=(2, 0),
    )
    sp._runner.reset(None, [spec] * sp.num_sim_envs)
    mask = sp._runner.buffers.feasible_mask[0].copy()
    cost = sp._runner.buffers.cost_to_lock[0].copy()
    return board, mask, cost


def test_decide_full_depth_returns_feasible_action_and_info() -> None:
    sp = _tiny_search(beam=4, deadline_ms=10_000.0)
    board, mask, cost = _root_state(sp)
    action, info = sp.decide(board, (0, 1), (2, 0), mask, cost, 2, 0, 5)
    assert action >= 0 and bool(mask[action])
    assert info["stage"] == "depth2"
    assert info["fallback_action"] >= 0 and bool(mask[info["fallback_action"]])
    assert info["nodes_expanded"] > 4  # ply-1 branches + at least some leaves
    assert info["value_best"] is not None
    assert info["value_best"] >= info["value_fallback"] - 1e-6
    assert info["agreed_with_policy"] == (action == info["fallback_action"])
    # Deterministic despite fresh sim seeds (leaf marginalization).
    action2, info2 = sp.decide(board, (0, 1), (2, 0), mask, cost, 2, 0, 5)
    assert action2 == action
    assert info2["value_best"] == pytest.approx(info["value_best"], abs=1e-4)
    sp.close()


def test_decide_deadline_zero_falls_back_to_policy_argmax() -> None:
    sp = _tiny_search(beam=4, deadline_ms=10_000.0)
    board, mask, cost = _root_state(sp)
    full_action, full_info = sp.decide(board, (0, 1), (2, 0), mask, cost, 2, 0, 5)
    action, info = sp.decide(board, (0, 1), (2, 0), mask, cost, 2, 0, 5, deadline_ms=0.0)
    assert info["stage"] == "fallback"
    assert action == info["fallback_action"] == full_info["fallback_action"]
    assert info["agreed_with_policy"] is True
    assert info["nodes_expanded"] == 0
    sp.close()


def test_decide_empty_mask_returns_minus_one() -> None:
    sp = _tiny_search(beam=4, deadline_ms=10_000.0)
    board, mask, cost = _root_state(sp)
    action, info = sp.decide(board, (0, 1), (2, 0), np.zeros_like(mask), cost, 2, 0, 5)
    assert action == -1
    assert info["stage"] == "no-feasible"
    sp.close()
