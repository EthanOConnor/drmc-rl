"""Opponent-board observability: VS obs layout, v1_vs aux, checkpoint surgery."""

from __future__ import annotations

import numpy as np
import pytest

import drmc_rl.game.specs.ram_to_state as ram_specs
from drmc_rl.envs.backends.drmario_pool import is_library_present


def _has_vspool_symbols() -> bool:
    if not is_library_present():
        return False
    try:
        import ctypes

        from drmc_rl.envs.backends.drmario_pool import resolve_library_path

        lib = ctypes.CDLL(str(resolve_library_path()))
        return hasattr(lib, "drm_vspool_create")
    except Exception:
        return False


needs_vspool = pytest.mark.skipif(
    not _has_vspool_symbols(),
    reason="cpp-vs-pool library missing (build with: make -C vendor/drmario_native libdrmario_pool)",
)


def _random_feasible_actions(infos, rng) -> np.ndarray:
    actions = []
    for info in infos:
        mask = np.asarray(info.get("placements/feasible_mask"), dtype=np.uint8)
        idxs = np.flatnonzero(mask.reshape(-1))
        if bool(info.get("placements/needs_action")) and idxs.size > 0:
            actions.append(int(rng.choice(idxs)))
        else:
            actions.append(-1)
    return np.asarray(actions, dtype=np.int32)


def _make_aux_shim(spec: str):
    from drmc_rl.training.algo.ppo_smdp import _AUX_DIM_BY_SPEC, SMDPPPOAdapter

    shim = SMDPPPOAdapter.__new__(SMDPPPOAdapter)
    shim.aux_spec = spec
    shim.aux_dim = int(_AUX_DIM_BY_SPEC[spec])
    return shim


@needs_vspool
def test_vs_opponent_obs_layout_and_scalars() -> None:
    prev_repr = ram_specs.get_state_representation()
    from drmc_rl.training.envs.drmario_vs_vec import DrMarioVsPoolVecEnv

    num_pairs = 2
    env = DrMarioVsPoolVecEnv(
        num_pairs=num_pairs,
        state_repr="bitplane_bottle_conn_mask_vs",
        level=5,
        speed_setting=2,
        randomize_rng=True,
    )
    rng = np.random.default_rng(0)
    try:
        obs, infos = env.reset(seed=0)
        assert obs.shape == (2 * num_pairs, 20, 16, 8)

        buf = env._runner.buffers
        for _ in range(30):
            for i in range(env.num_envs):
                opp = i ^ 1
                # Opponent planes == opponent side's own planes (post symmetry
                # reduction), same env step.
                np.testing.assert_array_equal(obs[i, 8:16], obs[opp, 0:8])
                # Feasible planes live at 16..19 (pre-reduction superset for
                # same-color pills, exact otherwise; same semantics as 12ch).
                mask = np.asarray(infos[i]["placements/feasible_mask"], dtype=np.float32)
                colors = np.asarray(infos[i]["next_pill_colors"])
                np.testing.assert_array_equal(obs[i, 16:18], mask[:2])
                if int(colors[0]) != int(colors[1]):
                    np.testing.assert_array_equal(obs[i, 18:20], mask[2:])
                else:
                    assert np.all(obs[i, 18:20] >= mask[2:])
                # Opponent scalars consistent with vspool exported state.
                assert int(infos[i]["vs/garbage_pending"]) == int(buf.garbage_pending[i])
                assert int(infos[i]["vs/garbage_pending_opp"]) == int(buf.garbage_pending[opp])
                assert int(infos[i]["vs/opponent_viruses_remaining"]) == int(buf.viruses_rem[opp])
                np.testing.assert_array_equal(
                    infos[i]["vs/opponent_pill_colors"], buf.pill_colors[opp]
                )
                np.testing.assert_array_equal(
                    infos[i]["vs/opponent_preview_colors"], buf.preview_colors[opp]
                )
            actions = _random_feasible_actions(infos, rng)
            obs, _r, _t, _tr, infos = env.step(actions)
    finally:
        env.close()
        ram_specs.set_state_representation(prev_repr)


@needs_vspool
def test_vs_obs_flag_off_unchanged() -> None:
    prev_repr = ram_specs.get_state_representation()
    from drmc_rl.training.envs.drmario_vs_vec import DrMarioVsPoolVecEnv

    env = DrMarioVsPoolVecEnv(num_pairs=1, state_repr="bitplane_bottle_conn_mask", level=0)
    try:
        obs, _infos = env.reset(seed=0)
        assert obs.shape == (2, 12, 16, 8)
        assert env.single_observation_space.shape == (12, 16, 8)
    finally:
        env.close()
        ram_specs.set_state_representation(prev_repr)


def test_aux_v1_vs_extends_v1() -> None:
    pytest.importorskip("torch")
    prev_repr = ram_specs.get_state_representation()
    ram_specs.set_state_representation("bitplane_bottle_conn_mask_vs")
    try:
        _check_aux_v1_vs()
    finally:
        ram_specs.set_state_representation(prev_repr)


def _check_aux_v1_vs() -> None:
    from drmc_rl.training.algo.ppo_smdp import _AUX_V1_DIM, _AUX_V1_VS_DIM

    shim_v1 = _make_aux_shim("v1")
    shim_vs = _make_aux_shim("v1_vs")

    rng = np.random.default_rng(1)
    B = 4
    obs = (rng.random((B, 20, 16, 8)) < 0.2).astype(np.float32)
    infos = []
    for _ in range(B):
        infos.append(
            {
                "pill/speed_setting": 2,
                "level": int(rng.integers(0, 21)),
                "task/frames_used": int(rng.integers(0, 4000)),
                "placements/options": int(rng.integers(0, 200)),
                "viruses_remaining": int(rng.integers(0, 10)),
                "vs/opponent_viruses_remaining": int(rng.integers(0, 60)),
                "vs/garbage_pending": int(rng.integers(0, 5)),
                "vs/garbage_pending_opp": int(rng.integers(0, 5)),
                "vs/opponent_pill_colors": rng.integers(0, 3, size=2),
                "vs/opponent_preview_colors": rng.integers(0, 3, size=2),
            }
        )
    # 1P-style info (no vs keys): the vs block must be all zeros.
    infos[0] = {"pill/speed_setting": 1}

    batched = shim_vs._build_aux_batch(obs, infos)
    assert batched.shape == (B, _AUX_V1_VS_DIM)
    v1 = shim_v1._build_aux_batch(obs, infos)
    np.testing.assert_allclose(batched[:, :_AUX_V1_DIM], v1, atol=1e-6)
    np.testing.assert_array_equal(batched[0, _AUX_V1_DIM:], 0.0)

    for i in range(B):
        single = shim_vs._build_aux_v1(obs[i], infos[i])
        np.testing.assert_allclose(batched[i], single, atol=1e-6, err_msg=f"env {i}")

    i = 1
    info = infos[i]
    vs_block = batched[i, _AUX_V1_DIM:]
    assert vs_block[0] == pytest.approx(min(1.0, info["vs/opponent_viruses_remaining"] / 84.0))
    assert vs_block[1] == pytest.approx(min(1.0, info["vs/garbage_pending"] / 4.0))
    assert vs_block[2] == pytest.approx(min(1.0, info["vs/garbage_pending_opp"] / 4.0))
    pill = np.asarray(info["vs/opponent_pill_colors"]).reshape(-1)
    preview = np.asarray(info["vs/opponent_preview_colors"]).reshape(-1)
    onehot = np.zeros(12, dtype=np.float32)
    onehot[int(pill[0])] = 1.0
    onehot[3 + int(pill[1])] = 1.0
    onehot[6 + int(preview[0])] = 1.0
    onehot[9 + int(preview[1])] = 1.0
    np.testing.assert_array_equal(vs_block[3:], onehot)


def test_expand_checkpoint_preserves_outputs() -> None:
    torch = pytest.importorskip("torch")
    from drmc_rl.models.policy.candidate_policy import CandidatePlacementPolicyNet
    from tools.expand_checkpoint import expand_checkpoint_payload

    torch.manual_seed(0)
    common = dict(
        board_encoder="cnn",
        encoder_blocks=1,
        d_model=32,
        pill_embed_dim=16,
        pill_embed_type="ordered_pair",
        pos_embed_dim=8,
        cost_embed_dim=8,
        cand_hidden_dim=32,
        patch_kernel=3,
    )
    old_net = CandidatePlacementPolicyNet(
        in_channels=12, board_channels=8, aux_dim=57, **common
    )
    new_net = CandidatePlacementPolicyNet(
        in_channels=20, board_channels=16, aux_dim=72, **common
    )

    payload = {
        "state_dict": old_net.state_dict(),
        "ema_state_dict": {k: v.clone() for k, v in old_net.state_dict().items()},
        "optimizer": {"dummy": True},
        "cfg": {"smdp_ppo": {"candidate_board_channels": 8}},
    }
    expand_checkpoint_payload(
        payload,
        new_board_channels=16,
        new_aux_dim=72,
        new_smdp_cfg={"candidate_board_channels": 16, "aux_spec": "v1_vs"},
    )
    assert "optimizer" not in payload
    assert payload["cfg"]["smdp_ppo"]["candidate_board_channels"] == 16
    new_net.load_state_dict(payload["state_dict"])
    # ema_state_dict expanded too and identical to the expanded state_dict here.
    for k, v in payload["ema_state_dict"].items():
        torch.testing.assert_close(v, payload["state_dict"][k], rtol=0, atol=0)

    rng = np.random.default_rng(2)
    B, K = 5, 16
    obs12 = (rng.random((B, 12, 16, 8)) < 0.25).astype(np.float32)
    # Expanded layout: own 0..7 unchanged, opponent planes 8..15 arbitrary
    # (zero weights make them irrelevant), feasible planes move to 16..19.
    obs20 = np.zeros((B, 20, 16, 8), dtype=np.float32)
    obs20[:, :8] = obs12[:, :8]
    obs20[:, 8:16] = (rng.random((B, 8, 16, 8)) < 0.25).astype(np.float32)
    obs20[:, 16:20] = obs12[:, 8:12]
    aux57 = rng.random((B, 57)).astype(np.float32)
    aux72 = np.concatenate(
        [aux57, rng.random((B, 15)).astype(np.float32)], axis=1
    )

    pills = torch.from_numpy(rng.integers(0, 3, size=(B, 2)))
    previews = torch.from_numpy(rng.integers(0, 3, size=(B, 2)))
    # Interior cells only so partner cells stay in bounds for any orientation.
    o = rng.integers(0, 4, size=(B, K))
    row = rng.integers(1, 15, size=(B, K))
    col = rng.integers(1, 7, size=(B, K))
    cand_actions = torch.from_numpy(((o * 16 + row) * 8 + col).astype(np.int32))
    cand_cost = torch.from_numpy(rng.integers(0, 64, size=(B, K)).astype(np.float32))
    cand_mask = torch.from_numpy(rng.random((B, K)) < 0.8)

    old_net.eval()
    new_net.eval()
    with torch.inference_mode():
        logits_old, value_old = old_net(
            torch.from_numpy(obs12), pills, previews, cand_actions, cand_cost, cand_mask,
            aux=torch.from_numpy(aux57),
        )
        logits_new, value_new = new_net(
            torch.from_numpy(obs20), pills, previews, cand_actions, cand_cost, cand_mask,
            aux=torch.from_numpy(aux72),
        )
    torch.testing.assert_close(logits_new, logits_old, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(value_new, value_old, rtol=1e-5, atol=1e-5)
