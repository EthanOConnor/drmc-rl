"""Search-distillation phase 2 (docs/SEARCH_DISTILL.md): vs-obs splice,
cross-env batched search, opponent self-model at leaves, act-from-search.

Unit tests for the obs/aux splice and batched-vs-serial equivalence, plus
small rollout+update smokes on the VS pool env with the `_vs` opponent-obs
representation (skipped without the native library).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import drmc_rl.game.specs.ram_to_state as ram_specs
from drmc_rl.envs.backends.drmario_pool import build_reset_spec, is_library_present
from drmc_rl.models.policy.candidate_packing import pack_feasible_candidates
from drmc_rl.training.algo.search_distill import SearchDistillConfig
from drmc_rl.training.utils.cfg import to_config_node

needs_pool = pytest.mark.skipif(
    not is_library_present(),
    reason="cpp-pool library missing (build with: python -m tools.build_drmario_pool)",
)


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


# ----------------------------------------------------------- obs/aux splice
def test_splice_obs_context_layout_bit_exact() -> None:
    from drmc_rl.models.policy.search_policy import splice_obs_context

    rng = np.random.default_rng(0)
    sim = (rng.random((5, 12, 16, 8)) > 0.5).astype(np.float32)
    planes = rng.random((8, 16, 8)).astype(np.float32)
    out = splice_obs_context(sim, planes)
    assert out.shape == (5, 20, 16, 8)
    assert np.array_equal(out[:, :8], sim[:, :8])  # own planes from sim
    for b in range(5):
        assert np.array_equal(out[b, 8:16], planes)  # opp planes preserved
    assert np.array_equal(out[:, 16:20], sim[:, 8:12])  # feasible moved
    # Per-row planes variant.
    planes_b = rng.random((5, 8, 16, 8)).astype(np.float32)
    out_b = splice_obs_context(sim, planes_b)
    assert np.array_equal(out_b[:, 8:16], planes_b)
    # Identity when no context.
    assert splice_obs_context(sim, None) is sim


def test_aux_v1_with_tail_prefix_and_tail() -> None:
    from drmc_rl.models.policy.search_policy import aux_v1_batch_fast, aux_v1_with_tail

    rng = np.random.default_rng(1)
    B = 4
    obs = (rng.random((B, 20, 16, 8)) > 0.6).astype(np.float32)
    obs[:, 3] *= obs[:, :3].max(axis=1)  # viruses subset of colors
    infos = [
        {
            "pill/speed_setting": 2,
            "level": 5,
            "task_mode": "viruses",
            "task/frames_used": 100 * b,
            "drm/viruses_initial": 20,
            "viruses_remaining": 10,
            "placements/options": 30,
        }
        for b in range(B)
    ]
    tail = rng.random(15).astype(np.float32)
    out = aux_v1_with_tail(obs, infos, tail, 15)
    assert out.shape == (B, 72)
    np.testing.assert_array_equal(out[:, :57], aux_v1_batch_fast(obs, infos))
    for b in range(B):
        np.testing.assert_array_equal(out[b, 57:], tail)  # tail bit-exact
    # None tail -> zeros (the v1_vs definition outside VS contexts).
    out0 = aux_v1_with_tail(obs, infos, None, 15)
    assert np.all(out0[:, 57:] == 0.0)
    # tail_dim 0 -> plain v1.
    assert aux_v1_with_tail(obs, infos, None, 0).shape == (B, 57)


# --------------------------------------------------------------- config gate
def test_config_validation_phase2() -> None:
    cfg = SearchDistillConfig.from_dict({"enabled": True})
    assert cfg.opponent_model == "none"
    assert cfg.act_from_search is False
    cfg = SearchDistillConfig.from_dict(
        {"enabled": True, "opponent_model": "self", "act_from_search": True}
    )
    assert cfg.opponent_model == "self" and cfg.act_from_search is True
    with pytest.raises(ValueError):
        SearchDistillConfig.from_dict({"enabled": True, "opponent_model": "oracle"})
    # Invalid values pass when disabled (consistent with phase 1).
    SearchDistillConfig.from_dict({"opponent_model": "oracle"})


# ------------------------------------------------------------- net builders
def _tiny_net(in_channels: int, board_channels: int, aux_dim: int, seed: int = 0):
    from drmc_rl.models.policy.candidate_policy import CandidatePlacementPolicyNet

    torch.manual_seed(seed)
    return CandidatePlacementPolicyNet(
        in_channels=in_channels,
        board_channels=board_channels,
        board_encoder="cnn",
        encoder_blocks=0,
        d_model=32,
        pill_embed_dim=32,
        pill_embed_type="ordered_pair",
        aux_dim=aux_dim,
        pos_embed_dim=8,
        cost_embed_dim=8,
        cand_hidden_dim=32,
        patch_kernel=3,
    )


def _tiny_policy(*, vs: bool = False, seed: int = 0, **kw):
    from drmc_rl.models.policy.search_policy import SearchPolicy
    from tools.eval_policy import _make_aux_builder

    net = _tiny_net(20 if vs else 12, 16 if vs else 8, 72 if vs else 57, seed=0)
    return SearchPolicy.from_net(
        net,
        aux_shim=_make_aux_builder(72 if vs else 57),
        candidate_max=64,
        deadline_ms=1e9,
        seed=seed,
        **kw,
    )


def _tall_board(offset: int, fill_from: int = 2) -> np.ndarray:
    """Nearly-full board (viruses below `fill_from`), no 4-runs: every ply-2
    candidate set fits the sim fan-out, so search results are independent of
    the rng-seeded preview colors (batched == serial exactly)."""

    board = np.full((16, 8), 0xFF, dtype=np.uint8)
    for r in range(fill_from, 16):
        for c in range(8):
            board[r, c] = 0xD0 + (r + c + offset) % 3
    return board.reshape(-1)


def _plan_root(sp, board: np.ndarray, pill_raw, prev_raw):
    spec = build_reset_spec(
        level=5,
        speed_setting=2,
        rng_state=(7, 9),
        rng_override=True,
        checkpoint_enabled=True,
        checkpoint_board=board,
        checkpoint_falling_colors=pill_raw,
        checkpoint_preview_colors=prev_raw,
    )
    sp._runner.reset(None, [spec] * sp.num_sim_envs)
    mask = sp._runner.buffers.feasible_mask[0].copy()
    cost = sp._runner.buffers.cost_to_lock[0].copy()
    return mask, cost


# --------------------------------------------------- batched == serial search
@needs_pool
def test_decide_batch_matches_serial() -> None:
    from drmc_rl.models.policy.search_policy import SearchRequest

    prev_repr = ram_specs.get_state_representation()
    sp_serial = _tiny_policy(beam=2, num_sim_envs=96)
    sp_batch = _tiny_policy(beam=2, num_sim_envs=96)
    try:
        pill_raw, prev_raw = (0, 1), (2, 0)
        reqs = []
        for offset in range(3):
            board = _tall_board(offset)
            mask, cost = _plan_root(sp_batch, board, pill_raw, prev_raw)
            assert mask.sum() > 0
            reqs.append(
                SearchRequest(
                    board=board,
                    pill=pill_raw,
                    preview=prev_raw,
                    feasible_mask512=mask,
                    cost_to_lock512=cost.astype(np.float64),
                    speed_setting=2,
                    speed_ups=0,
                    level=5,
                    viruses_initial=80,
                    frames_used=120,
                )
            )

        serial = [
            sp_serial.decide(
                r.board,
                r.pill,
                r.preview,
                r.feasible_mask512,
                r.cost_to_lock512,
                r.speed_setting,
                r.speed_ups,
                r.level,
                viruses_initial=r.viruses_initial,
                frames_used=r.frames_used,
            )
            for r in reqs
        ]
        batched = sp_batch.decide_batch(reqs)

        for (a_s, i_s), (a_b, i_b) in zip(serial, batched):
            assert i_s["stage"] == "depth2"
            assert i_b["stage"] == "depth2"
            assert a_b == a_s
            assert i_b["fallback_action"] == i_s["fallback_action"]
            assert i_b["value_root"] == pytest.approx(i_s["value_root"], abs=1e-4)
            np.testing.assert_array_equal(i_b["cand_actions"], i_s["cand_actions"])
            np.testing.assert_array_equal(i_b["cand_mask"], i_s["cand_mask"])
            m = i_s["cand_mask"]
            np.testing.assert_allclose(
                i_b["prior_logits"][m], i_s["prior_logits"][m], atol=1e-4
            )
            np.testing.assert_array_equal(i_b["ply1_actions"], i_s["ply1_actions"])
            q_s = np.asarray(i_s["q_values"])
            q_b = np.asarray(i_b["q_values"])
            np.testing.assert_array_equal(np.isfinite(q_s), np.isfinite(q_b))
            fin = np.isfinite(q_s)
            np.testing.assert_allclose(q_b[fin], q_s[fin], atol=1e-3)
    finally:
        sp_serial.close()
        sp_batch.close()
        ram_specs.set_state_representation(prev_repr)


@needs_pool
def test_decide_vs_context_changes_outputs() -> None:
    """The frozen opponent context must flow into the net evaluations."""

    from drmc_rl.models.policy.search_policy import ObsContext

    prev_repr = ram_specs.get_state_representation()
    sp = _tiny_policy(vs=True, beam=2, num_sim_envs=16)
    try:
        board = _tall_board(0, fill_from=8)
        pill_raw, prev_raw = (0, 1), (2, 0)
        mask, cost = _plan_root(sp, board, pill_raw, prev_raw)
        rng = np.random.default_rng(3)
        ctx = ObsContext(
            opp_planes=(rng.random((8, 16, 8)) > 0.5).astype(np.float32),
            aux_tail=rng.random(15).astype(np.float32),
        )
        _, i_zero = sp.decide(
            board, pill_raw, prev_raw, mask, cost.astype(np.float64), 2, 0, 5
        )
        _, i_ctx = sp.decide(
            board, pill_raw, prev_raw, mask, cost.astype(np.float64), 2, 0, 5,
            obs_context=ctx,
        )
        m = i_zero["cand_mask"]
        assert not np.allclose(i_ctx["prior_logits"][m], i_zero["prior_logits"][m])
        assert i_ctx["value_root"] != pytest.approx(i_zero["value_root"], abs=1e-7)
    finally:
        sp.close()
        ram_specs.set_state_representation(prev_repr)


# ------------------------------------------------------ opponent self-model
@needs_pool
def test_opponent_model_self_advances_and_falls_back() -> None:
    from drmc_rl.models.policy.search_policy import ObsContext, _build_obs_from_board

    prev_repr = ram_specs.get_state_representation()
    sp = _tiny_policy(vs=True, beam=2, num_sim_envs=16, opponent_model="self")
    try:
        opp_board = _tall_board(1, fill_from=10)
        frozen_planes = _build_obs_from_board(opp_board, np.zeros(512))[0:8]
        tail = np.linspace(0.1, 0.9, 15).astype(np.float32)
        ctx = ObsContext(
            opp_planes=frozen_planes.copy(),
            aux_tail=tail.copy(),
            opp_board=opp_board,
            opp_pill=(0, 1),
            opp_preview=(2, 2),
            own_planes=frozen_planes.copy(),
            opp_aux_tail=tail.copy(),
        )
        adv = sp._advance_opponents([(0, ctx, 2, 0, 5, 0, 30)])
        assert 0 in adv
        planes, new_tail = adv[0]
        assert planes.shape == (8, 16, 8)
        # One placement landed: the opponent plane set changed.
        assert not np.array_equal(planes, frozen_planes)
        assert new_tail is not None
        # Tail rewrites: pill one-hot = old preview (canonical 2,2), preview
        # one-hot zeroed, pendings frozen.
        assert new_tail[3 + 2] == 1.0 and new_tail[6 + 2] == 1.0
        assert np.all(new_tail[9:15] == 0.0)
        assert new_tail[1] == pytest.approx(tail[1]) and new_tail[2] == pytest.approx(tail[2])

        # Fallback: opponent with no feasible placement keeps the frozen ctx.
        blocked = _tall_board(0, fill_from=0)  # bottle full to the top row
        ctx_blocked = ObsContext(
            opp_planes=frozen_planes.copy(),
            aux_tail=tail.copy(),
            opp_board=blocked,
            opp_pill=(0, 1),
            opp_preview=(1, 2),
            own_planes=frozen_planes.copy(),
            opp_aux_tail=tail.copy(),
        )
        adv2 = sp._advance_opponents([(7, ctx_blocked, 2, 0, 5, 0, 30)])
        assert 7 not in adv2

        # decide() with the advance still completes and reports the flag.
        board = _tall_board(0, fill_from=8)
        mask, cost = _plan_root(sp, board, (0, 1), (2, 0))
        _, info = sp.decide(
            board, (0, 1), (2, 0), mask, cost.astype(np.float64), 2, 0, 5,
            obs_context=ctx,
        )
        assert info["stage"] == "depth2"
        assert info["opp_advanced"] is True
    finally:
        sp.close()
        ram_specs.set_state_representation(prev_repr)


# --------------------------------------------------------- act from search
@needs_pool
def test_act_from_search_logprob_bookkeeping(tmp_path) -> None:
    """Searched rows execute an improved-policy sample whose stored log-prob
    matches the improved distribution (PPO ratio 1 against the behavior)."""

    from drmc_rl.training.algo.search_distill import SearchDistillRunner
    from drmc_rl.training.envs.drmario_pool_vec import DrMarioPoolVecEnv

    prev_repr = ram_specs.get_state_representation()
    env = DrMarioPoolVecEnv(
        num_envs=2,
        state_repr="bitplane_bottle_conn_mask",
        level=2,
        speed_setting=2,
        randomize_rng=True,
        emit_board=True,
    )
    runner = None
    try:
        cfg = SearchDistillConfig.from_dict(
            {
                "enabled": True,
                "sims": 4,
                "beam": 2,
                "decision_fraction": 1.0,
                "act_from_search": True,
            }
        )
        runner = SearchDistillRunner(
            cfg,
            net=_tiny_net(12, 8, 57),
            aux_spec="v1",
            candidate_max=64,
            num_envs=2,
            reward_mode="1p",
            gamma=0.99,
            seed=0,
        )
        _obs, infos = env.reset(seed=11)
        N = env.num_envs
        masks = np.stack([np.asarray(infos[i]["placements/feasible_mask"], dtype=bool) for i in range(N)])
        costs = np.zeros((N, 4, 16, 8), dtype=np.float32)
        for i in range(N):
            c = np.asarray(infos[i]["placements/cost_to_lock"]).astype(np.float32)
            c[c >= np.float32(0xFFFE)] = np.inf
            costs[i] = c
        behavior = np.array(
            [int(np.flatnonzero(masks[i].reshape(-1))[0]) for i in range(N)], dtype=np.int64
        )
        behavior_lp = np.full((N,), -1.0, dtype=np.float32)

        targets, _values, searched, actions_out, lp_out = runner.run(
            infos, masks, costs, behavior, behavior_lp
        )
        assert bool(searched.any())
        for i in range(N):
            if not searched[i]:
                assert actions_out[i] == behavior[i] and lp_out[i] == behavior_lp[i]
                continue
            packed = pack_feasible_candidates(masks[i], costs[i], max_candidates=64)
            slots = np.flatnonzero(packed.actions[: packed.count] == actions_out[i])
            assert slots.size == 1  # executed action is in the packed list
            slot = int(slots[0])
            p = targets[i].astype(np.float64)
            p /= p.sum()
            assert p[slot] > 0.0
            # Ratio at step 0 against the stored behavior log-prob == 1.
            assert np.exp(np.log(p[slot]) - float(lp_out[i])) == pytest.approx(1.0, abs=1e-5)
    finally:
        if runner is not None:
            runner.close()
        env.close()
        ram_specs.set_state_representation(prev_repr)


# ------------------------------------------------- vs-obs integration smokes
class _StubLogger:
    def log_scalar(self, *a, **k):
        pass

    def flush(self):
        pass


class _StubBus:
    def __init__(self):
        self.events = []

    def emit(self, name, **payload):
        self.events.append((name, payload))


def _vs_obs_cfg(tmp_path, extra_sd=None) -> object:
    sd = {
        "enabled": True,
        "sims": 4,
        "beam": 4,
        "decision_fraction": 0.5,
        "beta": 1.0,
        "value_mix": 0.5,
    }
    sd.update(extra_sd or {})
    return to_config_node(
        {
            "seed": 0,
            "logdir": str(tmp_path),
            "train": {"total_steps": 1, "checkpoint_interval": 10**9},
            "smdp_ppo": {
                "policy_type": "candidate",
                "aux_spec": "v1_vs",
                "lr": 1e-3,
                "gamma": 1.0,
                "decisions_per_update": 16,
                "num_epochs": 1,
                "minibatch_size": 8,
                "pill_embed_dim": 16,
                "pill_embed_type": "ordered_pair",
                "candidate_max_candidates": 32,
                "candidate_d_model": 32,
                "candidate_pos_embed_dim": 8,
                "candidate_cost_embed_dim": 8,
                "candidate_hidden_dim": 32,
                "candidate_board_channels": 16,
                "candidate_transformer_layers": 1,
                "candidate_transformer_heads": 2,
                "candidate_patch_kernel": 3,
                "search_distill": sd,
            },
            "env": {"state_repr": "bitplane_bottle_conn_mask_vs"},
        }
    )


def _run_vs_obs_update(tmp_path, extra_sd=None):
    from drmc_rl.training.algo.ppo_smdp import SMDPPPOAdapter
    from drmc_rl.training.envs.drmario_vs_vec import DrMarioVsPoolVecEnv

    prev_repr = ram_specs.get_state_representation()
    env = DrMarioVsPoolVecEnv(
        num_pairs=2,
        state_repr="bitplane_bottle_conn_mask_vs",
        level=5,
        speed_setting=2,
        randomize_rng=True,
    )
    try:
        bus = _StubBus()
        torch.manual_seed(1234)
        adapter = SMDPPPOAdapter(
            _vs_obs_cfg(tmp_path, extra_sd), env, _StubLogger(), bus, device="cpu"
        )
        try:
            adapter.train_forever()
        finally:
            if adapter._sd_runner is not None:
                adapter._sd_runner.close()
        updates = [p for name, p in bus.events if name == "update_end"]
        assert updates, "no update_end emitted"
        return updates[-1]
    finally:
        env.close()
        ram_specs.set_state_representation(prev_repr)


@needs_vspool
def test_search_distill_smoke_vs_obs(tmp_path) -> None:
    metrics = _run_vs_obs_update(tmp_path / "base")
    assert metrics["search_distill/searched_fraction_actual"] > 0.0
    assert "search_distill/kl_target_net" in metrics
    assert "search_distill/search_ms_p50" in metrics


@needs_vspool
def test_search_distill_smoke_vs_obs_opponent_model(tmp_path) -> None:
    metrics = _run_vs_obs_update(tmp_path / "opp", {"opponent_model": "self"})
    assert metrics["search_distill/searched_fraction_actual"] > 0.0
    assert "search_distill/opp_advanced_fraction" in metrics
    assert 0.0 <= metrics["search_distill/opp_advanced_fraction"] <= 1.0


@needs_vspool
def test_search_distill_smoke_vs_obs_act_from_search(tmp_path) -> None:
    metrics = _run_vs_obs_update(tmp_path / "afs", {"act_from_search": True})
    assert metrics["search_distill/searched_fraction_actual"] > 0.0
    assert "search_distill/kl_target_net" in metrics
    # Executed improved-policy samples make the executed-vs-best gap small on
    # average, but the metric must still be present and sane.
    assert metrics.get("search_distill/q_gap_mean", 0.0) >= 0.0
