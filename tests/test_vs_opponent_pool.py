from __future__ import annotations

import numpy as np
import pytest

import drmc_rl.game.specs.ram_to_state as ram_specs
from drmc_rl.envs.backends.drmario_pool import is_library_present
from drmc_rl.training.envs.vs_opponents import _PFSP_MAX_WEIGHT, OpponentPool


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


# Tiny candidate policy config (arch-valid, fast to build) used both to build
# the random opponent net and as the cfg payload for snapshot round trips.
TINY_CFG = {
    "smdp_ppo": {
        "policy_type": "candidate",
        "aux_spec": "none",
        "candidate_board_channels": 4,
        "candidate_board_encoder": "cnn",
        "encoder_blocks": 1,
        "candidate_d_model": 32,
        "pill_embed_dim": 16,
        "pill_embed_type": "ordered_pair",
        "candidate_pos_embed_dim": 8,
        "candidate_cost_embed_dim": 8,
        "candidate_hidden_dim": 32,
        "candidate_transformer_layers": 1,
        "candidate_transformer_heads": 2,
        "candidate_transformer_ff_mult": 2,
        "candidate_patch_kernel": 3,
        "candidate_max_candidates": 32,
    }
}


def _tiny_net():
    from tools.eval_policy import _build_net_from_cfg

    return _build_net_from_cfg(TINY_CFG, 12, "cpu")


# ------------------------------------------------------------------ PFSP math
def test_pfsp_unplayed_gets_max_weight(tmp_path) -> None:
    pool = OpponentPool(tmp_path / "pool")
    played = pool.add_loaded("played", object())
    played.wins, played.games = 5.0, 10
    unplayed = pool.add_loaded("unplayed", object())

    assert unplayed.pfsp_weight() == pytest.approx(_PFSP_MAX_WEIGHT)
    assert unplayed.pfsp_weight() >= played.pfsp_weight()

    w = pool.pfsp_weights()
    assert w.shape == (2,)
    assert w.sum() == pytest.approx(1.0)
    assert w[1] >= w[0]


def test_pfsp_even_opponent_outweighs_dominated_one(tmp_path) -> None:
    pool = OpponentPool(tmp_path / "pool")
    even = pool.add_loaded("even", object())
    even.wins, even.games = 5.0, 10  # learner wins 50%
    crushed = pool.add_loaded("crushed", object())
    crushed.wins, crushed.games = 19.0, 20  # learner wins 95%

    w = pool.pfsp_weights()
    assert w[0] > w[1]
    # The floor keeps the dominated opponent sampled occasionally.
    assert w[1] > 0.0


def test_pfsp_sample_prefers_even_opponent(tmp_path) -> None:
    pool = OpponentPool(tmp_path / "pool")
    even = pool.add_loaded("even", object())
    even.wins, even.games = 50.0, 100
    crushed = pool.add_loaded("crushed", object())
    crushed.wins, crushed.games = 95.0, 100

    rng = np.random.default_rng(0)
    picks = [pool.sample(rng).id for _ in range(200)]
    assert picks.count("even") > picks.count("crushed")
    assert picks.count("crushed") > 0  # floor: never starved


# ----------------------------------------------------------- manifest persistence
def test_manifest_roundtrip(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from drmc_rl.training.utils.checkpoint_io import save_checkpoint

    seed_ckpt = tmp_path / "seed_step100.pt.gz"
    save_checkpoint({"state_dict": {"w": torch.zeros(1)}, "cfg": TINY_CFG, "step": 100}, seed_ckpt)

    pool_dir = tmp_path / "pool"
    pool = OpponentPool(pool_dir, max_pool=12)
    pool.seed([seed_ckpt], protected=[True])
    assert len(pool.entries) == 1
    entry = pool.entries[0]
    assert entry.protected
    assert entry.path is not None and entry.path.parent == pool_dir

    pool.record(entry.id, True)
    pool.record(entry.id, 0.5)
    pool.record(entry.id, False)

    reloaded = OpponentPool(pool_dir, max_pool=12)
    assert [e.id for e in reloaded.entries] == [entry.id]
    re_entry = reloaded.entries[0]
    assert re_entry.protected
    assert re_entry.wins == pytest.approx(1.5)
    assert re_entry.games == 3
    assert re_entry.path.is_file()


def test_afterstate_manifest_roundtrip_preserves_explicit_head(tmp_path) -> None:
    checkpoint = tmp_path / "v3.pt.gz"
    checkpoint.write_bytes(b"fixture")

    pool_dir = tmp_path / "pool"
    pool = OpponentPool(pool_dir, max_pool=12)
    pool.seed_afterstates(
        [
            {
                "id": "v3-style",
                "checkpoint": checkpoint,
                "selection": "style",
                "rating": 1875,
                "rating_sd": 75,
            }
        ]
    )
    entry = pool.entries[0]
    assert entry.kind == "afterstate"
    assert entry.selection == "style"
    assert entry.rating == 1875
    assert entry.protected

    reloaded = OpponentPool(pool_dir, max_pool=12)
    restored = reloaded.entries[0]
    assert restored.kind == "afterstate"
    assert restored.selection == "style"
    assert restored.rating == 1875
    assert restored.rating_sd == 75


def test_afterstate_seed_rejects_implicit_or_unknown_selection(tmp_path) -> None:
    checkpoint = tmp_path / "v3.pt.gz"
    checkpoint.write_bytes(b"fixture")
    pool = OpponentPool(tmp_path / "pool")

    with pytest.raises(ValueError, match="selection"):
        pool.seed_afterstates(
            [{"checkpoint": checkpoint, "selection": "strength-controller"}]
        )


def test_snapshot_eviction_keeps_protected_seed(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from drmc_rl.training.utils.checkpoint_io import save_checkpoint

    seed_ckpt = tmp_path / "champ_step1.pt.gz"
    save_checkpoint({"state_dict": {"w": torch.zeros(1)}, "cfg": TINY_CFG, "step": 1}, seed_ckpt)

    pool = OpponentPool(tmp_path / "pool", max_pool=2)
    pool.seed([seed_ckpt], protected=[True])
    sd = {"w": torch.zeros(1)}
    e1 = pool.snapshot(sd, TINY_CFG, step=10)
    e2 = pool.snapshot(sd, TINY_CFG, step=20)
    e3 = pool.snapshot(sd, TINY_CFG, step=30)

    ids = [e.id for e in pool.entries]
    assert len(pool.entries) == 2
    assert pool.entries[0].protected  # champion survives eviction
    assert ids == [pool.entries[0].id, e3.id]
    # Evicted snapshot files are removed from the pool dir.
    assert not e1.path.is_file()
    assert not e2.path.is_file()
    assert e3.path.is_file()


# ------------------------------------------------------------------ env smoke
@pytest.mark.skipif(
    not _has_vspool_symbols(),
    reason="cpp-vs-pool library missing (build with: make -C vendor/drmario_native libdrmario_pool)",
)
def test_opponent_pool_env_smoke(tmp_path) -> None:
    pytest.importorskip("torch")
    prev_repr = ram_specs.get_state_representation()
    from drmc_rl.training.envs.drmario_vs_vec import DrMarioVsPoolVecEnv

    net, aux_dim, cand_max = _tiny_net()
    pool = OpponentPool(tmp_path / "pool", max_pool=4)
    pool.add_loaded("rand", net, aux_dim=aux_dim, candidate_max=cand_max)

    num_pairs = 2
    env = DrMarioVsPoolVecEnv(
        num_pairs=num_pairs,
        state_repr="bitplane_bottle_conn_mask",
        level=0,
        speed_setting=2,
        randomize_rng=True,
        opponent_pool_cfg={"enabled": True, "pool": pool, "snapshot_every_matches": 1},
    )
    rng = np.random.default_rng(0)
    try:
        # Pool mode exposes only the learner sides.
        assert env.num_envs == num_pairs

        obs, infos = env.reset(seed=0)
        assert obs.shape == (num_pairs, 12, 16, 8)
        assert len(infos) == num_pairs
        for info in infos:
            assert int(info["vs/side"]) == 0
        assert all(e is not None for e in env._pair_opponents)

        def learner_actions(infos_):
            actions = []
            for info in infos_:
                mask = np.asarray(info["placements/feasible_mask"], dtype=np.uint8)
                idxs = np.flatnonzero(mask.reshape(-1))
                if bool(info.get("placements/needs_action")) and idxs.size > 0:
                    actions.append(int(rng.choice(idxs)))
                else:
                    actions.append(-1)
            return np.asarray(actions, dtype=np.int32)

        opponent_ids_seen = []
        matches_target = 2
        for _ in range(5000):
            obs, rewards, terminated, truncated, infos = env.step(learner_actions(infos))
            assert obs.shape == (num_pairs, 12, 16, 8)
            assert np.asarray(rewards).shape == (num_pairs,)
            assert len(infos) == num_pairs
            done = np.asarray(terminated) | np.asarray(truncated)
            for i in range(num_pairs):
                if done[i]:
                    assert "episode" in infos[i]
                    assert "drm" in infos[i]
                    opponent_ids_seen.append(infos[i].get("vs/opponent_id"))
            if env.get_vs_metrics().get("vs/matches_total", 0.0) >= matches_target:
                break

        metrics = env.get_vs_metrics()
        assert metrics["vs/matches_total"] >= matches_target
        assert metrics["vs/opponent_pool"] == 1.0
        assert "vs/win_rate" in metrics
        assert "vs/pool_winrate_min" in metrics and "vs/pool_winrate_max" in metrics

        # Opponent results were recorded and pairs resampled an opponent.
        entry = pool.get("rand")
        assert entry.games >= matches_target
        assert all(oid == "rand" for oid in opponent_ids_seen)
        assert all(e is not None for e in env._pair_opponents)

        # Pool mode grades only the learner side (one record per match).
        games = env.pop_skill_games()
        assert len(games) == int(metrics["vs/matches_total"])

        # Snapshot trigger: due after >= 1 match; writes a loadable checkpoint.
        assert env.maybe_snapshot(lambda: net.state_dict(), cfg=TINY_CFG, step=123)
        assert len(pool.entries) == 2
        snap = pool.entries[-1]
        assert snap.path is not None and snap.path.is_file()
        pool.ensure_loaded(snap)
        assert snap.net is not None
        # Counter reset: not due again until the next match completes.
        assert not env.maybe_snapshot(lambda: net.state_dict(), cfg=TINY_CFG, step=124)
    finally:
        env.close()
        ram_specs.set_state_representation(prev_repr)
