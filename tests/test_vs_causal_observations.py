import numpy as np
import pytest
from types import SimpleNamespace

from drmc_rl.envs.backends.drmario_pool import is_library_present
from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner
from drmc_rl.training.envs.drmario_vs_vec import DrMarioVsPoolVecEnv


pytestmark = pytest.mark.skipif(not is_library_present(), reason="native library missing")


def environment(*, public=True, pairs=1, direct=False, opponent_pool_cfg=None):
    return DrMarioVsPoolVecEnv(
        num_pairs=pairs,
        state_repr="bitplane_bottle_conn_mask_vs",
        level=14,
        speed_setting=2,
        public_observations=public,
        direct_policy_batch=direct,
        seed_provider=lambda _: (0x37, 0x91),
        frame_counter_provider=lambda _: 0,
        garbage_reward_coef=0.0,
        opponent_pool_cfg=opponent_pool_cfg,
    )


def test_public_ppo_rejects_legacy_native_observations():
    from drmc_rl.training.algo.ppo_smdp import SMDPPPOAdapter
    from drmc_rl.training.utils.cfg import to_config_node

    cfg = to_config_node(dict(smdp_ppo=dict(aux_spec="zero_v1_vs")))
    with pytest.raises(ValueError, match="public_observations=true"):
        SMDPPPOAdapter(
            cfg,
            SimpleNamespace(opponent_obs=True, public_observations=False),
            None,
            None,
            device="cpu",
        )


def test_vector_public_actor_cannot_observe_an_opponent_commitment():
    envs = [environment() for _ in (0, 1)]
    try:
        initial = [env.reset(seed=7)[0].copy() for env in envs]
        frontier = envs[0]._runner.buffers
        actions = np.flatnonzero(frontier.feasible_mask[1])
        order = np.argsort(frontier.cost_to_lock[1, actions])
        observed = []
        for env, action in zip(envs, actions[order[[0, -1]]], strict=True):
            obs, _, _, _, infos = env.step([-2, int(action)])
            observed.append(obs.copy())
            assert infos[0]["vs/observation_timeline"] == "causal-settled-pair-v1"
            np.testing.assert_array_equal(obs[0, 8:16], initial[0][0, 8:16])
            np.testing.assert_array_equal(infos[0]["vs/opponent_board"], env._public_boards[1])
        np.testing.assert_array_equal(observed[0][0], observed[1][0])
        assert envs[0]._runner.snapshot(0) != envs[1]._runner.snapshot(0)
    finally:
        for env in envs:
            env.close()


@pytest.mark.parametrize("direct", [False, True])
def test_causal_observations_preserve_batched_physics_and_clear_cache_on_reset(direct):
    reference = DrMarioVsPoolRunner(num_pairs=2, max_lock_frames=2048, max_wait_frames=6000)
    causal = environment(public=True, pairs=2, direct=direct)
    try:
        causal.reset(seed=9)
        for pair in range(2):
            reference.restore(pair, causal._runner.snapshot(pair))
        changed = False
        for _ in range(60):
            actions = []
            for side, mask in enumerate(causal._runner.buffers.feasible_mask):
                legal = np.flatnonzero(mask)
                if not len(legal):
                    actions.append(-1)
                    continue
                costs = causal._runner.buffers.cost_to_lock[side, legal]
                actions.append(int(legal[costs.argmax() if side % 2 else costs.argmin()]))
            send = np.asarray(actions, np.int32)
            send[causal._need_action == 0] = -2
            reset = causal._pending_reset.astype(np.uint8)
            specs = causal._build_reset_specs(reset_mask=reset) if reset.any() else None
            reference.step_strict(send, reset if specs is not None else None, specs)
            new = causal.step(actions)
            for pair in range(2):
                assert reference.snapshot(pair) == causal._runner.snapshot(pair)
            np.testing.assert_array_equal(
                new[2], np.repeat(reference.buffers.terminated, 2).astype(bool)
            )
            np.testing.assert_array_equal(
                new[3], np.repeat(reference.buffers.truncated, 2).astype(bool)
            )
            changed |= bool(np.any(new[0][:, 8:16] != new[0][[1, 0, 3, 2], :8]))
            if direct:
                assert not causal.policy_batch("zero_v1_vs").aux.any()
        assert changed
        # Exercise partial autoreset, including an opponent whose snapshot is old.
        causal._pending_reset[:] = [True, False]
        new = causal.step([-2] * 4)
        np.testing.assert_array_equal(new[0][:2, 8:16], new[0][[1, 0], :8])
        assert (causal._public_snapshot_frames[:2] == causal._public_frames[:2]).all()
    finally:
        reference.close()
        causal.close()


@pytest.mark.parametrize("direct,with_opponent", [(False, False), (True, False), (True, True)])
def test_public_native_ppo_updates_only_complete_placement_transitions(
    tmp_path, direct, with_opponent
):
    import torch
    from drmc_rl.training.algo.ppo_smdp import SMDPPPOAdapter
    from drmc_rl.training.utils.cfg import to_config_node

    torch.manual_seed(71)
    cfg = to_config_node(
        dict(
            seed=71,
            logdir=str(tmp_path),
            train=dict(total_steps=1, checkpoint_interval=10**9),
            env=dict(public_observations=True),
            smdp_ppo=dict(
                aux_spec="zero_v1_vs",
                gamma=1.0,
                policy_type="candidate",
                candidate_architecture="g5",
                candidate_board_channels=16,
                candidate_d_model=16,
                encoder_blocks=1,
                pill_embed_dim=8,
                candidate_hidden_dim=24,
                candidate_cross_layers=1,
                candidate_interaction_layers=1,
                candidate_transformer_heads=2,
                candidate_patch_kernel=3,
                candidate_critic_context="candidate_attention",
                decisions_per_update=16,
                num_epochs=1,
                minibatch_size=16,
                compile_mode="off",
            ),
        )
    )
    opponent_cfg = None
    if with_opponent:
        from tools.eval_policy import _build_net_from_cfg

        parent, _, _ = _build_net_from_cfg(cfg.to_dict(), 20, "cpu")
        path = tmp_path / "frozen.pt"
        torch.save(dict(cfg=cfg.to_dict(), state_dict=parent.state_dict()), path)
        opponent_cfg = dict(
            enabled=True,
            dir=str(tmp_path / "pool"),
            seed_paths=[str(path)],
            device="cpu",
            snapshot_every_matches=0,
        )
    env = environment(direct=direct, opponent_pool_cfg=opponent_cfg)
    events, batches = [], []
    try:
        adapter = SMDPPPOAdapter(
            cfg,
            env,
            SimpleNamespace(flush=lambda: None),
            SimpleNamespace(emit=lambda event, **data: events.append((event, data))),
            device="cpu",
        )
        adapter._log_metrics = lambda metrics: None
        update = adapter._update_policy

        def checked_update(batch):
            assert len(batch.actions) == 16
            assert batch.masks.reshape(16, -1).any(1).all()
            assert (batch.taus >= 0).all()
            batches.append(batch)
            np.testing.assert_array_equal(
                np.bincount(batch.env_ids), [16 // env.num_envs] * env.num_envs
            )
            if len(batches) == 1:
                # Exercise the next collection after a parameter update, when
                # another learner's unrecorded drain action may still be active.
                adapter.total_steps = adapter.global_step + 1
            metrics = update(batch)
            assert all(np.isfinite(value) for value in metrics.values())
            return metrics

        adapter._update_policy = checked_update
        adapter.train_forever()
        assert len(batches) == 2 and adapter.decision_step == 32
        report = next(data for event, data in events if event == "update_end")
        assert report["rollout/placement_decisions"] == 16
        assert report["rollout/pair_event_steps"] > 8
        assert report["candidate/truncation_frac"] == 0
    finally:
        env.close()
