"""Arm C: the full G5 public core with an additive, zero-initialized afterstate branch."""
from collections import Counter
import os

import numpy as np
import pytest
import torch

from drmc_rl.models.policy.afterstate_full_core import BRANCH_PREFIX, afterstate_full_config, from_g5

LIBRARY = os.environ.get("DRMC_FRAME_LIBRARY")


def _g5_cfg():
    return {"smdp_ppo": dict(
        policy_type="candidate", candidate_architecture="g5", aux_spec="public_pair_context_v3",
        candidate_board_channels=16, encoder_blocks=1, candidate_d_model=64, pill_embed_dim=16,
        candidate_hidden_dim=64, candidate_cross_layers=1, candidate_interaction_layers=1,
        candidate_transformer_heads=4, candidate_cross_ff_mult=4, candidate_max_candidates=512,
        candidate_terminal_wdl=True, candidate_wdl=True), "env": {"public_observations": True}}


def _nets(perturb=False):
    from tools.eval_policy import _build_net_from_cfg

    torch.manual_seed(0)
    g5, _, _ = _build_net_from_cfg(_g5_cfg(), 20, "cpu")
    cfg = afterstate_full_config(_g5_cfg())
    net = from_g5(g5, cfg["smdp_ppo"])
    if perturb:  # a trained branch: the decision now depends on the afterstates
        with torch.no_grad():
            net.afterstate.out.weight.normal_(0, 0.5)
    return cfg, g5.eval(), net.eval()


def _inputs(seed=5):
    from tests.test_afterstate_core import _inputs as afterstate_inputs

    return afterstate_inputs(np.random.default_rng(seed))


def test_initial_model_is_the_g5_core_and_every_g5_tensor_is_loaded():
    _cfg, g5, net = _nets()
    inputs, aux = _inputs()
    with torch.inference_mode():
        a = g5(*inputs, aux=aux, return_aux=True)
        b = net(*inputs, aux=aux, return_aux=True)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    for key in ("value_logits", "candidate_wdl_logits", "state_wdl_logits"):
        assert torch.equal(a[2][key], b[2][key])
    state = net.state_dict()
    assert all(torch.equal(state[k], v) for k, v in g5.state_dict().items())
    assert {k for k in state if k not in g5.state_dict()} == {k for k in state if k.startswith(BRANCH_PREFIX)}


def test_zero_projection_learns_and_trained_branch_uses_afterstates():
    _cfg, _g5, net = _nets()
    inputs, aux = _inputs(6)
    logits, value = net(*inputs, aux=aux)
    (logits.masked_fill(~inputs[5], 0).log_softmax(-1)[:, 0].sum() + value.sum()).backward()
    assert net.afterstate.out.weight.grad.abs().sum() > 0
    _cfg, _g5, net = _nets(perturb=True)
    tiles, facts = net.exact_afterstates(inputs[0], inputs[1], inputs[3], inputs[5])
    with torch.inference_mode():
        implicit, _ = net(*inputs, aux=aux)
        explicit, _ = net.forward_features(*inputs, aux, tiles, facts)
        changed, _ = net.forward_features(*inputs, aux, tiles, torch.zeros_like(facts))
        single, _ = net(*(t[1:2] for t in inputs), aux=aux[1:2])
    mask = inputs[5]
    assert torch.equal(implicit, explicit)
    assert not torch.allclose(implicit[mask], changed[mask])
    n = int(mask[1].sum())
    assert torch.allclose(single[0, :n], implicit[1, :n], atol=1e-4)


def test_checkpoint_loads_through_plain_policy_and_controller_core(tmp_path):
    from drmc_rl.models.policy.controller_core import ControllerCorePolicy
    from tools.vs_head_to_head import PlainPolicy

    cfg, _g5, net = _nets(perturb=True)
    path = tmp_path / "arm-c.pt"
    torch.save(dict(cfg=cfg, state_dict=net.state_dict()), path)
    inputs, aux = _inputs(7)
    with torch.inference_mode():
        loaded, _ = PlainPolicy(path, "cpu", public_only=True).net(*inputs, aux=aux)
        reference, _ = net(*inputs, aux=aux)
    assert torch.equal(loaded, reference)
    actor = ControllerCorePolicy(path, "cpu", training=True)
    records = []
    for b in range(inputs[0].shape[0]):
        n = int(inputs[5][b].sum())
        records.append(dict(
            observation=inputs[0][b].numpy().astype(np.uint8), pill=inputs[1][b].numpy().astype(np.int8),
            preview=inputs[2][b].numpy().astype(np.int8), actions=inputs[3][b, :n].numpy().astype(np.int16),
            costs=inputs[4][b, :n].numpy().astype(np.uint16), mask=np.ones(n, bool),
            public_context=aux[b].numpy(), base_logits=np.zeros(n, np.float32), slot=0))
    features, _ = actor.training_batch(records)
    logits, value = actor.training_forward(features)
    (logits.log_softmax(-1)[:, 0].sum() + value.sum()).backward()
    assert actor.net.afterstate.stem.weight.grad is not None


def test_retention_optimizer_gives_the_new_branch_its_own_group():
    from tools.train_controller_retention import optimizer_groups, set_learning_rates

    _cfg, _g5, net = _nets()
    config = dict(lr=3e-6, new_branch_prefix=BRANCH_PREFIX, new_branch_lr_multiplier=10.0,
                  new_branch_warmup_updates=4)
    optimizer = torch.optim.AdamW(optimizer_groups(net, config), lr=config["lr"], weight_decay=.001)
    trunk, branch = optimizer.param_groups
    assert sum(p.numel() for p in branch["params"]) == sum(
        p.numel() for n, p in net.named_parameters() if n.startswith(BRANCH_PREFIX))
    rates = []
    for update in (1, 2, 4, 9):
        set_learning_rates(optimizer, config, update)
        rates.append((trunk["lr"], branch["lr"]))
    assert all(t == 3e-6 for t, _ in rates)
    assert [round(b / 3e-6, 6) for _, b in rates] == [2.5, 5.0, 10.0, 10.0]
    plain = torch.optim.AdamW(optimizer_groups(net, dict(lr=3e-6)), lr=3e-6)
    assert len(plain.param_groups) == 1


@pytest.mark.skipif(not LIBRARY and not os.environ.get("DRMARIO_POOL_LIB"),
                    reason="requires the native frame library")
def test_lock_safe_pre_spawn_decisions_agree_with_spawn_when_the_prediction_holds(tmp_path):
    """Pre-spawn requests see the lock-predicted bottle; accepted ones match the spawn bottle.

    Mirrors the arena's lock_safe path (``EarlyRequests`` + ``early_public_view``,
    ``early_preview=repeat``). Whenever the prediction is validated at spawn, the
    own bottle planes, the exact afterstates and facts equal the spawn decision's,
    and with the rest of the public view held at spawn the decision is identical.
    """
    from drmc_rl.envs.backends.vs_frames import FrameVsPool
    from drmc_rl.execution.pace import resolve_pace
    from drmc_rl.human.backend import plan_candidates
    from drmc_rl.human.controller_context import controller_policy_inputs
    from drmc_rl.human.early_decision import EarlyRequests, early_public_view
    from drmc_rl.planning.native_reach import NativeReachabilityRunner
    from tools.vs_head_to_head import PlainPolicy

    cfg, _g5, net = _nets(perturb=True)
    path = tmp_path / "arm-c.pt"
    torch.save(dict(cfg=cfg, state_dict=net.state_dict()), path)
    policy = PlainPolicy(path, "cpu", public_only=True)
    pace, delay = resolve_pace("super_human"), 4
    planner = NativeReachabilityRunner()
    stats, checked = Counter(), 0

    def decide(state, candidate, view):
        obs, info = controller_policy_inputs(policy, candidate, state, pace, delay, delay, public=view,
                                             decision_delay_frames=delay)
        inputs, aux, _, _ = policy.model_inputs(obs, info)
        after = policy.net.exact_afterstates(inputs[0], inputs[1], inputs[3], inputs[5])
        with torch.inference_mode():
            logits, value = policy.net(*inputs, aux=aux)
        return inputs, after, logits, value

    try:
        with FrameVsPool(1, lib_path=LIBRARY) as pool:
            pool.reset([1234], level=14)
            early, was = EarlyRequests(2), [False, False]
            last = None
            for frame in range(3000):
                states = pool.states
                current = states[0]
                if was[0] and not current.falling:
                    early.on_lock(0, "lock_safe", frame, current, states[1], pool, stats)
                was[0], was[1] = current.falling, states[1].falling
                early.on_frame(0, "lock_safe", frame, current, states[1], pool, stats)
                key = (current.spawn_id, current.pill_counter_total)
                if current.falling and last != key:
                    last = key
                    request = early.resolve(0, "lock_safe", frame, current, states[1], stats)
                    if request is not None:
                        state = pool.semantic(0, public_context=True)
                        candidate = plan_candidates(planner, state, delay, pace)
                        spawn = decide(state, candidate, None)
                        pre = decide(state, candidate, early_public_view(
                            request["public"], board=request["board"], pill=state["pill"],
                            preview=tuple(state["pill"]), falling=state["falling"]))
                        held = decide(state, candidate, early_public_view(
                            state["public_pair_state"], board=request["board"], pill=state["pill"],
                            preview=tuple(state["preview"]), falling=state["falling"]))
                        assert request["board"] == bytes(current.board)
                        assert torch.equal(pre[0][0][:, :8], spawn[0][0][:, :8])
                        for i in (1, 3, 4, 5):
                            assert torch.equal(pre[0][i], spawn[0][i])
                        assert torch.equal(pre[1][0], spawn[1][0]) and torch.equal(pre[1][1], spawn[1][1])
                        assert torch.equal(held[2], spawn[2]) and torch.equal(held[3], spawn[3])
                        checked += 1
                pool.step([0x04, 0x04])
                if current.terminal or checked >= 8:
                    break
    finally:
        planner.close()
    assert checked >= 5 and stats["early_mismatch"] == 0
