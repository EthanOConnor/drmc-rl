from copy import deepcopy
import pytest
import torch

from drmc_rl.game.pair_state import (
    DecisionBoundary,
    PrivilegedPairState,
    PublicPairState,
    VisibleSideState,
)
from drmc_rl.search.native_pair import CAUSAL_PUBLIC_SCHEMA, NativePairSearchState, state_to_payload
from drmc_rl.training.quality_supervision import (
    forward,
    join_quality_rows,
    make_batch,
    quality_loss,
    split_games,
    upgrade_public_model,
)
from tools.eval_policy import _build_net_from_cfg


def data():
    sides = tuple(VisibleSideState(bytes([255] * 128), (0, 1), (2, 1), None) for _ in (0, 1))
    public = PublicPairState(2, 0, sides, DecisionBoundary.BOTH)
    private = PrivilegedPairState(
        public, (2, 2), (True, True), (0, 0), ("decision", "decision"), (None, None), b"opaque"
    )
    state = NativePairSearchState(
        private,
        ((0, 1, 2), (0, 1)),
        ((3, 5, 7), (4, 6)),
        public_observation_schema=CAUSAL_PUBLIC_SCHEMA,
    )
    source = dict(state_to_payload(state), id="a", game_id="g1", root_side=0)
    target = dict(
        schema="drmc-paired-terminal-quality-v1",
        source_id="a",
        game_id="g1",
        root_side=0,
        actions=[0, 1, 2],
        incumbent=0,
        posterior_enumerated=True,
        candidate_truncation=0,
        reference_prior=[0.5, 0.25, 0.25],
        policy_target=dict(probability=[0.51, 0.25, 0.24]),
        candidates=[
            dict(action=a, wdl=q, unknown_mass=0.0)
            for a, q in enumerate([[0.7, 0.2, 0.1], [0.4, 0.3, 0.3], [0.1, 0.2, 0.7]])
        ],
        member_sha256={"parent": "fixed"},
        continuations=[
            dict(actor="parent", opponent="parent", weight=1.0, execution="native-smdp-v1")
        ],
    )
    return source, target


def checkpoint():
    cfg = dict(
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
        aux_spec="zero_v1_vs",
    )
    net, _, _ = _build_net_from_cfg(cfg, 20, "cpu")
    return dict(cfg=cfg, state_dict=net.state_dict()), net


@pytest.mark.parametrize("mode", ["baseline", "critic", "context", "combined"])
def test_quality_heads_train_and_critic_migration_preserves_existing_policy(mode):
    source, target = data()
    rows = join_quality_rows([source], [target])
    parent, old = checkpoint()
    net, cfg = upgrade_public_model(parent, mode=mode, device="cpu")
    batch = make_batch(rows, schema=cfg["aux_spec"], device="cpu")
    output = forward(net, batch)
    if mode in ("baseline", "critic"):
        original = forward(old, batch)
        torch.testing.assert_close(output[0], original[0], rtol=0, atol=0)
        torch.testing.assert_close(output[1], original[1], rtol=0, atol=0)
    loss, metrics = quality_loss(output, batch, anchor_logp=output[0].detach().log_softmax(-1))
    loss.backward()
    assert torch.isfinite(loss) and metrics["anchor_kl"].abs() < 1e-7
    assert net.state_wdl_head.weight.grad.abs().sum() > 0
    assert net.candidate_wdl_head.weight.grad.abs().sum() > 0
    if mode in ("critic", "combined"):
        assert net.value_projection.weight.grad.abs().sum() > 0


def test_unknown_labels_reanalysis_mixing_and_game_leakage_fail_closed():
    source, target = data()
    bad = deepcopy(target)
    bad["candidates"][1]["unknown_mass"] = 0.5
    with pytest.raises(ValueError, match="censored"):
        join_quality_rows([source], [bad])
    second = dict(source, id="b", game_id="g2")
    target2 = dict(target, source_id="b", game_id="g2")
    rows = join_quality_rows([source, second], [target, target2])
    sharp = deepcopy(target)
    sharp["policy_target"]["probability"] = [1.0, 0.0, 0.0]
    assert len(join_quality_rows([source], [sharp])[0]["wdl"]) == 3
    train, validation = split_games(rows, seed=7)
    assert {r["game_id"] for r in train}.isdisjoint(r["game_id"] for r in validation)
    with pytest.raises(ValueError, match="pooled"):
        join_quality_rows(
            [source, second], [target, dict(target2, member_sha256={"parent": "changed"})]
        )
    duplicate = dict(rows[0], source_id="c")
    batch = make_batch([rows[0], duplicate, rows[1]], schema="zero_v1_vs", device="cpu")
    assert float(batch["weights"][:2].sum()) == pytest.approx(float(batch["weights"][2]))


@pytest.mark.parametrize("mode", ["baseline", "critic", "context", "combined"])
def test_quality_phases_preserve_learned_heads_and_effective_ema_parent(mode):
    parent, _ = checkpoint()
    parent["ema_state_dict"] = deepcopy(parent["state_dict"])
    key = "value_head.1.weight"
    parent["ema_state_dict"][key].add_(0.125)
    net, cfg = upgrade_public_model(parent, mode=mode, device="cpu")
    torch.testing.assert_close(net.state_dict()[key], parent["ema_state_dict"][key])
    assert cfg["env"]["public_observations"]
    with torch.no_grad():
        net.state_wdl_head.weight.fill_(0.25)
        net.candidate_wdl_head.bias.copy_(torch.tensor([0.2, -0.1, 0.4]))
    fitted = dict(cfg=cfg, state_dict=net.state_dict())
    continued, _ = upgrade_public_model(fitted, mode=mode, device="cpu")
    for name, value in net.state_dict().items():
        torch.testing.assert_close(continued.state_dict()[name], value, rtol=0, atol=0)
    other_mode = "combined" if mode != "combined" else "baseline"
    with pytest.raises(ValueError, match="retain its architecture"):
        upgrade_public_model(fitted, mode=other_mode, device="cpu")


def test_new_public_checkpoint_rejects_missing_timeline_even_without_public_only_flag():
    from tools.vs_head_to_head import PlainPolicy

    policy = PlainPolicy.__new__(PlainPolicy)
    policy.public_only = False
    policy.requires_causal_observations = True
    for info in ({}, {"vs/observation_timeline": "legacy-warp-buffer-v1"}):
        with pytest.raises(ValueError, match="explicit causal observation"):
            policy.score_and_value(None, [info])


def test_shared_new_critic_initialization_is_identical_across_ablations():
    parent, _ = checkpoint()
    critic, _ = upgrade_public_model(parent, mode="critic", device="cpu")
    combined, _ = upgrade_public_model(parent, mode="combined", device="cpu")
    for name, parameter in critic.value_query.state_dict().items():
        torch.testing.assert_close(
            parameter, combined.value_query.state_dict()[name], rtol=0, atol=0
        )
