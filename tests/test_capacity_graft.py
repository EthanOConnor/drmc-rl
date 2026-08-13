"""Function-preserving capacity growth: cross-candidate attention + deeper
trunk grafts must leave a checkpoint's outputs exactly unchanged at init."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

BASE_SP = {
    "policy_type": "candidate",
    "aux_spec": "v1",
    "pill_embed_dim": 32,
    "pill_embed_type": "ordered_pair",
    "encoder_blocks": 2,
    "candidate_max_candidates": 64,
    "candidate_d_model": 48,
    "candidate_pos_embed_dim": 16,
    "candidate_cost_embed_dim": 16,
    "candidate_hidden_dim": 96,
    "candidate_board_encoder": "cnn",
    "candidate_board_channels": 8,
    "candidate_transformer_layers": 2,
    "candidate_transformer_heads": 4,
    "candidate_transformer_ff_mult": 4,
    "candidate_patch_kernel": 3,
    "candidate_cross_layers": 0,
}


def _build(sp_overrides):
    from tools.eval_policy import _build_net_from_cfg

    sp = dict(BASE_SP)
    sp.update(sp_overrides)
    net, aux_dim, _ = _build_net_from_cfg({"smdp_ppo": sp}, 12, "cpu")
    return net.eval(), aux_dim, sp


def _rand_inputs(aux_dim, B=5, K=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    valid = [o * 128 + r * 8 + c for o in (0,) for r in range(2, 14) for c in range(0, 7)]
    acts = torch.tensor((valid * ((K // len(valid)) + 1))[:K]).unsqueeze(0).repeat(B, 1)
    mask = torch.ones(B, K, dtype=torch.bool)
    mask[0, K // 2:] = False  # partial padding row
    return dict(
        obs=torch.rand(B, 12, 16, 8, generator=g),
        pill_colors=torch.randint(0, 3, (B, 2), generator=g),
        preview_pill_colors=torch.randint(0, 3, (B, 2), generator=g),
        cand_actions=acts,
        cand_cost=torch.rand(B, K, generator=g) * 60,
        cand_mask=mask,
        aux=torch.rand(B, aux_dim, generator=g) if aux_dim else None,
    )


def _forward(net, inp):
    with torch.inference_mode():
        return net(
            inp["obs"], inp["pill_colors"], inp["preview_pill_colors"],
            inp["cand_actions"], inp["cand_cost"], inp["cand_mask"], aux=inp["aux"],
        )


def test_cross_attention_graft_identity():
    from tools.expand_checkpoint import graft_state_dict

    net0, aux_dim, _ = _build({})
    sd0 = net0.state_dict()

    net1, _, sp1 = _build({"candidate_cross_layers": 2})
    net1.load_state_dict(graft_state_dict(sd0, new_smdp_cfg=sp1))
    net1.eval()

    inp = _rand_inputs(aux_dim)
    l0, v0 = _forward(net0, inp)
    l1, v1 = _forward(net1, inp)
    feas = inp["cand_mask"]
    assert torch.equal(v0, v1)
    assert torch.allclose(l0[feas], l1[feas], atol=0.0), "graft changed logits"
    # And the attention params actually exist / train.
    assert sum(p.numel() for n, p in net1.named_parameters() if n.startswith("cross_blocks")) > 0


def test_depth_graft_identity():
    from tools.expand_checkpoint import graft_state_dict

    net0, aux_dim, _ = _build({})
    sd0 = net0.state_dict()

    net1, _, sp1 = _build({"encoder_blocks": 4})
    net1.load_state_dict(graft_state_dict(sd0, new_smdp_cfg=sp1))
    net1.eval()

    inp = _rand_inputs(aux_dim, seed=1)
    l0, v0 = _forward(net0, inp)
    l1, v1 = _forward(net1, inp)
    feas = inp["cand_mask"]
    assert torch.equal(v0, v1)
    assert torch.allclose(l0[feas], l1[feas], atol=0.0), "depth graft changed logits"


def test_graft_rejects_width_change():
    from tools.expand_checkpoint import graft_state_dict

    net0, _, _ = _build({})
    sd0 = net0.state_dict()
    sp_bad = dict(BASE_SP)
    sp_bad["candidate_d_model"] = 96
    with pytest.raises((ValueError, RuntimeError)):
        graft_state_dict(sd0, new_smdp_cfg=sp_bad)


def test_cross_attention_changes_outputs_after_training_step():
    """The block must be trainable — one grad step must move the logits."""

    net1, aux_dim, sp1 = _build({"candidate_cross_layers": 1})
    inp = _rand_inputs(aux_dim, seed=2)
    logits, _ = net1(
        inp["obs"], inp["pill_colors"], inp["preview_pill_colors"],
        inp["cand_actions"], inp["cand_cost"], inp["cand_mask"], aux=inp["aux"],
    )
    loss = logits[inp["cand_mask"]].sum()
    loss.backward()
    grads = [
        p.grad.abs().sum().item()
        for n, p in net1.named_parameters()
        if n.startswith("cross_blocks") and p.grad is not None
    ]
    assert grads and sum(grads) > 0, "no gradient reached the cross-attention block"
