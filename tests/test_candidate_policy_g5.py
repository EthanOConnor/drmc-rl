from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from drmc_rl.models.policy.candidate_policy_g5 import G5CandidatePlacementPolicyNet


def _net(**overrides):
    cfg = dict(
        in_channels=20,
        board_channels=16,
        encoder_blocks=2,
        d_model=64,
        pill_embed_dim=32,
        aux_dim=12,
        cand_hidden_dim=96,
        transformer_heads=4,
        cross_layers=2,
        interaction_layers=1,
        patch_kernel=3,
        value_atoms=21,
    )
    cfg.update(overrides)
    return G5CandidatePlacementPolicyNet(**cfg)


def _inputs(batch=3, width=32):
    mask = torch.zeros(batch, width, dtype=torch.bool)
    mask[:, :9] = True
    return dict(
        obs=torch.rand(batch, 20, 16, 8),
        pill_colors=torch.randint(0, 3, (batch, 2)),
        preview_pill_colors=torch.randint(0, 3, (batch, 2)),
        cand_actions=torch.arange(width).expand(batch, -1),
        cand_cost=torch.rand(batch, width) * 40,
        cand_mask=mask,
        aux=torch.rand(batch, 12),
    )


def test_g5_shapes_masking_and_distributional_value_loss():
    net = _net()
    inputs = _inputs()
    logits, value, extra = net(**inputs, return_aux=True)
    assert logits.shape == (3, 32)
    assert value.shape == (3, 1)
    assert extra["value_logits"].shape == (3, 21)
    assert torch.isfinite(logits[inputs["cand_mask"]]).all()
    assert (logits[~inputs["cand_mask"]] < -1e8).all()
    loss = net.distributional_value_loss(extra["value_logits"], torch.tensor([-1.0, 0.2, 1.0]))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()


def test_g5_valid_outputs_do_not_depend_on_padding_width():
    torch.manual_seed(4)
    net = _net().eval()
    inputs = _inputs(batch=2, width=16)

    def padded(width):
        amount = width - 16
        return net(
            inputs["obs"],
            inputs["pill_colors"],
            inputs["preview_pill_colors"],
            torch.nn.functional.pad(inputs["cand_actions"], (0, amount), value=-1),
            torch.nn.functional.pad(inputs["cand_cost"], (0, amount)),
            torch.nn.functional.pad(inputs["cand_mask"], (0, amount)),
            aux=inputs["aux"],
        )

    with torch.inference_mode():
        logits16, value16 = padded(16)
        logits32, value32 = padded(32)
    torch.testing.assert_close(logits16[:, :9], logits32[:, :9], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(value16, value32)


def test_g5_requires_full_vs_observation():
    with pytest.raises(ValueError, match="16 board channels"):
        _net(board_channels=8)


def test_g5_distributional_loss_accepts_bfloat16_logits():
    net = _net()
    logits = torch.randn(4, 21, dtype=torch.bfloat16, requires_grad=True)
    loss = net.distributional_value_loss(logits, torch.tensor([-1.0, -0.1, 0.4, 1.0]))
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_g5_efficient_variant_preserves_contract():
    net = _net(bottle_block="bottleneck", compact_candidate_features=True, cross_ff_mult=2)
    logits, value = net(**_inputs())
    assert logits.shape == (3, 32)
    assert value.shape == (3, 1)
