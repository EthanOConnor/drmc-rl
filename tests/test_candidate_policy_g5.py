from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from drmc_rl.models.policy.candidate_policy_g5 import G5CandidatePlacementPolicyNet
from drmc_rl.training.algo.ppo_smdp import (
    TeacherDistillConfig,
    _candidate_bucketed_indices,
    _teacher_policy_targets,
)


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


def test_candidate_bucket_order_is_lossless_and_width_local():
    counts = torch.tensor([3, 65, 31, 33, 96, 5, 64, 120, 32, 34])
    mask = torch.arange(128).unsqueeze(0) < counts.unsqueeze(1)
    torch.manual_seed(8)
    order = _candidate_bucketed_indices(mask)
    assert sorted(order.tolist()) == list(range(len(counts)))
    ordered_buckets = ((counts[order] - 1) // 32).tolist()
    # Each width group is contiguous even though group and row order vary.
    transitions = sum(a != b for a, b in zip(ordered_buckets, ordered_buckets[1:]))
    assert transitions == len(set(ordered_buckets)) - 1


def test_teacher_targets_are_masked_normalized_and_padded():
    logits = torch.tensor([[1.0, 3.0, 99.0], [2.0, -1.0, 0.0]])
    mask = torch.tensor([[True, True, False], [True, False, True]])
    targets = _teacher_policy_targets(logits, mask, target_width=8, temperature=1.0)
    assert targets.shape == (2, 8)
    torch.testing.assert_close(targets.sum(dim=1), torch.ones(2))
    assert targets[0, 2] == 0 and targets[1, 1] == 0
    assert not targets[:, 3:].any()


def test_teacher_distill_config_requires_checkpoint_and_temperature():
    with pytest.raises(ValueError, match="checkpoint"):
        TeacherDistillConfig.from_dict({"enabled": True})
    with pytest.raises(ValueError, match="temperature"):
        TeacherDistillConfig.from_dict(
            {"enabled": True, "checkpoint": "teacher.pt.gz", "temperature": 0}
        )
