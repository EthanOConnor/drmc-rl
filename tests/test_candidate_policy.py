"""Tests for candidate-scoring placement policy + candidate packing helpers."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.policy.candidate_packing import pack_feasible_candidates
from models.policy.candidate_policy import CandidatePlacementPolicyNet
from models.policy.placement_heads import OrderedPairEmbedding


def test_pack_feasible_candidates_sorted_and_padded():
    mask = np.zeros((4, 16, 8), dtype=bool)
    cost = np.full((4, 16, 8), np.inf, dtype=np.float32)

    # Three feasible actions with distinct costs.
    idxs = [0, 7, 128 + 3]  # (o,row,col) -> flat indices
    costs = [10.0, 3.0, 7.0]
    for a, c in zip(idxs, costs, strict=True):
        o = a // (16 * 8)
        rem = a % (16 * 8)
        r = rem // 8
        col = rem % 8
        mask[o, r, col] = True
        cost[o, r, col] = c

    packed = pack_feasible_candidates(mask, cost, max_candidates=8, sort_by_cost=True)
    assert packed.actions.shape == (8,)
    assert packed.mask.shape == (8,)
    assert packed.cost.shape == (8,)
    assert packed.count == 3

    valid_actions = packed.actions[: packed.count]
    valid_costs = packed.cost[: packed.count]
    assert np.all(packed.mask[: packed.count])
    assert np.all(packed.actions[packed.count :] == -1)
    assert np.all(~packed.mask[packed.count :])

    # Sorted ascending by cost.
    assert np.all(valid_costs[:-1] <= valid_costs[1:])
    # Same set as input.
    assert set(int(a) for a in valid_actions.tolist()) == set(idxs)


def test_pack_feasible_candidates_deterministic_ties():
    mask = np.zeros((4, 16, 8), dtype=bool)
    cost = np.full((4, 16, 8), np.inf, dtype=np.float32)

    # Many feasible actions with identical costs to stress tie-breaking.
    idxs = [0, 1, 2, 3, 5, 7, 128 + 0, 128 + 7, 256 + 10, 384 + 42]
    for a in idxs:
        o = a // (16 * 8)
        rem = a % (16 * 8)
        r = rem // 8
        c = rem % 8
        mask[o, r, c] = True
        cost[o, r, c] = 5.0

    packed1 = pack_feasible_candidates(mask, cost, max_candidates=64, sort_by_cost=True)
    packed2 = pack_feasible_candidates(mask, cost, max_candidates=64, sort_by_cost=True)

    assert packed1.count == len(idxs)
    assert packed2.count == len(idxs)
    assert np.array_equal(packed1.actions, packed2.actions)
    assert np.array_equal(packed1.mask, packed2.mask)
    assert np.array_equal(packed1.cost, packed2.cost)

    # With identical costs, ordering should be by macro action id.
    assert packed1.actions[: packed1.count].tolist() == sorted(idxs)


def test_candidate_policy_forward_shapes_and_masking():
    net = CandidatePlacementPolicyNet(
        in_channels=4,
        board_channels=4,
        board_encoder="cnn",
        encoder_blocks=0,
        d_model=64,
        pill_embed_dim=64,
        aux_dim=0,
        transformer_layers=2,
        transformer_heads=2,
        transformer_ff_mult=2,
        patch_kernel=3,
    )
    net.eval()

    B = 2
    K = 16
    board = torch.randn(B, 4, 16, 8)
    colors = torch.randint(0, 3, (B, 2))
    preview = torch.randint(0, 3, (B, 2))

    cand_actions = torch.full((B, K), -1, dtype=torch.int64)
    cand_mask = torch.zeros((B, K), dtype=torch.bool)
    cand_cost = torch.zeros((B, K), dtype=torch.float32)

    # Two valid candidates per batch element.
    cand_actions[0, 0] = 0
    cand_actions[0, 1] = 1
    cand_mask[0, 0] = True
    cand_mask[0, 1] = True
    cand_cost[0, 0] = 5.0
    cand_cost[0, 1] = 9.0

    cand_actions[1, 0] = 128
    cand_actions[1, 1] = 129
    cand_mask[1, 0] = True
    cand_mask[1, 1] = True
    cand_cost[1, 0] = 2.0
    cand_cost[1, 1] = 4.0

    with torch.no_grad():
        logits, value = net(board, colors, preview, cand_actions, cand_cost, cand_mask)

    assert logits.shape == (B, K)
    assert value.shape == (B, 1)
    assert torch.isfinite(logits[:, :2]).all()
    assert torch.isfinite(value).all()

    # Padding slots are masked to large negative logits.
    assert (logits[0, 2:] < -1e8).all()
    assert (logits[1, 2:] < -1e8).all()


def test_candidate_policy_forward_shapes_and_masking_col_transformer():
    net = CandidatePlacementPolicyNet(
        in_channels=4,
        board_channels=4,
        board_encoder="col_transformer",
        encoder_blocks=0,
        d_model=64,
        pill_embed_dim=64,
        aux_dim=0,
        transformer_layers=2,
        transformer_heads=2,
        transformer_ff_mult=2,
        patch_kernel=3,
    )
    net.eval()

    B = 2
    K = 16
    board = torch.randn(B, 4, 16, 8)
    colors = torch.randint(0, 3, (B, 2))
    preview = torch.randint(0, 3, (B, 2))

    cand_actions = torch.full((B, K), -1, dtype=torch.int64)
    cand_mask = torch.zeros((B, K), dtype=torch.bool)
    cand_cost = torch.zeros((B, K), dtype=torch.float32)

    cand_actions[0, 0] = 0
    cand_actions[0, 1] = 1
    cand_mask[0, 0] = True
    cand_mask[0, 1] = True
    cand_cost[0, 0] = 5.0
    cand_cost[0, 1] = 9.0

    cand_actions[1, 0] = 128
    cand_actions[1, 1] = 129
    cand_mask[1, 0] = True
    cand_mask[1, 1] = True
    cand_cost[1, 0] = 2.0
    cand_cost[1, 1] = 4.0

    with torch.no_grad():
        logits, value = net(board, colors, preview, cand_actions, cand_cost, cand_mask)

    assert logits.shape == (B, K)
    assert value.shape == (B, 1)
    assert torch.isfinite(logits[:, :2]).all()
    assert torch.isfinite(value).all()
    assert (logits[0, 2:] < -1e8).all()
    assert (logits[1, 2:] < -1e8).all()


def test_candidate_patch_gather_matches_reference():
    net = CandidatePlacementPolicyNet(
        in_channels=4,
        board_channels=4,
        board_encoder="cnn",
        encoder_blocks=0,
        d_model=64,
        pill_embed_dim=64,
        aux_dim=0,
        transformer_layers=2,
        transformer_heads=2,
        transformer_ff_mult=2,
        patch_kernel=3,
    )
    net.eval()

    B = 2
    P = 4
    K = 5
    planes = torch.randn(B, P, 16, 8)
    row = torch.tensor([[0, 1, 5, 15, 7], [2, 3, 0, 14, 15]], dtype=torch.int64)
    col = torch.tensor([[0, 7, 4, 2, 6], [7, 0, 3, 5, 1]], dtype=torch.int64)

    with torch.no_grad():
        out = net._gather_patch_planes(planes, row, col)

    # Naive reference using explicit padding + slicing.
    r = 1
    planes_p = torch.nn.functional.pad(planes, (r, r, r, r), mode="constant", value=0.0)
    ref = torch.zeros(B, K, P * 3 * 3, dtype=planes.dtype)
    for b in range(B):
        for k in range(K):
            rr = int(row[b, k].item())
            cc = int(col[b, k].item())
            patch = planes_p[b, :, rr : rr + 3, cc : cc + 3]  # [P,3,3]
            ref[b, k] = patch.reshape(P, -1).reshape(-1)

    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)


def test_candidate_policy_can_use_ordered_pill_embedding():
    net = CandidatePlacementPolicyNet(
        in_channels=4,
        board_channels=4,
        board_encoder="cnn",
        encoder_blocks=0,
        d_model=64,
        pill_embed_dim=64,
        pill_embed_type="ordered_onehot",
        aux_dim=0,
        transformer_layers=2,
        transformer_heads=2,
        transformer_ff_mult=2,
        patch_kernel=3,
    )
    assert isinstance(net.pill_embedding, OrderedPairEmbedding)
    assert isinstance(net.preview_embedding, OrderedPairEmbedding)
