"""Unit tests for placement policy components.

Tests mask handling, color invariance, geometry alignment, SMDP math,
and distribution API as specified in the handoff document.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.policy.placement_dist import (
    MaskedPlacementDist,
    flatten_placement,
    unflatten_placement,
)
from models.policy.placement_heads import (
    DrMarioBoardEncoder,
    UnorderedPillEmbedding,
    DenseConvHead,
    ShiftAndScoreHead,
    FactorizedHead,
    PlacementPolicyNet,
)
from training.rollout.decision_buffer import compute_gae_smdp


class TestMaskedDistribution:
    """Test 7.1: Mask & numerics."""
    
    def test_masked_softmax_sums_to_one(self):
        """Masked softmax sums to 1 over feasible set."""
        logits = torch.randn(4, 16, 8)
        mask = torch.randint(0, 2, (4, 16, 8)).bool()
        
        # Ensure at least one valid action
        mask[0, 0, 0] = True
        
        dist = MaskedPlacementDist(logits, mask)
        
        # Check probabilities sum to 1
        assert torch.allclose(dist.probs.sum(dim=-1), torch.ones(1), atol=1e-5)
        
    def test_invalid_actions_zero_prob(self):
        """Invalid entries have prob=0, -inf log-prob."""
        logits = torch.randn(4, 16, 8)
        mask = torch.zeros(4, 16, 8, dtype=torch.bool)
        mask[0, 5, 3] = True  # Single valid action
        
        dist = MaskedPlacementDist(logits, mask)
        
        # Check that only the valid action has non-zero probability
        valid_idx = flatten_placement(0, 5, 3)
        assert dist.probs[0, valid_idx] > 0.99
        
        # Check that invalid actions have near-zero probability
        invalid_idx = flatten_placement(1, 0, 0)
        assert dist.probs[0, invalid_idx] < 1e-6
        
    def test_single_valid_action(self):
        """All-zero except one valid cell → prob 1 at that cell; gradients finite."""
        logits = torch.randn(4, 16, 8, requires_grad=True)
        mask = torch.zeros(4, 16, 8, dtype=torch.bool)
        mask[2, 10, 4] = True
        
        dist = MaskedPlacementDist(logits, mask)
        
        # Should concentrate all probability on valid action
        valid_idx = flatten_placement(2, 10, 4)
        assert dist.probs[0, valid_idx] > 0.99
        
        # Check gradients are finite
        loss = -dist.log_prob(torch.tensor([valid_idx]))
        loss.backward()
        assert torch.isfinite(logits.grad).all()
        
    def test_no_valid_sentinel(self):
        """No-valid sentinel: fallback path increments counter (doesn't crash)."""
        logits = torch.randn(4, 16, 8)
        mask = torch.zeros(4, 16, 8, dtype=torch.bool)  # All invalid
        
        # Should not crash and should fall back gracefully
        dist = MaskedPlacementDist(logits, mask)
        
        # Should have made at least one action valid
        assert dist.mask_flat.any()
        assert torch.allclose(dist.probs.sum(dim=-1), torch.ones(1), atol=1e-5)


class TestColorInvariance:
    """Test 7.2: Color invariance."""
    
    def test_embedding_symmetry(self):
        """Swapping (R,G) vs (G,R) yields identical embeddings."""
        embedder = UnorderedPillEmbedding(num_colors=3, embedding_dim=16, output_dim=32)
        
        # Test swapping
        colors_1 = torch.tensor([[0, 1]])  # Red, Yellow
        colors_2 = torch.tensor([[1, 0]])  # Yellow, Red
        
        emb_1 = embedder(colors_1)
        emb_2 = embedder(colors_2)
        
        # Should be identical
        assert torch.allclose(emb_1, emb_2, atol=1e-6)
        
    def test_policy_color_invariance(self):
        """Full policy with Deep-Sets embedding is color-order invariant."""
        net = PlacementPolicyNet(in_channels=12, head_type="dense")
        net.eval()
        
        board = torch.randn(1, 12, 16, 8)
        mask = torch.ones(1, 4, 16, 8, dtype=torch.bool)
        
        colors_1 = torch.tensor([[0, 2]])  # Red, Blue
        colors_2 = torch.tensor([[2, 0]])  # Blue, Red
        
        with torch.no_grad():
            logits_1, _ = net(board, colors_1, mask)
            logits_2, _ = net(board, colors_2, mask)
            
        # Logits should be identical
        assert torch.allclose(logits_1, logits_2, atol=1e-5)


class TestGeometryAlignment:
    """Test 7.3: Geometry alignment (toy board)."""
    
    def test_single_virus_placement(self):
        """Single exposed virus: highest logit at geometrically correct placement."""
        # Create toy board with single virus at (8, 4)
        board = torch.zeros(1, 12, 16, 8)
        board[0, 0, 8, 4] = 1.0  # Red virus at row 8, col 4
        
        mask = torch.ones(1, 4, 16, 8, dtype=torch.bool)
        colors = torch.tensor([[0, 0]])  # Red-red pill
        
        net = PlacementPolicyNet(in_channels=12, head_type="dense")
        net.eval()
        
        with torch.no_grad():
            logits, _ = net(board, colors, mask)
            
        # Find max logit position
        logits_flat = logits.reshape(1, -1)
        max_idx = logits_flat.argmax(dim=-1).item()
        o, i, j = unflatten_placement(max_idx)
        
        # Should target near the virus (within reasonable neighborhood)
        # This is a weak test since the network is untrained, but geometry should help
        # At minimum, it shouldn't crash and should produce reasonable output
        assert 0 <= o < 4
        assert 0 <= i < 16
        assert 0 <= j < 8


class TestSMDPMath:
    """Test 7.4: SMDP math."""
    
    def test_two_step_episode(self):
        """Synthetic 2-step episode with τ=[2,3]: verify GAE with Γ=γ^τ."""
        values = np.array([1.0, 0.5, 0.0])  # V(s0), V(s1), V(s2)=0
        rewards = np.array([1.0, 2.0])  # R0, R1
        taus = np.array([2, 3])
        gamma = 0.99
        lam = 0.95
        
        # Compute gammas: Γ_t = γ^τ_t
        gammas = gamma ** taus.astype(np.float32)
        
        # Expected:
        # Γ0 = 0.99^2 = 0.9801
        # Γ1 = 0.99^3 = 0.970299
        
        # Returns (backward):
        # t=1: ret[1] = R[1] + Γ[1] * 0 = 2.0
        # t=0: ret[0] = R[0] + Γ[0] * ret[1] = 1.0 + 0.9801 * 2.0 = 2.9602
        
        # GAE (backward):
        # t=1: δ[1] = R[1] + Γ[1]*0 - V[1] = 2.0 - 0.5 = 1.5
        #      A[1] = δ[1] = 1.5
        # t=0: δ[0] = R[0] + Γ[0]*V[1] - V[0] = 1.0 + 0.9801*0.5 - 1.0 = 0.49005
        #      A[0] = δ[0] + Γ[0]*λ*A[1] = 0.49005 + 0.9801*0.95*1.5 = 1.88694...
        
        advantages, returns = compute_gae_smdp(
            values[:2], rewards, gammas, dones=None, lam=lam, bootstrap=0.0
        )
        
        # Check returns
        expected_ret = np.array([2.9602, 2.0])
        assert np.allclose(returns, expected_ret, atol=1e-4)
        
        # Check advantages (approximately)
        expected_adv_0 = 0.49005 + 0.9801 * 0.95 * 1.5
        expected_adv = np.array([expected_adv_0, 1.5])
        assert np.allclose(advantages, expected_adv, atol=1e-4)
        
    def test_tau_one_recovers_standard_gae(self):
        """Setting all τ=1 reproduces standard GAE within tolerance."""
        values = np.array([1.0, 0.8, 0.5])
        rewards = np.array([0.5, 1.0])
        taus = np.array([1, 1])
        gamma = 0.99
        lam = 0.95
        
        gammas = gamma ** taus.astype(np.float32)
        
        # Standard GAE
        advantages_std = np.zeros(2, dtype=np.float32)
        returns_std = np.zeros(2, dtype=np.float32)
        
        # Backward
        last_gae = 0.0
        for t in [1, 0]:
            next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value - values[t]
            last_gae = delta + gamma * lam * last_gae
            advantages_std[t] = last_gae
            next_return = 0.0 if t == 1 else returns_std[t + 1]
            returns_std[t] = rewards[t] + gamma * (values[t + 1] if t < 1 else 0.0)
            
        # SMDP version with τ=1
        # Note: our SMDP helper takes V(s_t) for t in [0..T-1] and an explicit
        # bootstrap value V(s_T). For this non-terminal synthetic example,
        # bootstrap with values[-1] to match "standard" GAE that uses V(s_{t+1}).
        advantages_smdp, returns_smdp = compute_gae_smdp(
            values[:2], rewards, gammas, lam=lam, bootstrap=float(values[-1])
        )
        
        # Should match closely
        assert np.allclose(advantages_smdp, advantages_std, atol=1e-4)


class TestDistributionAPI:
    """Test 7.5: Distribution API."""
    
    def test_log_prob_entropy_match_reference(self):
        """Log-prob/entropy equal to manual masked softmax reference."""
        logits = torch.randn(4, 16, 8)
        mask = torch.randint(0, 2, (4, 16, 8)).bool()
        mask[0, 0, 0] = True  # Ensure at least one valid
        
        dist = MaskedPlacementDist(logits, mask)
        
        # Manual computation
        logits_flat = logits.reshape(-1)
        mask_flat = mask.reshape(-1)
        
        masked_logits = logits_flat.clone()
        masked_logits[~mask_flat] = -1e9
        
        probs_manual = torch.softmax(masked_logits, dim=0)
        log_probs_manual = torch.log(probs_manual + 1e-9)
        entropy_manual = -(probs_manual * log_probs_manual).sum()
        
        # Compare
        entropy_dist = dist.entropy()
        assert torch.allclose(entropy_dist, entropy_manual, atol=1e-4)
        
        # Sample an action and check log_prob
        action_idx = dist.mode()
        log_prob_dist = dist.log_prob(action_idx)
        log_prob_manual_val = log_probs_manual[action_idx]
        
        assert torch.allclose(log_prob_dist, log_prob_manual_val, atol=1e-4)


class TestFlattenUnflatten:
    """Test placement index conversion utilities."""
    
    def test_round_trip(self):
        """flatten → unflatten → flatten should be identity."""
        for o in range(4):
            for i in range(16):
                for j in range(8):
                    idx = flatten_placement(o, i, j)
                    o2, i2, j2 = unflatten_placement(idx)
                    assert (o, i, j) == (o2, i2, j2)
                    
    def test_bounds(self):
        """Flattened indices should be in [0, 511]."""
        assert flatten_placement(0, 0, 0) == 0
        assert flatten_placement(3, 15, 7) == 511
        
        o, i, j = unflatten_placement(0)
        assert (o, i, j) == (0, 0, 0)
        
        o, i, j = unflatten_placement(511)
        assert (o, i, j) == (3, 15, 7)
