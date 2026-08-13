"""Smoke test for placement policy system.

Quick integration test to verify all components work together.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from drmc_rl.models.policy.placement_dist import MaskedPlacementDist
from drmc_rl.models.policy.placement_heads import PlacementPolicyNet
from drmc_rl.training.rollout.decision_buffer import DecisionRolloutBuffer, DecisionStep


class TestPlacementPolicySmoke:
    """Smoke tests for placement policy components."""
    
    def test_policy_forward_pass(self):
        """Policy forward pass completes without errors."""
        net = PlacementPolicyNet(in_channels=12, head_type="dense")
        net.eval()
        
        # Dummy inputs
        board = torch.randn(2, 12, 16, 8)
        pill_colors = torch.randint(0, 3, (2, 2))
        preview_colors = torch.randint(0, 3, (2, 2))
        mask = torch.ones(2, 4, 16, 8, dtype=torch.bool)
        
        with torch.no_grad():
            logits, values = net(board, pill_colors, preview_colors, mask)
            
        assert logits.shape == (2, 4, 16, 8)
        assert values.shape == (2, 1)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(values).all()
        
    def test_masked_sampling(self):
        """Masked sampling works with sparse masks."""
        logits = torch.randn(4, 16, 8)
        
        # Create sparse mask
        mask = torch.zeros(4, 16, 8, dtype=torch.bool)
        mask[0, 5, 3] = True
        mask[1, 10, 2] = True
        mask[2, 3, 7] = True
        
        dist = MaskedPlacementDist(logits, mask)
        
        # Sample
        action_idx, log_prob = dist.sample(deterministic=False)
        
        assert isinstance(action_idx.item(), int)
        assert torch.isfinite(log_prob)
        
        # Check that sampled action is valid
        o, i, j = (
            action_idx.item() // (16 * 8),
            (action_idx.item() % (16 * 8)) // 8,
            action_idx.item() % 8,
        )
        assert mask[o, i, j]
        
    def test_decision_buffer_workflow(self):
        """Decision buffer stores and retrieves batches correctly."""
        buffer = DecisionRolloutBuffer(
            capacity=10,
            obs_shape=(12, 16, 8),
            num_envs=1,
            gamma=0.99,
            gae_lambda=0.95,
        )
        
        # Add steps
        for _ in range(5):
            step = DecisionStep(
                obs=np.random.randn(12, 16, 8).astype(np.float32),
                mask=np.ones((4, 16, 8), dtype=bool),
                pill_colors=np.array([0, 1], dtype=np.int64),
                preview_pill_colors=np.array([2, 0], dtype=np.int64),
                action=42,
                log_prob=-1.5,
                value=2.3,
                tau=10,
                reward=0.5,
                obs_next=np.random.randn(12, 16, 8).astype(np.float32),
                done=False,
            )
            buffer.add(step)
            
        assert len(buffer) == 5
        
        # Get batch
        batch = buffer.get_batch(bootstrap_value=0.0)
        
        assert batch.observations.shape == (5, 12, 16, 8)
        assert batch.actions.shape == (5,)
        assert batch.advantages is not None
        assert batch.returns is not None
        assert batch.gammas is not None
        
        # Check shapes match
        assert batch.advantages.shape == (5,)
        assert batch.returns.shape == (5,)
        assert batch.gammas.shape == (5,)

    def test_decision_buffer_batched_append(self):
        buffer = DecisionRolloutBuffer(
            capacity=8,
            obs_shape=(4, 16, 8),
            num_envs=3,
            aux_dim=2,
            store_costs_to_lock=True,
            search_target_dim=5,
        )
        rng = np.random.default_rng(3)
        obs = rng.random((3, 4, 16, 8), dtype=np.float32)
        next_obs = rng.random((3, 4, 16, 8), dtype=np.float32)
        buffer.add_arrays(
            observations=obs,
            masks=np.ones((3, 4, 16, 8), dtype=bool),
            costs_to_lock=np.ones((3, 4, 16, 8), dtype=np.float32),
            pill_colors=np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64),
            preview_pill_colors=np.array([[2, 1], [0, 2], [1, 0]], dtype=np.int64),
            aux=np.arange(6, dtype=np.float32).reshape(3, 2),
            actions=np.array([4, 5, 6]),
            log_probs=np.array([-1.0, -2.0, -3.0], dtype=np.float32),
            values=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            taus=np.array([10, 20, 30]),
            rewards=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            observations_next=next_obs,
            dones=np.array([False, True, False]),
            env_ids=np.arange(3, dtype=np.int32),
            search_targets=np.full((3, 5), 0.2, dtype=np.float32),
            search_values=np.array([0.4, 0.5, 0.6], dtype=np.float32),
            search_mask=np.array([True, False, True]),
        )

        assert len(buffer) == 3
        np.testing.assert_array_equal(buffer.observations[:3], obs)
        np.testing.assert_array_equal(buffer.observations_next[:3], next_obs)
        np.testing.assert_array_equal(buffer.actions[:3], [4, 5, 6])
        np.testing.assert_array_equal(buffer.search_mask[:3], [True, False, True])
        
    def test_all_heads_produce_valid_output(self):
        """All three policy heads produce valid outputs."""
        heads = ["dense", "shift_score", "factorized"]
        
        for head_type in heads:
            net = PlacementPolicyNet(in_channels=12, head_type=head_type)
            net.eval()
            
            board = torch.randn(1, 12, 16, 8)
            colors = torch.tensor([[0, 2]])
            preview = torch.tensor([[1, 1]])
            mask = torch.ones(1, 4, 16, 8, dtype=torch.bool)
            
            with torch.no_grad():
                logits, values = net(board, colors, preview, mask)
                
            assert logits.shape == (1, 4, 16, 8), f"{head_type} failed"
            assert values.shape == (1, 1), f"{head_type} failed"
            assert torch.isfinite(logits).all(), f"{head_type} produced non-finite logits"
            assert torch.isfinite(values).all(), f"{head_type} produced non-finite values"
            
    def test_gradient_flow(self):
        """Gradients flow through policy network correctly."""
        net = PlacementPolicyNet(in_channels=12, head_type="dense")
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        
        # Forward pass
        board = torch.randn(2, 12, 16, 8)
        colors = torch.randint(0, 3, (2, 2))
        preview = torch.randint(0, 3, (2, 2))
        mask = torch.ones(2, 4, 16, 8, dtype=torch.bool)
        
        logits, values = net(board, colors, preview, mask)
        
        # Create distributions and sample
        losses = []
        for i in range(2):
            dist = MaskedPlacementDist(logits[i], mask[i])
            action, log_prob = dist.sample()
            losses.append(-log_prob)
            
        loss = torch.stack(losses).mean() + values.mean()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that gradients exist and are finite
        for name, param in net.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
