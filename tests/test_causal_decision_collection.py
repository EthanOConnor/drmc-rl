import numpy as np
import pytest

from drmc_rl.training.rollout.causal_decisions import CausalDecisionCollector
from drmc_rl.training.rollout.decision_buffer import DecisionRolloutBuffer


def selected():
    mask = np.zeros((2, 4, 16, 8), bool)
    mask[:, 0, 0, 0] = True
    return (
        np.zeros(2, np.int64),
        np.zeros(2),
        np.array([0.1, 0.2]),
        mask,
        np.ones_like(mask, dtype=np.float32),
        np.zeros((2, 2), np.int64),
        np.zeros((2, 2), np.int64),
        None,
    )


def test_wait_rewards_terminal_credit_and_per_learner_bootstrap_are_preserved():
    buffer = DecisionRolloutBuffer(
        2, (1, 16, 8), num_envs=2, gamma=1.0, gae_lambda=1.0, store_costs_to_lock=True
    )
    collector = CausalDecisionCollector(buffer)
    obs = np.zeros((2, 1, 16, 8), np.float32)
    collector.begin(obs, selected(), np.array([True, True]), [10, 10])
    obs.fill(7)  # The native/vector observation storage is reused in place.
    collector.advance([0.2, 0.1], [False, False], obs, [15, 15])
    collector.arrive(obs, [True, False], [15, 15], [0.4, 999.0])
    assert buffer.size == 1 and not collector.full
    # The faster learner's unrecorded drain actions cannot alter its bootstrap.
    collector.begin(obs, selected(), np.array([True, False]), [15, 15])
    collector.advance([5.0, 0.3], [False, True], obs, [25, 25])
    assert collector.full and all(step is None for step in collector.pending)
    batch = buffer.get_batch(bootstrap_value=collector.bootstrap)
    np.testing.assert_array_equal(batch.taus, [5, 15])
    np.testing.assert_allclose(batch.rewards, [0.2, 0.4])
    np.testing.assert_allclose(batch.returns, [0.6, 0.4])
    np.testing.assert_allclose(batch.advantages, [0.5, 0.2])
    assert not batch.observations.any() and (batch.observations_next == 7).all()
    np.testing.assert_array_equal(batch.dones, [False, True])


def test_pending_decision_cannot_cross_an_unreported_reset():
    buffer = DecisionRolloutBuffer(2, (1, 16, 8), num_envs=2, gamma=1.0)
    collector = CausalDecisionCollector(buffer)
    obs = np.zeros((2, 1, 16, 8), np.float32)
    collector.begin(obs, selected(), np.array([True, True]), [10, 10])
    with pytest.raises(RuntimeError, match="crossed a reset"):
        collector.arrive(obs, [True, False], [2, 12], [0.0, 0.0])


def test_discounted_rewards_need_an_explicit_event_timing_contract():
    with pytest.raises(ValueError, match="gamma one"):
        CausalDecisionCollector(DecisionRolloutBuffer(2, (1, 16, 8), gamma=0.99))
