from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from training import discounting
from training.speedrun_experiment import PolicyGradientAgent


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_policy_gradient_agent_zeroes_nan_rewards(monkeypatch):

    action_space = SimpleNamespace(n=3)
    prototype_obs = np.zeros((4,), dtype=np.float32)
    agent = PolicyGradientAgent(
        action_space,
        prototype_obs,
        gamma=0.9,
        learning_rate=1e-3,
        batch_runs=1,
    )

    captured: dict[str, np.ndarray] = {}

    def _capture_discounted_returns(rewards_tensor, gamma, dones=None, bootstrap=None):
        captured["rewards"] = rewards_tensor.detach().cpu()
        return discounting.discounted_returns_torch(
            rewards_tensor, gamma, dones=dones, bootstrap=bootstrap
        )

    monkeypatch.setattr(
        "training.speedrun_experiment.discounted_returns_torch",
        _capture_discounted_returns,
    )

    agent.begin_episode()
    obs = np.zeros_like(prototype_obs)
    agent.select_action(obs, {}, None)
    agent.observe_step(1.0, False, {}, None)
    agent.select_action(obs, {}, None)
    agent.observe_step(float("nan"), True, {}, None)

    agent.end_episode(0.0)

    rewards_tensor = captured["rewards"]
    assert rewards_tensor.shape[0] == 2
    assert not np.isnan(rewards_tensor.numpy()).any()

    assert agent._nan_rewards_seen == 1
    assert agent._nan_reward_episodes == 1
    assert agent._total_reward_steps == 2

    metrics = agent.latest_metrics()
    assert metrics["rewards/nan_rewards_total"] == pytest.approx(1.0)
    assert metrics["rewards/nan_reward_episodes"] == pytest.approx(1.0)
    assert metrics["rewards/nan_reward_last_episode"] == 1.0
    assert metrics["rewards/nan_reward_rate"] == pytest.approx(0.5)
    assert metrics["alerts/nan_reward_rate_high"] == 1.0
