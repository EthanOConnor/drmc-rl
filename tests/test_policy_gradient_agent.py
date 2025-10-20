from types import SimpleNamespace

import numpy as np
import pytest


def test_policy_gradient_end_episode_handles_reused_action_tensor_backward() -> None:
    torch = pytest.importorskip("torch")

    from training.speedrun_experiment import PolicyGradientAgent

    action_space = SimpleNamespace(n=3)
    prototype_obs = np.zeros((4,), dtype=np.float32)
    agent = PolicyGradientAgent(
        action_space,
        prototype_obs=prototype_obs,
        gamma=0.9,
        learning_rate=1e-3,
        policy_arch="mlp",
        obs_mode="state",
        batch_runs=1,
    )

    context_id = 0
    agent.begin_episode(context_id=context_id)
    ctx = agent._get_context(context_id)

    obs_tensor = torch.zeros_like(torch.from_numpy(prototype_obs))
    for action_val in [0, 1, 2, 1]:
        ctx.observations.append(obs_tensor.clone())
        ctx.actions.append(int(action_val))
        ctx.rewards.append(1.0)
        ctx.reset_flags.append(False)

    agent.end_episode(0.0, context_id=context_id)

    assert any(
        param.grad is not None and torch.all(torch.isfinite(param.grad))
        for param in agent.model.parameters()
    )
