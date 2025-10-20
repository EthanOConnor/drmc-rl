import math

import numpy as np
import pytest

from training.discounting import (
    discounted_returns_list,
    discounted_returns_mlx,
    discounted_returns_torch,
)


def test_discounted_returns_matches_expected_values():
    rewards = [1.0, 2.0, 3.0]
    gamma = 0.5
    expected = [2.75, 3.5, 3.0]
    returns = discounted_returns_list(rewards, gamma)
    assert np.allclose(returns, expected)


def test_discounted_returns_stays_finite_for_long_sequences():
    gamma = 0.99
    steps = 50000
    rewards = [1.0] * steps
    returns = discounted_returns_list(rewards, gamma)

    assert math.isfinite(returns[0])
    assert math.isfinite(returns[-1])

    expected_first = (1.0 - gamma ** steps) / (1.0 - gamma)
    assert math.isclose(returns[0], expected_first, rel_tol=1e-3)
    assert math.isclose(returns[-1], 1.0, rel_tol=1e-6)


def test_naive_power_method_underflows_for_context():
    gamma = 0.99
    steps = 50000
    rewards = np.ones(steps, dtype=np.float32)
    pow_gamma = np.power(gamma, np.arange(steps, dtype=np.float32))
    discounted = rewards * pow_gamma
    cumulative = np.cumsum(discounted[::-1])[::-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        naive = cumulative / pow_gamma

    assert not np.isfinite(naive[-1])


def test_discounted_returns_list_supports_dones():
    rewards = [1.0, 2.0, 3.0, 4.0]
    dones = [False, True, False, False]
    gamma = 0.9
    bootstrap = 5.0

    returns = discounted_returns_list(rewards, gamma, dones=dones, bootstrap=bootstrap)

    running = bootstrap
    expected = [0.0] * len(rewards)
    for idx in range(len(rewards) - 1, -1, -1):
        mask = 0.0 if dones[idx] else 1.0
        running = rewards[idx] + gamma * running * mask
        expected[idx] = running

    assert np.allclose(returns, expected)


def test_discounted_returns_torch_handles_terminals_and_bootstrap():
    torch = pytest.importorskip("torch")

    rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    dones = torch.tensor([False, True, False])
    gamma = 0.9
    bootstrap = torch.tensor(5.0, dtype=torch.float32)

    returns = discounted_returns_torch(rewards, gamma, dones=dones, bootstrap=bootstrap)

    running = bootstrap.item()
    expected = [0.0] * rewards.shape[0]
    dones_np = dones.to(dtype=torch.bool).cpu().numpy()
    rewards_np = rewards.cpu().numpy()
    for idx in range(len(expected) - 1, -1, -1):
        mask = 0.0 if dones_np[idx] else 1.0
        running = rewards_np[idx] + gamma * running * mask
        expected[idx] = running

    assert returns.device == rewards.device
    assert returns.dtype == rewards.dtype
    assert torch.allclose(returns, torch.tensor(expected, dtype=rewards.dtype, device=rewards.device))


def test_discounted_returns_torch_batch_shape():
    torch = pytest.importorskip("torch")

    rewards = torch.tensor(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=torch.float32
    )
    dones = torch.tensor(
        [[False, False], [True, False], [False, True]], dtype=torch.bool
    )
    gamma = 0.8
    bootstrap = torch.tensor([5.0, 7.0], dtype=torch.float32)

    returns = discounted_returns_torch(rewards, gamma, dones=dones, bootstrap=bootstrap)

    running = bootstrap.cpu().numpy().astype(np.float64)
    expected = np.zeros_like(rewards.cpu().numpy())
    dones_np = dones.cpu().numpy()
    rewards_np = rewards.cpu().numpy()
    for idx in range(rewards.shape[0] - 1, -1, -1):
        mask = np.where(dones_np[idx], 0.0, 1.0)
        running = rewards_np[idx] + gamma * running * mask
        expected[idx] = running

    assert returns.shape == rewards.shape
    assert np.allclose(returns.cpu().numpy(), expected)


def test_discounted_returns_mlx_handles_terminals_and_bootstrap():
    mx = pytest.importorskip("mlx.core")

    rewards = mx.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=mx.float32)
    dones = mx.array([[False, False], [True, False], [False, True]], dtype=mx.bool_)
    bootstrap = mx.array([5.0, 7.0], dtype=mx.float32)
    gamma = 0.8

    returns = discounted_returns_mlx(rewards, gamma, dones=dones, bootstrap=bootstrap)

    rewards_np = np.asarray(rewards, dtype=np.float64)
    dones_np = np.asarray(dones, dtype=bool)
    running = np.asarray(bootstrap, dtype=np.float64)
    expected = np.zeros_like(rewards_np)
    for idx in range(rewards_np.shape[0] - 1, -1, -1):
        mask = np.where(dones_np[idx], 0.0, 1.0)
        running = rewards_np[idx] + gamma * running * mask
        expected[idx] = running

    assert np.allclose(np.asarray(returns), expected)
