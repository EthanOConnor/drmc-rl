from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from drmc_rl.training.algo.ppo_smdp import _DeviceRolloutBuffer, _DeviceRolloutWave


def _wave(offset: int) -> _DeviceRolloutWave:
    count = 3
    return _DeviceRolloutWave(
        observations=torch.full((count, 4, 2, 2), float(offset)),
        pill_colors=torch.full((count, 2), offset, dtype=torch.int64),
        preview_pill_colors=torch.full((count, 2), offset + 1, dtype=torch.int64),
        aux=torch.full((count, 2), float(offset + 2)),
        actions=torch.arange(offset, offset + count),
        log_probs=torch.arange(count, dtype=torch.float32) + offset,
        candidate_actions=torch.arange(12, dtype=torch.int32).reshape(count, 4) + offset,
        candidate_mask=torch.ones((count, 4), dtype=torch.bool),
        candidate_cost=torch.arange(12, dtype=torch.float32).reshape(count, 4) + offset,
    )


def test_device_rollout_reuses_policy_inputs_and_supports_action_replacement():
    buffer = _DeviceRolloutBuffer(
        capacity=6,
        obs_shape=(4, 2, 2),
        aux_dim=2,
        candidate_max=4,
        device=torch.device("cpu"),
    )
    first = _wave(0)
    buffer.add(
        first,
        actions=np.array([50, 51, 52]),
        log_probs=np.array([5.0, 5.1, 5.2], dtype=np.float32),
        replace_policy_outputs=False,
    )
    second = _wave(10)
    buffer.add(
        second,
        actions=np.array([50, 51, 52]),
        log_probs=np.array([5.0, 5.1, 5.2], dtype=np.float32),
        replace_policy_outputs=True,
    )

    batch = buffer.batch(6)
    torch.testing.assert_close(batch.observations[:3], first.observations)
    torch.testing.assert_close(batch.candidate_cost[3:], second.candidate_cost)
    torch.testing.assert_close(batch.actions[:3], first.actions)
    torch.testing.assert_close(batch.actions[3:], torch.tensor([50, 51, 52]))
    torch.testing.assert_close(
        batch.log_probs[3:], torch.tensor([5.0, 5.1, 5.2], dtype=torch.float32)
    )

    buffer.clear()
    assert buffer.size == 0
