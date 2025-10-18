from __future__ import annotations

import numpy as np

from envs.retro.drmario_env import DrMarioRetroEnv


def test_mock_backend_reset_and_step():
    env = DrMarioRetroEnv(obs_mode="state", backend="mock", include_risk_tau=False)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert info["viruses_remaining"] == env._viruses_remaining  # internal consistency

    action = env.action_space.sample()
    obs_next, reward, done, trunc, step_info = env.step(action)
    assert isinstance(obs_next, np.ndarray)
    assert isinstance(reward, float)
    assert done in {True, False}
    assert trunc in {True, False}
    assert "backend_active" in step_info
    assert not step_info["backend_active"]

    env.close()
