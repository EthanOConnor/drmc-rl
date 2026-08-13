import gymnasium as gym
import numpy as np

from drmc_rl.envs.libretro.intent_env import DrMarioIntentEnv, IntentAction
from drmc_rl.envs.libretro.registration import register_env_id


def make_mock_env():
    register_env_id()
    env = gym.make(
        "DrMarioLibretroEnv-v0",
        obs_mode="state",
        backend="mock",
        include_risk_tau=False,
    )
    return env


def test_intent_env_basic_step():
    base_env = make_mock_env()
    env = DrMarioIntentEnv(base_env)
    obs, info = env.reset(seed=0)
    assert env.action_space.n == IntentAction.size()
    assert isinstance(obs, np.ndarray)
    assert "intent/active" in info

    # A few noop steps should be stable
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(IntentAction.NOOP)
        assert not terminated
        assert not truncated
        assert "intent/active" in info


def test_intent_env_left_right():
    base_env = make_mock_env()
    env = DrMarioIntentEnv(base_env)
    env.reset(seed=1)

    # Issue a left intent followed by noop frames to ensure translator handles stickiness
    env.step(IntentAction.LEFT1)
    for _ in range(4):
        env.step(IntentAction.NOOP)

    env.step(IntentAction.RIGHT1)
    for _ in range(4):
        env.step(IntentAction.NOOP)
