from __future__ import annotations

import numpy as np

from training.envs import make_vec_env
from training.utils.cfg import load_and_merge_cfg, to_config_node


def test_make_vec_env_builds_real_retro_env_with_mock_backend() -> None:
    cfg_dict = load_and_merge_cfg("training/configs/base.yaml", None)
    cfg_dict.setdefault("env", {})["id"] = "DrMarioRetroEnv-v0"
    cfg_dict["env"]["backend"] = "mock"
    cfg_dict["env"]["obs_mode"] = "state"
    cfg_dict["env"]["include_risk_tau"] = False
    cfg_dict["env"]["frame_stack"] = 1  # request last-frame wrapper
    cfg_dict["env"]["num_envs"] = 1
    cfg_dict["env"]["vectorization"] = "sync"
    cfg = to_config_node(cfg_dict)

    env = make_vec_env(cfg)
    try:
        assert hasattr(env, "single_observation_space")
        obs, infos = env.reset(seed=123)
        assert isinstance(infos, (list, tuple))
        assert obs.shape[0] == 1
        assert obs.shape[1:] == env.single_observation_space.shape

        actions = np.zeros(env.num_envs, dtype=np.int64)
        obs2, rewards, terminated, truncated, infos2 = env.step(actions)
        assert obs2.shape[0] == 1
        assert rewards.shape == (1,)
        assert terminated.shape == (1,)
        assert truncated.shape == (1,)
        assert isinstance(infos2, (list, tuple))
        assert len(infos2) == 1
    finally:
        if hasattr(env, "close"):
            env.close()

