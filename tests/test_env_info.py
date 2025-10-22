from __future__ import annotations

from training.envs.dr_mario_vec import DummyVecEnv, VecEnvConfig


def test_env_info_schema() -> None:
    env = DummyVecEnv(VecEnvConfig(num_envs=1, episode_length=3))
    obs, infos = env.reset()
    assert obs.shape[0] == 1
    done_info = None
    for _ in range(4):
        obs, rewards, terminated, truncated, infos = env.step([0])
        if infos[0]:
            done_info = infos[0]
            break
    assert done_info is not None, "Expected episode termination"
    assert "episode" in done_info
    assert "drm" in done_info
    drm = done_info["drm"]
    expected_keys = {
        "viruses_cleared",
        "lines_cleared",
        "drop_speed_level",
        "top_out",
        "combo_max",
        "risk",
    }
    assert expected_keys <= drm.keys()
    assert isinstance(drm["top_out"], bool)
    assert isinstance(drm["viruses_cleared"], int)
    assert isinstance(drm["risk"], float)
