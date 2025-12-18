from __future__ import annotations

from envs.retro.drmario_env import DrMarioRetroEnv


def test_negative_level_task_mapping_matches_and_viruses() -> None:
    env = DrMarioRetroEnv(obs_mode="state", backend="mock", auto_start=False)

    env.level = -10
    env._configure_task_from_level()
    assert env._task_goal_mode == "matches"
    assert env._synthetic_virus_target == 0
    assert env._match_target == 1

    env.level = -4
    env._configure_task_from_level()
    assert env._task_goal_mode == "matches"
    assert env._synthetic_virus_target == 0
    assert env._match_target == 7

    env.level = -3
    env._configure_task_from_level()
    assert env._task_goal_mode == "viruses"
    assert env._synthetic_virus_target == 1
    assert env._match_target is None

    env.level = -1
    env._configure_task_from_level()
    assert env._task_goal_mode == "viruses"
    assert env._synthetic_virus_target == 3
    assert env._match_target is None

