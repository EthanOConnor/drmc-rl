from __future__ import annotations

import numpy as np

from training.envs.curriculum import CurriculumConfig, ScriptedCurriculum


def test_curriculum_advances_on_success_rate() -> None:
    cfg = CurriculumConfig(
        enabled=True,
        start_level=-4,
        max_level=0,
        success_threshold=0.9,
        window_episodes=10,
        min_episodes=10,
        rehearsal_prob=0.0,
        seed=123,
    )
    cur = ScriptedCurriculum(cfg)
    assert cur.current_level == -4

    # 10/10 successes -> advance to -3
    for _ in range(10):
        advanced = cur.note_episode(level=-4, success=True)
    assert advanced is True
    assert cur.current_level == -3

    # 9/10 successes meets threshold -> advance again.
    for i in range(10):
        cur.note_episode(level=-3, success=(i != 0))
    assert cur.current_level == -2

    # Not enough successes -> do not advance.
    for i in range(10):
        cur.note_episode(level=-2, success=(i % 2 == 0))
    assert cur.current_level == -2


def test_curriculum_rehearsal_sampling_stays_in_range() -> None:
    cfg = CurriculumConfig(
        enabled=True,
        start_level=-4,
        max_level=0,
        success_threshold=0.9,
        window_episodes=5,
        min_episodes=5,
        rehearsal_prob=1.0,  # always rehearsal when possible
        seed=0,
    )
    cur = ScriptedCurriculum(cfg)
    cur.current_level = -2

    rng = np.random.default_rng(0)
    samples = [cur.sample_level(rng) for _ in range(50)]
    assert all(s in (-4, -3) for s in samples)

