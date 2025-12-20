from __future__ import annotations

import numpy as np

from training.envs.curriculum import CurriculumConfig, ScriptedCurriculum


def test_curriculum_advances_on_confidence_bound() -> None:
    cfg = CurriculumConfig(
        enabled=True,
        start_level=-4,
        max_level=-2,
        success_threshold=0.5,
        confidence_sigmas=1.0,
        rehearsal_prob=0.0,
        seed=123,
    )
    cur = ScriptedCurriculum(cfg)
    assert cur.current_level == -4

    # With success_threshold=0.5 and 1-sigma Wilson LB, the default confidence
    # window size is small; a short streak of successes should advance.
    for _ in range(3):
        advanced = cur.note_episode(level=-4, success=True)
    assert advanced is True
    assert cur.current_level == -3

    # 2/3 successes within the window should NOT advance.
    for success in (True, False, True):
        cur.note_episode(level=-3, success=bool(success))
    assert cur.current_level == -3

    # A short success streak should advance to -2.
    advanced = False
    for _ in range(3):
        advanced = bool(advanced or cur.note_episode(level=-3, success=True))
    assert advanced is True
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
