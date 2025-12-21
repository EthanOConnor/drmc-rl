from __future__ import annotations

import math
import numpy as np

from training.envs.curriculum import CurriculumConfig, LnHopBackCurriculum, ScriptedCurriculum


def test_curriculum_advances_on_confidence_bound() -> None:
    cfg = CurriculumConfig(
        enabled=True,
        start_level=-4,
        max_level=-2,
        success_threshold=0.5,
        confidence_sigmas=1.0,
        confidence_ema_half_life_episodes=0.0,
        min_episodes=3,
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


def test_ln_hop_back_uses_exponent_multiplier_by_default() -> None:
    cfg = CurriculumConfig(
        enabled=True,
        mode="ln_hop_back",
        start_level=-15,
        max_level=-13,
        probe_threshold=0.01,
        ln_hop_back_max_k=5,
        rehearsal_prob=0.0,
        seed=0,
    )
    cur = LnHopBackCurriculum(cfg)

    exp_mult = float(cfg.pass_ramp_exponent_multiplier)
    expected = {k: float(1.0 - math.exp(-float(k) * exp_mult)) for k in (1, 2, 3)}

    assert any(s.level == -15 and abs(float(s.threshold) - float(cfg.probe_threshold)) < 1e-12 for s in cur._stages)
    for k, thr in expected.items():
        assert any(s.level == -15 and abs(float(s.threshold) - float(thr)) < 1e-12 for s in cur._stages), k
