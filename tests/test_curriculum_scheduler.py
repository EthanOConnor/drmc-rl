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


def test_ln_hop_back_skips_hop_back_when_probe_already_meets_pass_target() -> None:
    cfg = CurriculumConfig(
        enabled=True,
        mode="ln_hop_back",
        start_level=-4,
        max_level=-2,
        probe_threshold=0.5,
        confidence_sigmas=0.0,
        min_episodes=1,
        min_stage_decisions=0,
        rehearsal_prob=0.0,
        seed=0,
    )
    cur = LnHopBackCurriculum(cfg)
    assert cur.current_level == -4
    assert cur.stage_index == 0

    # One success meets both the probe target (0.5) and the k=1 pass target
    # (1-exp(-m)) when m>0, so we should skip the hop-back chain and advance
    # directly to the next frontier's probe stage (level -3).
    advanced = cur.note_episode(level=-4, success=True)
    assert advanced is True
    assert cur.current_level == -3
    assert cur.stage_index == 2  # probe(-4), hop(-4) skipped, probe(-3)


def test_ln_hop_back_starts_hop_back_at_third_highest_mastered_level() -> None:
    cfg = CurriculumConfig(
        enabled=True,
        mode="ln_hop_back",
        start_level=-6,
        max_level=-2,
        probe_threshold=0.5,
        # Make the "pass target" near-1 so probe stages won't fast-skip.
        pass_ramp_exponent_multiplier=10.0,
        confidence_sigmas=0.0,
        min_episodes=2,
        min_stage_decisions=0,
        rehearsal_prob=0.0,
        seed=0,
    )
    cur = LnHopBackCurriculum(cfg)

    def pass_probe(level: int) -> None:
        assert cur.current_level == int(level)
        assert cur._stages[cur.stage_index].is_probe
        # Exactly meets the probe threshold (0.5) but not the near-1 pass target.
        assert cur.note_episode(level=level, success=True) is False
        assert cur.note_episode(level=level, success=False) is True

    def pass_stage(level: int) -> None:
        assert cur.current_level == int(level)
        assert not cur._stages[cur.stage_index].is_probe
        assert cur.note_episode(level=level, success=True) is False
        assert cur.note_episode(level=level, success=True) is True

    # Build mastery up through level -3 via hop-back stages (mastered: -6,-5,-4,-3).
    pass_probe(-6)
    pass_stage(-6)
    pass_probe(-5)
    pass_stage(-6)
    pass_stage(-5)
    pass_probe(-4)
    pass_stage(-6)
    pass_stage(-5)
    pass_stage(-4)
    pass_probe(-3)
    pass_stage(-6)
    pass_stage(-5)
    pass_stage(-4)
    pass_stage(-3)

    assert cur.current_level == -2
    assert cur._stages[cur.stage_index].is_probe

    # Failing to meet the pass target at the -2 probe should hop back to the 3rd-highest
    # mastered level (mastered: -6,-5,-4,-3 => 3rd-highest is -5).
    pass_probe(-2)
    assert cur.current_level == -5
    assert not cur._stages[cur.stage_index].is_probe


def test_ln_hop_back_bails_out_of_stuck_probe_stage() -> None:
    cfg = CurriculumConfig(
        enabled=True,
        mode="ln_hop_back",
        start_level=-4,
        max_level=-3,
        probe_threshold=1.0,
        confidence_sigmas=0.0,
        min_episodes=1,
        min_stage_decisions=0,
        rehearsal_prob=0.0,
        seed=0,
        run_total_steps=1000,
        ln_hop_back_bailout_fraction=0.01,  # => 10 frames
    )
    cur = LnHopBackCurriculum(cfg)
    assert cur.stage_index == 0
    assert cur._stages[cur.stage_index].is_probe

    assert cur.note_episode(level=-4, success=False, episode_frames=5) is False
    advanced = cur.note_episode(level=-4, success=False, episode_frames=5)
    assert advanced is True
    assert cur.stage_index == 1  # jumped into hop-back stage
    assert not cur._stages[cur.stage_index].is_probe
