"""Skill-grade inputs logged by the VS env match tools/skill_grade.py's fit data.

The model was fit on fightcadeRatings crown rows, where every metric is what
the graded side inflicted/sent. These tests pin the env's features to those
definitions on a synthetic match with known volleys, for both sides.
"""

from __future__ import annotations

import math
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from drmc_rl.envs.backends.drmario_vs_pool import VS_OUTCOME_LOSS, VS_OUTCOME_WIN, VsVolley
from drmc_rl.training.envs.drmario_vs_vec import (
    SKILL_FPS,
    DrMarioVsPoolVecEnv,
    skill_game_features,
)
from tools import skill_grade


def _salt(depth: int) -> int:
    return 16 * depth


def _volley(receiver: int, size: int, depth: int, *, pair: int = 0) -> VsVolley:
    cols = tuple(range(size))
    return VsVolley(pair=pair, receiver=receiver, size=size, cols=cols,
                    colors=(0,) * size, frame=0, salt_frames=_salt(depth))


def test_skill_game_features_known_volleys() -> None:
    # Sent volleys of 2/3/4 pieces whose deepest pieces fell 5, 16 and 1 rows.
    feats = skill_game_features(
        length_s=120.0,
        volleys_sent=3,
        garbage_sent=2 + 3 + 4,
        salt_frames_inflicted=_salt(5) + _salt(16) + _salt(1),
        pills=95,
        speed_setting=2,
    )
    assert feats["cpm"] == pytest.approx(1.5)
    assert feats["cur"] == pytest.approx(3.0)  # pieces per volley, not 0/1
    assert feats["salt_per_min"] == pytest.approx(16 * 22 / 60.0988 / 2.0)
    assert feats["garbage_per_min"] == pytest.approx(4.5)
    assert feats["pills_per_min"] == pytest.approx(47.5)
    assert feats["spd"] == 31 + 9  # HI base + one speed-up per 10 pills


def test_skill_game_features_edges() -> None:
    quiet = skill_game_features(length_s=60.0, volleys_sent=0, garbage_sent=0,
                                salt_frames_inflicted=0, pills=600, speed_setting=0)
    assert quiet["cur"] == 0.0 and quiet["cpm"] == 0.0 and quiet["salt_per_min"] == 0.0
    assert quiet["spd"] == 15 + 49  # LOW base, speed-ups capped
    unknown = skill_game_features(length_s=60.0, volleys_sent=1, garbage_sent=2,
                                  salt_frames_inflicted=None, pills=10, speed_setting=2)
    assert math.isnan(unknown["salt_per_min"])


def _bare_env(*, pool_mode: bool = False) -> DrMarioVsPoolVecEnv:
    """Env with only the match-accounting state, no native runner."""
    env = object.__new__(DrMarioVsPoolVecEnv)
    n = 2
    for name in ("_garbage_sent_prev", "_volleys_sent_round", "_volleys_recv_round",
                 "_garbage_recv_round", "_garbage_volley_sent_round",
                 "_salt_frames_sent_round", "_ep_pills", "_ep_viruses_cleared"):
        setattr(env, name, np.zeros((n,), dtype=np.int64))
    env._salt_unknown_round = np.zeros((n,), dtype=bool)
    for name in ("_win_p1", "_draws", "_match_len_sec", "_clear_wins", "_topout_wins",
                 "_horizon_matches", "_pills_per_match", "_garbage_per_match",
                 "_viruses_per_match"):
        setattr(env, name, deque(maxlen=100))
    env._matches_total = 0
    env._volleys_total = 0
    env._skill_games = []
    env._opp_pool = object() if pool_mode else None
    env.speed_setting = 2
    env._runner = SimpleNamespace(buffers=SimpleNamespace(viruses_rem=np.array([0, 7])))
    return env


def _play_synthetic_match(env: DrMarioVsPoolVecEnv) -> float:
    # Side 0 sends three volleys into side 1's field (receiver=1); side 1
    # sends one small volley back into side 0's field (receiver=0).
    env._accumulate_volleys([
        _volley(receiver=1, size=2, depth=5),
        _volley(receiver=1, size=4, depth=16),
        _volley(receiver=0, size=2, depth=3),
        _volley(receiver=1, size=3, depth=1),
    ])
    # The native per-side totals the env mirrors in _garbage_sent_prev.
    env._garbage_sent_prev[:] = [9, 2]
    env._ep_pills[:] = [95, 88]
    pair_clock = int(round(120.0 * SKILL_FPS))
    env._record_match(0, np.array([VS_OUTCOME_WIN, VS_OUTCOME_LOSS]), pair_clock)
    return pair_clock / SKILL_FPS / 60.0


def test_env_credits_inflicted_salt_to_the_sender_for_both_sides() -> None:
    env = _bare_env()
    minutes = _play_synthetic_match(env)
    games = env.pop_skill_games()
    assert len(games) == 2
    p1, p2 = games

    # Side 0 inflicted three volleys (depths 5, 16, 1); received one (depth 3).
    assert p1["cpm"] == pytest.approx(3 / minutes)
    assert p1["cur"] == pytest.approx(3.0)
    assert p1["salt_per_min"] == pytest.approx(16 * 22 / 60.0988 / minutes)
    assert p1["garbage_per_min"] == pytest.approx(9 / minutes)
    assert p1["won"] == 1.0

    # Side 1 inflicted only the depth-3 pair; its SALT must not include the
    # 22 rows it received.
    assert p2["cpm"] == pytest.approx(1 / minutes)
    assert p2["cur"] == pytest.approx(2.0)
    assert p2["salt_per_min"] == pytest.approx(16 * 3 / 60.0988 / minutes)
    assert p2["garbage_per_min"] == pytest.approx(2 / minutes)
    assert p2["won"] == 0.0

    # Same match; both records feed the grader in skill_grade's feature order.
    assert p1["match"] == p2["match"]
    for g in games:
        vec, missing = skill_grade._game_features(g)
        assert not missing
        assert vec == pytest.approx([g[k] for k in skill_grade.BASE_FEATURES])


def test_env_pool_mode_grades_only_the_learner() -> None:
    env = _bare_env(pool_mode=True)
    _play_synthetic_match(env)
    games = env.pop_skill_games()
    assert len(games) == 1
    assert games[0]["cur"] == pytest.approx(3.0)


def test_env_marks_salt_unknown_for_libraries_without_it() -> None:
    env = _bare_env()
    env._accumulate_volleys([
        VsVolley(pair=0, receiver=1, size=2, cols=(0, 1), colors=(0, 0), frame=0),
    ])
    env._record_match(0, np.array([VS_OUTCOME_WIN, VS_OUTCOME_LOSS]), 3600)
    p1, p2 = env.pop_skill_games()
    assert math.isnan(p1["salt_per_min"])
    assert p2["salt_per_min"] == 0.0  # side 1 sent nothing, so nothing unknown


def test_grade_inputs_are_graded_by_the_fit_model() -> None:
    model_path = skill_grade.DEFAULT_MODEL
    if not model_path.is_file():
        pytest.skip("no fitted skill-grade model")
    import json

    model = json.loads(model_path.read_text())
    env = _bare_env()
    _play_synthetic_match(env)
    games = env.pop_skill_games()
    X = np.asarray([[g[k] for k in skill_grade.BASE_FEATURES] for g in games])
    # Synthetic inputs are inside the human feature range the model was fit on.
    for j, name in enumerate(skill_grade.BASE_FEATURES):
        lo, hi = model["feature_range"][name]
        assert (lo <= X[:, j]).all() and (X[:, j] <= hi).all(), name
    rating, _converged, _iters = skill_grade.self_play_rating(model, X)
    assert math.isfinite(rating)
