
import numpy as np
import pytest

from drmc_rl.arena.sequential import (
    SequentialRule, confidence_sequence, decide, evaluate, fixed_interval, seed_scores,
)

GRID = np.linspace(0.0, 1.0, 801)


def _seed_scores(rng, runs, seeds, win_both, lose_both):
    """Side-swapped seed scores: 1 (won both), 0 (lost both), otherwise a split 0.5."""
    u = rng.random((runs, seeds))
    return np.where(u < win_both, 1.0, np.where(u < win_both + lose_both, 0.0, 0.5))


def _run(rule, scores, look):
    """The arena's decision procedure: confidence-sequence looks, fixed interval at the budget."""
    seeds = len(scores)
    low, high = confidence_sequence(scores, scores, rule.interim_alpha, rule.bet_cap, GRID)
    for t in range(look, seeds, look):
        decision = decide(rule, low[t - 1], high[t - 1])
        if decision:
            return decision, t
    return decide(rule, *fixed_interval(scores, scores, rule.final_alpha)), seeds


def _rate(decisions, value):
    return sum(d == value for d, _ in decisions) / len(decisions)


def test_threshold_false_pass_rate_on_the_boundary_is_nominal():
    # mu = 0.5 + (0.10 - 0.20) / 2 = 0.45 exactly on the pre-registered bar.
    rule = SequentialRule("threshold", threshold=0.45, alpha=0.025)
    rng = np.random.default_rng(20260924)
    runs = [_run(rule, x, 16) for x in _seed_scores(rng, 800, 128, 0.10, 0.20)]
    # 800 runs: binomial SE at 2.5% is 0.55%; allow three SEs of simulation noise.
    assert _rate(runs, "pass") <= 0.025 + 3 * 0.0055


def test_confidence_sequence_is_valid_when_peeking_after_every_seed():
    # Always-valid: stopping at the first seed whose lower bound clears mu.
    alpha, rng = 0.05, np.random.default_rng(7)
    misses = 0
    scores = _seed_scores(rng, 600, 256, 0.15, 0.15)  # mu = 0.5
    for x in scores:
        low, high = confidence_sequence(x, x, alpha, 0.9, GRID)
        misses += bool((low > 0.5).any())
    assert misses / len(scores) <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / len(scores))


def test_equivalence_false_equivalent_rate_on_the_margin_is_nominal():
    rule = SequentialRule("equivalence", margin=0.05, alpha=0.025)
    rng = np.random.default_rng(11)
    runs = [_run(rule, x, 16) for x in _seed_scores(rng, 800, 128, 0.15, 0.05)]  # mu = 0.55
    assert _rate(runs, "equivalent") <= 0.025 + 3 * 0.0055


def test_clear_answers_stop_early_and_keep_fixed_design_power():
    rng = np.random.default_rng(3)
    fail = SequentialRule("threshold", threshold=0.5)
    runs = [_run(fail, x, 16) for x in _seed_scores(rng, 200, 192, 0.03, 0.27)]  # mu = 0.38
    assert _rate(runs, "fail") == 1.0
    assert np.mean([t for _, t in runs]) < 0.6 * 192
    easy = SequentialRule("threshold", threshold=0.45)
    runs = [_run(easy, x, 16) for x in _seed_scores(rng, 200, 192, 0.15, 0.05)]  # mu = 0.55
    assert _rate(runs, "pass") == 1.0
    assert np.mean([t for _, t in runs]) < 0.6 * 192
    near = [_run(easy, x, 16) for x in _seed_scores(rng, 300, 192, 0.10, 0.10)]  # mu = 0.50
    assert _rate(near, "pass") > 0.75  # fixed 2.5% design: ~0.88


def test_all_split_mirror_pairs_decide_equivalence_before_the_budget():
    rule = SequentialRule("equivalence", margin=0.05)
    rows = [dict(seed=s, side=side, index=2*s+side, score=float(side), reason="topout")
            for s in range(1, 129) for side in (0, 1)]
    verdict = evaluate(rule, rows, 384)
    assert verdict["decision"] == "equivalent" and verdict["early"] and verdict["games"] == 256


def test_rows_use_complete_seeds_and_bound_censored_games():
    rows = [dict(seed=9, side=0, index=0, score=1.0, reason="clear"),
            dict(seed=9, side=1, index=1, score=None, reason="timeout"),
            dict(seed=4, side=0, index=2, score=0.0, reason="topout")]
    lower, upper, censored = seed_scores(rows)
    assert lower.tolist() == [0.5] and upper.tolist() == [1.0] and censored == 1
    state = evaluate(SequentialRule("threshold", threshold=0.45), rows, 128)
    assert state["seeds"] == 1 and state["decision"] is None and not state["stop"]
    assert state["games"] == 3 and state["budget_games"] == 128


def test_budget_without_a_decision_reads_as_the_preregistered_fail():
    rows = [dict(seed=s, side=side, index=2*s+side, score=[1.0, 0.0, 1.0, 1.0][(s + side) % 4], reason="clear")
            for s in range(8) for side in (0, 1)]
    state = evaluate(SequentialRule("threshold", threshold=0.7), rows, 16)
    assert state["stop"] and not state["early"] and state["decision"] == "fail"
    assert state["final_interval"] is not None


@pytest.mark.parametrize("bad", [dict(question="sign"), dict(question="threshold"),
                                 dict(question="equivalence", margin=0.6),
                                 dict(question="threshold", threshold=0.5, bet_cap=1.0),
                                 dict(question="threshold", threshold=0.5, early_alpha=0.05),
                                 dict(question="threshold", threshold=0.5, extra=1)])
def test_invalid_rules_are_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        SequentialRule.from_config(bad)
