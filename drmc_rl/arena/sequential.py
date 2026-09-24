"""Always-valid sequential stopping for side-swapped arena comparisons.

The unit of evidence is one reset seed: the candidate's mean score over its
two side-swapped games, a bounded value in [0, 1]. Seeds are drawn
independently of play, so seed scores are i.i.d. and a betting confidence
sequence (Waudby-Smith & Ramdas, 2023, predictable plug-in hedged capital)
gives lower and upper bounds that hold simultaneously at every number of seeds.
The arena may therefore look after every batch, in any batch sizes, and stop
as soon as the pre-registered question is decided without inflating error.

Error is split between the two kinds of look. Interim looks use the
confidence sequence at ``early_alpha`` per side (default one fifth of
``alpha``). The final look at the game budget uses the ordinary fixed-sample
one-sided Student interval on seed scores at ``alpha - early_alpha``. A wrong
answer requires the sequence to miss at some look or the fixed interval to
miss at the budget, so by the union bound each one-sided error is at most
``alpha``, while the final look loses little power relative to the fixed
design (its critical value moves from 1.96 to about 2.05 at 2.5%/2.0%).

For a threshold question a wrong PASS needs mu < threshold and a missed lower
bound, so a false PASS has probability at most ``alpha``. An early FAIL means
the upper bound is below the bar (also at most ``alpha`` wrong); a FAIL at the
budget is the pre-registered reading "lower bound below the bar" and carries
the design's ordinary power, not an error guarantee. For the equivalence
question (is the score within ``margin`` of ``centre``), a wrong
"equivalent" has probability at most ``alpha``; a wrong "different" at most
``2 * alpha`` when the true score is inside the band. ``alpha = 0.025`` per
side matches a two-sided 95% interval. For a family of comparisons that must
hold simultaneously (for example seven paces) divide ``alpha`` by the family
size.

The bet size is capped by a constant below one, not by ``c / m``, so every
capital process is monotone in the candidate mean. The rejected set is then an
interval and a grid evaluation is conservative: the reported bound is a grid
point already rejected, never an interpolated value. Censored games (timeouts)
enter lower bounds as losses and upper bounds as wins. The fixed interval's
variance includes one prior pseudo-observation of variance 1/4 (as in the
betting plug-in), so an all-draw run never reports zero width.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import NormalDist

import numpy as np

QUESTIONS = ("threshold", "equivalence")
_GRID = np.linspace(0.0, 1.0, 4001)


@dataclass(frozen=True)
class SequentialRule:
    question: str
    alpha: float = 0.025
    threshold: float | None = None
    margin: float | None = None
    centre: float = 0.5
    bet_cap: float = 0.9
    early_alpha: float | None = None

    def __post_init__(self):
        if self.question not in QUESTIONS:
            raise ValueError(f"sequential question must be one of {QUESTIONS}")
        if not 0 < self.alpha < 0.5:
            raise ValueError("sequential alpha is a one-sided error in (0, 0.5)")
        if self.early_alpha is not None and not 0 < self.early_alpha < self.alpha:
            raise ValueError("early_alpha must be a strict part of alpha")
        if not 0 < self.bet_cap < 1:
            raise ValueError("bet_cap must be in (0, 1) to keep capital monotone")
        if self.question == "threshold" and (self.threshold is None or not 0 < self.threshold < 1):
            raise ValueError("threshold questions need a bar in (0, 1)")
        if self.question == "equivalence" and (
                self.margin is None or not 0 < self.margin
                or not 0 <= self.centre - self.margin < self.centre + self.margin <= 1):
            raise ValueError("equivalence questions need a margin inside [0, 1] around the centre")

    @property
    def interim_alpha(self):
        return self.alpha / 5 if self.early_alpha is None else self.early_alpha

    @property
    def final_alpha(self):
        return self.alpha - self.interim_alpha

    @classmethod
    def from_config(cls, value):
        if value is None or isinstance(value, cls):
            return value
        allowed = {"question", "alpha", "threshold", "margin", "centre", "bet_cap", "early_alpha"}
        unknown = set(value) - allowed - {"look_games", "identity_probe_games"}
        if unknown:
            raise ValueError(f"unknown sequential settings: {sorted(unknown)}")
        return cls(**{k: v for k, v in value.items() if k in allowed})


def seed_scores(rows):
    """Complete side-swapped seeds in schedule order: (lower, upper, censored games).

    Incomplete seeds are ignored until their partner game arrives. Order is
    the scheduled game index, which is fixed before any outcome is seen.
    """
    seeds = {}
    for row in rows:
        seeds.setdefault(row["seed"], {})[row["side"]] = row
    complete = sorted((min(r["index"] for r in games.values()), games)
                      for games in seeds.values() if set(games) == {0, 1})
    lower, upper, censored = [], [], 0
    for _, games in complete:
        low = high = 0.0
        for row in games.values():
            if row.get("score") is None or row.get("reason") == "timeout":
                censored += 1
                high += 1.0
            else:
                low += float(row["score"])
                high += float(row["score"])
        lower.append(low / 2)
        upper.append(high / 2)
    return np.asarray(lower, float), np.asarray(upper, float), censored


def _lower_sequence(x, alpha, cap, grid=_GRID):
    """Running lower confidence bound for the mean of i.i.d. values in [0, 1]."""
    n = len(x)
    if n == 0:
        return np.zeros(0)
    t = np.arange(1, n + 1, dtype=float)
    # Predictable estimates use only x_1..x_{t-1} (WSR eq. 26 with priors 1/2, 1/4).
    mean_after = (0.5 + np.cumsum(x)) / (t + 1)
    square = np.cumsum((x - mean_after) ** 2)
    variance_before = (0.25 + np.concatenate(([0.0], square[:-1]))) / t
    bet = np.sqrt(2 * math.log(1 / alpha) / (variance_before * t * np.log1p(t)))
    bet = np.minimum(bet, cap)
    capital = np.cumsum(np.log1p(bet[:, None] * (x[:, None] - grid[None, :])), axis=0)
    rejected = capital >= math.log(1 / alpha)
    # Capital is non-increasing in m, so rejected grid points form a prefix.
    count = rejected.sum(axis=1)
    bound = np.where(count > 0, grid[np.maximum(count - 1, 0)], 0.0)
    return np.maximum.accumulate(bound)


def confidence_sequence(lower_scores, upper_scores, alpha, cap=0.9, grid=_GRID):
    """Simultaneous running (lower, upper) bounds after each complete seed."""
    low = _lower_sequence(np.asarray(lower_scores, float), alpha, cap, grid)
    high = 1.0 - _lower_sequence(1.0 - np.asarray(upper_scores, float), alpha, cap, grid)
    return low, high


def decide(rule: SequentialRule, low: float, high: float):
    if rule.question == "threshold":
        if low >= rule.threshold:
            return "pass"
        if high < rule.threshold:
            return "fail"
        return None
    band = (rule.centre - rule.margin, rule.centre + rule.margin)
    if band[0] < low and high < band[1]:
        return "equivalent"
    if low > band[1] or high < band[0]:
        return "different"
    return None


def _t_quantile(p, df):
    """Upper-tail Student quantile (Cornish-Fisher; <1e-3 absolute error for df >= 10)."""
    z = NormalDist().inv_cdf(1 - p)
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def fixed_interval(lower_scores, upper_scores, alpha):
    """One-sided Student bounds at the budget, with a 1/4 variance pseudo-observation."""
    bounds = []
    for x in (np.asarray(lower_scores, float), np.asarray(upper_scores, float)):
        n = len(x)
        if n < 2:
            return 0.0, 1.0
        variance = (0.25 + np.sum((x - x.mean()) ** 2)) / n
        bounds.append((x.mean(), _t_quantile(alpha, n - 1) * math.sqrt(variance / n)))
    return max(0.0, bounds[0][0] - bounds[0][1]), min(1.0, bounds[1][0] + bounds[1][1])


def evaluate(rule: SequentialRule, rows, budget_games: int):
    """Decision state for one comparison after the games recorded so far.

    Looks may happen at any batch boundary. Before the budget the decision
    uses the confidence sequence after the last complete seed; at the budget
    it uses the fixed-sample interval. ``stop`` is true once a decision is
    reached or the budget is exhausted.
    """
    lower, upper, censored = seed_scores(rows)
    pairs, games = len(lower), len(rows)
    exhausted = games >= budget_games
    low, high = confidence_sequence(lower, upper, rule.interim_alpha, rule.bet_cap)
    sequence = (float(low[-1]), float(high[-1])) if pairs else (0.0, 1.0)
    decision = decide(rule, *sequence)
    final = None
    if decision is None and exhausted:
        final = fixed_interval(lower, upper, rule.final_alpha)
        decision = decide(rule, *final) or ("fail" if rule.question == "threshold" else "undecided")
    return dict(
        rule=asdict(rule), decision=decision, stop=decision is not None,
        early=decision is not None and final is None and not exhausted,
        games=games, budget_games=int(budget_games), seeds=pairs, censored=censored,
        mean=float(np.mean(lower)) if pairs else None,
        confidence_sequence=[round(sequence[0], 6), round(sequence[1], 6)],
        final_interval=None if final is None else [round(final[0], 6), round(final[1], 6)],
    )
