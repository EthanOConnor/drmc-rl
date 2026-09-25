"""Anchored Bradley-Terry ratings with paired-seed robust uncertainty.

Model (one condition key at a time)::

    P(i beats j) = sigmoid(theta_i - theta_j),   theta_anchor = 0,
    theta_i ~ Normal(0, prior_sd^2)  (weak; only keeps separated records finite)

A draw scores 0.5 (a half win). The fit is the posterior mode (Newton's
method, deterministic), the uncertainty its Laplace curvature, widened to the
cluster-robust (sandwich) variance when games within a side-swapped seed pair
are correlated more than the model assumes. Each seed pair is one cluster; a
pair with a censored (timed-out) game is excluded entirely so every rated
result stays paired and side-balanced. Entrants not connected to the anchor by
games under this condition are reported as unanchored, never guessed.

Ratings are displayed on the Elo scale (400/ln 10 per logit) with the anchor
at ``anchor_rating``. Everything is a pure function of the game journal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import NormalDist

import numpy as np

ELO_SCALE = 400.0 / math.log(10.0)
MODEL = "anchored-bradley-terry-sandwich-v1"


@dataclass
class PairStats:
    """Complete seed pairs of one canonical pairing (i < j) under one condition."""
    pairs: int = 0        # side-swapped seed pairs (two games each)
    score: float = 0.0    # total score of entrant i
    square: float = 0.0   # sum over pairs of (pair score of i)^2
    draws: int = 0

    def add(self, pair_score: float, draws: int = 0):
        self.pairs += 1
        self.score += pair_score
        self.square += pair_score * pair_score
        self.draws += draws

    @property
    def games(self):
        return 2 * self.pairs


@dataclass
class Rating:
    entrant: str
    rating: float
    se: float
    games: int
    opponents: int
    score: float
    theta: float = 0.0
    anchored: bool = True

    def to_dict(self):
        return dict(entrant=self.entrant, rating=self.rating, se=self.se, games=self.games,
                    opponents=self.opponents, score=round(self.score, 4), anchored=self.anchored)


@dataclass
class Fit:
    anchor: str
    ratings: dict[str, Rating] = field(default_factory=dict)
    unanchored: list[str] = field(default_factory=list)
    games: int = 0
    iterations: int = 0
    covariance: np.ndarray | None = None
    index: dict[str, int] = field(default_factory=dict)

    def expected(self, a: str, b: str) -> float | None:
        """Model probability that ``a`` beats ``b`` (None when either is unanchored)."""
        if a not in self.ratings or b not in self.ratings:
            return None
        return 1.0 / (1.0 + math.exp(-(self.ratings[a].theta - self.ratings[b].theta)))

    def difference_se(self, a: str, b: str) -> float | None:
        """Standard error of rating(a) - rating(b) on the Elo scale."""
        if a not in self.ratings or b not in self.ratings or self.covariance is None:
            return None
        def row(e):
            v = np.zeros(len(self.index))
            if e in self.index:
                v[self.index[e]] = 1.0
            return v
        d = row(a) - row(b)
        return float(math.sqrt(max(d @ self.covariance @ d, 0.0)) * ELO_SCALE)


def _components(entrants, pairs):
    parent = {e: e for e in entrants}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for (i, j), stats in pairs.items():
        if stats.pairs:
            parent[find(i)] = find(j)
    return find


def fit(pairs: dict[tuple[str, str], PairStats], anchor: str, *, anchor_rating: float = 1500.0,
        prior_sd: float = 4.0, tolerance: float = 1e-10, max_iterations: int = 100) -> Fit:
    """Fit one condition. ``pairs`` maps canonical (i, j) with i < j to PairStats."""
    entrants = sorted({e for key in pairs for e in key} | {anchor})
    find = _components(entrants, pairs)
    root = find(anchor)
    connected = [e for e in entrants if find(e) == root]
    result = Fit(anchor=anchor, unanchored=[e for e in entrants if find(e) != root])
    free = [e for e in connected if e != anchor]
    index = {e: k for k, e in enumerate(free)}
    edges = [(index.get(i, -1), index.get(j, -1), s) for (i, j), s in sorted(pairs.items())
             if s.pairs and find(i) == root]
    n = len(free)
    theta = np.zeros(n)
    precision = 1.0 / (prior_sd * prior_sd)

    def full(t, k):
        return 0.0 if k < 0 else t[k]
    iterations = 0
    hessian = np.eye(n) * precision
    for iterations in range(1, max_iterations + 1):
        gradient = -precision * theta
        hessian = np.eye(n) * precision
        for i, j, s in edges:
            p = 1.0 / (1.0 + math.exp(-(full(theta, i) - full(theta, j))))
            residual = s.score - s.games * p
            w = s.games * p * (1.0 - p)
            if i >= 0:
                gradient[i] += residual
                hessian[i, i] += w
            if j >= 0:
                gradient[j] -= residual
                hessian[j, j] += w
            if i >= 0 and j >= 0:
                hessian[i, j] -= w
                hessian[j, i] -= w
        if n == 0:
            break
        step = np.linalg.solve(hessian, gradient)
        # Damped Newton keeps near-separated records stable.
        largest = float(np.max(np.abs(step)))
        if largest > 2.0:
            step *= 2.0 / largest
        theta += step
        if largest < tolerance:
            break
    covariance = np.linalg.inv(hessian) if n else np.zeros((0, 0))
    meat = np.zeros((n, n))
    for i, j, s in edges:
        p = 1.0 / (1.0 + math.exp(-(full(theta, i) - full(theta, j))))
        # Sum over seed-pair clusters of (pair score - 2p)^2.
        spread = max(s.square - 4.0 * p * s.score + 4.0 * p * p * s.pairs, 0.0)
        if i >= 0:
            meat[i, i] += spread
        if j >= 0:
            meat[j, j] += spread
        if i >= 0 and j >= 0:
            meat[i, j] -= spread
            meat[j, i] -= spread
    sandwich = covariance @ meat @ covariance if n else covariance
    variance = np.maximum(np.diag(covariance), np.diag(sandwich)) if n else np.zeros(0)
    # The reported covariance is the model covariance scaled to the robust variances.
    if n:
        scale = np.sqrt(variance / np.maximum(np.diag(covariance), 1e-300))
        covariance = covariance * np.outer(scale, scale)
    games, score, opponents = {}, {}, {}
    for (i, j), s in pairs.items():
        if not s.pairs or find(i) != root:
            continue
        for e, other, got in ((i, j, s.score), (j, i, s.games - s.score)):
            games[e] = games.get(e, 0) + s.games
            score[e] = score.get(e, 0.0) + got
            opponents.setdefault(e, set()).add(other)
    for e in connected:
        k = index.get(e, -1)
        t = full(theta, k)
        result.ratings[e] = Rating(entrant=e, rating=anchor_rating + ELO_SCALE * t,
                                   se=0.0 if k < 0 else float(math.sqrt(variance[k])) * ELO_SCALE,
                                   games=games.get(e, 0), opponents=len(opponents.get(e, ())),
                                   score=score.get(e, 0.0) / max(games.get(e, 0), 1), theta=t)
    result.games = sum(s.games for _, _, s in edges)
    result.iterations = iterations
    result.covariance = covariance
    result.index = index
    return result


def pooled(fits: dict[str, Fit], conditions: list[str], *, min_games: int = 1,
           weights: list[float] | None = None) -> dict[str, dict]:
    """Mean rating over a named condition set, only for entrants rated under every one.

    No imputation: an entrant missing (or unanchored, or under ``min_games``)
    in any condition of the set has no pooled rating. The per-condition errors
    are treated as independent (different games): with weights w (default all
    1), rating = sum(w r) / sum(w) and se = sqrt(sum(w^2 se^2)) / sum(w).
    """
    out = {}
    if not conditions or any(c not in fits for c in conditions):
        return out
    w = [1.0] * len(conditions) if weights is None else [float(x) for x in weights]
    total = sum(w)
    entrants = set.intersection(*[set(fits[c].ratings) for c in conditions])
    for e in sorted(entrants):
        rows = [fits[c].ratings[e] for c in conditions]
        if any(r.games < min_games for r in rows) and e not in {fits[c].anchor for c in conditions}:
            continue
        out[e] = dict(entrant=e, rating=sum(x * r.rating for x, r in zip(w, rows)) / total,
                      se=math.sqrt(sum((x * r.se) ** 2 for x, r in zip(w, rows))) / total,
                      games=sum(r.games for r in rows), conditions=len(rows))
    return out


def pooled_difference_se(fits: dict[str, Fit], conditions: list[str], a: str, b: str,
                         weights: list[float] | None = None) -> float | None:
    """SE of pooled(a) - pooled(b): per-condition difference variances (with covariance), weighted."""
    parts = [fits[c].difference_se(a, b) for c in conditions]
    if not parts or any(p is None for p in parts):
        return None
    w = [1.0] * len(parts) if weights is None else [float(x) for x in weights]
    return math.sqrt(sum((x * p) ** 2 for x, p in zip(w, parts))) / sum(w)


def superiority(difference: float, se: float | None) -> float | None:
    """Likelihood of superiority: P(true rating difference > 0) under the normal posterior approximation."""
    if se is None:
        return None
    if se <= 0:
        return 1.0 if difference > 0 else 0.0 if difference < 0 else 0.5
    return NormalDist().cdf(difference / se)
