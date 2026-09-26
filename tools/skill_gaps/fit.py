"""Fit a skill-gap knob model: human P(event within k placements | afterstate), player-held-out.

    python -m tools.skill_gaps.fit --rows 'data/human-2026-0[4-9].npz' --event vclear --k 4 \
        --min-rating 2000 --fraction 0.25 --sign 1 --out drmc_rl/style/models/gap_vclear_k4_v1.json \
        --report report.json

Rows come from ``tools.skill_gaps.extract human``. The label for placement t is "the event
happens at one of the same player's placements t+1..t+k" (the placement's own outcome is
scored exactly by the knob's immediate rule, not the model). Players are split by hash
(``group % 5 == 0`` is the held-out test set). The fit is an L2 logistic regression (C=1 on
standardized columns, the same objective as scikit-learn's default) solved by Newton steps in
numpy, so no extra dependency is needed. The output is a ``drmc-gap-knob-v1`` spec.
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np

from drmc_rl.style import gap_knob as gk
from drmc_rl.style import showy_knob as sk
from tools.skill_gaps import features as sf

EVENTS = {
    # name: per-row event on (rows d, gap-stat dict f)
    "vclear": lambda d, f: d["vcleared"] > 0,
    "vclear2": lambda d, f: d["vcleared"] >= 2,
    "danger": lambda d, f: f["h34"] >= 12,
    "cover": lambda d, f: f["d_covered"] > 0,
    "attack": lambda d, f: d["lines"] >= 2,
    "attack3": lambda d, f: d["lines"] >= 3,
    "isolate": lambda d, f: f["d_isolated"] > 0,
    "strand": lambda d, f: f["edge_strand"] > 0,
}


def window(seq, event, k):
    n = len(seq)
    last = np.r_[np.flatnonzero(np.diff(seq)), n - 1]
    end = np.repeat(last, np.diff(np.r_[-1, last]))
    cs = np.r_[0, np.cumsum(event.astype(np.int64))]
    idx = np.arange(n)
    lo, hi = np.minimum(idx + 1, end + 1), np.minimum(idx + k, end) + 1
    return (cs[np.maximum(hi, lo)] - cs[lo]) > 0


def row_stats(d):
    a = sf.board_stats(d["after"])
    r = sf.board_stats(d["root"])
    return dict(h34=a["h34"], d_covered=a["covered_viruses"] - r["covered_viruses"],
                d_isolated=a["isolated"] - r["isolated"], edge_strand=a["edge_strand"])


def dataset(paths, event, k, fraction, min_rating, feature_set, max_viruses=None, pool_group=None, seed=1):
    rng = np.random.default_rng(seed)
    names = gk.all_names()
    X, Y, G = [], [], []
    for path in sorted(paths):
        d = dict(np.load(path))
        f = row_stats(d)
        y = window(d["seq"], EVENTS[event](d, f), k)
        ok = (d["group"] == pool_group) if pool_group is not None else (d["rating"] >= min_rating)
        if max_viruses is not None:
            ok &= ((d["root"] & 0xF0) == 0xD0).sum(axis=1) <= max_viruses
        keep = np.flatnonzero(ok & (rng.random(len(y)) < fraction))
        after = d["after"][keep]
        board = sk.board_features(after)
        trig = sk.trigger_features(after) if feature_set in ("showy+trig", "all") else np.zeros((len(keep), len(sk.TRIGGER_NAMES)), np.float32)
        gap = gk.gap_features(after) if feature_set in ("gap", "all") else np.zeros((len(keep), len(gk.GAP_NAMES)), np.float32)
        X.append(np.concatenate([board, trig, gap], axis=1))
        Y.append(y[keep])
        G.append((d["seq"] if pool_group is not None else d["group"])[keep])
        print(path, len(keep), float(y[keep].mean()), flush=True)
    cols = {"showy": list(sk.FEATURE_NAMES), "showy+trig": list(sk.FEATURE_NAMES) + list(sk.TRIGGER_NAMES),
            "gap": list(sk.FEATURE_NAMES) + list(gk.GAP_NAMES), "all": names}[feature_set]
    idx = [names.index(c) for c in cols]
    return np.concatenate(X)[:, idx], np.concatenate(Y), np.concatenate(G), cols


def logistic(Z, y, C=1.0, iters=50):
    """L2 logistic regression (penalty ||w||^2 / 2C, intercept unpenalized) by Newton steps."""
    n, p = Z.shape
    A = np.c_[Z, np.ones(n)]
    w = np.zeros(p + 1)
    reg = np.r_[np.full(p, 1.0 / C), 0.0]
    for _ in range(iters):
        q = 1 / (1 + np.exp(-(A @ w)))
        g = A.T @ (q - y) + reg * w
        H = (A * (q * (1 - q))[:, None]).T @ A + np.diag(reg + 1e-9)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w[:-1], float(w[-1])


def auc(y, p):
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    # average ties
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    pos = y.astype(bool)
    npos, nneg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def average_precision(y, p):
    order = np.argsort(-p, kind="mergesort")
    ys = y[order].astype(np.float64)
    tp = np.cumsum(ys)
    precision = tp / np.arange(1, len(ys) + 1)
    return float((precision * ys).sum() / max(ys.sum(), 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", required=True)
    ap.add_argument("--event", required=True, choices=sorted(EVENTS))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--sign", type=float, default=1.0)
    ap.add_argument("--features", default="all", choices=("showy", "showy+trig", "gap", "all"))
    ap.add_argument("--fraction", type=float, default=0.25)
    ap.add_argument("--min-rating", type=float, default=2000.0)
    ap.add_argument("--max-viruses", type=int, help="fit only on decisions with at most this many viruses (stored: "
                    "the knob is inert on bottles with more)")
    ap.add_argument("--pool-group", type=int, help="fit on pool rows of this entrant index instead of rated humans")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report")
    args = ap.parse_args(argv)
    X, y, g, names = dataset(glob.glob(args.rows), args.event, args.k, args.fraction, args.min_rating, args.features,
                             args.max_viruses, args.pool_group)
    y = y.astype(np.float64)
    test = g % 5 == 0          # held-out players (pool: held-out sequences)
    mean, scale = X[~test].mean(0), X[~test].std(0) + 1e-6
    Z = (X - mean) / scale
    coef, intercept = logistic(Z[~test], y[~test])
    p = 1 / (1 + np.exp(-(Z[test] @ coef + intercept)))
    q = np.quantile(p, np.linspace(0, 1, 11))
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 9)
    report = dict(event=args.event, k=args.k, features=args.features, train=int((~test).sum()), test=int(test.sum()),
                  base_rate=float(y[test].mean()), auc=auc(y[test], p), ap=average_precision(y[test], p),
                  top_decile_rate=float(y[test][p >= q[9]].mean()), bottom_decile_rate=float(y[test][p <= q[1]].mean()),
                  calibration=[[float(p[b == i].mean()), float(y[test][b == i].mean())] for i in range(10)],
                  coef=sorted(((n, float(c)) for n, c in zip(names, coef)), key=lambda t: -abs(t[1]))[:15])
    spec = dict(schema=gk.SCHEMA, label=f"{args.event}_k{args.k}", sign=args.sign, features=names,
                mean=mean.astype(float).tolist(), scale=scale.astype(float).tolist(), coef=coef.tolist(),
                intercept=intercept, auc_heldout=report["auc"], ap_heldout=report["ap"], base_rate=report["base_rate"],
                **({"max_root_viruses": args.max_viruses} if args.max_viruses is not None else {}),
                source="pool" if args.pool_group is not None else f"human>={args.min_rating:g}")
    with open(args.out, "w") as f:
        json.dump(spec, f)
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ("calibration", "coef")}))


if __name__ == "__main__":
    main()
