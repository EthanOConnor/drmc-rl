"""Window labels, board + trigger features and a held-out logistic fit for the showy knob.

    python -m tools.showy_knob.fit --extracts 'ex/2026-*.npz' --label t2_k4 --fraction 0.35 \
        --out drmc_rl/style/models/showy_t2k4_v1.json --report fit_report.json

Labels (per placement t, same player and game): ``t2_kK`` = a T2+ clear
(showiness >= 30) in placements t+1..t+K; ``t3_kK`` >= 42; ``t1_kK`` >= 27;
``hc_kK`` = a horizontal clear with 2+ lines. The placement's own clear is not
part of the label: the knob scores it exactly. Players are split held-out by
hash (pid % 5 == 0 is the test set). Needs scikit-learn (fit time only; the
runtime model is pure numpy).
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np

from drmc_rl.style import showy_knob as sk

EVENTS = dict(t1=lambda s, h, l: s >= 27, t2=lambda s, h, l: s >= 30, t3=lambda s, h, l: s >= 42,
              hc=lambda s, h, l: (h > 0) & (l >= 2))


def window_labels(seq, event, k):
    n = len(seq)
    last = np.r_[np.flatnonzero(np.diff(seq)), n - 1]
    end = np.repeat(last, np.diff(np.r_[-1, last]))
    idx = np.arange(n)
    cs = np.cumsum(event.astype(np.int32))
    return (cs[np.minimum(idx + k, end)] - cs[idx]) > 0


def dataset(paths, labels, fraction, seed=1):
    rng = np.random.default_rng(seed)
    X, Y, P = [], {k: [] for k in labels}, []
    for path in sorted(paths):
        d = np.load(path)
        keep = np.flatnonzero(d["valid"])
        keep = keep[rng.random(len(keep)) < fraction]
        after = d["after"][keep]
        X.append(np.concatenate([sk.board_features(after), sk.trigger_features(after)], axis=1))
        for lab in labels:
            name, k = lab.split("_k")
            Y[lab].append(window_labels(d["seq"], EVENTS[name](d["score"], d["horiz"], d["lines"]), int(k))[keep])
        P.append(d["pid"][keep])
    return np.concatenate(X), {k: np.concatenate(v) for k, v in Y.items()}, np.concatenate(P)


def main(argv=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--extracts", required=True)
    ap.add_argument("--label", default="t2_k4")
    ap.add_argument("--fraction", type=float, default=0.35)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report")
    args = ap.parse_args(argv)
    X, Y, pid = dataset(glob.glob(args.extracts), [args.label], args.fraction)
    names = list(sk.FEATURE_NAMES) + list(sk.TRIGGER_NAMES)
    y, test = Y[args.label], pid % 5 == 0
    mean, scale = X[~test].mean(0), X[~test].std(0) + 1e-6
    Z = (X - mean) / scale
    lr = LogisticRegression(C=1.0, max_iter=400).fit(Z[~test], y[~test])
    p = lr.predict_proba(Z[test])[:, 1]
    q = np.quantile(p, np.linspace(0, 1, 11))
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 9)
    report = dict(label=args.label, train=int((~test).sum()), test=int(test.sum()), base_rate=float(y[test].mean()),
                  auc=float(roc_auc_score(y[test], p)), ap=float(average_precision_score(y[test], p)),
                  top1pct_rate=float(y[test][p >= np.quantile(p, 0.99)].mean()),
                  calibration=[[float(p[b == i].mean()), float(y[test][b == i].mean())] for i in range(10)],
                  coef=sorted(((n, float(c)) for n, c in zip(names, lr.coef_[0])), key=lambda t: -abs(t[1])))
    spec = dict(schema=sk.SCHEMA, label=args.label, features=names, mean=mean.tolist(), scale=scale.tolist(),
                coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]), auc_heldout=report["auc"],
                base_rate=report["base_rate"])
    with open(args.out, "w") as f:
        json.dump(spec, f)
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ("coef", "calibration")}))


if __name__ == "__main__":
    main()
