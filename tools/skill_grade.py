"""Skill grading: estimate a Dr. Mario VS player's fightcadeRatings WHR rating
from in-game performance metrics.

Purpose: grade our RL agent during self-play training. The agent cannot play
on Fightcade, so we regress DrMC-style play metrics (CPM, CUR, SPD, SALT/min,
pills/min, garbage/min) onto the corrected WHR-C rating (``eC``) of human
players from the fightcadeRatings project, then apply that regression to the
agent's own per-game metrics.

Data source (read-only): ../fightcadeRatings/data/drmario.sqlite (table
``crown``, per-crown metrics for winner and loser) joined to
../fightcadeRatings/data/out/players.json (per-player WHR-C trajectory,
``traj[].eC`` at ``traj[].day``; the rating nearest the crown's day is used).
Degenerate crowns (length < 30 s or < 5 pills on either side — known
extraction artifacts) are dropped. Each valid crown yields two samples
(winner-side, loser-side). The win/loss outcome is deliberately NOT a
feature: we grade playstyle, not results.

Model: ridge regression on standardized degree-2 features (base + squares +
pairwise interactions), numpy-only. Samples are weighted 1/n_crowns(player)
so frequent players don't dominate, and cross-validation folds are grouped
by player so a player's rating never appears in both train and test.

HONEST CAVEATS
--------------
1. This grades *playstyle metrics*, not head-to-head strength. Two players
   with identical CPM/SPD profiles can have very different ratings; the CV
   MAE quantifies exactly how blurry this lens is. Treat outputs as a coarse
   skill estimate with the reported uncertainty band, not a true rating.
2. Ratings beyond the human metric range are EXTRAPOLATION. If the agent's
   CPM or SPD exceeds anything in the human corpus, the linear-in-features
   model will happily emit a number that no human data supports. Grade mode
   flags out-of-range features per game.
3. The human samples come from games against other humans; self-play metrics
   (e.g., SALT inflicted on an equally strong copy of yourself) are not
   distributed identically. Comparisons across training stages are more
   meaningful than the absolute level.
4. The corpus is small (~hundreds of crowns, ~dozens of rated players) and
   growing; refit periodically (``fit`` mode is cheap).

Usage:
    python tools/skill_grade.py fit
        [--db ../fightcadeRatings/data/drmario.sqlite]
        [--players ../fightcadeRatings/data/out/players.json]
        [--out data/skill_grade_model.json]
    python tools/skill_grade.py grade games.jsonl [--model data/skill_grade_model.json]

Grade input: JSON list or JSONL, one object per game, with either the rate
features directly (cpm, cur, spd, salt_per_min, pills_per_min,
garbage_per_min) or raw totals plus length_s (salt, cur, cpm, spd, pills,
garbage, length_s) from which rates are derived.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO.parent / "fightcadeRatings" / "data" / "drmario.sqlite"
DEFAULT_PLAYERS = REPO.parent / "fightcadeRatings" / "data" / "out" / "players.json"
DEFAULT_MODEL = REPO / "data" / "skill_grade_model.json"

BASE_FEATURES = ["cpm", "cur", "spd", "salt_per_min", "pills_per_min", "garbage_per_min"]

# Filters for known extraction artifacts in the crown table.
MIN_LENGTH_S = 30.0
MIN_PILLS = 5


# ---------------------------------------------------------------- dataset

def _rating_lookup(players_path: Path):
    """name -> (sorted days, eC array) from WHR-C trajectories."""
    players = json.loads(players_path.read_text())
    table = {}
    for p in players:
        traj = p.get("traj") or []
        days = [t["day"] for t in traj]
        ecs = [t["eC"] for t in traj]
        if days:
            table[p["name"]] = (days, ecs)
    return table


def _rating_at(table, name, day):
    entry = table.get(name)
    if entry is None:
        return None
    days, ecs = entry
    i = bisect.bisect_left(days, day)
    if i == 0:
        return ecs[0]
    if i == len(days):
        return ecs[-1]
    # nearest day
    return ecs[i] if days[i] - day < day - days[i - 1] else ecs[i - 1]


def build_dataset(db_path: Path, players_path: Path):
    """Per-crown per-player samples: (features, rating, player name)."""
    table = _rating_lookup(players_path)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT day, winner, loser, length_s,"
        " salt_w, salt_l, cur_w, cur_l, cpm_w, cpm_l, spd_w, spd_l,"
        " pills_w, pills_l, garbage_w, garbage_l FROM crown"
        " WHERE length_s >= ? AND pills_w >= ? AND pills_l >= ?",
        (MIN_LENGTH_S, MIN_PILLS, MIN_PILLS),
    ).fetchall()
    con.close()

    X, y, groups = [], [], []
    n_crowns = 0
    for (day, winner, loser, length_s, salt_w, salt_l, cur_w, cur_l, cpm_w,
         cpm_l, spd_w, spd_l, pills_w, pills_l, garbage_w, garbage_l) in rows:
        minutes = length_s / 60.0
        n_crowns += 1
        for name, salt, cur, cpm, spd, pills, garbage in (
            (winner, salt_w, cur_w, cpm_w, spd_w, pills_w, garbage_w),
            (loser, salt_l, cur_l, cpm_l, spd_l, pills_l, garbage_l),
        ):
            rating = _rating_at(table, name, day)
            if rating is None:
                continue
            X.append([cpm, cur, spd, salt / minutes, pills / minutes, garbage / minutes])
            y.append(rating)
            groups.append(name)
    return np.asarray(X, float), np.asarray(y, float), groups, n_crowns


# ---------------------------------------------------------------- model

def expand(X):
    """Degree-2 expansion: base, squares, pairwise interactions."""
    cols = [X]
    names = list(BASE_FEATURES)
    n = X.shape[1]
    for i in range(n):
        for j in range(i, n):
            cols.append((X[:, i] * X[:, j])[:, None])
            names.append(
                f"{BASE_FEATURES[i]}^2" if i == j
                else f"{BASE_FEATURES[i]}*{BASE_FEATURES[j]}"
            )
    return np.hstack(cols), names


def ridge_fit(Z, y, w, alpha):
    """Weighted ridge on standardized Z; returns (mu, sd, coef, intercept)."""
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0)
    sd[sd == 0] = 1.0
    Zs = (Z - mu) / sd
    sw = np.sqrt(w)
    A = Zs * sw[:, None]
    b = (y - np.average(y, weights=w)) * sw
    coef = np.linalg.solve(A.T @ A + alpha * np.eye(Z.shape[1]), A.T @ b)
    intercept = np.average(y - Zs @ coef, weights=w)
    return mu, sd, coef, intercept


def ridge_predict(model, Z):
    Zs = (Z - np.asarray(model["mu"])) / np.asarray(model["sd"])
    return Zs @ np.asarray(model["coef"]) + model["intercept"]


def group_kfold(groups, k):
    """Deterministic grouped folds, balanced by group sample count."""
    from collections import defaultdict
    idx_by_g = defaultdict(list)
    for i, g in enumerate(groups):
        idx_by_g[g].append(i)
    # largest groups first, assign to lightest fold
    folds = [[] for _ in range(k)]
    for g in sorted(idx_by_g, key=lambda g: -len(idx_by_g[g])):
        tgt = min(range(k), key=lambda f: len(folds[f]))
        folds[tgt].extend(idx_by_g[g])
    return [np.asarray(f) for f in folds if f]


def cross_validate(Z, y, w, groups, alpha, k=5):
    folds = group_kfold(groups, k)
    preds = np.full(len(y), np.nan)
    for test in folds:
        train = np.setdiff1d(np.arange(len(y)), test)
        mu, sd, coef, b0 = ridge_fit(Z[train], y[train], w[train], alpha)
        m = {"mu": mu, "sd": sd, "coef": coef, "intercept": b0}
        preds[test] = ridge_predict(m, Z[test])
    err = preds - y
    mae = float(np.average(np.abs(err), weights=w))
    resid_std = float(math.sqrt(np.average(err**2, weights=w)))
    return mae, resid_std, preds


# ---------------------------------------------------------------- modes

def cmd_fit(args):
    X, y, groups, n_crowns = build_dataset(Path(args.db), Path(args.players))
    n_players = len(set(groups))
    print(f"dataset: {len(y)} samples from {n_crowns} crowns, {n_players} players")
    print(f"target (WHR-C eC) range: {y.min():.0f}..{y.max():.0f}, mean {y.mean():.0f}")

    # per-player weights: each player counts equally
    from collections import Counter
    counts = Counter(groups)
    w = np.asarray([1.0 / counts[g] for g in groups])
    w *= len(w) / w.sum()

    # direction sanity checks: weighted correlation of base features vs rating
    print("feature -> rating correlations (weighted):")
    for j, name in enumerate(BASE_FEATURES):
        xm = np.average(X[:, j], weights=w)
        ym = np.average(y, weights=w)
        cov = np.average((X[:, j] - xm) * (y - ym), weights=w)
        sx = math.sqrt(np.average((X[:, j] - xm) ** 2, weights=w))
        sy = math.sqrt(np.average((y - ym) ** 2, weights=w))
        print(f"  {name:16s} r = {cov / (sx * sy):+.3f}")

    Z, feat_names = expand(X)
    best = None
    for alpha in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
        mae, resid_std, _ = cross_validate(Z, y, w, groups, alpha)
        print(f"  alpha={alpha:<6g} CV MAE={mae:7.1f}  resid std={resid_std:7.1f}")
        if best is None or mae < best[1]:
            best = (alpha, mae, resid_std)
    alpha, cv_mae, resid_std = best
    print(f"selected alpha={alpha}: CV MAE={cv_mae:.1f} Elo, residual std={resid_std:.1f} Elo")

    mu, sd, coef, b0 = ridge_fit(Z, y, w, alpha)
    model = {
        "kind": "ridge_poly2",
        "base_features": BASE_FEATURES,
        "feature_names": feat_names,
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "coef": coef.tolist(),
        "intercept": float(b0),
        "alpha": alpha,
        "cv_mae": round(cv_mae, 2),
        "resid_std": round(resid_std, 2),
        "n_samples": int(len(y)),
        "n_crowns": int(n_crowns),
        "n_players": int(n_players),
        "target_range": [float(y.min()), float(y.max())],
        "feature_range": {
            name: [float(X[:, j].min()), float(X[:, j].max())]
            for j, name in enumerate(BASE_FEATURES)
        },
        "filters": {"min_length_s": MIN_LENGTH_S, "min_pills": MIN_PILLS},
        "rating_basis": "WHR-C (corrected), set-basis, players.json traj eC nearest day",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, indent=1))
    print(f"model written to {out}")


def _game_features(rec):
    """Extract base feature vector from one game record; returns (vec, missing)."""
    vec, missing = [], []
    minutes = None
    if "length_s" in rec:
        minutes = float(rec["length_s"]) / 60.0
    for name in BASE_FEATURES:
        if name in rec:
            vec.append(float(rec[name]))
            continue
        raw = name.replace("_per_min", "")
        if name.endswith("_per_min") and raw in rec and minutes:
            vec.append(float(rec[raw]) / minutes)
        else:
            missing.append(name)
            vec.append(math.nan)
    return vec, missing


def cmd_grade(args):
    model = json.loads(Path(args.model).read_text())
    text = Path(args.input).read_text() if args.input != "-" else sys.stdin.read()
    text = text.strip()
    if text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    rows, ok_idx = [], []
    for i, rec in enumerate(records):
        vec, missing = _game_features(rec)
        if missing:
            print(json.dumps({"index": i, "error": f"missing features: {missing}"}))
            continue
        rows.append(vec)
        ok_idx.append(i)

    if not rows:
        return
    X = np.asarray(rows, float)
    Z, _ = expand(X)
    preds = ridge_predict(model, Z)
    franges = model["feature_range"]
    sigma = model["resid_std"]

    per_game = []
    for vec, pred, i in zip(X, preds, ok_idx):
        oob = [
            name for j, name in enumerate(BASE_FEATURES)
            if not (franges[name][0] <= vec[j] <= franges[name][1])
        ]
        out = {
            "index": i,
            "rating": round(float(pred), 1),
            "sigma": sigma,
            "extrapolated_features": oob,
        }
        per_game.append(out)
        print(json.dumps(out))

    n = len(per_game)
    mean = float(np.mean([g["rating"] for g in per_game]))
    summary = {
        "n_games": n,
        "rating_mean": round(mean, 1),
        # per-game noise shrinks with n; model bias (resid_std is dominated by
        # between-player error) does not, so report both.
        "sem": round(sigma / math.sqrt(n), 1),
        "model_sigma": sigma,
        "extrapolated_games": sum(1 for g in per_game if g["extrapolated_features"]),
    }
    print(json.dumps({"summary": summary}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("fit", help="fit rating regression from fightcadeRatings data")
    f.add_argument("--db", default=str(DEFAULT_DB))
    f.add_argument("--players", default=str(DEFAULT_PLAYERS))
    f.add_argument("--out", default=str(DEFAULT_MODEL))
    g = sub.add_parser("grade", help="grade agent games (JSON list or JSONL; '-' = stdin)")
    g.add_argument("input")
    g.add_argument("--model", default=str(DEFAULT_MODEL))
    args = ap.parse_args()
    if args.mode == "fit":
        cmd_fit(args)
    else:
        cmd_grade(args)


if __name__ == "__main__":
    main()
