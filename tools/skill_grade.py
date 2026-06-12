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

To control for opponent skill, each sample also carries ``opp_whr`` — the
OTHER side's eC rating at the crown's date (same nearest-day join). Samples
whose opponent is unrated are dropped. ``opp_whr`` is standardized and
included in the degree-2 expansion like every other base feature.

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
        [--opp-whr RATING | --self-play]

Grade input: JSON list or JSONL, one object per game, with either the rate
features directly (cpm, cur, spd, salt_per_min, pills_per_min,
garbage_per_min) or raw totals plus length_s (salt, cur, cpm, spd, pills,
garbage, length_s) from which rates are derived.

Grading a version-2 model requires an opponent rating. ``--opp-whr R`` uses
an explicit value; the default (``--self-play``) treats the opponent as a
copy of the graded agent and solves the fixed point r = f(metrics,
opp_whr=r) by iterating from the corpus mean rating until |Δ| < 1 Elo or 20
iterations. f is quadratic in opp_whr, so plain iteration converges whenever
|df/d opp_whr| < 1 near the fixed point; if successive deltas flip sign
(oscillation), the step is damped by 0.5 (cumulatively), which restores
convergence for any locally stable fixed point. Version-1 model files (no
``version`` field) have no opp_whr feature; grade detects this and ignores
the opponent options with a warning.
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
OPP_FEATURE = "opp_whr"
FIT_FEATURES = BASE_FEATURES + [OPP_FEATURE]
MODEL_VERSION = 2

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
    """Per-crown per-player samples: (features incl. opp_whr, rating, player name).

    A sample is kept only if BOTH sides are rated: the player's own rating is
    the target, the opponent's rating is the ``opp_whr`` feature.
    """
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
    n_dropped_unrated_opp = 0
    for (day, winner, loser, length_s, salt_w, salt_l, cur_w, cur_l, cpm_w,
         cpm_l, spd_w, spd_l, pills_w, pills_l, garbage_w, garbage_l) in rows:
        minutes = length_s / 60.0
        n_crowns += 1
        rating_w = _rating_at(table, winner, day)
        rating_l = _rating_at(table, loser, day)
        for name, rating, opp_rating, salt, cur, cpm, spd, pills, garbage in (
            (winner, rating_w, rating_l, salt_w, cur_w, cpm_w, spd_w, pills_w, garbage_w),
            (loser, rating_l, rating_w, salt_l, cur_l, cpm_l, spd_l, pills_l, garbage_l),
        ):
            if rating is None:
                continue
            if opp_rating is None:
                n_dropped_unrated_opp += 1
                continue
            X.append([cpm, cur, spd, salt / minutes, pills / minutes,
                      garbage / minutes, opp_rating])
            y.append(rating)
            groups.append(name)
    if n_dropped_unrated_opp:
        print(f"dropped {n_dropped_unrated_opp} samples with unrated opponent")
    return np.asarray(X, float), np.asarray(y, float), groups, n_crowns


# ---------------------------------------------------------------- model

def expand(X, base_names=None):
    """Degree-2 expansion: base, squares, pairwise interactions."""
    if base_names is None:
        base_names = BASE_FEATURES
    cols = [X]
    names = list(base_names)
    n = X.shape[1]
    for i in range(n):
        for j in range(i, n):
            cols.append((X[:, i] * X[:, j])[:, None])
            names.append(
                f"{base_names[i]}^2" if i == j
                else f"{base_names[i]}*{base_names[j]}"
            )
    return np.hstack(cols), names


def ridge_fit(Z, y, w, alpha):
    """Weighted ridge on standardized Z; returns (mu, sd, coef, intercept).

    ``alpha`` may be a scalar or a per-column penalty vector.
    """
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0)
    sd[sd == 0] = 1.0
    Zs = (Z - mu) / sd
    sw = np.sqrt(w)
    A = Zs * sw[:, None]
    b = (y - np.average(y, weights=w)) * sw
    pen = np.diag(np.broadcast_to(np.asarray(alpha, float), (Z.shape[1],)))
    coef = np.linalg.solve(A.T @ A + pen, A.T @ b)
    intercept = np.average(y - Zs @ coef, weights=w)
    return mu, sd, coef, intercept


def ridge_predict(model, Z):
    Zs = (Z - np.asarray(model["mu"])) / np.asarray(model["sd"])
    return Zs @ np.asarray(model["coef"]) + model["intercept"]


def model_version(model) -> int:
    return int(model.get("version", 1))


def predict_rating(model, X, opp_whr=None):
    """Per-sample rating predictions from metric features X (n, 6).

    For version>=2 models a scalar ``opp_whr`` (opponent rating) is required
    and is appended as the last base feature; version-1 models ignore it.
    """
    X = np.asarray(X, float)
    base = model["base_features"]
    if model_version(model) >= 2:
        if opp_whr is None:
            raise ValueError("opp_whr is required for version>=2 models")
        X = np.hstack([X, np.full((X.shape[0], 1), float(opp_whr))])
    Z, _ = expand(X, base)
    return ridge_predict(model, Z)


def self_play_rating(model, X, tol=1.0, max_iter=20):
    """Self-play rating: solve r = mean(f(metrics, opp_whr=r)).

    Iterates from the corpus mean rating. f is quadratic in opp_whr, so plain
    fixed-point iteration converges when |df/dr| < 1 near the solution; when
    successive deltas flip sign (oscillation) the step is damped by 0.5
    (cumulatively), which restores convergence for any locally stable fixed
    point. Stops when |Δ| < tol (Elo) or after max_iter iterations.

    Version-1 models have no opp_whr feature; the mean prediction is returned
    directly. Returns (rating, converged, n_iterations).
    """
    X = np.asarray(X, float)
    if model_version(model) < 2:
        return float(np.mean(predict_rating(model, X))), True, 1
    r = float(model["y_mean"])
    damp = 1.0
    prev_delta = None
    for it in range(1, max_iter + 1):
        delta = float(np.mean(predict_rating(model, X, opp_whr=r))) - r
        if prev_delta is not None and delta * prev_delta < 0:
            damp *= 0.5
        r += damp * delta
        if abs(delta) < tol:
            return r, True, it
        prev_delta = delta
    return r, False, max_iter


SELFPLAY_SLOPE_MAX = 0.7


def selfplay_slope(model, X_metrics, y_mean, span=1000.0, step=250.0):
    """Max |d pred / d opp_whr| of the self-play fixed-point map.

    Evaluated numerically at the given metric rows over opp_whr in
    [y_mean - span, y_mean + span]. The fixed point r = f(opp_whr=r) is only
    well-conditioned when this slope stays clearly below 1; corpora with
    strong matchmaking correlation push fits toward the degenerate
    "your rating == opponent's rating" shortcut (slope -> 1), which collapses
    self-play grading.
    """
    rs = np.arange(y_mean - span, y_mean + span + 1, step)
    preds = [float(np.mean(predict_rating(model, X_metrics, opp_whr=r))) for r in rs]
    return max(
        abs((preds[i + 1] - preds[i]) / (rs[i + 1] - rs[i]))
        for i in range(len(rs) - 1)
    )


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
    for j, name in enumerate(FIT_FEATURES):
        xm = np.average(X[:, j], weights=w)
        ym = np.average(y, weights=w)
        cov = np.average((X[:, j] - xm) * (y - ym), weights=w)
        sx = math.sqrt(np.average((X[:, j] - xm) ** 2, weights=w))
        sy = math.sqrt(np.average((y - ym) ** 2, weights=w))
        print(f"  {name:16s} r = {cov / (sx * sy):+.3f}")

    def sweep(Z):
        best = None
        for alpha in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
            mae, resid_std, _ = cross_validate(Z, y, w, groups, alpha)
            print(f"  alpha={alpha:<6g} CV MAE={mae:7.1f}  resid std={resid_std:7.1f}")
            if best is None or mae < best[1]:
                best = (alpha, mae, resid_std)
        return best

    # baseline: old config (metrics only, no opp_whr) on the same samples
    print("old config (no opp_whr):")
    Z_old, _ = expand(X[:, : len(BASE_FEATURES)], BASE_FEATURES)
    _, cv_mae_old, _ = sweep(Z_old)

    print("new config (with opp_whr):")
    Z, feat_names = expand(X, FIT_FEATURES)
    opp_cols = np.asarray([OPP_FEATURE in n for n in feat_names])
    X_mean = np.average(X[:, : len(BASE_FEATURES)], axis=0, weights=w)[None, :]
    y_mean = float(np.average(y, weights=w))

    # Sweep (base alpha) x (extra penalty on opp_whr-involving terms); admit
    # only fits whose self-play map slope stays below SELFPLAY_SLOPE_MAX —
    # matchmaking correlation otherwise drives the fit toward the degenerate
    # "rating == opponent rating" shortcut and self-play grading collapses
    # (observed 2026-06-12: fixed point 2500 -> 312 on a corpus refresh).
    best = None  # (mae, resid_std, alpha, mult, slope)
    for mult in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
        for alpha in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
            avec = np.where(opp_cols, alpha * mult, alpha)
            mae, resid_std, _ = cross_validate(Z, y, w, groups, avec)
            mu, sd, coef, b0 = ridge_fit(Z, y, w, avec)
            m = {"version": MODEL_VERSION, "base_features": FIT_FEATURES,
                 "mu": mu.tolist(), "sd": sd.tolist(), "coef": coef.tolist(),
                 "intercept": float(b0)}
            slope = selfplay_slope(m, X_mean, y_mean)
            ok = slope <= SELFPLAY_SLOPE_MAX
            print(f"  alpha={alpha:<6g} opp_mult={mult:<5g} CV MAE={mae:7.1f}  "
                  f"slope={slope:.2f}{'' if ok else '  (rejected)'}")
            if ok and (best is None or mae < best[0]):
                best = (mae, resid_std, alpha, mult, slope)
    if best is None:
        raise SystemExit("no admissible fit: every candidate exceeded "
                         f"self-play slope {SELFPLAY_SLOPE_MAX}")
    cv_mae, resid_std, alpha, opp_mult, slope = best
    print(f"selected alpha={alpha} opp_mult={opp_mult}: CV MAE={cv_mae:.1f} Elo, "
          f"residual std={resid_std:.1f} Elo, self-play slope={slope:.2f}")
    print(f"CV MAE old (no opp_whr) {cv_mae_old:.1f} -> new (with opp_whr) {cv_mae:.1f} Elo")

    alpha_vec = np.where(opp_cols, alpha * opp_mult, alpha)
    mu, sd, coef, b0 = ridge_fit(Z, y, w, alpha_vec)
    j_opp = feat_names.index(OPP_FEATURE)
    print(f"opp_whr standardized coef: {coef[j_opp]:+.1f} Elo per sd "
          f"(sd = {sd[j_opp]:.1f} Elo)")
    model = {
        "version": MODEL_VERSION,
        "kind": "ridge_poly2",
        "base_features": FIT_FEATURES,
        "metric_features": BASE_FEATURES,
        "feature_names": feat_names,
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "coef": coef.tolist(),
        "intercept": float(b0),
        "alpha": alpha,
        "opp_alpha_mult": opp_mult,
        "selfplay_slope": round(slope, 3),
        "cv_mae": round(cv_mae, 2),
        "cv_mae_no_opp": round(cv_mae_old, 2),
        "resid_std": round(resid_std, 2),
        "n_samples": int(len(y)),
        "n_crowns": int(n_crowns),
        "n_players": int(n_players),
        "target_range": [float(y.min()), float(y.max())],
        "y_mean": float(y.mean()),
        "feature_range": {
            name: [float(X[:, j].min()), float(X[:, j].max())]
            for j, name in enumerate(FIT_FEATURES)
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
    version = model_version(model)
    self_play = None
    if version < 2:
        if args.opp_whr is not None or args.self_play:
            print("note: model file is version 1 (no opp_whr feature); "
                  "opponent options ignored — refit to use them", file=sys.stderr)
        opp_used = None
        preds = predict_rating(model, X)
        mode = "v1_no_opp"
    elif args.opp_whr is not None:
        opp_used = float(args.opp_whr)
        preds = predict_rating(model, X, opp_whr=opp_used)
        mode = "fixed_opp"
    else:
        r, converged, iters = self_play_rating(model, X)
        opp_used = r
        preds = predict_rating(model, X, opp_whr=r)
        self_play = {"rating": round(r, 1), "converged": converged, "iterations": iters}
        mode = "self_play"
    franges = model["feature_range"]
    sigma = model["resid_std"]
    opp_oob = (
        opp_used is not None
        and OPP_FEATURE in franges
        and not (franges[OPP_FEATURE][0] <= opp_used <= franges[OPP_FEATURE][1])
    )

    per_game = []
    for vec, pred, i in zip(X, preds, ok_idx):
        oob = [
            name for j, name in enumerate(BASE_FEATURES)
            if not (franges[name][0] <= vec[j] <= franges[name][1])
        ]
        if opp_oob:
            oob.append(OPP_FEATURE)
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
        "mode": mode,
    }
    if opp_used is not None:
        summary["opp_whr"] = round(opp_used, 1)
    if self_play is not None:
        summary["self_play"] = self_play
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
    g.add_argument("--opp-whr", type=float, default=None,
                   help="explicit opponent rating (Elo)")
    g.add_argument("--self-play", action="store_true",
                   help="opponent is the graded agent itself; solve the rating"
                        " fixed point (default when --opp-whr is absent)")
    args = ap.parse_args()
    if args.mode == "grade" and args.opp_whr is not None and args.self_play:
        ap.error("--opp-whr and --self-play are mutually exclusive")
    if args.mode == "fit":
        cmd_fit(args)
    else:
        cmd_grade(args)


if __name__ == "__main__":
    main()
