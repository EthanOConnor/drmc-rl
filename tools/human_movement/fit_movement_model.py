"""Fit the compact per-placement human movement model (drmc-human-movement-model-v1).

    python -m tools.human_movement.fit_movement_model FEATURES.parquet SLACK.parquet OUT.json \
        [--holdout-folds 0,1,2,3] [--top-split]

FEATURES.parquet comes from ``movement_features.py``; SLACK.parquet (``slack.py`` output) supplies
the raw inputs used only for tap hold lengths. The output is a small JSON of quantile tables and
probabilities, one profile per named pace, plus a Gaussian-copula style model. Sloth and Super Human
are extrapolated from the Relaxed and Top Humans bands (see ``EXTRAPOLATE``); Frame Perfect has no
profile because it keeps exact machine movement.

``--holdout-folds`` excludes those player folds (blake2b(handle) % 20) from fitting, for validation.
``--top-split`` additionally keeps only every other top-band player (by index) so the remaining
top players can serve as a held-out comparison; the corpus has too few 2250+ players for folds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit, logit, ndtri

KNOTS = [0.005, 0.025] + [round(0.05 * i, 2) for i in range(1, 20)] + [0.975, 0.995]
GRAVITY = ("slow", "mid", "fast")  # ROM speed threshold >= 13, 7..12, <= 6 (frames per row = threshold + 1)
ROWS = ("0-1", "2-3", "4-6", "7-9", "10+")  # rows still to fall once steering is complete
DEPTH = ("1-4", "5-8", "9-12", "13-16")  # rows the pill falls to its lock (lock row + 1)
PAUSE_GAP = 17  # an inter-press gap of this many frames or more is a pause
# Calibrated bands. Centers are placement-weighted; windows chosen so the four measured paces are
# roughly evenly spaced in log(reaction), log(time lost), log(press gap) and log(descent slack).
BANDS = {
    "relaxed": (1100, 1300),
    "normal": (1550, 1700),
    "fast": (1900, 2050),
    "top_humans": (2250, 9999),
}
# Extrapolated ends: (base band, rating trend window, rating offset, per-metric notes).
EXTRAPOLATE = {
    "sloth": ("relaxed", (1000, 1500), -300),
    "super_human": ("top_humans", (1850, 2500), +300),
}
LIMITS = {  # physical/behavioural bounds applied after extrapolation (frames)
    "reaction": (0, 90), "gap": (1, 40), "pause": (PAUSE_GAP, 120), "slack": (0, 200),
}
SUPER_HUMAN_SLACK = 0.75
STYLE_DIMS = ("reaction", "gap", "das", "slack", "correction", "pause")


def gravity_of(threshold):
    t = np.asarray(threshold)
    return np.select([t >= 13, t >= 7], ["slow", "mid"], "fast")


def rows_of(remaining):
    r = np.asarray(remaining)
    return np.select([r <= 1, r <= 3, r <= 6, r <= 9], list(ROWS[:4]), ROWS[4])


def quantiles(values):
    v = np.asarray(values, dtype=float)
    return [round(float(x), 2) for x in np.quantile(v, KNOTS)]


def cell_tables(frame, value, keys, minimum=150):
    """Quantiles per cell, falling back to coarser marginals for sparse cells."""
    out = {}
    marginal = quantiles(frame[value])
    coarse = {k: quantiles(g[value]) for k, g in frame.groupby(keys[0]) if len(g) >= minimum}
    levels = [GRAVITY] + [
        {"depth": DEPTH, "rows": ROWS}[k] for k in keys[1:]
    ]
    for combo in np.array(np.meshgrid(*levels, indexing="ij")).reshape(len(levels), -1).T:
        mask = np.ones(len(frame), dtype=bool)
        for k, level in zip(keys, combo):
            mask &= frame[k].to_numpy() == level
        key = "|".join(combo)
        out[key] = quantiles(frame[value][mask]) if mask.sum() >= minimum else coarse.get(combo[0], marginal)
    return out


def probabilities(frame, flag, key="g", minimum=150):
    overall = float(frame[flag].mean())
    return {level: round(float(g[flag].mean()) if len(g) >= minimum else overall, 4)
            for level, g in [(level, frame[frame[key] == level]) for level in GRAVITY]}


def depth_probabilities(frame, flag, minimum=150):
    by_gravity = probabilities(frame, flag)
    out = {}
    for g in GRAVITY:
        for depth in DEPTH:
            cell = frame[(frame.g == g) & (frame.depth == depth)]
            out[f"{g}|{depth}"] = round(float(cell[flag].mean()), 4) if len(cell) >= minimum else by_gravity[g]
    return out


def rows_probabilities(frame, flag, minimum=150):
    by_gravity = probabilities(frame, flag)
    out = {}
    for g in GRAVITY:
        for rows in ROWS:
            cell = frame[(frame.g == g) & (frame.rows == rows)]
            out[f"{g}|{rows}"] = round(float(cell[flag].mean()), 4) if len(cell) >= minimum else by_gravity[g]
    return out


def depth_of(rows_fallen):
    r = np.asarray(rows_fallen)
    return np.select([r <= 4, r <= 8, r <= 12], list(DEPTH[:3]), DEPTH[3])


def hold_lengths(rle, held, rng, limit=60000):
    """Tap hold lengths (frames a fresh L/R or A/B press stays down, if released within 15)."""
    index = np.arange(len(rle))
    if len(index) > limit:
        index = rng.choice(index, limit, replace=False)
    lat, rot = [], []
    for i in index:
        raw = bytearray()
        for n, b in struct.iter_unpack("<HB", rle[i]):
            raw.extend(bytes((b,)) * n)
        raw = raw[:-1]
        prev = int(held[i])
        for f, b in enumerate(raw):
            new = b & ~prev
            for bit, sink in ((1, lat), (2, lat), (64, rot), (128, rot)):
                if new & bit:
                    j = f
                    while j < len(raw) and raw[j] & bit:
                        j += 1
                    if j < len(raw) and j - f < 16:
                        sink.append(j - f)
            prev = b
    return {"lateral": quantiles(lat), "rotation": quantiles(rot)}


def band_profile(x, holds):
    """Tables for one calibrated rating window."""
    x = x.copy()
    moved = x[x.reaction >= 0]
    gaps = x[x.gaps.map(len) > 0]
    gap_rows = pd.DataFrame({
        "g": np.repeat(gaps.g.to_numpy(), gaps.gaps.map(len).to_numpy()),
        "depth": np.repeat(gaps.depth.to_numpy(), gaps.gaps.map(len).to_numpy()),
        "gap": np.concatenate(gaps.gaps.to_numpy()).astype(float),
    })
    rhythm = gap_rows[gap_rows.gap < PAUSE_GAP]
    pauses = gap_rows[gap_rows.gap >= PAUSE_GAP]
    two = x[x.presses_lat + x.presses_rot >= 2].copy()
    two["paused"] = two.gaps.map(lambda g: bool(len(g)) and max(g) >= PAUSE_GAP).astype(float)
    # Auto-repeat (a direction held >= 16 frames) by sideways distance and whether the target is
    # against a wall, over every sideways placement (fresh presses and holds carried over spawn).
    moving = x[x.dx != 0].copy()
    moving["adx"] = np.minimum(moving.dx.abs(), 4)
    lock_x = moving.dx + 3
    moving["wall"] = (lock_x == 0) | (lock_x == np.where(moving.lock_rot % 2 == 1, 7, 6))
    das = {}
    for level in (1, 2, 3, 4):
        for wall in (False, True):
            g = moving[(moving.adx == level) & (moving.wall == wall)]
            if len(g) >= 100:
                das[f"{level}{'w' if wall else ''}"] = round(float(g.das.mean()), 4)
        g = moving[moving.adx == level]
        das[str(level)] = round(float(g.das.mean()), 4) if len(g) >= 100 else 0.0
    both = x[(x.rot_need > 0) & (x.dx != 0) & (x.presses_lat >= 1) & (x.presses_rot >= 1)]
    corrected = x[x.corrected == 1]
    mix = {
        "overshoot": float((corrected.overshoot > 0).mean()),
        "reversal": float(((corrected.reversals > 0) & (corrected.overshoot == 0)).mean()),
        "extra_rotation": float((corrected.extra_rot >= 2).mean()),
        "late_lateral": float((corrected.late_lateral > 0).mean()),
    }
    total = sum(mix.values()) or 1.0
    return {
        "source": {"placements": int(len(x)), "players": int(x.player_index.nunique()),
                   "rating_median": round(float(x.rating.median()), 1)},
        "reaction": cell_tables(moved, "reaction", ("g", "depth")),
        "gap": cell_tables(rhythm, "gap", ("g", "depth")),
        "pause_p": depth_probabilities(two, "paused"),
        "pause": quantiles(pauses.gap) if len(pauses) >= 50 else quantiles([PAUSE_GAP, 2 * PAUSE_GAP]),
        "das_p": das,
        "rot_first_p": round(float(both.rot_first.mean()), 4),
        # Descent: P(no soft drop at all), else frames beyond the fastest drop with Down used.
        "no_down_p": rows_probabilities(x.assign(no_down=(x.down_frames == 0).astype(float)), "no_down"),
        "slack": cell_tables(x[x.down_frames > 0], "descent_slack", ("g", "rows")),
        "correction_p": probabilities(x, "corrected"),
        "correction_mix": {k: round(v / total, 4) for k, v in mix.items()},
        "hold": holds,
    }


def _scale_tables(tables, factor, lo, hi):
    if isinstance(tables, dict):
        return {k: _scale_tables(v, factor, lo, hi) for k, v in tables.items()}
    return [round(float(np.clip(v * factor, lo, hi)), 2) for v in tables]


def _shift_p(probs, delta, cap=None):
    if isinstance(probs, dict):
        return {k: _shift_p(v, delta, cap) for k, v in probs.items()}
    p = float(expit(logit(np.clip(probs, 1e-3, 1 - 1e-3)) + delta))
    return round(min(p, cap) if cap is not None else p, 4)


def trend(d, window, metric, valid=None):
    """Per-100-rating slope of log(player-median metric) (continuous) or logit(rate) (binary)."""
    x = d[(d.rating >= window[0]) & (d.rating < window[1])]
    if valid is not None:
        x = x[valid(x)]
    x = x.assign(bin=(x.rating // 50) * 50)
    per = x.groupby(["bin", "player_index"])[metric].agg(["median", "mean", "count"])
    per = per[per["count"] >= 30].reset_index()
    binary = set(np.unique(x[metric])) <= {0.0, 1.0}
    stat = per.groupby("bin")["mean" if binary else "median"].median()
    y = logit(np.clip(stat.to_numpy(), 1e-3, 1 - 1e-3)) if binary else np.log(np.maximum(stat.to_numpy(), 0.5))
    slope = np.polyfit(stat.index.to_numpy() / 100.0, y, 1)[0]
    return float(slope)


def extrapolate(d, profiles, name):
    base, window, offset = EXTRAPOLATE[name]
    src = json.loads(json.dumps(profiles[base]))
    k = offset / 100.0
    slopes = {
        "reaction": trend(d, window, "reaction", lambda x: x.reaction >= 0),
        "gap": trend(d, window, "gap_median", lambda x: x.gap_median >= 0),
        "slack": trend(d, window, "descent_slack"),
        "pause": trend(d.assign(paused=d.idle_max >= 10), window, "paused"),
        "das": trend(d[d.presses_lat >= 1], window, "das"),
        "no_down": trend(d.assign(no_down=(d.down_frames == 0).astype(float)), window, "no_down"),
    }
    factors = {m: float(np.exp(slopes[m] * k)) for m in ("reaction", "gap", "slack")}
    if name == "super_human":
        # Never slower than the top band. The measured descent-slack trend is flat at the top, so
        # "less time lost than the best humans" is an explicit assumption: slack x0.75.
        factors = {m: min(f, 1.0) for m, f in factors.items()}
        factors["slack"] = min(factors["slack"], SUPER_HUMAN_SLACK)
    src["reaction"] = _scale_tables(src["reaction"], factors["reaction"], *LIMITS["reaction"])
    src["gap"] = _scale_tables(src["gap"], factors["gap"], *LIMITS["gap"])
    src["slack"] = _scale_tables(src["slack"], factors["slack"], *LIMITS["slack"])
    src["no_down_p"] = _shift_p(src["no_down_p"], slopes["no_down"] * k, cap=0.8)
    src["hold"] = _scale_tables(src["hold"], factors["gap"], 1, 15)
    pause_shift = slopes["pause"] * k
    if name == "super_human":
        pause_shift = min(pause_shift, 0.0)
    src["pause_p"] = _shift_p(src["pause_p"], pause_shift, cap=0.6)
    src["pause"] = _scale_tables(src["pause"], max(factors["gap"], 1.0) if name == "sloth" else 1.0,
                                 *LIMITS["pause"])
    if name == "super_human":
        # Keep imperfect human-style corrections, at half the top band's rate (assumption).
        src["correction_p"] = {g: round(p * 0.5, 4) for g, p in src["correction_p"].items()}
    else:
        src["correction_p"] = _shift_p(src["correction_p"], trend(d, window, "corrected") * k, cap=0.5)
    src["source"] = {"extrapolated_from": base, "trend_window": list(window), "rating_offset": offset,
                     "log_slopes_per_100": {m: round(v, 4) for m, v in slopes.items()},
                     "time_factors": {m: round(v, 3) for m, v in factors.items()},
                     "pause_logit_shift": round(pause_shift, 3)}
    return src


def normal_scores(frame, metric, keys, rng):
    """Randomized mid-rank normal scores within cells (ties broken at random)."""
    z = np.full(len(frame), np.nan)
    values = frame[metric].to_numpy(dtype=float)
    groups = frame.groupby(keys, observed=True).indices
    for idx in groups.values():
        v = values[idx] + rng.uniform(0, 1e-3, len(idx))
        order = np.argsort(np.argsort(v))
        z[idx] = ndtri((order + 0.5) / len(idx))
    return z


def fit_style(d, rng):
    """Between-player (per game) and within-player (per placement) covariance of normal scores."""
    x = d.copy()
    x["bin"] = (x.rating // 100).astype(int)
    x["rr"] = rows_of(x.rows_fallen - 1 - x.steer_y)
    x["paused"] = x.gaps.map(lambda g: bool(len(g)) and max(g) >= PAUSE_GAP).astype(float)
    z = np.column_stack([
        np.where(x.reaction >= 0, normal_scores(x.assign(r=x.reaction.where(x.reaction >= 0, 1e6)), "r", ["bin", "g", "depth"], rng), np.nan),
        np.where(x.gap_median >= 0, normal_scores(x.assign(r=x.gap_median.where(x.gap_median >= 0, 1e6)), "r", ["bin", "g", "depth"], rng), np.nan),
        np.where(x.dx != 0, normal_scores(x, "das", ["bin", "g"], rng), np.nan),
        normal_scores(x, "descent_slack", ["bin", "g", "rr"], rng),
        normal_scores(x, "corrected", ["bin", "g"], rng),
        np.where(x.presses_lat + x.presses_rot >= 2, normal_scores(x, "paused", ["bin", "g", "depth"], rng), np.nan),
    ])
    groups = pd.Series(list(zip(x.player_index, x.bin))).factorize()[0]
    means, weights = [], []
    for gid in np.unique(groups):
        rows = z[groups == gid]
        if len(rows) < 200:
            continue
        means.append(np.nanmean(rows, axis=0))
        weights.append(np.sum(~np.isnan(rows), axis=0))
    means, weights = np.array(means), np.array(weights)
    between = np.cov(np.nan_to_num(means).T)
    noise = np.mean(1.0 / np.maximum(weights, 1), axis=0)  # each dim has unit total variance
    between[np.diag_indices(len(STYLE_DIMS))] = np.maximum(np.diag(between) - noise, 0.01)
    d_between = np.sqrt(np.diag(between))
    corr = between / np.outer(d_between, d_between)
    w, v = np.linalg.eigh(corr)
    corr = v @ np.diag(np.maximum(w, 1e-3)) @ v.T
    corr /= np.sqrt(np.outer(np.diag(corr), np.diag(corr)))
    var = np.minimum(d_between ** 2, 0.6)
    between = corr * np.sqrt(np.outer(var, var))
    within_rows = []
    for gid in np.unique(groups):
        rows = z[groups == gid]
        if len(rows) >= 200:
            within_rows.append(rows - np.nanmean(rows, axis=0))
    residual = np.concatenate(within_rows)
    within = pd.DataFrame(residual).corr(min_periods=1000).to_numpy()
    within = np.nan_to_num(within)
    np.fill_diagonal(within, 1.0)
    w, v = np.linalg.eigh(within)
    within = v @ np.diag(np.maximum(w, 1e-3)) @ v.T
    within /= np.sqrt(np.outer(np.diag(within), np.diag(within)))
    scale = np.sqrt(1.0 - var)
    within = within * np.outer(scale, scale)
    # Within-placement agreement of consecutive press gaps, for per-press draws.
    pairs = [(g[i], g[i + 1]) for g in x.gaps if len(g) >= 2 for i in range(len(g) - 1)
             if g[i] < PAUSE_GAP and g[i + 1] < PAUSE_GAP]
    pairs = np.array(pairs[:400000], dtype=float)
    ra = np.argsort(np.argsort(pairs[:, 0] + rng.uniform(0, 1e-3, len(pairs))))
    rb = np.argsort(np.argsort(pairs[:, 1] + rng.uniform(0, 1e-3, len(pairs))))
    gap_corr = float(np.corrcoef(ra, rb)[0, 1])
    return {"dims": list(STYLE_DIMS),
            "between_cov": np.round(between, 4).tolist(), "within_cov": np.round(within, 4).tolist(),
            "between_sd": np.round(np.sqrt(var), 3).tolist(), "gap_press_corr": round(gap_corr, 3),
            "groups": int(len(means)),
            "method": "Gaussian copula on randomized mid-rank normal scores within rating-bin x situation"
                      " cells; between = cov of player x 100-rating-bin means (>=200 placements) minus"
                      " sampling noise; within = residual correlation scaled to unit total variance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features")
    parser.add_argument("slack")
    parser.add_argument("output")
    parser.add_argument("--holdout-folds", default="")
    parser.add_argument("--top-split", action="store_true")
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    d = pq.read_table(args.features).to_pandas()
    players = pq.read_table(args.features + ".players.parquet").to_pandas()
    # movement_features keeps slack.py rows with a fastest route and a rating curve, in order.
    raw = pq.read_table(args.slack, columns=["player", "fastest_frames", "rle", "held_before_spawn"]).to_pandas()
    raw = raw[raw.fastest_frames.notna() & raw.player.isin(set(players.player))]
    if len(raw) != len(d):
        raise ValueError("features and slack rows are not aligned")
    rle, held_before = raw.rle.to_numpy(), raw.held_before_spawn.to_numpy()
    del raw
    d = d[d.rating_sd <= 150].copy()
    d["g"] = gravity_of(d.threshold)
    d["depth"] = depth_of(d.rows_fallen)
    d["rows"] = rows_of(d.rows_fallen - 1 - d.steer_y)
    held = [int(f) for f in args.holdout_folds.split(",") if f]
    if held:
        d = d[~d.player_fold.isin(held)]
    if args.top_split:
        top = d.rating >= BANDS["top_humans"][0]
        d = d[~top | (d.player_index % 2 == 0)]
    profiles = {}
    for band, (lo, hi) in BANDS.items():
        x = d[(d.rating >= lo) & (d.rating < hi)]
        profiles[band] = band_profile(x, hold_lengths(rle[x.index.to_numpy()], held_before[x.index.to_numpy()], rng))
        profiles[band]["source"]["rating_window"] = [lo, min(hi, 3000)]
    for name in EXTRAPOLATE:
        profiles[name] = extrapolate(d, profiles, name)
    for name, p in profiles.items():
        # Hard floors for validating generated scripts: reaction p0.5 of the profile, one-frame
        # controller changes (humans do press different buttons on consecutive frames).
        p["floor"] = {"reaction_frames": int(max(0, min(np.floor(v[0]) for v in p["reaction"].values()))),
                      "edge_interval": 1, "motion_interval": 1, "max_buttons": 2}
    order = ["sloth", "relaxed", "normal", "fast", "top_humans", "super_human"]
    model = {
        "schema": "drmc-human-movement-model-v1",
        "knots": KNOTS,
        "cells": {"gravity": {"slow": "threshold>=13", "mid": "7<=threshold<=12", "fast": "threshold<=6"},
                  "depth": list(DEPTH), "rows_remaining": list(ROWS), "pause_gap": PAUSE_GAP},
        "profiles": {name: profiles[name] for name in order},
        "style": fit_style(d, rng),
        "fit": {"features_sha256": hashlib.sha256(Path(args.features).read_bytes()).hexdigest(),
                "placements": int(len(d)), "players": int(d.player_index.nunique()),
                "holdout_folds": held, "top_split": bool(args.top_split), "seed": args.seed,
                "player_fold": "blake2b(handle, 8 bytes, little endian) % 20",
                "rating_filter": "skill_sd <= 150"},
    }
    text = json.dumps(model, separators=(",", ":"))
    model["id"] = "human-movement-v1-" + hashlib.sha256(text.encode()).hexdigest()[:12]
    Path(args.output).write_text(json.dumps(model, separators=(",", ":")) + "\n")
    print(args.output, len(json.dumps(model)), "bytes", model["id"])


if __name__ == "__main__":
    main()
