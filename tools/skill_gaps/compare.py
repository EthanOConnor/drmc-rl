"""Gap table: AI entrants vs strong humans on every ``tools.skill_gaps.features`` statistic.

    python -m tools.skill_gaps.compare --human feats-human.npz --pool feats-pool.npz --out gaps.json

Groups: humans 1800-2000 (H18), >= 2000 (H20) and the top 5 by rating (TOP5, among players
with enough placements); pool entrants by index. Per statistic and group: the pooled mean over
placements with a cluster-bootstrap 95% interval (clusters: players for humans, sequences for
the pool). Gaps are standardized by the between-game sd of H20's per-sequence means
(``d = (AI - H20) / sd``). Two strength-plausibility views come from the humans alone:
``rho``, the Spearman correlation of a player's mean with their rating (players with 1000+
placements), and ``win_d``, the within-game winner-minus-loser difference of per-sequence
means over each side's first 40 placements (games with both sides in the data), in H20 sd units.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

CONDITIONAL = {
    # name: (base statistic, row condition)
    "opp_h34|attack": ("opp_h34", lambda f: f["attack"] > 0),
    "attack|opp_h34>=10": ("attack", lambda f: f["opp_h34"] >= 10),
    "attack|opp_h34<=6": ("attack", lambda f: f["opp_h34"] <= 6),
    "quad|opp_h34>=10": ("quad", lambda f: f["opp_h34"] >= 10),
    "vcleared|V<=4": ("vcleared", lambda f: f["vir_root"] <= 4),
    "vcleared|V5-12": ("vcleared", lambda f: (f["vir_root"] >= 5) & (f["vir_root"] <= 12)),
    "vcleared|V>=25": ("vcleared", lambda f: f["vir_root"] >= 25),
    "attack|V<=4": ("attack", lambda f: f["vir_root"] <= 4),
    "attack|V>=25": ("attack", lambda f: f["vir_root"] >= 25),
    "nonvirus_clear|V<=4": ("nonvirus_clear", lambda f: f["vir_root"] <= 4),
    "h34|V<=4": ("h34", lambda f: f["vir_root"] <= 4),
    "pill_cells|V<=4": ("pill_cells", lambda f: f["vir_root"] <= 4),
    "d_covered|V>=10": ("d_covered", lambda f: f["vir_root"] >= 10),
    "garb_holes": ("garb_holes", None),
    "h34|garb_in>0": ("h34", lambda f: np.r_[False, f["garb_in"][:-1] > 0] & np.r_[False, f["seq"][1:] == f["seq"][:-1]]),
    "danger12|opp_h34>=10": ("danger12", lambda f: f["opp_h34"] >= 10),
}
SKIP = {"seq", "game", "t", "won", "group", "rating", "pace", "lines", "rounds", "score", "hlines", "after", "meta", "col"}


def stat_columns(f):
    cols = {k: f[k].astype(np.float64) for k in f if k not in SKIP}
    cols["col_entropy"] = None           # sequence-level, handled separately
    for name, (base, cond) in CONDITIONAL.items():
        if cond is None:
            continue
        cols[name] = np.where(cond(f), f[base], np.nan).astype(np.float64)
    del cols["col_entropy"]
    return cols


def cluster_mean(values, clusters, boots=200, seed=0):
    ok = ~np.isnan(values)
    if not ok.any():
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n=0)
    v, c = values[ok], clusters[ok]
    uniq, inv = np.unique(c, return_inverse=True)
    s = np.bincount(inv, weights=v, minlength=len(uniq))
    n = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=boots)
    bs = (w @ s) / np.maximum(w @ n, 1)
    return dict(mean=float(s.sum() / n.sum()), lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)),
                n=int(n.sum()))


def seq_means(values, seq, min_rows=20):
    ok = ~np.isnan(values)
    uniq, inv = np.unique(seq[ok], return_inverse=True)
    s = np.bincount(inv, weights=values[ok], minlength=len(uniq))
    n = np.bincount(inv, minlength=len(uniq))
    keep = n >= min_rows
    return uniq[keep], s[keep] / n[keep]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 3 else np.nan


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--human", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--min-player-rows", type=int, default=1000)
    args = ap.parse_args(argv)
    H = {k: v for k, v in np.load(args.human).items() if k not in ("meta", "after")}
    P = {k: v for k, v in np.load(args.pool).items() if k not in ("meta", "after")}
    hmeta = json.loads(str(np.load(args.human)["meta"]))
    pmeta = json.loads(str(np.load(args.pool)["meta"]))
    entrants = pmeta["sources"][0]["entrants"]
    Hc, Pc = stat_columns(H), stat_columns(P)
    # human groups
    player = H["group"]
    uniq, inv = np.unique(player, return_inverse=True)
    counts = np.bincount(inv)
    mean_rating = np.bincount(inv, weights=H["rating"]) / counts
    eligible = (counts >= 3000) & (mean_rating >= 2000)
    top = uniq[eligible][np.argsort(-mean_rating[eligible])[:args.top]]
    names = hmeta["players"]
    groups = {
        "H18": (H, Hc, (H["rating"] < 2000), player),
        "H20": (H, Hc, (H["rating"] >= 2000), player),
        "TOP5": (H, Hc, np.isin(player, top), player),
    }
    for i, e in enumerate(entrants):
        short = {"bigclear-turbo-f00125000000": "turbo", "bigclear-turbo-f00125000000+showy-quad@1:1.5": "turbo+quad",
                 "champion-retention-mixed-v2": "champion"}.get(e, e)
        groups[short] = (P, Pc, P["group"] == i, P["seq"])
    stats = sorted(Hc)
    table = {}
    # H20 between-game sd per statistic, skill gradient and win association
    h20 = H["rating"] >= 2000
    rated = counts >= args.min_player_rows
    early = H["t"] < 40
    both = None
    for s in stats:
        row = {}
        for g, (src, cols, mask, clus) in groups.items():
            v = np.where(mask, cols[s], np.nan)
            row[g] = cluster_mean(v, clus)
        _, m = seq_means(np.where(h20, Hc[s], np.nan), H["seq"])
        sd = float(np.std(m)) if len(m) > 2 else np.nan
        row["sd_game"] = sd
        for g in groups:
            if g.startswith("H") or g == "TOP5":
                continue
            row[f"d_{g}"] = (row[g]["mean"] - row["H20"]["mean"]) / sd if sd and sd > 0 else np.nan
        row["d_TOP5"] = (row["TOP5"]["mean"] - row["H20"]["mean"]) / sd if sd and sd > 0 else np.nan
        # skill gradient over players
        ok = ~np.isnan(Hc[s])
        ps = np.bincount(inv[ok], weights=Hc[s][ok], minlength=len(uniq))
        pn = np.bincount(inv[ok], minlength=len(uniq))
        sel = rated & (pn >= args.min_player_rows // 4)
        row["rho"] = spearman(ps[sel] / pn[sel], mean_rating[sel])
        # within-game winner - loser over the first 40 placements
        seqs, m40 = seq_means(np.where(early, Hc[s], np.nan), H["seq"], min_rows=10)
        if both is None:
            first = np.r_[True, H["seq"][1:] != H["seq"][:-1]]
            seq_game = dict(zip(H["seq"][first], H["game"][first]))
            seq_won = dict(zip(H["seq"][first], H["won"][first]))
            both = (seq_game, seq_won)
        seq_game, seq_won = both
        by_game = {}
        for q, val in zip(seqs, m40):
            by_game.setdefault(seq_game[q], []).append((seq_won[q], val))
        diffs = [(a[1] - b[1]) if a[0] == 1 else (b[1] - a[1]) for pair in by_game.values() if len(pair) == 2
                 for a, b in [pair] if a[0] != b[0]]
        diffs = np.asarray(diffs)
        row["win_d"] = float(diffs.mean() / sd) if len(diffs) and sd and sd > 0 else np.nan
        row["win_se"] = float(diffs.std() / np.sqrt(len(diffs)) / sd) if len(diffs) > 1 and sd and sd > 0 else np.nan
        row["win_pairs"] = int(len(diffs))
        table[s] = row
    out = dict(groups={g: int(m.sum()) for g, (_, _, m, _) in groups.items()},
               top=[names.get(str(int(t)), str(int(t))) for t in top], stats=table)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    cols = [g for g in groups]
    print(f"{'statistic':<22}" + "".join(f"{g:>12}" for g in cols) + f"{'d_turbo':>9}{'d_quad':>9}{'d_top5':>9}{'rho':>7}{'win_d':>8}")
    order = sorted(stats, key=lambda s: -abs(np.nan_to_num(table[s].get("d_turbo+quad", 0))))
    for s in order:
        r = table[s]
        print(f"{s:<22}" + "".join(f"{r[g]['mean']:>12.4g}" for g in cols)
              + f"{r.get('d_turbo', np.nan):>9.2f}{r.get('d_turbo+quad', np.nan):>9.2f}{r['d_TOP5']:>9.2f}{r['rho']:>7.2f}{r['win_d']:>8.2f}")
    print("groups:", out["groups"], "top:", out["top"])


if __name__ == "__main__":
    main()
