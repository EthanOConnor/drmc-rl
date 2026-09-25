"""Big-clear rate and size in rating-pool games (sampled move traces).

The pool keeps full move traces for one seed pair in 16 (``traces/<condition>/``,
id ``condition/a/b/seed/side``). Every placement of both entrants is resolved
exactly (``drmc_rl.eval.big_clear``), so per entrant this reports placements,
clears, T1/T2/T3 clears per 100 placements, the mean score of its clears, the
share of games with a T1+ clear, and what arm (b)'s bonus would pay per game
(``--bonus`` JSON spec). Intervals bootstrap whole traced games (both sides of
a game move together), 2000 draws.

``--compare A B``: A minus B (and A/B ratio) of the T1+ and T2+ rates, from
the games each played in the same conditions (not necessarily against each
other).

  python -m tools.big_clear_pool_metrics --traces DATA/pool-traces --out metrics.json [--entrants 'armA-*' ...]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import fnmatch
import gzip
import json
from pathlib import Path

import numpy as np

from drmc_rl.eval import big_clear as bc
from drmc_rl.training.showiness import validate_spec

DEFAULT_BONUS = dict(threshold=20.0, base=0.05, per_point=0.005, event_cap=0.15, game_cap=0.30)


def game_sides(path: Path, spec: dict | None):
    data = json.load(gzip.open(path))
    condition, a, b, seed, _ = data["id"].split("/")
    side_a = int(data["game"]["side"])
    out = []
    for entrant, side in ((a, side_a), (b, 1 - side_a)):
        scores, placements, bonus = [], 0, 0.0
        for move in data["moves"]:
            if int(move["side"]) != side:
                continue
            placements += 1
            try:
                f = bc.placement_features(bytes(move["board"]), move["pill"], int(move["placement"]["action"]))
            except ValueError:
                continue
            if f.rounds:
                scores.append(f.score())
                if spec:
                    bonus += bc.showiness_bonus(f, spec)
        if spec:
            bonus = min(bonus, spec["game_cap"])
        out.append(dict(entrant=entrant, opponent=b if entrant == a else a, condition=condition, seed=int(seed),
                        placements=placements, scores=scores, bonus=bonus,
                        score=data["game"]["score"] if entrant == a else 1 - data["game"]["score"]))
    return out


def _rate(games, bar):
    p = sum(g["placements"] for g in games)
    return 100.0 * sum(sum(s >= bar for s in g["scores"]) for g in games) / max(1, p)


def summarize(games: list[dict], draws: int = 2000, seed: int = 0) -> dict:
    t1, t2, t3 = (bar for _, bar in bc.TIERS)
    rng = np.random.default_rng(seed)

    def stats(sample):
        clears = [s for g in sample for s in g["scores"]]
        return dict(
            T1_per_100=_rate(sample, t1), T2_per_100=_rate(sample, t2), T3_per_100=_rate(sample, t3),
            clear_mean_score=float(np.mean(clears)) if clears else 0.0,
            games_with_T1=float(np.mean([any(s >= t1 for s in g["scores"]) for g in sample])),
            bonus_per_game=float(np.mean([g["bonus"] for g in sample])),
            bonus_games=float(np.mean([g["bonus"] > 0 for g in sample])),
            score=float(np.mean([g["score"] for g in sample])))

    point = stats(games)
    boot = defaultdict(list)
    for _ in range(draws):
        sample = [games[i] for i in rng.integers(0, len(games), len(games))]
        for k, v in stats(sample).items():
            boot[k].append(v)
    result = {k: [round(point[k], 4), *[round(float(x), 4) for x in np.percentile(boot[k], [2.5, 97.5])]]
              for k in point}
    result.update(games=len(games), placements=sum(g["placements"] for g in games),
                  clears=sum(len(g["scores"]) for g in games),
                  max_score=max((s for g in games for s in g["scores"]), default=0.0))
    return result


def compare(a: list[dict], b: list[dict], draws: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for name, bar in (("T1", bc.TIERS[0][1]), ("T2", bc.TIERS[1][1])):
        diff, ratio = [], []
        for _ in range(draws):
            sa = [a[i] for i in rng.integers(0, len(a), len(a))]
            sb = [b[i] for i in rng.integers(0, len(b), len(b))]
            ra, rb = _rate(sa, bar), _rate(sb, bar)
            diff.append(ra - rb)
            ratio.append(ra / rb if rb > 0 else np.nan)
        point_a, point_b = _rate(a, bar), _rate(b, bar)
        out[name] = dict(difference_per_100=[round(point_a - point_b, 4), *[round(float(x), 4) for x in np.nanpercentile(diff, [2.5, 97.5])]],
                         ratio=[round(point_a / point_b, 4) if point_b else None,
                                *[round(float(x), 4) for x in np.nanpercentile(ratio, [2.5, 97.5])]])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--out")
    parser.add_argument("--entrants", nargs="*", default=["*"])
    parser.add_argument("--conditions", nargs="*", default=None, help="condition keys (trace directory names)")
    parser.add_argument("--bonus", default=json.dumps(DEFAULT_BONUS))
    parser.add_argument("--compare", nargs=2, action="append", default=[], metavar=("A", "B"))
    args = parser.parse_args()
    spec = validate_spec(json.loads(args.bonus)) if args.bonus else None
    games = defaultdict(list)
    for path in sorted(Path(args.traces).rglob("*.json.gz")):
        if args.conditions and path.parent.name not in args.conditions:
            continue
        for side in game_sides(path, spec):
            if any(fnmatch.fnmatch(side["entrant"], p) for p in args.entrants):
                games[side["entrant"]].append(side)
    report = dict(schema="drmc-big-clear-pool-metrics-v1", tiers=bc.TIERS, bonus=spec,
                  entrants={e: summarize(g) for e, g in sorted(games.items())},
                  comparisons={f"{a} - {b}": compare(games[a], games[b]) for a, b in args.compare
                               if games.get(a) and games.get(b)})
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    for e, s in report["entrants"].items():
        print(f"{e:45s} games {s['games']:5d}  T1/100 {s['T1_per_100']}  T2/100 {s['T2_per_100']}  "
              f"T3/100 {s['T3_per_100']}  bonus/game {s['bonus_per_game']}  games_w_T1 {s['games_with_T1']}")
    if report["comparisons"]:
        print(json.dumps(report["comparisons"], indent=1))


if __name__ == "__main__":
    main()
