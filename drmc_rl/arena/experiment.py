"""Small, read-only view of the active experiment and its latest results."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from drmc_rl.arena.ratings import ELO_SCALE, PairCounts, fit_laplace_ratings


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".next")
    tmp.write_text(json.dumps(value, indent=2, default=lambda a: a.tolist()) + "\n")
    tmp.replace(path)


def score_interval(records):
    grouped = {}
    for row in records:
        grouped.setdefault(row["seed"], []).append(row["score"])
    values = np.asarray([np.mean(v) for v in grouped.values() if len(v) == 2])
    if len(values) < 16:
        return None
    # Resample whole paired seeds, never placements or individual swapped games.
    rng = np.random.default_rng(71891)
    samples = values[rng.integers(len(values), size=(4000, len(values)))].mean(axis=1)
    low, high = np.quantile(samples, [.025, .975])
    # A bootstrap can collapse when every paired seed gives the same score.
    # Enclose it in a conservative Wilson interval using whole pairs as the
    # effective sample count; never report certainty from a small all-win run.
    p, n, z = float(values.mean()), len(values), 1.95996398454
    centre = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [float(min(low, max(0, centre-half))), float(max(high, min(1, centre+half)))]


def relative_ratings(comparisons, records, anchor="baseline8"):
    """Estimate each connected field without mixing levels or pace settings.

    Each complete, side-swapped seed contributes one effective observation:
    its two game likelihoods receive half weight. This conservative composite
    likelihood avoids treating correlated sides as two independent trials.
    Ratings are experiment-relative; there is no human-rating calibration.
    """
    groups = {}
    for match in comparisons.values():
        key = (match.get("rating_group", "Screening"), match["level"], match.get("pace", "frame_perfect"))
        seeds = {}
        for row in records.values():
            if row["comparison"] == match["id"]:
                seeds.setdefault(row["seed"], {})[row["side"]] = row["score"]
        for sides in seeds.values():
            if set(sides) == {0, 1}:
                groups.setdefault(key, []).append((match["a"], match["b"], list(sides.values())))
    result = []
    for (label, level, pace), rows in groups.items():
        connected = {anchor}
        while True:
            expanded = connected | {v for a,b,_ in rows if a in connected or b in connected for v in (a,b)}
            if expanded == connected:
                break
            connected = expanded
        if len(connected) < 2:
            continue
        agents = sorted(connected)
        counts, games = {}, dict.fromkeys(agents, 0)
        for a,b,scores in rows:
            if a not in connected:
                continue
            games[a] += 2
            games[b] += 2
            i,j = agents.index(a), agents.index(b)
            if i > j:
                i,j = j,i
                scores = [1-s for s in scores]
            bucket = counts.setdefault((i,j), [0,0,0])
            for score in scores:
                bucket[0 if score == 1 else 1 if score == .5 else 2] += 1
        # PairCounts accepts numerical weights; no lineage relationships are
        # asserted between execution variants of the same frozen checkpoint.
        fit = fit_laplace_ratings(len(agents), [PairCounts(i,j,*(n/2 for n in wdl))
            for (i,j),wdl in counts.items()], [None]*len(agents), samples=4096)
        draws = fit.samples.skills
        relative = (draws - draws[:, [agents.index(anchor)]]) * ELO_SCALE
        ratings = []
        for i,id in enumerate(agents):
            low, median, high = np.quantile(relative[:,i], [.025,.5,.975])
            ratings.append({"id": id, "elo": round(float(median)), "low": round(float(low)),
                            "high": round(float(high)), "games": games[id]})
        result.append({"label": label, "level": level, "pace": pace, "anchor": anchor,
            "ratings": sorted(ratings, key=lambda r: -r["elo"]),
            "matchups": [{"a": agents[i], "b": agents[j], "games": sum(wdl),
                "score": (wdl[0]+.5*wdl[1])/sum(wdl)} for (i,j),wdl in counts.items()],
            "method": "Davidson fit, 95% Laplace intervals; one effective observation per side-swapped seed pair"})
    return result


def read_experiment(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"active": False}
    plan = json.loads(path.read_text())
    if not isinstance(plan, dict):
        raise ValueError("experiment must be an object")
    result_path = path.with_name("results.json")
    results = json.loads(result_path.read_text()) if result_path.is_file() else {}
    if not isinstance(results, dict):
        raise ValueError("results must be an object")
    # Results cannot overwrite the operator's task, goals, or stage descriptions.
    return {
        **plan,
        "active": True,
        "results": results,
        "served_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
