"""Small, read-only view of the active experiment and its latest results."""

from __future__ import annotations

import json
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from drmc_rl.arena.ratings import ELO_SCALE, PairCounts, fit_laplace_ratings


def execution_key(profile):
    """Keep unrecorded historical presets separate from measured motor limits."""
    if profile is None:
        return ""
    return hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".next")
    tmp.write_text(json.dumps(value, indent=2, default=lambda a: a.tolist()) + "\n")
    tmp.replace(path)


def score_interval(records):
    # Complete-case intervals hide potentially outcome-dependent censoring.
    # Retain the attempted games and report their identification bounds instead.
    if any(row.get("reason") == "timeout" or row.get("score") is None for row in records):
        return None
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


def outcome_summary(records, *, include_interval=True):
    completed = [r for r in records if r.get("reason") != "timeout" and r.get("score") is not None]
    n, censored = len(records), len(records) - len(completed)
    total = sum(r["score"] for r in completed)
    return dict(
        wins=sum(r["score"] == 1 for r in completed),
        losses=sum(r["score"] == 0 for r in completed),
        draws=sum(r["score"] == 0.5 for r in completed),
        censored=censored,
        observed_score=total / len(completed) if completed else None,
        score_bounds=[total / n, (total + censored) / n] if n else None,
        score_ci=score_interval(records) if include_interval else None,
    )


def relative_ratings(comparisons, records, anchor="baseline8", *, unified=False):
    """Estimate each connected field without mixing levels or pace settings.

    Each complete, side-swapped seed contributes one effective observation:
    its two game likelihoods receive half weight. This conservative composite
    likelihood avoids treating correlated sides as two independent trials.
    Ratings are experiment-relative; there is no human-rating calibration.
    """
    groups = {}
    by_comparison = {}
    for row in records.values():
        by_comparison.setdefault(row["comparison"], []).append(row)
    for match in comparisons.values():
        match_rows = by_comparison.get(match["id"], ())
        profile_key = execution_key(match.get("execution_profile"))
        if profile_key and any(r.get("execution_key") != profile_key for r in match_rows):
            raise ValueError("rating records disagree with their execution profile")
        if any(r.get("reason") == "timeout" or r.get("score") is None for r in match_rows):
            continue  # No strength inference from an outcome-censored edge.
        key = ("Live tournament" if unified else match.get("rating_group", "Screening"),
               match["level"], match.get("pace", "frame_perfect"),
               profile_key)
        seeds = {}
        for row in match_rows:
            seeds.setdefault(row["seed"], {})[row["side"]] = row["score"]
        for sides in seeds.values():
            if set(sides) == {0, 1}:
                groups.setdefault(key, []).append((match["a"], match["b"], list(sides.values())))
    result = []
    for (label, level, pace, profile_key), rows in groups.items():
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
            differences = {}
            for j,reference in enumerate(agents):
                gap = (draws[:,i]-draws[:,j])*ELO_SCALE
                lo,mid,hi = np.quantile(gap,[.025,.5,.975])
                differences[reference] = {"elo":round(float(mid)),"low":round(float(lo)),"high":round(float(hi))}
            ratings.append({"id": id, "elo": round(float(median)), "low": round(float(low)),
                            "high": round(float(high)), "games": games[id],"differences":differences})
        profile = next((m.get("execution_profile") for m in comparisons.values()
                        if execution_key(m.get("execution_profile")) == profile_key), None)
        result.append({"label": label, "level": level, "pace": pace, "anchor": anchor,
            "execution_key": profile_key, "execution_profile": profile,
            "ratings": sorted(ratings, key=lambda r: -r["elo"]),
            "matchups": [{"a": agents[i], "b": agents[j], "games": sum(wdl),
                "score": (wdl[0]+.5*wdl[1])/sum(wdl)} for (i,j),wdl in counts.items()],
            "method": "Davidson fit, 95% Laplace intervals; one effective observation per side-swapped seed pair"})
    return result


def experiment_health(training, pipeline, now):
    """Reported failures outrank the plan; silence is uncertainty, not success."""
    health = {"status":pipeline.get("status") or training.get("status"), "severity":"ok"}
    try:
        updated = datetime.fromisoformat(training["updated_at"])
        health["training_age_seconds"] = max(0, int((now-updated).total_seconds()))
    except (KeyError, TypeError, ValueError):
        pass
    if training.get("status") == "Failed":
        health.update(status="Training stopped", severity="failed",
                      message=training.get("error") or "The training worker reported a failure.")
    elif pipeline.get("status") == "Failed":
        health.update(status="Study stopped", severity="failed",
                      message=pipeline.get("error") or "The study supervisor reported a failure.")
    elif training.get("status") == "Running":
        # A completed auxiliary study cannot mark the active outcome trainer
        # complete. Failures above still take precedence over this live status.
        health["status"] = "Running"
        age = health.get("training_age_seconds")
        # Allow normal long batches; do not equate a connected web server with
        # a healthy trainer or remote synchronization feed.
        threshold = max(180, 3*float(training.get("batch_seconds") or 0))
        if age is None or age > threshold:
            health.update(status="Training update overdue", severity="stale",
                          message="Training or its remote feed may have stalled. Displayed counts are the last reported progress.")
    return health


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
    training_path = path.parent / plan.get("training_file", "training.json")
    training = json.loads(training_path.read_text()) if training_path.is_file() else {}
    training_runs = []
    for run in plan.get("training_runs", []):
        run_path = path.parent / run["path"]
        state = json.loads(run_path.read_text()) if run_path.is_file() else {"status": "Queued"}
        training_runs.append({**run, **state})
    if training_runs:
        training = next(
            (r for r in training_runs if r.get("status") in ("Running", "Failed")),
            next((r for r in training_runs if r.get("status") == "Queued"), training_runs[-1]),
        )
    pipeline_path = path.parent / plan.get("pipeline_file", "pipeline.json")
    pipeline = json.loads(pipeline_path.read_text()) if pipeline_path.is_file() else {}
    research_runs = []
    for run in plan.get("research_runs", []):
        run_path = path.parent / run["path"]
        state = json.loads(run_path.read_text()) if run_path.is_file() else {"status": "Queued"}
        research_runs.append({**run, **state})
    now = datetime.now(timezone.utc)
    # Results cannot overwrite the operator's task, goals, or stage descriptions.
    return {
        **plan,
        "active": True,
        "results": results,
        "training": training,
        "training_runs": training_runs,
        "pipeline": pipeline,
        "research_runs": research_runs,
        "health": experiment_health(training, pipeline, now),
        "served_at": now.isoformat(timespec="seconds"),
    }
