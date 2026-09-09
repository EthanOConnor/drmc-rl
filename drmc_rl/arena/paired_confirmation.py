"""Predeclared, whole-game noninferiority for persistent policies.

Each sample is the side-swapped candidate-minus-reference score on one common
seed against the same frozen opponent. These are fixed confirmation analyses,
not anytime-valid monitors or confidence statements about local action Q.
"""

from __future__ import annotations

import math

import numpy as np


def lower_bound(differences, *, alpha, seed_design):
    values = np.asarray(differences, np.float64)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or (np.abs(values) > 1).any()
        or not 0 < alpha < 1
    ):
        raise ValueError("paired scores require finite differences in [-1,1] and valid alpha")
    n = len(values)
    if n < 2:
        return -1.0
    if seed_design == "independent":
        # Maurer & Pontil (2009), Theorem 4, after mapping [-1,1] to
        # [0,1]. The finite-sample term prevents certainty at zero variance.
        log = math.log(2 / alpha)
        radius = math.sqrt(2 * float(values.var(ddof=1)) * log / n) + 14 * log / (3 * (n - 1))
    elif seed_design == "uniform_without_replacement":
        # Range-only Hoeffding remains conservative for simple random sampling
        # without replacement. Do not apply an iid variance bound silently.
        radius = math.sqrt(2 * math.log(1 / alpha) / n)
    else:
        raise ValueError("confirmation must declare its seed sampling design")
    return max(-1.0, float(values.mean()) - radius)


def confirm(records, plan):
    baseline = plan["baseline"]
    candidates = plan["candidates"]
    opponents = plan["opponents"]
    conditions = plan["conditions"]
    seeds = plan["confirmation_seeds"]
    margin = float(plan["score_margin"])
    alpha = float(plan.get("alpha", 0.05))
    if not 0 <= margin < 1 or not 0 < alpha < 1:
        raise ValueError("predeclared noninferiority margin/alpha is invalid")
    if not candidates or not opponents or not conditions or len(seeds) < 2:
        raise ValueError(
            "confirmation plan must declare candidates, opponents, conditions and seeds"
        )
    if (
        len(set(seeds)) != len(seeds)
        or len(set(candidates)) != len(candidates)
        or baseline in candidates
    ):
        raise ValueError("confirmation identities must be unique")
    if len(set(opponents)) != len(opponents):
        raise ValueError("duplicate opponent hypothesis")
    condition_keys = [(int(c["level"]), int(c["speed"]), str(c["pace"])) for c in conditions]
    if len(set(condition_keys)) != len(condition_keys):
        raise ValueError("duplicate condition hypothesis")
    seed_design = plan["seed_design"]
    if seed_design not in ("independent", "uniform_without_replacement"):
        raise ValueError("confirmation must declare its seed sampling design")
    seed_set = set(seeds)
    rows = {}
    agents = {baseline, *candidates}
    for record in records:
        key = (
            record["agent"],
            record["opponent"],
            int(record["level"]),
            int(record["speed"]),
            record["pace"],
            record["seed"],
            int(record["side"]),
        )
        if (
            key[0] not in agents
            or key[1] not in opponents
            or key[2:5] not in condition_keys
            or key[5] not in seed_set
        ):
            continue
        if key[-1] not in (0, 1) or record["score"] not in (None, 0.0, 0.5, 1.0):
            raise ValueError("confirmation records require natural WDL scores or None")
        if key in rows:
            raise ValueError("duplicate confirmation game cannot count as new evidence")
        rows[key] = None if record.get("reason") == "timeout" else record["score"]
    family = len(candidates) * len(opponents) * len(conditions)
    reports = []
    for candidate in candidates:
        for opponent in opponents:
            for condition in condition_keys:
                values = []
                missing = censored = 0
                for seed in seeds:
                    keys = [
                        (agent, opponent, *condition, seed, side)
                        for agent in (candidate, baseline)
                        for side in (0, 1)
                    ]
                    missing += sum(key not in rows for key in keys)
                    censored += sum(key in rows and rows[key] is None for key in keys)
                    if all(key in rows and rows[key] is not None for key in keys):
                        scores = [rows[key] for key in keys]
                        values.append((scores[0] + scores[1] - scores[2] - scores[3]) / 2)
                complete = not (missing or censored)
                low = (
                    lower_bound(values, alpha=alpha / family, seed_design=seed_design)
                    if complete
                    else None
                )
                reports.append(
                    dict(
                        candidate=candidate,
                        opponent=opponent,
                        level=condition[0],
                        speed=condition[1],
                        pace=condition[2],
                        paired_seeds=len(values),
                        missing_games=missing,
                        censored_games=censored,
                        score_difference=float(np.mean(values)) if complete else None,
                        lower_bound=low,
                        noninferior=bool(low > -margin) if complete else None,
                    )
                )
    return dict(
        schema="drmc-whole-game-noninferiority-v1",
        comparisons=reports,
        family_size=family,
        alpha=alpha,
        score_margin=margin,
        seed_design=seed_design,
        fixed_confirmation_only=True,
        noninferior=all(r["noninferior"] is True for r in reports),
        calibrated_local_regret=False,
        blind_preference_gain=None,
    )
