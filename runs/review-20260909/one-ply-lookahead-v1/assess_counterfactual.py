"""Score the lookahead scorers against exact counterfactual rollout outcomes.

usage: assess_counterfactual.py CF.jsonl [CF.jsonl ...] > summary.json

gain: outcome of the scorer's pick minus the root argmax's outcome, averaged
over decisions (standard error clustered by seed). pairwise: accuracy on kept
pairs with different outcomes (ties count half). Root logits are the plain
core's own ranking, so a useful value scorer must beat them.
"""
import json
import sys
from collections import defaultdict

import numpy as np

points = [json.loads(line) for path in sys.argv[1:] for line in open(path)]
points = [p for p in points if None not in p["outcomes"]]
SCORERS = ("logit", "value_f1", "value_f2", "settled_value", "followup_logit")


def scorer_values(p, name):
    if "+" in name:
        base, weight = name.split("+")
        return np.asarray(p[base]) + float(weight) * np.asarray(p["logit"])
    return np.asarray(p[name], dtype=float)


def summarize(group):
    names = list(SCORERS) + [f"{v}+{w}" for v in ("value_f1", "settled_value") for w in (0.02, 0.05, 0.1, 0.2)]
    out = {"decisions": len(group), "seeds": len({p["seed"] for p in group}),
           "decisions_with_outcome_spread": sum(len(set(p["outcomes"])) > 1 for p in group),
           "root_argmax_win_rate": float(np.mean([p["outcomes"][0] for p in group])),
           "kept_mean_win_rate": float(np.mean([np.mean(p["outcomes"]) for p in group]))}
    for name in names:
        gains, by_seed, pairs = [], defaultdict(list), []
        for p in group:
            values, outcomes = scorer_values(p, name), np.asarray(p["outcomes"])
            pick = int(np.argmax(values))   # first maximum: ties keep the root order
            gain = outcomes[pick] - outcomes[0]
            gains.append(gain)
            by_seed[p["seed"]].append(gain)
            for i in range(len(outcomes)):
                for j in range(i + 1, len(outcomes)):
                    if outcomes[i] != outcomes[j]:
                        better = i if outcomes[i] > outcomes[j] else j
                        worse = j if better == i else i
                        pairs.append(1.0 if values[better] > values[worse] else 0.5 if values[better] == values[worse] else 0.0)
        sums = np.asarray([sum(v) for v in by_seed.values()])
        counts = np.asarray([len(v) for v in by_seed.values()])
        mean = sums.sum() / counts.sum()
        # Ratio-estimator cluster SE.
        se = np.sqrt(np.sum((sums - mean * counts) ** 2) / (len(counts) * (len(counts) - 1))) / counts.mean() if len(counts) > 1 else None
        out[name] = dict(gain=round(float(mean), 4), gain_se=None if se is None else round(float(se), 4),
                         changed=round(float(np.mean([int(np.argmax(scorer_values(p, name))) != 0 for p in group])), 3),
                         pairwise=round(float(np.mean(pairs)), 4) if pairs else None, pairs=len(pairs))
    # Value calibration on the realized (argmax-continuation) outcome.
    values = np.asarray([p["root_value"] for p in group])
    wins = np.asarray([p["outcomes"][0] for p in group])
    pos, neg = values[wins == 1], values[wins == 0]
    out["root_value_auc"] = (round(float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])), 4)
                             if len(pos) and len(neg) else None)
    out["root_value_brier"] = round(float(np.mean(((values + 1) / 2 - wins) ** 2)), 4)
    return out


summary = {}
for pace in sorted({p["pace"] for p in points}):
    summary[pace] = summarize([p for p in points if p["pace"] == pace])
print(json.dumps(summary, indent=1))
