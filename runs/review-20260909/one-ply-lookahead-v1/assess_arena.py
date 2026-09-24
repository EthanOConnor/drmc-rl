"""Summarize lookahead arena studies: paired scores, lookahead activity, value head quality.

usage: assess_arena.py STUDY_DIR [STUDY_DIR ...]

Value quality uses every journaled root value (both entrants): AUC of the
side-to-move value for that side's eventual win, and the spread of lookahead
Q across kept placements next to the one-decision value change of the
realized line (|V(next own decision) - V(now)|), a noise floor for ranking.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    values = np.concatenate([pos, neg])
    ranks = values.argsort().argsort() + 1.0
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


summary = {}
for directory in map(Path, sys.argv[1:]):
    results = json.loads((directory / "results.json").read_text())
    for match in results["tournaments"]:
        pos, neg, spread, step, changed, kept = [], [], [], [], 0, 0
        for trace in sorted((directory / "moves").glob(f"{match['id']}-*.json.gz")):
            record = json.loads(gzip.decompress(trace.read_bytes()))
            game = record["game"]
            if game["score"] is None or game["score"] == 0.5:
                continue
            for physical in (0, 1):
                a_side = physical == game["side"]
                won = (game["score"] == 1) == a_side
                values = [m["value"] for m in record["moves"] if m["side"] == physical and "value" in m]
                (pos if won else neg).extend(values)
                step.extend(np.abs(np.diff(values)).tolist())
                for m in record["moves"]:
                    if m["side"] == physical and "lookahead" in m and "q" in m["lookahead"]:
                        q = [x for x in m["lookahead"]["q"] if -1.5 < x < 1.5]
                        if len(q) > 1:
                            spread.append(max(q) - min(q))
                        changed += int(m["lookahead"].get("changed", False))
                        kept += 1
        summary[match["id"]] = dict(
            pace=match["pace"], games=match["played"], score=match["observed_score"], ci=match["score_ci"],
            wins=match["wins"], losses=match["losses"], draws=match["draws"], censored=match["censored"],
            lookahead_decisions=kept, changed_rate=round(changed / kept, 3) if kept else None,
            value_auc=None if not pos else round(auc(np.array(pos), np.array(neg)), 4),
            q_spread_median=round(float(np.median(spread)), 4) if spread else None,
            value_step_median=round(float(np.median(step)), 4) if step else None)
print(json.dumps(summary, indent=1))
