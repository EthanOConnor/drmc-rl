"""Audit whether search leaves match the frozen decision-value model's inputs.

The real frozen mixture supplies expansion priors. Leaf values are suppressed:
this search does not prune by value, so doing so preserves the visited states
while avoiding irrelevant value forwards. This produces diagnostics, not labels.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from drmc_rl.search.joint_event import LEAF_VALUE_CONTRACT, JointEventSearch, SearchConfig, WDL
from drmc_rl.search.strong_league_memberwise import frozen_strong_league_belief_factory


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def audit(args):
    import torch

    torch.set_num_threads(2)
    model, decode = frozen_strong_league_belief_factory(args)
    config = SearchConfig(depth_events=args.depth_events, own_beam=512,
                          opponent_beam=args.opponent_beam, chance_beam=9, max_nodes=100000)
    totals = Counter()
    current = Counter()
    examples = []

    def evaluate(state, root_side):
        need = state.privileged.need_action
        row = {
            "root_ready": bool(need[root_side]),
            "opponent_ready": bool(need[1-root_side]),
            "root_has_candidates": bool(state.legal_actions_by_side[root_side]),
            "neither_ready": not any(need),
            "acting_value_side_has_candidates": bool(
                state.legal_actions_by_side[root_side if need[root_side] else 1-root_side]
            ) if any(need) else False,
        }
        model.runner.restore(0, state.privileged.engine_checkpoint)
        reveal = model.runner.search_reveal_info(0)
        row["pending_reveal"] = reveal is not None
        current["leaves"] += 1
        current[f"boundary/{state.privileged.decision_boundary.value}"] += 1
        for key, value in row.items():
            current[key] += int(value)
        if len(examples) < 8 and not row["root_ready"]:
            examples.append({**row, "boundary": state.privileged.decision_boundary.value,
                             "root_side": root_side, "reveal": reveal,
                             "candidate_counts": [len(v) for v in state.legal_actions_by_side],
                             "public_state": state.privileged.public.to_dict()})
        return WDL(.5, 0, .5)

    model.evaluate = evaluate
    cells = Counter()
    reports = []
    try:
        with gzip.open(args.state_bank, "rt") as handle:
            for line in handle:
                row = json.loads(line)
                cell = (row["level"], row["speed_setting"], row["tactical_stratum"])
                if cells[cell] >= args.per_cell:
                    continue
                cells[cell] += 1
                current.clear()
                result = JointEventSearch(model, config).search(decode(row), root_side=int(row["root_side"]))
                if result.budget_exhausted:
                    raise RuntimeError(f"search budget exhausted for {row['id']}")
                totals.update(current)
                reports.append({"source_id": row["id"], "stratum": cell, "nodes": result.nodes,
                                "root_candidates": len(result.actions), "leaf_counts": dict(current)})
                if len(reports) % 5 == 0:
                    print(f"audited {len(reports)} states; leaves={totals['leaves']}; "
                          f"root_not_ready={totals['leaves']-totals['root_ready']}", flush=True)
        repo = Path(__file__).resolve().parents[1]
        manifest = {
            "schema": "drmc-search-boundary-audit-v1", "diagnostic_only": True,
            "checkpoint_mixture_sha256": sha256(args.mixture_manifest),
            "calibration_sha256": sha256(args.wdl_calibration),
            "state_bank_sha256": sha256(args.state_bank),
            "source_code_sha256": {path: sha256(repo/path) for path in (
                "tools/audit_search_boundaries.py", "drmc_rl/search/joint_event.py",
                "drmc_rl/search/native_pair.py", "drmc_rl/search/belief_native_pair.py",
                "drmc_rl/search/strong_league.py", "drmc_rl/game/observation.py")},
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "native_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo/"vendor/drmario_native", text=True).strip(),
            "search": asdict(config), "prior": "frozen-aggregate-strong-league",
            "leaf_value_contract": LEAF_VALUE_CONTRACT,
            "leaf_values": "suppressed; fixed-prior search does not prune by backed-up value",
            "states": len(reports), "cells": len(cells), "counts": dict(totals),
            "per_state": reports, "root_inactive_examples": examples,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2)+"\n")
        print(json.dumps({k:v for k,v in manifest.items() if k not in ("per_state", "root_inactive_examples")}, indent=2))
    finally:
        model.runner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--mixture-manifest", type=Path, required=True)
    parser.add_argument("--wdl-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--depth-events", type=int, default=2)
    parser.add_argument("--opponent-beam", type=int, default=8)
    parser.add_argument("--per-cell", type=int, default=1)
    args = parser.parse_args()
    if args.per_cell < 1:
        parser.error("--per-cell must be positive")
    audit(args)


if __name__ == "__main__":
    main()
