"""Verify and summarize one deterministic sharded counterfactual pilot."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from drmc_rl.teachers.counterfactual_release import (
    RELEASE_SCHEMA,
    canonical_json,
    sha256_file,
    source_identity,
)


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def _check_probability_triplet(win: float, draw: float, loss: float, *, where: str) -> None:
    values = (float(win), float(draw), float(loss))
    if not all(math.isfinite(value) and -1e-8 <= value <= 1.0 + 1e-8 for value in values):
        raise ValueError(f"invalid W/D/L at {where}: {values}")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=2e-6):
        raise ValueError(f"unnormalized W/D/L at {where}: {values}")


def _source_index(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {source_identity(row): row for row in rows}


def audit(
    manifest_paths: list[Path],
    calibration_path: Path | None,
    source_path: Path | None = None,
) -> dict[str, Any]:
    if not manifest_paths:
        raise ValueError("at least one shard manifest is required")
    manifests = [json.loads(path.read_text()) for path in manifest_paths]
    for path, manifest in zip(manifest_paths, manifests, strict=True):
        if manifest.get("schema") != RELEASE_SCHEMA:
            raise ValueError(f"unsupported release schema in {path}")

    reference = dict(manifests[0]["settings"])
    reference.pop("shard_index", None)
    shard_indices: set[int] = set()
    source_ids: set[str] = set()
    strata: Counter[tuple[str, ...]] = Counter()
    candidate_counts: list[float] = []
    nodes: list[float] = []
    root_wins: list[float] = []
    root_draws: list[float] = []
    root_losses: list[float] = []
    win_spreads: list[float] = []
    max_regrets: list[float] = []
    chance_nodes = 0
    chance_outcomes = 0
    candidate_labels = 0
    budget_exhausted = 0
    uncertainty_available = 0
    chance_branched_states = 0
    source_rows = _source_index(source_path)

    for manifest_path, manifest in zip(manifest_paths, manifests, strict=True):
        settings = dict(manifest["settings"])
        shard_index = int(settings.pop("shard_index"))
        if settings != reference:
            raise ValueError(f"shard settings differ beyond shard_index: {manifest_path}")
        if shard_index in shard_indices:
            raise ValueError(f"duplicate shard index {shard_index}")
        shard_indices.add(shard_index)
        base = manifest_path.parent
        manifest_rows = 0
        manifest_candidates = 0
        for part in manifest["parts"]:
            part_path = base / str(part["file"])
            if sha256_file(part_path) != part["sha256"]:
                raise ValueError(f"chunk hash mismatch: {part_path}")
            with gzip.open(part_path, "rt", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    row = json.loads(line)
                    where = f"{part_path.name}:{line_number}"
                    source_id = str(row["metadata"]["source_id"])
                    if source_id in source_ids:
                        raise ValueError(f"duplicate source identity: {source_id}")
                    source_ids.add(source_id)
                    if source_rows:
                        source = source_rows.get(source_id)
                        if source is None:
                            raise ValueError(f"release source is absent from source bank: {source_id}")
                        stratum = (
                            str(source["level"]),
                            str(source["speed"]),
                            str(source["tactical_stratum"]),
                        )
                    else:
                        stratum = tuple(str(item) for item in row["metadata"]["stratum"])
                    strata[stratum] += 1
                    candidates = list(row["candidates"])
                    if not candidates:
                        raise ValueError(f"empty candidate set at {where}")
                    actions = [int(item["action"]) for item in candidates]
                    if len(actions) != len(set(actions)):
                        raise ValueError(f"duplicate candidate action at {where}")
                    if source_rows:
                        source_root = int(source.get("root_side", reference["root_side"]))
                        expected_actions = {
                            int(action)
                            for action in source["legal_actions_by_side"][source_root]
                        }
                        if set(actions) != expected_actions:
                            raise ValueError(f"candidate coverage mismatch at {where}")
                    ranks = sorted(int(item["rank"]) for item in candidates)
                    if ranks != list(range(1, len(candidates) + 1)):
                        raise ValueError(f"candidate ranks are not a permutation at {where}")
                    if not math.isclose(
                        sum(float(item["policy_target"]) for item in candidates),
                        1.0,
                        rel_tol=0.0,
                        abs_tol=2e-6,
                    ):
                        raise ValueError(f"policy target is not normalized at {where}")
                    regrets = [float(item["regret_win_logit"]) for item in candidates]
                    if min(regrets) < -2e-6 or not math.isclose(
                        min(regrets), 0.0, rel_tol=0.0, abs_tol=2e-6
                    ):
                        raise ValueError(f"invalid candidate regret at {where}")
                    best = next(item for item in candidates if int(item["rank"]) == 1)
                    if int(best["action"]) != int(row["best_action"]):
                        raise ValueError(f"best action/rank mismatch at {where}")
                    for index, item in enumerate(candidates):
                        _check_probability_triplet(
                            item["win"], item["draw"], item["loss"], where=f"{where}#{index}"
                        )
                    _check_probability_triplet(
                        row["root_win"], row["root_draw"], row["root_loss"], where=where
                    )
                    row_chance_nodes = int(row["chance_nodes"])
                    row_chance_outcomes = int(row["chance_outcomes"])
                    if row_chance_outcomes != 9 * row_chance_nodes:
                        raise ValueError(f"incomplete reveal branching at {where}")
                    chance_branched_states += int(row_chance_nodes > 0)
                    candidate_counts.append(float(len(candidates)))
                    nodes.append(float(row["nodes"]))
                    root_wins.append(float(row["root_win"]))
                    root_draws.append(float(row["root_draw"]))
                    root_losses.append(float(row["root_loss"]))
                    wins = [float(item["win"]) for item in candidates]
                    win_spreads.append(max(wins) - min(wins))
                    max_regrets.append(max(regrets))
                    chance_nodes += row_chance_nodes
                    chance_outcomes += row_chance_outcomes
                    budget_exhausted += int(bool(row["budget_exhausted"]))
                    uncertainty_available += int(bool(row["uncertainty_available"]))
                    manifest_rows += 1
                    manifest_candidates += len(candidates)
        if manifest_rows != int(manifest["selected_states"]):
            raise ValueError(f"row count mismatch in {manifest_path}")
        if manifest_candidates != int(manifest["candidate_labels"]):
            raise ValueError(f"candidate count mismatch in {manifest_path}")
        candidate_labels += manifest_candidates

    expected_shards = int(reference["num_shards"])
    if shard_indices != set(range(expected_shards)):
        raise ValueError(f"incomplete shard set: {sorted(shard_indices)} of {expected_shards}")
    if source_rows and source_ids != set(source_rows):
        raise ValueError("release identities do not exactly cover the supplied source bank")
    digest_rows = sorted(
        (int(manifest["settings"]["shard_index"]), str(manifest["release_sha256"]))
        for manifest in manifests
    )
    result: dict[str, Any] = {
        "schema": "drmc-counterfactual-pilot-audit-v1",
        "aggregate_sha256": hashlib.sha256(canonical_json(digest_rows)).hexdigest(),
        "shard_release_sha256": [item[1] for item in digest_rows],
        "states": len(source_ids),
        "candidate_labels": candidate_labels,
        "full_candidate_coverage": bool(source_rows),
        "budget_exhausted": budget_exhausted,
        "unique_source_ids": len(source_ids),
        "strata": {"/".join(key): value for key, value in sorted(strata.items())},
        "candidate_count": _summary(candidate_counts),
        "nodes": _summary(nodes),
        "chance_nodes": chance_nodes,
        "chance_outcomes": chance_outcomes,
        "all_reveals_nine_way": chance_outcomes == 9 * chance_nodes,
        "chance_branched_states": chance_branched_states,
        "pre_reveal_terminal_states": len(source_ids) - chance_branched_states,
        "root_win": _summary(root_wins),
        "root_draw": _summary(root_draws),
        "root_loss": _summary(root_losses),
        "candidate_win_spread": _summary(win_spreads),
        "max_win_logit_regret": _summary(max_regrets),
        "uncertainty_available_states": uncertainty_available,
        "settings": reference,
    }
    if calibration_path is not None:
        calibration = json.loads(calibration_path.read_text())
        result["calibration"] = {
            "sha256": sha256_file(calibration_path),
            "collection": calibration["collection"],
            "heldout_metrics": calibration["heldout_metrics"],
            "parameters": calibration["parameters"],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="+")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.manifest, args.calibration, args.source)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
