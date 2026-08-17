"""Verified loading and deterministic comparison of counterfactual releases."""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from drmc_rl.teachers.counterfactual_release import RELEASE_SCHEMA, sha256_file


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {key: math.nan for key in ("min", "p10", "median", "mean", "p90", "p95", "max")}
    return {
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _probability(candidate: Mapping[str, Any]) -> np.ndarray:
    result = np.asarray(
        (candidate["win"], candidate["draw"], candidate["loss"]), dtype=np.float64
    )
    if not np.isfinite(result).all() or (result < -1e-9).any():
        raise ValueError(f"invalid candidate W/D/L {result.tolist()}")
    total = float(result.sum())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=2e-6):
        raise ValueError(f"candidate W/D/L does not sum to one: {result.tolist()}")
    return np.maximum(result, 0.0) / max(total, 1e-12)


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    epsilon = 1e-12
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left /= left.sum()
    right /= right.sum()
    mean = 0.5 * (left + right)
    return float(
        0.5 * np.sum(left * np.log(np.maximum(left, epsilon) / np.maximum(mean, epsilon)))
        + 0.5 * np.sum(right * np.log(np.maximum(right, epsilon) / np.maximum(mean, epsilon)))
    )


def _spearman(left: Sequence[int], right: Sequence[int]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.size < 2:
        return 1.0
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.sqrt(np.square(a).sum() * np.square(b).sum()))
    return 1.0 if denominator <= 0 else float((a @ b) / denominator)


@dataclass(frozen=True, slots=True)
class ReleaseState:
    source_id: str
    stratum: tuple[str, ...]
    row: Mapping[str, Any]
    candidates: Mapping[int, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class ReleaseDataset:
    settings: Mapping[str, Any]
    states: Mapping[str, ReleaseState]
    manifest_paths: tuple[Path, ...]
    release_sha256: tuple[str, ...]

    @property
    def chance_model(self) -> str:
        return str(self.settings.get("chance_model", "independent-uniform-ordered-pair-v0"))

    @property
    def information_scope(self) -> str:
        return str(self.settings.get("information_scope", "unspecified"))


def load_release(manifest_paths: Iterable[str | Path]) -> ReleaseDataset:
    paths = tuple(Path(path) for path in manifest_paths)
    if not paths:
        raise ValueError("at least one release manifest is required")
    manifests = [json.loads(path.read_text()) for path in paths]
    for path, manifest in zip(paths, manifests, strict=True):
        if manifest.get("schema") != RELEASE_SCHEMA:
            raise ValueError(f"unsupported release schema in {path}")
    reference = dict(manifests[0]["settings"])
    reference.pop("shard_index", None)
    expected_shards = int(reference["num_shards"])
    shard_indices: set[int] = set()
    states: dict[str, ReleaseState] = {}
    release_ids: list[tuple[int, str]] = []
    for path, manifest in zip(paths, manifests, strict=True):
        settings = dict(manifest["settings"])
        shard_index = int(settings.pop("shard_index"))
        if settings != reference:
            raise ValueError(f"release settings differ beyond shard_index: {path}")
        if shard_index in shard_indices:
            raise ValueError(f"duplicate shard index {shard_index}")
        shard_indices.add(shard_index)
        release_ids.append((shard_index, str(manifest["release_sha256"])))
        row_count = 0
        candidate_count = 0
        for part in manifest["parts"]:
            part_path = path.parent / str(part["file"])
            if sha256_file(part_path) != str(part["sha256"]):
                raise ValueError(f"release part hash mismatch: {part_path}")
            with gzip.open(part_path, "rt", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    metadata = row.get("metadata") or {}
                    source_id = str(metadata.get("source_id", ""))
                    if not source_id:
                        raise ValueError(f"missing source_id at {part_path}:{line_number}")
                    if source_id in states:
                        raise ValueError(f"duplicate source_id {source_id!r}")
                    raw_candidates = list(row.get("candidates") or ())
                    candidates = {int(item["action"]): item for item in raw_candidates}
                    if not candidates or len(candidates) != len(raw_candidates):
                        raise ValueError(f"invalid candidate set at {part_path}:{line_number}")
                    for candidate in candidates.values():
                        _probability(candidate)
                    policy_mass = sum(float(item["policy_target"]) for item in candidates.values())
                    if not math.isclose(policy_mass, 1.0, rel_tol=0.0, abs_tol=2e-6):
                        raise ValueError(f"policy target is not normalized for {source_id}")
                    states[source_id] = ReleaseState(
                        source_id=source_id,
                        stratum=tuple(str(item) for item in metadata.get("stratum", ())),
                        row=row,
                        candidates=candidates,
                    )
                    row_count += 1
                    candidate_count += len(candidates)
        if row_count != int(manifest["selected_states"]):
            raise ValueError(f"release row count mismatch in {path}")
        if candidate_count != int(manifest["candidate_labels"]):
            raise ValueError(f"release candidate count mismatch in {path}")
    if shard_indices != set(range(expected_shards)):
        raise ValueError(f"incomplete shard set: {sorted(shard_indices)} of {expected_shards}")
    return ReleaseDataset(
        settings=reference,
        states=states,
        manifest_paths=paths,
        release_sha256=tuple(value for _index, value in sorted(release_ids)),
    )


def _state_comparison(reference: ReleaseState, candidate: ReleaseState) -> dict[str, float | bool]:
    if set(reference.candidates) != set(candidate.candidates):
        raise ValueError(f"candidate action sets differ for source {reference.source_id}")
    actions = sorted(reference.candidates)
    ref_policy = np.asarray(
        [float(reference.candidates[action]["policy_target"]) for action in actions]
    )
    cand_policy = np.asarray(
        [float(candidate.candidates[action]["policy_target"]) for action in actions]
    )
    ref_wdl = np.asarray([_probability(reference.candidates[action]) for action in actions])
    cand_wdl = np.asarray([_probability(candidate.candidates[action]) for action in actions])
    ref_ranks = [int(reference.candidates[action]["rank"]) for action in actions]
    cand_ranks = [int(candidate.candidates[action]["rank"]) for action in actions]
    delta = np.abs(ref_wdl - cand_wdl)
    return {
        "top1_agree": int(reference.row["best_action"]) == int(candidate.row["best_action"]),
        "max_win_delta": float(delta[:, 0].max(initial=0.0)),
        "mean_win_delta": float(delta[:, 0].mean()),
        "max_wdl_delta": float(delta.max(initial=0.0)),
        "mean_wdl_delta": float(delta.mean()),
        "policy_js": _js_divergence(ref_policy, cand_policy),
        "rank_spearman": _spearman(ref_ranks, cand_ranks),
    }


def compare_releases(reference: ReleaseDataset, candidate: ReleaseDataset) -> dict[str, Any]:
    if set(reference.states) != set(candidate.states):
        missing = sorted(set(reference.states) - set(candidate.states))[:8]
        extra = sorted(set(candidate.states) - set(reference.states))[:8]
        raise ValueError(f"release source sets differ; missing={missing}, extra={extra}")
    records: list[tuple[tuple[str, ...], dict[str, float | bool]]] = []
    for source_id in sorted(reference.states):
        records.append(
            (
                reference.states[source_id].stratum,
                _state_comparison(reference.states[source_id], candidate.states[source_id]),
            )
        )

    def aggregate(rows: Sequence[dict[str, float | bool]]) -> dict[str, Any]:
        return {
            "states": len(rows),
            "top1_agreement": float(np.mean([bool(row["top1_agree"]) for row in rows])),
            "max_win_delta": _summary([float(row["max_win_delta"]) for row in rows]),
            "mean_win_delta": _summary([float(row["mean_win_delta"]) for row in rows]),
            "max_wdl_delta": _summary([float(row["max_wdl_delta"]) for row in rows]),
            "mean_wdl_delta": _summary([float(row["mean_wdl_delta"]) for row in rows]),
            "policy_js": _summary([float(row["policy_js"]) for row in rows]),
            "rank_spearman": _summary([float(row["rank_spearman"]) for row in rows]),
        }

    by_stratum: dict[str, list[dict[str, float | bool]]] = {}
    for stratum, row in records:
        by_stratum.setdefault("/".join(stratum) or "unspecified", []).append(row)
    return {
        "schema": "drmc-counterfactual-release-comparison-v1",
        "reference": {
            "release_sha256": list(reference.release_sha256),
            "search": reference.settings.get("search"),
            "chance_model": reference.chance_model,
            "information_scope": reference.information_scope,
        },
        "candidate": {
            "release_sha256": list(candidate.release_sha256),
            "search": candidate.settings.get("search"),
            "chance_model": candidate.chance_model,
            "information_scope": candidate.information_scope,
        },
        "aggregate": aggregate([row for _stratum, row in records]),
        "by_stratum": {
            key: aggregate(value) for key, value in sorted(by_stratum.items())
        },
    }


def compare_beam_sweep(
    releases: Mapping[int, ReleaseDataset], *, reference_beam: int | None = None
) -> dict[str, Any]:
    if len(releases) < 2:
        raise ValueError("beam sweep requires at least two releases")
    beams = sorted(int(beam) for beam in releases)
    reference = int(reference_beam if reference_beam is not None else max(beams))
    if reference not in releases:
        raise ValueError("reference beam is absent from releases")
    return {
        "schema": "drmc-counterfactual-opponent-beam-sweep-v1",
        "reference_beam": reference,
        "beams": beams,
        "comparisons": {
            str(beam): compare_releases(releases[reference], releases[beam])
            for beam in beams
            if beam != reference
        },
    }


__all__ = [
    "ReleaseDataset",
    "ReleaseState",
    "compare_beam_sweep",
    "compare_releases",
    "load_release",
]
