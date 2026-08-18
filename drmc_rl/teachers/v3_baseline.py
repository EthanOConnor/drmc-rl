"""Build a leakage-resistant frozen-V3 W/D/L bootstrap baseline.

V3 emits an uncalibrated scalar outcome logit for each legal afterstate.  This
module fits one positive-slope Davidson link on independent natural games and
then applies that frozen link to a disjoint evaluation bank.  The resulting
rows can be compared directly with an audited counterfactual release.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from drmc_rl.eval.wdl_calibration import (
    DavidsonParameters,
    calibration_report,
    probabilities,
)
from drmc_rl.search.native_pair import state_from_payload
from drmc_rl.teachers.counterfactual_release import canonical_json, sha256_file

V3_BASELINE_CALIBRATION_SCHEMA = "drmc-v3-wdl-baseline-calibration-v1"
V3_BASELINE_MANIFEST_SCHEMA = "drmc-v3-wdl-bootstrap-manifest-v1"


@dataclass(frozen=True, slots=True)
class V3ScoreRow:
    source_id: str
    game_id: str
    outcome: int
    observed_action: int
    score: float
    stratum: tuple[str, ...] = ()


def _outcome(value: object) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"win", "w"}:
            return 0
        if normalized in {"draw", "d"}:
            return 1
        if normalized in {"loss", "l"}:
            return 2
    result = int(value)
    if result not in (0, 1, 2):
        raise ValueError(f"invalid natural outcome {value!r}")
    return result


def game_set_sha256(game_ids: Iterable[str]) -> str:
    return hashlib.sha256(canonical_json(sorted(set(game_ids)))).hexdigest()


def _open_rows(path: Path):
    return (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open("r", encoding="utf-8")
    )


def load_source_rows(path: Path) -> list[dict[str, Any]]:
    with _open_rows(path) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"V3 source bank is empty or malformed: {path}")
    return rows


def score_source_rows(
    runtime: Any,
    rows: Iterable[Mapping[str, Any]],
    *,
    batch_size: int = 64,
) -> list[V3ScoreRow]:
    """Score the recorded rollout action at each exact PairState boundary."""

    from drmc_rl.search.strong_league import board_bytes_to_semantic_planes

    source = list(rows)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    result: list[V3ScoreRow] = []
    seen: set[str] = set()
    for offset in range(0, len(source), batch_size):
        batch = source[offset : offset + batch_size]
        requests: list[dict[str, Any]] = []
        metadata: list[tuple[Mapping[str, Any], tuple[int, ...], int]] = []
        for payload in batch:
            source_id = str(payload.get("id", payload.get("source_id", "")))
            game_id = str(payload.get("game_id", ""))
            if not source_id or source_id in seen or not game_id:
                raise ValueError("V3 source rows require unique IDs and stable game_id values")
            if (
                not bool(payload.get("natural_outcome_available", True))
                or payload.get("outcome") is None
            ):
                raise ValueError(f"source row {source_id} lacks a natural terminal outcome")
            seen.add(source_id)
            state = state_from_payload(payload)
            side = int(payload["root_side"])
            legal = state.legal_actions_by_side[side]
            costs = state.action_costs_by_side[side]
            observed = int(payload["observed_action"])
            if observed not in legal:
                raise ValueError(f"observed action {observed} is not legal for {source_id}")
            own = state.privileged.public.sides[side]
            opponent = state.privileged.public.sides[1 - side]
            requests.append(
                {
                    "board_planes": board_bytes_to_semantic_planes(own.board),
                    "opponent_board_planes": board_bytes_to_semantic_planes(opponent.board),
                    "opponent_state_age_frames": abs(
                        int(state.privileged.public.observable_clock_delta_frames)
                    ),
                    "pill": np.asarray(own.pill, dtype=np.int64),
                    "preview": np.asarray(own.preview, dtype=np.int64),
                    "candidate_actions": np.asarray(legal, dtype=np.int64),
                    "candidate_costs": np.asarray(costs, dtype=np.float32),
                    "candidate_mask": np.ones(len(legal), dtype=np.bool_),
                    "rating": float(runtime.condition.mean),
                    "speed": int(state.speed_setting),
                    "speed_ups": int(payload.get("speed_ups", 0)),
                }
            )
            metadata.append((payload, legal, observed))
        outputs = runtime.score_batch(requests)
        for (payload, legal, observed), output in zip(metadata, outputs, strict=True):
            values = np.asarray(output["outcome_logit"], dtype=np.float64)
            index = legal.index(observed)
            score = float(values[index])
            if not np.isfinite(score):
                raise ValueError(f"non-finite V3 score for source {payload.get('id')}")
            raw_stratum = payload.get("stratum")
            stratum = (
                tuple(str(item) for item in raw_stratum)
                if isinstance(raw_stratum, (list, tuple))
                else (
                    str(payload.get("level", "")),
                    str(payload.get("speed", payload.get("speed_setting", ""))),
                    str(payload.get("tactical_stratum", "")),
                )
            )
            result.append(
                V3ScoreRow(
                    source_id=str(payload.get("id", payload.get("source_id"))),
                    game_id=str(payload["game_id"]),
                    outcome=_outcome(payload["outcome"]),
                    observed_action=observed,
                    score=score,
                    stratum=stratum,
                )
            )
    return result


def build_v3_baseline(
    calibration_rows: Iterable[V3ScoreRow],
    evaluation_rows: Iterable[V3ScoreRow],
    *,
    seed: int = 20260816,
    folds: int = 5,
    bootstrap_samples: int = 4000,
    min_calibration_games: int = 192,
    min_calibration_draw_games: int = 0,
    min_evaluation_games: int = 48,
    min_evaluation_draw_games: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    calibration = list(calibration_rows)
    evaluation = list(evaluation_rows)
    if not calibration or not evaluation:
        raise ValueError("calibration and evaluation rows must both be non-empty")
    calibration_games = {row.game_id for row in calibration}
    evaluation_games = {row.game_id for row in evaluation}
    overlap = calibration_games & evaluation_games
    if overlap:
        raise ValueError(f"calibration/evaluation game leakage: {sorted(overlap)[:8]}")

    def draw_games(rows: list[V3ScoreRow]) -> int:
        return len({row.game_id for row in rows if row.outcome == 1})

    if len(calibration_games) < min_calibration_games:
        raise ValueError("insufficient independent V3 calibration games")
    if draw_games(calibration) < min_calibration_draw_games:
        raise ValueError("insufficient natural draws for V3 calibration")
    if len(evaluation_games) < min_evaluation_games:
        raise ValueError("insufficient independent V3 evaluation games")
    if draw_games(evaluation) < min_evaluation_draw_games:
        raise ValueError("V3 evaluation requires natural draw evidence")

    scores = np.asarray([row.score for row in calibration], dtype=np.float64)
    outcomes = np.asarray([row.outcome for row in calibration], dtype=np.int64)
    groups = np.asarray([row.game_id for row in calibration])
    report = calibration_report(
        scores,
        outcomes,
        groups,
        seed=seed,
        folds=folds,
        bootstrap_samples=bootstrap_samples,
    )
    parameters = DavidsonParameters(**report["parameters"])
    evaluation_probability = probabilities(
        np.asarray([row.score for row in evaluation], dtype=np.float64), parameters
    )
    output_rows = [
        {
            "source_id": row.source_id,
            "game_id": row.game_id,
            "outcome": row.outcome,
            "observed_action": row.observed_action,
            "v3_outcome_logit": row.score,
            "baseline_wdl": [float(item) for item in probability],
            "stratum": list(row.stratum),
        }
        for row, probability in zip(evaluation, evaluation_probability, strict=True)
    ]
    artifact = {
        "schema": V3_BASELINE_CALIBRATION_SCHEMA,
        "link": "positive-slope-davidson-wdl",
        "score": "frozen-v3-observed-action-outcome-logit",
        "parameters": report["parameters"],
        "grouped_calibration": report,
        "calibration_rows": len(calibration),
        "calibration_games": len(calibration_games),
        "calibration_draw_games": draw_games(calibration),
        "evaluation_rows": len(evaluation),
        "evaluation_games": len(evaluation_games),
        "evaluation_draw_games": draw_games(evaluation),
        "calibration_game_set_sha256": game_set_sha256(calibration_games),
        "evaluation_game_set_sha256": game_set_sha256(evaluation_games),
        "game_sets_disjoint": True,
        "seed": int(seed),
    }
    return artifact, output_rows


def verify_v3_baseline_manifest(rows_path: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != V3_BASELINE_MANIFEST_SCHEMA:
        raise ValueError("unsupported V3 bootstrap manifest schema")
    if sha256_file(rows_path) != str(manifest.get("rows_sha256", "")):
        raise ValueError("V3 bootstrap rows hash mismatch")
    calibration_path = Path(str(manifest["calibration_artifact"]))
    if not calibration_path.is_absolute():
        calibration_path = manifest_path.parent / calibration_path
    if sha256_file(calibration_path) != str(manifest.get("calibration_sha256", "")):
        raise ValueError("V3 bootstrap calibration hash mismatch")
    calibration = json.loads(calibration_path.read_text())
    if calibration.get("schema") != V3_BASELINE_CALIBRATION_SCHEMA:
        raise ValueError("unsupported V3 bootstrap calibration schema")
    if not bool(calibration.get("game_sets_disjoint")):
        raise ValueError("V3 bootstrap calibration/evaluation split is not disjoint")
    for field in (
        "checkpoint_sha256",
        "calibration_game_set_sha256",
        "evaluation_game_set_sha256",
        "calibration_games",
        "calibration_draw_games",
        "evaluation_games",
        "evaluation_draw_games",
    ):
        if manifest.get(field) != calibration.get(field):
            raise ValueError(f"V3 bootstrap manifest/calibration mismatch for {field}")
    return manifest


__all__ = [
    "V3_BASELINE_CALIBRATION_SCHEMA",
    "V3_BASELINE_MANIFEST_SCHEMA",
    "V3ScoreRow",
    "build_v3_baseline",
    "game_set_sha256",
    "load_source_rows",
    "score_source_rows",
    "verify_v3_baseline_manifest",
]
