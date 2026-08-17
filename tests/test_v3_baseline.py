from __future__ import annotations

import json

import pytest

from drmc_rl.teachers.counterfactual_release import sha256_file
from drmc_rl.teachers.v3_baseline import (
    V3_BASELINE_CALIBRATION_SCHEMA,
    V3_BASELINE_MANIFEST_SCHEMA,
    V3ScoreRow,
    build_v3_baseline,
    verify_v3_baseline_manifest,
)


def _rows(prefix: str, games: int, draws: set[int]) -> list[V3ScoreRow]:
    rows = []
    for game in range(games):
        if game in draws:
            outcome, score = 1, 0.0
        elif game % 2:
            outcome, score = 2, -1.5
        else:
            outcome, score = 0, 1.5
        rows.append(
            V3ScoreRow(
                source_id=f"{prefix}-state-{game}",
                game_id=f"{prefix}-game-{game}",
                outcome=outcome,
                observed_action=game,
                score=score,
                stratum=("10", "2", "midgame"),
            )
        )
    return rows


def test_v3_baseline_fits_on_disjoint_games_and_normalizes_wdl() -> None:
    calibration = _rows("cal", 20, {0, 5, 10, 15})
    evaluation = _rows("eval", 8, {0, 4})
    artifact, rows = build_v3_baseline(
        calibration,
        evaluation,
        seed=7,
        folds=4,
        bootstrap_samples=80,
        min_calibration_games=20,
        min_calibration_draw_games=4,
        min_evaluation_games=8,
        min_evaluation_draw_games=2,
    )
    assert artifact["game_sets_disjoint"] is True
    assert artifact["parameters"]["slope"] > 0
    assert artifact["calibration_game_set_sha256"] != artifact["evaluation_game_set_sha256"]
    assert len(rows) == 8
    assert all(sum(row["baseline_wdl"]) == pytest.approx(1.0) for row in rows)


def test_v3_baseline_rejects_game_leakage() -> None:
    calibration = _rows("same", 8, {0, 4})
    with pytest.raises(ValueError, match="leakage"):
        build_v3_baseline(
            calibration,
            calibration,
            min_calibration_games=1,
            min_calibration_draw_games=1,
            min_evaluation_games=1,
            min_evaluation_draw_games=1,
        )


def test_v3_manifest_binds_rows_calibration_and_split(tmp_path) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_text("{}\n")
    calibration = tmp_path / "calibration.json"
    shared = {
        "checkpoint_sha256": "a" * 64,
        "calibration_game_set_sha256": "b" * 64,
        "evaluation_game_set_sha256": "c" * 64,
        "calibration_games": 192,
        "calibration_draw_games": 8,
        "evaluation_games": 48,
        "evaluation_draw_games": 2,
    }
    calibration.write_text(
        json.dumps(
            {
                "schema": V3_BASELINE_CALIBRATION_SCHEMA,
                "game_sets_disjoint": True,
                **shared,
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": V3_BASELINE_MANIFEST_SCHEMA,
                "rows_sha256": sha256_file(rows),
                "calibration_artifact": calibration.name,
                "calibration_sha256": sha256_file(calibration),
                "game_sets_disjoint": True,
                **shared,
            }
        )
    )
    assert verify_v3_baseline_manifest(rows, manifest)["evaluation_games"] == 48
    rows.write_text("tampered\n")
    with pytest.raises(ValueError, match="rows hash"):
        verify_v3_baseline_manifest(rows, manifest)
