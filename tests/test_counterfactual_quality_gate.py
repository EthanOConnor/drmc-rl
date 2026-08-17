from __future__ import annotations

from copy import deepcopy

from drmc_rl.search.pill_belief import CHANCE_MODEL_ID
from drmc_rl.teachers.quality_gate import QualityGateThresholds, evaluate_quality_gate

MIXTURE_HASH = "a" * 64
CALIBRATION_HASH = "b" * 64
BANK_HASH = "c" * 64
SOURCE_MANIFEST_HASH = "d" * 64
RELEASE_HASH = "e" * 64
V3_ROWS_HASH = "1" * 64
V3_CALIBRATION_HASH = "2" * 64
V3_CHECKPOINT_HASH = "3" * 64
V3_CALIBRATION_BANK_HASH = "4" * 64
V3_EVALUATION_BANK_HASH = "5" * 64
V3_CALIBRATION_GAMES_HASH = "6" * 64
V3_EVALUATION_GAMES_HASH = "7" * 64
TEACHERS = ("rewarm-900m", "plus-1b", "plus-750m", "plus-500m")
STRATA = tuple(
    f"{level}/{speed}/{tactical}"
    for level in (5, 10, 15, 20)
    for speed in (0, 1, 2)
    for tactical in (
        "midgame",
        "high-pressure",
        "topout-defense",
        "incoming-garbage",
        "race-finish",
    )
)


def _paired():
    return {
        "brier": {
            "delta": -0.02,
            "ci95_low": -0.03,
            "ci95_high": -0.01,
        },
        "log_loss": {
            "delta": -0.03,
            "ci95_low": -0.05,
            "ci95_high": -0.01,
        },
    }


def _member_calibration():
    return {
        "parameters": {"slope": 1.2, "bias": 0.0, "draw_logit": -1.5},
        "crossfit": {
            "folds": [{"fold": index} for index in range(5)],
            "all_training_folds_draw_identifiable": True,
        },
    }


def _passing_inputs():
    audit = {
        "schema": "drmc-counterfactual-pilot-audit-v3",
        "states": 1200,
        "candidate_labels": 42000,
        "full_candidate_coverage": True,
        "budget_exhausted": 0,
        "chance_model": CHANCE_MODEL_ID,
        "chance_model_gate_eligible": True,
        "source_reserve_belief_complete": True,
        "source_reserve_belief_states": 1200,
        "memberwise_complete_states": 1200,
        "memberwise_complete_candidates": 42000,
        "teacher_count_min": 4,
        "teacher_count_max": 4,
        "teacher_ids": list(TEACHERS),
        "teacher_weights": [0.4, 0.3, 0.2, 0.1],
        "information_scope": "privileged-pending-attack-continuation-v1",
        "repository_dirty_shards": 0,
        "shard_release_sha256": [RELEASE_HASH],
        "settings": {
            "input_sha256": BANK_HASH,
            "mixture_manifest_sha256": MIXTURE_HASH,
            "wdl_calibration_sha256": CALIBRATION_HASH,
            "search": {
                "depth_events": 2,
                "own_beam": 512,
                "opponent_beam": 8,
                "chance_beam": 9,
                "max_nodes": 100000,
            },
        },
    }
    calibration = {
        "schema": "drmc-strong-league-wdl-calibration-v3",
        "mixture_manifest_sha256": MIXTURE_HASH,
        "heldout_metrics": {
            "validation_games": 192,
            "natural_draw_games": 8,
            "draw_identifiable": True,
            "paired_game_bootstrap": _paired(),
        },
        "grouped_calibration": {
            "schema": "drmc-grouped-davidson-calibration-v3",
            "crossfit": {
                "folds": [{"fold": index} for index in range(5)],
                "all_training_folds_draw_identifiable": True,
            },
        },
        "collection": {
            "strata": [
                {"level": level, "speed": speed, "games": 16}
                for level in (5, 10, 15, 20)
                for speed in (0, 1, 2)
            ]
        },
        "member_calibrations": {teacher: _member_calibration() for teacher in TEACHERS},
    }
    by_stratum = {
        key: {
            "states": 20,
            "top1_agreement": 0.96,
            "max_win_delta": {"p95": 0.018},
            "policy_js": {"p95": 0.008},
        }
        for key in STRATA
    }
    beam_sweep = {
        "schema": "drmc-counterfactual-opponent-beam-sweep-v2",
        "reference_beam": 8,
        "beams": [1, 4, 8],
        "comparisons": {
            "4": {
                "settings_compatible": True,
                "reference": {"release_sha256": [RELEASE_HASH]},
                "aggregate": {
                    "states": 1200,
                    "top1_agreement": 0.97,
                    "max_win_delta": {"p95": 0.015},
                    "policy_js": {"p95": 0.006},
                },
                "by_stratum": by_stratum,
            },
            "1": {
                "settings_compatible": True,
                "reference": {"release_sha256": [RELEASE_HASH]},
                "aggregate": {"states": 1200},
                "by_stratum": by_stratum,
            },
        },
    }
    bank = {
        "schema": "drmc-balanced-pair-state-bank-v2",
        "sha256": BANK_HASH,
        "states": 1200,
        "quota_shortfall": 0,
        "strata": {key: 20 for key in STRATA},
        "quota": {key: 20 for key in STRATA},
        "rollout_policy": "frozen-strong-league-mixture-argmax",
        "rollout_policy_manifest_sha256": MIXTURE_HASH,
        "chance_model": CHANCE_MODEL_ID,
        "reserve_initial_board_conditioned": True,
        "diagnostic_only": False,
        "source_manifest_sha256": SOURCE_MANIFEST_HASH,
        "source_sampling": "whole-game-global-tactical-round-robin-v1",
        "source_diagnostic_only": False,
    }
    bootstrap = {
        "games": 48,
        "outcomes": {"win": 24, "draw": 2, "loss": 22},
        "release_sha256": [RELEASE_HASH],
        "chance_model": CHANCE_MODEL_ID,
        "information_scope": "privileged-pending-attack-continuation-v1",
        "paired_game_bootstrap": _paired(),
        "baseline_provenance": {
            "schema": "drmc-v3-wdl-bootstrap-manifest-v1",
            "rows_sha256": V3_ROWS_HASH,
            "calibration_sha256": V3_CALIBRATION_HASH,
            "checkpoint_sha256": V3_CHECKPOINT_HASH,
            "calibration_bank_sha256": V3_CALIBRATION_BANK_HASH,
            "evaluation_bank_sha256": V3_EVALUATION_BANK_HASH,
            "calibration_game_set_sha256": V3_CALIBRATION_GAMES_HASH,
            "evaluation_game_set_sha256": V3_EVALUATION_GAMES_HASH,
            "game_sets_disjoint": True,
            "calibration_games": 192,
            "calibration_draw_games": 8,
            "evaluation_games": 48,
            "evaluation_draw_games": 2,
            "diagnostic_only": False,
            "repository_dirty": False,
        },
    }
    return audit, calibration, beam_sweep, bank, bootstrap


def test_quality_gate_passes_only_complete_coherent_evidence_bundle() -> None:
    audit, calibration, beam_sweep, bank, bootstrap = _passing_inputs()
    report = evaluate_quality_gate(
        audit=audit,
        calibration=calibration,
        beam_sweep=beam_sweep,
        bank_manifest=bank,
        bootstrap=bootstrap,
        thresholds=QualityGateThresholds(),
    )
    assert report.passed is True
    assert report.status == "passed"
    assert report.schema == "drmc-v3-counterfactual-quality-gate-v2"
    assert all(check.passed for check in report.checks)


def test_quality_gate_rejects_legacy_uniform_reveal_and_missing_draws() -> None:
    audit, calibration, beam_sweep, bank, bootstrap = _passing_inputs()
    audit["chance_model"] = "independent-uniform-ordered-pair-v0"
    audit["chance_model_gate_eligible"] = False
    calibration["heldout_metrics"]["natural_draw_games"] = 0
    calibration["heldout_metrics"]["draw_identifiable"] = False
    report = evaluate_quality_gate(
        audit=audit,
        calibration=calibration,
        beam_sweep=beam_sweep,
        bank_manifest=bank,
        bootstrap=bootstrap,
    )
    failed = {check.id for check in report.checks if not check.passed}
    assert "reserve-chance-model" in failed
    assert "draw-identifiability" in failed
    assert report.status == "staged"


def test_quality_gate_rejects_cross_artifact_provenance_drift() -> None:
    audit, calibration, beam_sweep, bank, bootstrap = _passing_inputs()
    drifted = deepcopy(calibration)
    drifted["mixture_manifest_sha256"] = "f" * 64
    beam_sweep["comparisons"]["4"]["reference"]["release_sha256"] = ["0" * 64]
    bootstrap["release_sha256"] = ["1" * 64]
    bank["sha256"] = "2" * 64
    report = evaluate_quality_gate(
        audit=audit,
        calibration=drifted,
        beam_sweep=beam_sweep,
        bank_manifest=bank,
        bootstrap=bootstrap,
    )
    failed = {check.id for check in report.checks if not check.passed}
    assert "calibration-mixture-match" in failed
    assert "beam-reference-release-match" in failed
    assert "bootstrap-release-match" in failed
    assert "balanced-bank-release-input-match" in failed


def test_quality_gate_rejects_pruned_chance_and_bad_tactical_cell() -> None:
    audit, calibration, beam_sweep, bank, bootstrap = _passing_inputs()
    audit["settings"]["search"]["chance_beam"] = 4
    beam_sweep["comparisons"]["4"]["by_stratum"][STRATA[0]]["top1_agreement"] = 0.5
    report = evaluate_quality_gate(
        audit=audit,
        calibration=calibration,
        beam_sweep=beam_sweep,
        bank_manifest=bank,
        bootstrap=bootstrap,
    )
    failed = {check.id for check in report.checks if not check.passed}
    assert "full-chance-support" in failed
    assert "beam-stratum-convergence" in failed


def test_quality_gate_rejects_duplicate_calibration_cell_and_missing_bank_cell() -> None:
    audit, calibration, beam_sweep, bank, bootstrap = _passing_inputs()
    calibration["collection"]["strata"][-1] = dict(calibration["collection"]["strata"][0])
    removed = STRATA[-1]
    bank["states"] -= bank["strata"].pop(removed)
    bank["quota"].pop(removed)
    report = evaluate_quality_gate(
        audit=audit,
        calibration=calibration,
        beam_sweep=beam_sweep,
        bank_manifest=bank,
        bootstrap=bootstrap,
    )
    failed = {check.id for check in report.checks if not check.passed}
    assert "calibration-strata" in failed
    assert "balanced-bank-quotas" in failed
