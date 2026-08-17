from __future__ import annotations

from drmc_rl.search.pill_belief import CHANCE_MODEL_ID
from drmc_rl.teachers.quality_gate import QualityGateThresholds, evaluate_quality_gate


def _paired():
    return {
        "brier": {"delta": -0.02, "ci95_low": -0.03, "ci95_high": -0.01},
        "log_loss": {"delta": -0.03, "ci95_low": -0.05, "ci95_high": -0.01},
    }


def _passing_inputs():
    audit = {
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
        "information_scope": "privileged-pending-attack-continuation-v1",
    }
    calibration = {
        "heldout_metrics": {
            "validation_games": 72,
            "natural_draw_games": 4,
            "draw_identifiable": True,
            "paired_game_bootstrap": _paired(),
        }
    }
    beam_sweep = {
        "reference_beam": 8,
        "comparisons": {
            "4": {
                "aggregate": {
                    "top1_agreement": 0.97,
                    "max_win_delta": {"p95": 0.015},
                    "policy_js": {"p95": 0.006},
                }
            }
        },
    }
    bank = {
        "states": 1440,
        "quota_shortfall": 0,
        "rollout_policy": "frozen-strong-league-mixture-argmax",
        "chance_model": CHANCE_MODEL_ID,
    }
    bootstrap = {"games": 72, "paired_game_bootstrap": _paired()}
    return audit, calibration, beam_sweep, bank, bootstrap


def test_quality_gate_passes_only_complete_evidence_bundle() -> None:
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
