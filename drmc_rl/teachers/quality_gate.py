"""Executable promotion gate for mature counterfactual quality targets."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from drmc_rl.search.pill_belief import CHANCE_MODEL_ID


@dataclass(frozen=True, slots=True)
class QualityGateThresholds:
    min_states: int = 1024
    max_states: int = 2048
    min_calibration_games: int = 48
    min_natural_draw_games: int = 1
    min_beam_top1_agreement: float = 0.95
    max_beam_win_delta_p95: float = 0.02
    max_beam_policy_js_p95: float = 0.01
    require_bootstrap_ci_improvement: bool = True
    reference_beam: int = 8
    convergence_beam: int = 4

    def __post_init__(self) -> None:
        if not 1 <= self.min_states <= self.max_states:
            raise ValueError("state thresholds are invalid")
        if self.min_calibration_games < 2 or self.min_natural_draw_games < 0:
            raise ValueError("calibration thresholds are invalid")
        if not 0 <= self.min_beam_top1_agreement <= 1:
            raise ValueError("beam agreement threshold must be in [0,1]")
        if self.max_beam_win_delta_p95 < 0 or self.max_beam_policy_js_p95 < 0:
            raise ValueError("beam delta thresholds must be non-negative")


@dataclass(frozen=True, slots=True)
class GateCheck:
    id: str
    passed: bool
    observed: object
    required: object
    detail: str


@dataclass(frozen=True, slots=True)
class QualityGateReport:
    schema: str
    status: str
    passed: bool
    thresholds: Mapping[str, object]
    checks: tuple[GateCheck, ...]
    input_sha256: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nested(payload: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = payload
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return default
        value = value[component]
    return value


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _metric_ci_improves(payload: Mapping[str, Any], metric: str) -> tuple[bool, object]:
    value = _nested(payload, f"paired_game_bootstrap.{metric}")
    if not isinstance(value, Mapping):
        value = _nested(payload, f"crossfit.paired_game_bootstrap.{metric}")
    if not isinstance(value, Mapping):
        return False, None
    high = value.get("ci95_high")
    return bool(_finite(high) and float(high) < 0.0), dict(value)


def evaluate_quality_gate(
    *,
    audit: Mapping[str, Any] | None,
    calibration: Mapping[str, Any] | None,
    beam_sweep: Mapping[str, Any] | None,
    bank_manifest: Mapping[str, Any] | None,
    bootstrap: Mapping[str, Any] | None,
    input_sha256: Mapping[str, str] | None = None,
    thresholds: QualityGateThresholds | None = None,
) -> QualityGateReport:
    limit = thresholds or QualityGateThresholds()
    checks: list[GateCheck] = []

    def check(identifier: str, passed: bool, observed: object, required: object, detail: str):
        checks.append(GateCheck(identifier, bool(passed), observed, required, detail))

    if audit is None:
        check("audit-present", False, None, "audit artifact", "counterfactual audit is missing")
    else:
        states = int(audit.get("states", 0))
        candidates = int(audit.get("candidate_labels", 0))
        check(
            "full-candidate-coverage",
            bool(audit.get("full_candidate_coverage")) and candidates > 0,
            {"states": states, "candidate_labels": candidates},
            "all source legal actions exactly once",
            "coverage is checked against the supplied source bank",
        )
        check(
            "search-budget",
            int(audit.get("budget_exhausted", -1)) == 0,
            audit.get("budget_exhausted"),
            0,
            "no incomplete search may enter a promoted release",
        )
        chance_model = str(audit.get("chance_model", ""))
        check(
            "reserve-chance-model",
            chance_model == CHANCE_MODEL_ID and bool(audit.get("chance_model_gate_eligible")),
            chance_model,
            CHANCE_MODEL_ID,
            "independent 1/9 branching is mechanics-only and cannot pass quality promotion",
        )
        check(
            "reserve-belief-history",
            bool(audit.get("source_reserve_belief_complete")),
            audit.get("source_reserve_belief_states"),
            states,
            "every source state carries the public reveal history used by the seed posterior",
        )
        check(
            "member-wise-uncertainty",
            int(audit.get("memberwise_complete_states", 0)) == states
            and int(audit.get("memberwise_complete_candidates", 0)) == candidates,
            {
                "states": audit.get("memberwise_complete_states"),
                "candidates": audit.get("memberwise_complete_candidates"),
            },
            {"states": states, "candidates": candidates},
            "each candidate must export every frozen member W/D/L and finite disagreement",
        )
        check(
            "information-scope-declared",
            str(audit.get("information_scope", "unspecified")) != "unspecified",
            audit.get("information_scope"),
            "explicit public or privileged teacher scope",
            "teacher-only privileged inputs must not be confused with deployable search inputs",
        )

    if calibration is None:
        check("calibration-present", False, None, "grouped calibration", "calibration is missing")
    else:
        heldout = calibration.get("heldout_metrics") or calibration
        games = int(heldout.get("validation_games", heldout.get("games", 0)))
        draws = int(heldout.get("natural_draw_games", 0))
        check(
            "calibration-games",
            games >= limit.min_calibration_games,
            games,
            f">={limit.min_calibration_games}",
            "validation must be separated by whole natural game",
        )
        check(
            "draw-identifiability",
            bool(heldout.get("draw_identifiable")) and draws >= limit.min_natural_draw_games,
            {"draw_games": draws, "identifiable": heldout.get("draw_identifiable")},
            f">={limit.min_natural_draw_games} natural draw games",
            "the Davidson draw parameter is not identified by win/loss-only evidence",
        )
        for metric in ("brier", "log_loss"):
            improved, observed = _metric_ci_improves(heldout, metric)
            check(
                f"calibration-{metric}-paired-ci",
                improved if limit.require_bootstrap_ci_improvement else observed is not None,
                observed,
                "calibrated-minus-baseline game-bootstrap CI95 high < 0",
                "calibration must improve on held-out whole games, not just fitted rows",
            )

    if beam_sweep is None:
        check("beam-sweep-present", False, None, "opponent beam 1/4/8 sweep", "beam sweep is missing")
    else:
        reference = int(beam_sweep.get("reference_beam", -1))
        comparisons = beam_sweep.get("comparisons") or {}
        comparison = comparisons.get(str(limit.convergence_beam)) if isinstance(comparisons, Mapping) else None
        check(
            "beam-reference",
            reference == limit.reference_beam,
            reference,
            limit.reference_beam,
            "convergence is judged against the declared high-beam release",
        )
        if not isinstance(comparison, Mapping):
            check(
                "beam-convergence-comparison",
                False,
                None,
                f"beam {limit.convergence_beam} versus {limit.reference_beam}",
                "required aligned comparison is absent",
            )
        else:
            aggregate = comparison.get("aggregate") or {}
            agreement = float(aggregate.get("top1_agreement", math.nan))
            win_p95 = float(_nested(aggregate, "max_win_delta.p95", math.nan))
            js_p95 = float(_nested(aggregate, "policy_js.p95", math.nan))
            check(
                "beam-top1-convergence",
                _finite(agreement) and agreement >= limit.min_beam_top1_agreement,
                agreement,
                f">={limit.min_beam_top1_agreement}",
                "best-action stability is measured on identical source/action sets",
            )
            check(
                "beam-value-convergence",
                _finite(win_p95) and win_p95 <= limit.max_beam_win_delta_p95,
                win_p95,
                f"<={limit.max_beam_win_delta_p95}",
                "p95 state-wise maximum candidate win-probability change",
            )
            check(
                "beam-policy-convergence",
                _finite(js_p95) and js_p95 <= limit.max_beam_policy_js_p95,
                js_p95,
                f"<={limit.max_beam_policy_js_p95}",
                "p95 Jensen-Shannon divergence of root policy targets",
            )

    if bank_manifest is None:
        check("balanced-bank-present", False, None, "balanced bank manifest", "state bank is missing")
    else:
        states = int(bank_manifest.get("states", 0))
        shortfall = int(bank_manifest.get("quota_shortfall", -1))
        policy = str(bank_manifest.get("rollout_policy", ""))
        check(
            "balanced-bank-size",
            limit.min_states <= states <= limit.max_states,
            states,
            f"[{limit.min_states},{limit.max_states}]",
            "promotion bank is large enough for strata without becoming an unreviewed corpus dump",
        )
        check(
            "balanced-bank-quotas",
            shortfall == 0,
            shortfall,
            0,
            "all declared level/speed/tactical quotas must be filled",
        )
        check(
            "balanced-bank-policy",
            bool(policy) and "random" not in policy.lower(),
            policy,
            "frozen competitive rollout policy",
            "random-action banks distort the tactical state distribution",
        )
        check(
            "balanced-bank-reserve-belief",
            str(bank_manifest.get("chance_model", "")) == CHANCE_MODEL_ID,
            bank_manifest.get("chance_model"),
            CHANCE_MODEL_ID,
            "bank rows must retain public reserve-belief history",
        )

    if bootstrap is None:
        check(
            "bootstrap-comparison-present",
            False,
            None,
            "observed-action/V3 bootstrap comparison",
            "direct bootstrap comparison is missing",
        )
    else:
        games = int(bootstrap.get("games", 0))
        check(
            "bootstrap-independent-games",
            games >= limit.min_calibration_games,
            games,
            f">={limit.min_calibration_games}",
            "comparison uncertainty is grouped by game/player split rather than decision rows",
        )
        for metric in ("brier", "log_loss"):
            improved, observed = _metric_ci_improves(bootstrap, metric)
            check(
                f"bootstrap-{metric}-paired-ci",
                improved if limit.require_bootstrap_ci_improvement else observed is not None,
                observed,
                "counterfactual-minus-V3 game-bootstrap CI95 high < 0",
                "the mature teacher must beat the observed-action/V3 bootstrap directly",
            )

    passed = bool(checks) and all(item.passed for item in checks)
    return QualityGateReport(
        schema="drmc-v3-counterfactual-quality-gate-v1",
        status="passed" if passed else "staged",
        passed=passed,
        thresholds=asdict(limit),
        checks=tuple(checks),
        input_sha256=dict(input_sha256 or {}),
    )


def load_and_evaluate(
    *,
    audit_path: Path | None,
    calibration_path: Path | None,
    beam_sweep_path: Path | None,
    bank_manifest_path: Path | None,
    bootstrap_path: Path | None,
    thresholds: QualityGateThresholds | None = None,
) -> QualityGateReport:
    paths = {
        "audit": audit_path,
        "calibration": calibration_path,
        "beam_sweep": beam_sweep_path,
        "bank_manifest": bank_manifest_path,
        "bootstrap": bootstrap_path,
    }
    payloads = {
        key: None if path is None else json.loads(path.read_text())
        for key, path in paths.items()
    }
    hashes = {key: _sha256(path) for key, path in paths.items() if path is not None}
    return evaluate_quality_gate(
        audit=payloads["audit"],
        calibration=payloads["calibration"],
        beam_sweep=payloads["beam_sweep"],
        bank_manifest=payloads["bank_manifest"],
        bootstrap=payloads["bootstrap"],
        input_sha256=hashes,
        thresholds=thresholds,
    )


__all__ = [
    "GateCheck",
    "QualityGateReport",
    "QualityGateThresholds",
    "evaluate_quality_gate",
    "load_and_evaluate",
]
