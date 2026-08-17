"""Executable, fail-closed promotion gate for mature counterfactual targets.

The gate verifies not only point metrics, but the chain of provenance connecting
the balanced source bank, frozen continuation mixture, grouped calibration,
member-wise beam-8 release, beam convergence, and observed-action/V3 bootstrap.
A clean mechanics pilot cannot pass by supplying unrelated or hand-mixed
artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from drmc_rl.search.pill_belief import CHANCE_MODEL_ID

_ALLOWED_INFORMATION_SCOPES = {
    "public-pair-state-v2",
    "privileged-pending-attack-continuation-v1",
}


@dataclass(frozen=True, slots=True)
class QualityGateThresholds:
    min_states: int = 1024
    max_states: int = 2048
    min_calibration_games: int = 192
    min_calibration_strata: int = 12
    min_calibration_games_per_stratum: int = 16
    min_calibration_folds: int = 5
    min_natural_draw_games: int = 8
    min_bootstrap_games: int = 48
    min_bootstrap_draw_games: int = 1
    min_beam_top1_agreement: float = 0.95
    max_beam_win_delta_p95: float = 0.02
    max_beam_policy_js_p95: float = 0.01
    min_stratum_top1_agreement: float = 0.85
    max_stratum_win_delta_p95: float = 0.04
    max_stratum_policy_js_p95: float = 0.02
    require_bootstrap_ci_improvement: bool = True
    reference_beam: int = 8
    convergence_beam: int = 4
    required_beams: tuple[int, ...] = (1, 4, 8)

    def __post_init__(self) -> None:
        if not 1 <= self.min_states <= self.max_states:
            raise ValueError("state thresholds are invalid")
        if (
            self.min_calibration_games < 2
            or self.min_calibration_strata < 1
            or self.min_calibration_games_per_stratum < 1
            or self.min_calibration_folds < 2
            or self.min_natural_draw_games < 1
            or self.min_bootstrap_games < 2
            or self.min_bootstrap_draw_games < 0
        ):
            raise ValueError("calibration/bootstrap thresholds are invalid")
        for value, label in (
            (self.min_beam_top1_agreement, "beam agreement"),
            (self.min_stratum_top1_agreement, "stratum agreement"),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{label} threshold must be in [0,1]")
        if any(
            value < 0
            for value in (
                self.max_beam_win_delta_p95,
                self.max_beam_policy_js_p95,
                self.max_stratum_win_delta_p95,
                self.max_stratum_policy_js_p95,
            )
        ):
            raise ValueError("beam delta thresholds must be non-negative")
        if self.reference_beam < 1 or self.convergence_beam < 1:
            raise ValueError("beam thresholds must be positive")
        required = tuple(sorted(set(int(item) for item in self.required_beams)))
        if any(item < 1 for item in required):
            raise ValueError("required beams must be positive")
        if self.reference_beam not in required or self.convergence_beam not in required:
            raise ValueError("reference and convergence beams must be required")
        object.__setattr__(self, "required_beams", required)


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


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text.lower())


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


def _release_hashes(payload: Mapping[str, Any], path: str) -> tuple[str, ...]:
    raw = _nested(payload, path, ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    return tuple(str(item) for item in raw)


def _member_calibration_valid(payload: Mapping[str, Any], *, min_folds: int) -> bool:
    parameters = payload.get("parameters")
    if not isinstance(parameters, Mapping):
        return False
    if not (
        _finite(parameters.get("slope"))
        and float(parameters["slope"]) > 0
        and _finite(parameters.get("bias"))
        and _finite(parameters.get("draw_logit"))
    ):
        return False
    crossfit = payload.get("crossfit")
    if not isinstance(crossfit, Mapping):
        return False
    folds = crossfit.get("folds")
    if not isinstance(folds, Sequence) or len(folds) < min_folds:
        return False
    return bool(crossfit.get("all_training_folds_draw_identifiable"))


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
    hashes = dict(input_sha256 or {})
    checks: list[GateCheck] = []

    def check(
        identifier: str,
        passed: bool,
        observed: object,
        required: object,
        detail: str,
    ) -> None:
        checks.append(GateCheck(identifier, bool(passed), observed, required, detail))

    audit_states = 0
    audit_candidates = 0
    audit_settings: Mapping[str, Any] = {}
    audit_release_hashes: tuple[str, ...] = ()
    teacher_ids: tuple[str, ...] = ()

    if audit is None:
        check(
            "audit-present",
            False,
            None,
            "audit artifact",
            "counterfactual audit is missing",
        )
    else:
        audit_states = int(audit.get("states", 0))
        audit_candidates = int(audit.get("candidate_labels", 0))
        audit_settings_raw = audit.get("settings")
        audit_settings = audit_settings_raw if isinstance(audit_settings_raw, Mapping) else {}
        audit_release_hashes = tuple(str(item) for item in audit.get("shard_release_sha256", ()))
        teacher_ids = tuple(str(item) for item in audit.get("teacher_ids", ()))
        check(
            "audit-schema",
            str(audit.get("schema", "")) in {
                "drmc-counterfactual-pilot-audit-v2",
                "drmc-counterfactual-pilot-audit-v3",
            },
            audit.get("schema"),
            "counterfactual audit v2+",
            "unknown audit schemas cannot promote a release",
        )
        check(
            "full-candidate-coverage",
            bool(audit.get("full_candidate_coverage")) and audit_candidates > 0,
            {"states": audit_states, "candidate_labels": audit_candidates},
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
            chance_model == CHANCE_MODEL_ID
            and bool(audit.get("chance_model_gate_eligible")),
            chance_model,
            CHANCE_MODEL_ID,
            "independent one-ninth branching is mechanics-only",
        )
        chance_beam = int(_nested(audit_settings, "search.chance_beam", 0))
        check(
            "full-chance-support",
            chance_beam >= 9,
            chance_beam,
            ">=9",
            "the posterior has at most nine supported pills; promotion cannot prune support",
        )
        check(
            "reserve-belief-history",
            bool(audit.get("source_reserve_belief_complete")),
            audit.get("source_reserve_belief_states"),
            audit_states,
            "every source state carries validated public reveal history",
        )
        check(
            "member-wise-uncertainty",
            int(audit.get("memberwise_complete_states", 0)) == audit_states
            and int(audit.get("memberwise_complete_candidates", 0))
            == audit_candidates
            and int(audit.get("teacher_count_min", 0)) >= 2
            and int(audit.get("teacher_count_min", 0))
            == int(audit.get("teacher_count_max", -1)),
            {
                "states": audit.get("memberwise_complete_states"),
                "candidates": audit.get("memberwise_complete_candidates"),
                "teacher_count_min": audit.get("teacher_count_min"),
                "teacher_count_max": audit.get("teacher_count_max"),
            },
            {
                "states": audit_states,
                "candidates": audit_candidates,
                "teacher_count": ">=2 and constant",
            },
            "each candidate must export every frozen member W/D/L and finite disagreement",
        )
        scope = str(audit.get("information_scope", "unspecified"))
        check(
            "information-scope-declared",
            scope in _ALLOWED_INFORMATION_SCOPES,
            scope,
            sorted(_ALLOWED_INFORMATION_SCOPES),
            "teacher-only privileged inputs must not be confused with deployable search",
        )
        check(
            "clean-release-repository",
            int(audit.get("repository_dirty_shards", -1)) == 0,
            audit.get("repository_dirty_shards"),
            0,
            "promotion artifacts must be produced from committed code",
        )
        mixture_hash = audit_settings.get("mixture_manifest_sha256")
        calibration_hash = audit_settings.get("wdl_calibration_sha256")
        check(
            "audit-mixture-hash",
            _is_sha256(mixture_hash),
            mixture_hash,
            "SHA-256",
            "the frozen continuation mixture must be content-addressed",
        )
        check(
            "audit-calibration-hash",
            _is_sha256(calibration_hash),
            calibration_hash,
            "SHA-256",
            "the W/D/L link used by search must be content-addressed",
        )
        if "calibration" in hashes:
            check(
                "audit-calibration-artifact-match",
                str(calibration_hash) == hashes["calibration"],
                calibration_hash,
                hashes["calibration"],
                "the audited release must use the supplied calibration artifact",
            )

    if calibration is None:
        check(
            "calibration-present",
            False,
            None,
            "grouped calibration",
            "calibration is missing",
        )
    else:
        heldout_raw = calibration.get("heldout_metrics") or calibration
        heldout = heldout_raw if isinstance(heldout_raw, Mapping) else {}
        grouped_raw = calibration.get("grouped_calibration") or calibration
        grouped = grouped_raw if isinstance(grouped_raw, Mapping) else {}
        crossfit_raw = grouped.get("crossfit")
        crossfit = crossfit_raw if isinstance(crossfit_raw, Mapping) else {}
        collection_raw = calibration.get("collection")
        collection = collection_raw if isinstance(collection_raw, Mapping) else {}
        games = int(heldout.get("validation_games", heldout.get("games", 0)))
        draws = int(heldout.get("natural_draw_games", 0))
        check(
            "calibration-schema",
            str(calibration.get("schema", ""))
            in {
                "drmc-strong-league-wdl-calibration-v2",
                "drmc-strong-league-wdl-calibration-v3",
            }
            and str(grouped.get("schema", ""))
            in {
                "drmc-grouped-davidson-calibration-v2",
                "drmc-grouped-davidson-calibration-v3",
            },
            {
                "outer": calibration.get("schema"),
                "grouped": grouped.get("schema"),
            },
            "Strong League calibration v2+/grouped v2+",
            "unknown calibration schemas cannot promote a release",
        )
        check(
            "calibration-games",
            games >= limit.min_calibration_games,
            games,
            f">={limit.min_calibration_games}",
            "validation must be separated by whole natural game",
        )
        strata_raw = collection.get("strata")
        strata = list(strata_raw) if isinstance(strata_raw, Sequence) else []
        per_stratum = [
            int(item.get("games", 0))
            for item in strata
            if isinstance(item, Mapping)
        ]
        check(
            "calibration-strata",
            len(per_stratum) >= limit.min_calibration_strata
            and min(per_stratum, default=0)
            >= limit.min_calibration_games_per_stratum,
            {
                "strata": len(per_stratum),
                "min_games": min(per_stratum, default=0),
            },
            {
                "strata": f">={limit.min_calibration_strata}",
                "min_games": f">={limit.min_calibration_games_per_stratum}",
            },
            "all level/speed cells need enough independent games",
        )
        folds_raw = crossfit.get("folds")
        folds = list(folds_raw) if isinstance(folds_raw, Sequence) else []
        check(
            "calibration-crossfit-folds",
            len(folds) >= limit.min_calibration_folds,
            len(folds),
            f">={limit.min_calibration_folds}",
            "grouped validation must have multiple independently fitted folds",
        )
        check(
            "calibration-draw-folds",
            bool(crossfit.get("all_training_folds_draw_identifiable")),
            crossfit.get("all_training_folds_draw_identifiable"),
            True,
            "every training fold must contain natural draw evidence",
        )
        check(
            "draw-identifiability",
            bool(heldout.get("draw_identifiable"))
            and draws >= limit.min_natural_draw_games,
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
                "calibration must improve on held-out whole games",
            )
        member_calibrations_raw = calibration.get("member_calibrations")
        member_calibrations = (
            member_calibrations_raw
            if isinstance(member_calibrations_raw, Mapping)
            else {}
        )
        expected_members = set(teacher_ids)
        observed_members = {str(key) for key in member_calibrations}
        check(
            "member-calibration-coverage",
            bool(expected_members)
            and observed_members == expected_members
            and all(
                isinstance(value, Mapping)
                and _member_calibration_valid(
                    value, min_folds=limit.min_calibration_folds
                )
                for value in member_calibrations.values()
            ),
            sorted(observed_members),
            sorted(expected_members),
            "every exported teacher member needs a finite draw-aware grouped link",
        )
        calibration_mixture_hash = calibration.get("mixture_manifest_sha256")
        audit_mixture_hash = audit_settings.get("mixture_manifest_sha256")
        check(
            "calibration-mixture-match",
            _is_sha256(calibration_mixture_hash)
            and str(calibration_mixture_hash) == str(audit_mixture_hash),
            calibration_mixture_hash,
            audit_mixture_hash,
            "calibration and search must refer to the same frozen mixture manifest",
        )

    if beam_sweep is None:
        check(
            "beam-sweep-present",
            False,
            None,
            "opponent beam 1/4/8 sweep",
            "beam sweep is missing",
        )
    else:
        schema = str(beam_sweep.get("schema", ""))
        reference = int(beam_sweep.get("reference_beam", -1))
        beams = tuple(sorted(int(item) for item in beam_sweep.get("beams", ())))
        comparisons_raw = beam_sweep.get("comparisons")
        comparisons = comparisons_raw if isinstance(comparisons_raw, Mapping) else {}
        comparison_raw = comparisons.get(str(limit.convergence_beam))
        comparison = comparison_raw if isinstance(comparison_raw, Mapping) else None
        check(
            "beam-sweep-schema",
            schema == "drmc-counterfactual-opponent-beam-sweep-v2",
            schema,
            "drmc-counterfactual-opponent-beam-sweep-v2",
            "v2 records strict non-beam settings comparability",
        )
        check(
            "beam-coverage",
            set(limit.required_beams).issubset(beams),
            beams,
            limit.required_beams,
            "beam 1 is sensitivity evidence; beams 4 and 8 establish convergence",
        )
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
            check(
                "beam-settings-compatible",
                bool(comparison.get("settings_compatible")),
                comparison.get("settings_compatible"),
                True,
                "only opponent_beam may differ between compared releases",
            )
            aggregate_raw = comparison.get("aggregate")
            aggregate = aggregate_raw if isinstance(aggregate_raw, Mapping) else {}
            agreement = float(aggregate.get("top1_agreement", math.nan))
            win_p95 = float(_nested(aggregate, "max_win_delta.p95", math.nan))
            js_p95 = float(_nested(aggregate, "policy_js.p95", math.nan))
            check(
                "beam-state-coverage",
                int(aggregate.get("states", -1)) == audit_states,
                aggregate.get("states"),
                audit_states,
                "beam convergence must cover the audited beam-8 state set",
            )
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
            by_stratum_raw = comparison.get("by_stratum")
            by_stratum = by_stratum_raw if isinstance(by_stratum_raw, Mapping) else {}
            expected_strata = {
                str(key)
                for key, value in (
                    (bank_manifest.get("strata") or {}).items()
                    if isinstance(bank_manifest, Mapping)
                    and isinstance(bank_manifest.get("strata"), Mapping)
                    else ()
                )
                if int(value) > 0
            }
            observed_strata = {str(key) for key in by_stratum}
            check(
                "beam-stratum-coverage",
                bool(expected_strata) and observed_strata == expected_strata,
                sorted(observed_strata),
                sorted(expected_strata),
                "aggregate convergence may not hide an omitted tactical cell",
            )
            stratum_failures: list[str] = []
            for key, value in by_stratum.items():
                if not isinstance(value, Mapping):
                    stratum_failures.append(str(key))
                    continue
                top1 = float(value.get("top1_agreement", math.nan))
                state_win_p95 = float(_nested(value, "max_win_delta.p95", math.nan))
                state_js_p95 = float(_nested(value, "policy_js.p95", math.nan))
                if not (
                    _finite(top1)
                    and top1 >= limit.min_stratum_top1_agreement
                    and _finite(state_win_p95)
                    and state_win_p95 <= limit.max_stratum_win_delta_p95
                    and _finite(state_js_p95)
                    and state_js_p95 <= limit.max_stratum_policy_js_p95
                ):
                    stratum_failures.append(str(key))
            check(
                "beam-stratum-convergence",
                bool(by_stratum) and not stratum_failures,
                stratum_failures,
                {
                    "top1": f">={limit.min_stratum_top1_agreement}",
                    "win_p95": f"<={limit.max_stratum_win_delta_p95}",
                    "policy_js_p95": f"<={limit.max_stratum_policy_js_p95}",
                },
                "each declared tactical cell must converge, not only the aggregate",
            )
            reference_hashes = _release_hashes(comparison, "reference.release_sha256")
            check(
                "beam-reference-release-match",
                bool(audit_release_hashes)
                and tuple(reference_hashes) == tuple(audit_release_hashes),
                reference_hashes,
                audit_release_hashes,
                "beam sweep reference must be the audited beam-8 release",
            )

    if bank_manifest is None:
        check(
            "balanced-bank-present",
            False,
            None,
            "balanced bank manifest",
            "state bank is missing",
        )
    else:
        states = int(bank_manifest.get("states", 0))
        shortfall = int(bank_manifest.get("quota_shortfall", -1))
        policy = str(bank_manifest.get("rollout_policy", ""))
        check(
            "balanced-bank-size",
            limit.min_states <= states <= limit.max_states,
            states,
            f"[{limit.min_states},{limit.max_states}]",
            "promotion bank is large enough for strata without becoming an unreviewed dump",
        )
        check(
            "balanced-bank-audit-match",
            states == audit_states,
            states,
            audit_states,
            "audited release and balanced bank must contain the same state count",
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
            bool(policy)
            and "random" not in policy.lower()
            and not bool(bank_manifest.get("diagnostic_only")),
            {
                "policy": policy,
                "diagnostic_only": bank_manifest.get("diagnostic_only"),
            },
            "frozen competitive rollout policy, non-diagnostic",
            "random-action banks distort the tactical state distribution",
        )
        check(
            "balanced-bank-reserve-belief",
            str(bank_manifest.get("chance_model", "")) == CHANCE_MODEL_ID,
            bank_manifest.get("chance_model"),
            CHANCE_MODEL_ID,
            "bank rows must retain public reserve-belief history",
        )
        source_manifest_hash = bank_manifest.get("source_manifest_sha256")
        check(
            "balanced-bank-source-provenance",
            _is_sha256(source_manifest_hash)
            and str(bank_manifest.get("source_sampling", ""))
            == "whole-game-global-tactical-round-robin-v1"
            and not bool(bank_manifest.get("source_diagnostic_only")),
            {
                "source_manifest_sha256": source_manifest_hash,
                "source_sampling": bank_manifest.get("source_sampling"),
                "source_diagnostic_only": bank_manifest.get("source_diagnostic_only"),
            },
            "hashed non-diagnostic whole-game source manifest",
            "the final bank must inherit verifiable whole-game sampling provenance",
        )
        check(
            "balanced-bank-release-input-match",
            str(bank_manifest.get("sha256", ""))
            == str(audit_settings.get("input_sha256", "")),
            bank_manifest.get("sha256"),
            audit_settings.get("input_sha256"),
            "the audited release must consume the supplied balanced bank bytes",
        )
        rollout_hash = bank_manifest.get("rollout_policy_manifest_sha256")
        check(
            "balanced-bank-mixture-match",
            _is_sha256(rollout_hash)
            and str(rollout_hash)
            == str(audit_settings.get("mixture_manifest_sha256", "")),
            rollout_hash,
            audit_settings.get("mixture_manifest_sha256"),
            "bank rollout and search continuation must use the same frozen mixture",
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
        outcomes_raw = bootstrap.get("outcomes")
        outcomes = outcomes_raw if isinstance(outcomes_raw, Mapping) else {}
        check(
            "bootstrap-independent-games",
            games >= limit.min_bootstrap_games,
            games,
            f">={limit.min_bootstrap_games}",
            "comparison uncertainty is grouped by held-out game",
        )
        check(
            "bootstrap-draw-coverage",
            int(outcomes.get("draw", 0)) >= limit.min_bootstrap_draw_games,
            outcomes.get("draw", 0),
            f">={limit.min_bootstrap_draw_games}",
            "W/D/L comparison should include natural held-out draws",
        )
        bootstrap_release_hashes = tuple(
            str(item) for item in bootstrap.get("release_sha256", ())
        )
        check(
            "bootstrap-release-match",
            bool(audit_release_hashes)
            and bootstrap_release_hashes == audit_release_hashes,
            bootstrap_release_hashes,
            audit_release_hashes,
            "bootstrap comparison must evaluate the audited beam-8 release",
        )
        check(
            "bootstrap-information-scope-match",
            str(bootstrap.get("chance_model", ""))
            == str(audit.get("chance_model", "") if audit else "")
            and str(bootstrap.get("information_scope", ""))
            == str(audit.get("information_scope", "") if audit else ""),
            {
                "chance_model": bootstrap.get("chance_model"),
                "information_scope": bootstrap.get("information_scope"),
            },
            {
                "chance_model": audit.get("chance_model") if audit else None,
                "information_scope": audit.get("information_scope") if audit else None,
            },
            "bootstrap metrics must refer to the same teacher information contract",
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
        schema="drmc-v3-counterfactual-quality-gate-v2",
        status="passed" if passed else "staged",
        passed=passed,
        thresholds=asdict(limit),
        checks=tuple(checks),
        input_sha256=hashes,
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
    hashes = {
        key: _sha256(path) for key, path in paths.items() if path is not None
    }
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
