"""Evaluate and write the v3-counterfactual-quality promotion evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drmc_rl.teachers.quality_gate import QualityGateThresholds, load_and_evaluate


def main() -> None:
    defaults = QualityGateThresholds()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--beam-sweep", type=Path)
    parser.add_argument("--bank-manifest", type=Path)
    parser.add_argument("--bootstrap", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-states", type=int, default=defaults.min_states)
    parser.add_argument("--max-states", type=int, default=defaults.max_states)
    parser.add_argument(
        "--min-calibration-games",
        type=int,
        default=defaults.min_calibration_games,
    )
    parser.add_argument(
        "--min-calibration-strata",
        type=int,
        default=defaults.min_calibration_strata,
    )
    parser.add_argument(
        "--min-calibration-games-per-stratum",
        type=int,
        default=defaults.min_calibration_games_per_stratum,
    )
    parser.add_argument(
        "--min-calibration-folds",
        type=int,
        default=defaults.min_calibration_folds,
    )
    parser.add_argument(
        "--min-natural-draw-games",
        type=int,
        default=defaults.min_natural_draw_games,
    )
    parser.add_argument(
        "--min-bootstrap-games",
        type=int,
        default=defaults.min_bootstrap_games,
    )
    parser.add_argument(
        "--min-bootstrap-draw-games",
        type=int,
        default=defaults.min_bootstrap_draw_games,
    )
    parser.add_argument(
        "--min-beam-top1-agreement",
        type=float,
        default=defaults.min_beam_top1_agreement,
    )
    parser.add_argument(
        "--max-beam-win-delta-p95",
        type=float,
        default=defaults.max_beam_win_delta_p95,
    )
    parser.add_argument(
        "--max-beam-policy-js-p95",
        type=float,
        default=defaults.max_beam_policy_js_p95,
    )
    parser.add_argument(
        "--min-stratum-top1-agreement",
        type=float,
        default=defaults.min_stratum_top1_agreement,
    )
    parser.add_argument(
        "--max-stratum-win-delta-p95",
        type=float,
        default=defaults.max_stratum_win_delta_p95,
    )
    parser.add_argument(
        "--max-stratum-policy-js-p95",
        type=float,
        default=defaults.max_stratum_policy_js_p95,
    )
    parser.add_argument(
        "--reference-beam", type=int, default=defaults.reference_beam
    )
    parser.add_argument(
        "--convergence-beam", type=int, default=defaults.convergence_beam
    )
    parser.add_argument(
        "--required-beam",
        type=int,
        action="append",
        default=[],
        help="repeat to override the required beam set (default: 1,4,8)",
    )
    parser.add_argument(
        "--allow-point-improvement",
        action="store_true",
        help="diagnostic only: do not require paired CI to exclude zero",
    )
    args = parser.parse_args()
    thresholds = QualityGateThresholds(
        min_states=args.min_states,
        max_states=args.max_states,
        min_calibration_games=args.min_calibration_games,
        min_calibration_strata=args.min_calibration_strata,
        min_calibration_games_per_stratum=(
            args.min_calibration_games_per_stratum
        ),
        min_calibration_folds=args.min_calibration_folds,
        min_natural_draw_games=args.min_natural_draw_games,
        min_bootstrap_games=args.min_bootstrap_games,
        min_bootstrap_draw_games=args.min_bootstrap_draw_games,
        min_beam_top1_agreement=args.min_beam_top1_agreement,
        max_beam_win_delta_p95=args.max_beam_win_delta_p95,
        max_beam_policy_js_p95=args.max_beam_policy_js_p95,
        min_stratum_top1_agreement=args.min_stratum_top1_agreement,
        max_stratum_win_delta_p95=args.max_stratum_win_delta_p95,
        max_stratum_policy_js_p95=args.max_stratum_policy_js_p95,
        require_bootstrap_ci_improvement=not args.allow_point_improvement,
        reference_beam=args.reference_beam,
        convergence_beam=args.convergence_beam,
        required_beams=tuple(args.required_beam or defaults.required_beams),
    )
    report = load_and_evaluate(
        audit_path=args.audit,
        calibration_path=args.calibration,
        beam_sweep_path=args.beam_sweep,
        bank_manifest_path=args.bank_manifest,
        bootstrap_path=args.bootstrap,
        thresholds=thresholds,
    )
    report.write(args.output)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    raise SystemExit(0 if report.passed else 2)


if __name__ == "__main__":
    main()
