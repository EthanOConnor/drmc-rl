"""Evaluate and write the v3-counterfactual-quality promotion evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drmc_rl.teachers.quality_gate import QualityGateThresholds, load_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--beam-sweep", type=Path)
    parser.add_argument("--bank-manifest", type=Path)
    parser.add_argument("--bootstrap", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-states", type=int, default=1024)
    parser.add_argument("--max-states", type=int, default=2048)
    parser.add_argument("--min-calibration-games", type=int, default=48)
    parser.add_argument("--min-natural-draw-games", type=int, default=1)
    parser.add_argument("--min-beam-top1-agreement", type=float, default=0.95)
    parser.add_argument("--max-beam-win-delta-p95", type=float, default=0.02)
    parser.add_argument("--max-beam-policy-js-p95", type=float, default=0.01)
    parser.add_argument("--reference-beam", type=int, default=8)
    parser.add_argument("--convergence-beam", type=int, default=4)
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
        min_natural_draw_games=args.min_natural_draw_games,
        min_beam_top1_agreement=args.min_beam_top1_agreement,
        max_beam_win_delta_p95=args.max_beam_win_delta_p95,
        max_beam_policy_js_p95=args.max_beam_policy_js_p95,
        require_bootstrap_ci_improvement=not args.allow_point_improvement,
        reference_beam=args.reference_beam,
        convergence_beam=args.convergence_beam,
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
