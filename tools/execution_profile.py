"""Fit or validate named human controller-operation envelopes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from drmc_rl.execution.profile import ExecutionProfile, profile_from_json, script_metrics

METRICS = (
    "reaction_frames",
    "total_edges",
    "peak_edges_250ms",
    "peak_edges_1s",
    "peak_edges_10s",
    "max_simultaneous_buttons",
    "direction_reversals",
    "correction_bursts",
    "complexity",
)


def _scripts(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if "script" not in payload:
                raise ValueError(f"line {line_number} is missing script")
            yield payload, np.asarray(payload["script"], dtype=np.uint8)


def fit(args: argparse.Namespace) -> None:
    rows = [script_metrics(script, fps=args.fps) for _payload, script in _scripts(args.input)]
    if len(rows) < args.min_scripts:
        raise ValueError(f"need at least {args.min_scripts} scripts, got {len(rows)}")
    q = float(args.quantile)
    values = {name: np.asarray([getattr(row, name) for row in rows], dtype=float) for name in METRICS}
    inter = np.asarray(
        [row.min_inter_edge_frames for row in rows if row.min_inter_edge_frames is not None],
        dtype=float,
    )
    distribution = {
        name: (float(array.mean()), float(array.std())) for name, array in values.items()
    }
    profile = ExecutionProfile(
        id=args.id,
        description=args.description,
        fps=args.fps,
        min_reaction_frames=max(0, int(np.floor(np.quantile(values["reaction_frames"], 1 - q)))),
        max_reaction_frames=int(np.ceil(np.quantile(values["reaction_frames"], q))),
        min_inter_edge_frames=0 if inter.size == 0 else int(np.floor(np.quantile(inter, 1 - q))),
        max_edges_250ms=int(np.ceil(np.quantile(values["peak_edges_250ms"], q))),
        max_edges_1s=int(np.ceil(np.quantile(values["peak_edges_1s"], q))),
        max_edges_10s=int(np.ceil(np.quantile(values["peak_edges_10s"], q))),
        max_simultaneous_buttons=int(np.ceil(np.quantile(values["max_simultaneous_buttons"], q))),
        max_direction_reversals=int(np.ceil(np.quantile(values["direction_reversals"], q))),
        max_correction_bursts=int(np.ceil(np.quantile(values["correction_bursts"], q))),
        max_complexity=float(np.quantile(values["complexity"], q)),
        distribution_targets=distribution,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile.write(args.output)
    print(json.dumps(profile.to_dict(), indent=2, sort_keys=True))


def validate(args: argparse.Namespace) -> None:
    profile = profile_from_json(args.profile)
    total = valid = 0
    violations: dict[str, int] = {}
    for _payload, script in _scripts(args.input):
        total += 1
        result = profile.validate(script)
        valid += int(result.valid)
        for violation in result.violations:
            violations[violation] = violations.get(violation, 0) + 1
    print(
        json.dumps(
            {
                "profile": profile.id,
                "scripts": total,
                "valid": valid,
                "valid_fraction": valid / max(total, 1),
                "violations": violations,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--input", type=Path, required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    fit_parser.add_argument("--id", required=True)
    fit_parser.add_argument("--description", required=True)
    fit_parser.add_argument("--quantile", type=float, default=0.99)
    fit_parser.add_argument("--fps", type=float, default=60.1)
    fit_parser.add_argument("--min-scripts", type=int, default=100)
    fit_parser.set_defaults(func=fit)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--input", type=Path, required=True)
    validate_parser.add_argument("--profile", type=Path, required=True)
    validate_parser.set_defaults(func=validate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
