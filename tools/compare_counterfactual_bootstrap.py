"""Compare counterfactual observed-action W/D/L against the frozen V3 bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drmc_rl.teachers.bootstrap_comparison import compare_bootstrap, load_bootstrap_bundle
from drmc_rl.teachers.release_analysis import load_release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="+")
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--bootstrap-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    release = load_release(args.manifest)
    rows, provenance = load_bootstrap_bundle(args.bootstrap, args.bootstrap_manifest)
    result = compare_bootstrap(
        release,
        rows,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        baseline_provenance=provenance,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
