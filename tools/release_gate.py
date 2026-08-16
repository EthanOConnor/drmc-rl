"""Evaluate predeclared product release evidence from a JSON specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drmc_rl.eval.release_gates import (
    competitive_release_gate,
    execution_release_gate,
    trainer_release_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("competitive", "execution", "trainer"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("release-gate input must be a JSON object")
    if args.kind == "competitive":
        evidence = competitive_release_gate(**payload)
    elif args.kind == "execution":
        evidence = execution_release_gate(**payload)
    else:
        evidence = trainer_release_gate(**payload)
    result = {"schema": f"drmc-{args.kind}-release-gate-v1", **evidence.to_dict()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not evidence.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
