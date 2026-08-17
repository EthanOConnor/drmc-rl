"""Compare aligned counterfactual releases, especially opponent-beam sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drmc_rl.teachers.release_analysis import compare_beam_sweep, load_release


def _parse_release(value: str) -> tuple[int, list[Path]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("release must be BEAM=manifest[,manifest...]")
    raw_beam, raw_paths = value.split("=", 1)
    try:
        beam = int(raw_beam)
    except ValueError as error:
        raise argparse.ArgumentTypeError("release beam must be an integer") from error
    paths = [Path(item) for item in raw_paths.split(",") if item]
    if beam < 1 or not paths:
        raise argparse.ArgumentTypeError("release requires a positive beam and manifest paths")
    return beam, paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release",
        action="append",
        required=True,
        help="BEAM=manifest[,manifest...] (repeat for each beam)",
    )
    parser.add_argument("--reference-beam", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    releases = {}
    for raw in args.release:
        beam, paths = _parse_release(raw)
        if beam in releases:
            parser.error(f"duplicate beam {beam}")
        releases[beam] = load_release(paths)
    result = compare_beam_sweep(releases, reference_beam=args.reference_beam)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
