#!/usr/bin/env python3
"""Measure cadence and human-vs-model calibration from consented PP replays."""

from __future__ import annotations

import argparse
import json

from drmc_rl.human.professor_corpus import corpus_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", help="ppcorpus1 JSONL from drmariostats")
    parser.add_argument("--output", help="write JSON report instead of stdout")
    args = parser.parse_args()
    encoded = json.dumps(corpus_report(args.export), indent=2, sort_keys=True) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
