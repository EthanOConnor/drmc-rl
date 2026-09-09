"""Analyze a frozen whole-game confirmation plan and standardized game journal."""

import argparse
import json
from pathlib import Path

from drmc_rl.arena.paired_confirmation import confirm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    games = [json.loads(line) for line in args.games.read_text().splitlines() if line.strip()]
    report = confirm(games, plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
