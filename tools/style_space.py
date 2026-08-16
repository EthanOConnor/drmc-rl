"""Fit a rating-residualized style latent from corpus feature rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from drmc_rl.human.style import StyleSpace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="NPZ with features, ratings, player_ids")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimensions", type=int, default=6)
    parser.add_argument("--min-decisions", type=int, default=50)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        names = None
        if "feature_names" in data:
            names = [str(item) for item in data["feature_names"]]
        space = StyleSpace.fit(
            data["features"],
            data["ratings"],
            data["player_ids"],
            feature_names=names,
            dimensions=args.dimensions,
            min_decisions_per_player=args.min_decisions,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    space.write(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "dimensions": space.dimensions,
                "players": len(space.player_ids),
                "explained_variance": space.explained_variance.tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
