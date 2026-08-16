"""Compute a PSRO-lite population mixture from a JSON or CSV payoff matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from drmc_rl.arena.meta_strategy import (
    antisymmetrize_pairwise_payoff,
    solve_entropy_regularized_zero_sum,
    write_result,
)


def _load(path: Path) -> tuple[list[str] | None, np.ndarray]:
    if path.suffix.lower() == ".csv":
        return None, np.loadtxt(path, delimiter=",")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return None, np.asarray(payload, dtype=np.float64)
    if not isinstance(payload, dict) or "payoff" not in payload:
        raise ValueError("JSON payoff input must be a matrix or {agents, payoff}")
    agents = None if payload.get("agents") is None else [str(item) for item in payload["agents"]]
    return agents, np.asarray(payload["payoff"], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--floor", type=float, default=0.002)
    parser.add_argument("--no-antisymmetrize", action="store_true")
    args = parser.parse_args()
    agents, matrix = _load(args.payoff)
    if not args.no_antisymmetrize and matrix.shape[0] == matrix.shape[1]:
        matrix = antisymmetrize_pairwise_payoff(matrix)
    result = solve_entropy_regularized_zero_sum(
        matrix,
        iterations=args.iterations,
        temperature=args.temperature,
        floor=args.floor,
    )
    write_result(result, args.output, agents=agents)
    print(json.dumps(result.to_dict(agents), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
