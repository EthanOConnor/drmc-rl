"""Generate strict joint-event search targets through an explicit state adapter."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import importlib
import json
from pathlib import Path

from drmc_rl.search.joint_event import JointEventSearch, SearchConfig


def _open(path: Path, mode: str):
    return gzip.open(path, mode + "t", encoding="utf-8") if path.suffix == ".gz" else path.open(mode, encoding="utf-8")


def _adapter(spec: str, args):
    if ":" not in spec:
        raise ValueError("adapter must be module:function")
    module_name, function_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    model, decode = factory(args)
    return model, decode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--root-side", type=int, choices=(0, 1), default=None)
    parser.add_argument("--max-nodes", type=int, default=200000)
    parser.add_argument("--depth-events", type=int, default=6)
    parser.add_argument("--own-beam", type=int, default=64)
    parser.add_argument("--opponent-beam", type=int, default=16)
    parser.add_argument("--opponent-mode", choices=("expectation", "minimax", "mixed"), default="expectation")
    parser.add_argument("--matrix-iterations", type=int, default=2048)
    parser.add_argument("--matrix-temperature", type=float, default=0.001)
    parser.add_argument("--matrix-gap-tolerance", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--checkpoint", help="forwarded to adapter")
    parser.add_argument(
        "--frontier-batch-size",
        type=int,
        default=0,
        help="opt-in cooperative batching; zero retains the recursive reference",
    )
    args, _unknown = parser.parse_known_args()
    model, decode = _adapter(args.adapter, args)
    search = JointEventSearch(
        model,
        SearchConfig(
            depth_events=args.depth_events,
            own_beam=args.own_beam,
            opponent_beam=args.opponent_beam,
            opponent_mode=args.opponent_mode,
            policy_temperature=args.temperature,
            max_nodes=args.max_nodes,
            matrix_iterations=args.matrix_iterations,
            matrix_temperature=args.matrix_temperature,
            matrix_gap_tolerance=args.matrix_gap_tolerance,
        ),
    )
    if args.frontier_batch_size:
        from drmc_rl.search.queued_event import QueuedJointEventSearch

        search = QueuedJointEventSearch(model, search.config, batch_size=args.frontier_batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with _open(args.states, "r") as source, _open(args.output, "w") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            root = decode(payload)
            root_side = (
                int(payload.get("root_side", 0)) if args.root_side is None else args.root_side
            )
            result = search.search(
                root, root_side=root_side, root_actions=tuple(model.legal_actions(root, root_side))
            )
            usable = not result.budget_exhausted and result.equilibrium_converged
            target.write(
                json.dumps(
                    {
                        "schema": "drmc-joint-search-target-v2",
                        "search_config": asdict(search.config),
                        "source_id": payload.get("id"),
                        "source_line": line_number,
                        "root_side": root_side,
                        "actions": list(result.actions),
                        "policy_target": None
                        if not usable
                        else result.policy_target.tolist(),
                        "utilities": None if not usable else result.utilities.tolist(),
                        "wdl": None
                        if not usable
                        else [
                            {"win": value.win, "draw": value.draw, "loss": value.loss}
                            for value in result.values
                        ],
                        "best_action": None if not usable else result.best_action,
                        "action_selection": result.action_selection,
                        "opponent_actions": list(result.opponent_actions),
                        "opponent_policy": list(result.opponent_policy) if usable else None,
                        "matrix_games": result.matrix_games,
                        "matrix_solve_ms": result.matrix_solve_ms,
                        "equilibrium_gap": result.equilibrium_gap,
                        "equilibrium_converged": result.equilibrium_converged,
                        "root_value": None
                        if not usable
                        else {
                            "win": result.root_value.win,
                            "draw": result.root_value.draw,
                            "loss": result.root_value.loss,
                        },
                        "nodes": result.nodes,
                        "cache_hits": result.cache_hits,
                        "budget_exhausted": result.budget_exhausted,
                        "usable_for_training": usable,
                        "unsearched_actions": sorted(
                            set(model.legal_actions(root, root_side)) - set(result.actions)
                        ),
                        "frontier_batches": getattr(search, "inference_batches", 0),
                    },
                    sort_keys=True,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
