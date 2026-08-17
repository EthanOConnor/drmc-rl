"""Generate a versioned counterfactual target release through a model adapter.

The adapter argument is ``module:function``. The function receives parsed CLI
arguments and returns ``(model_or_models, decode_state)``. ``decode_state``
turns each input JSON object into the backend's restorable state. Keeping the
adapter explicit prevents this tool from silently reading hidden state or a
legacy own-board simulator.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from drmc_rl.search.joint_event import SearchConfig
from drmc_rl.teachers.counterfactual import CounterfactualTeacher, WeightedTeacherModels
from drmc_rl.teachers.counterfactual_release import (
    ReleaseSettings,
    build_release,
    sha256_file,
)


def _load_adapter(spec: str, args: argparse.Namespace):
    if ":" not in spec:
        raise ValueError("adapter must be module:function")
    module_name, function_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    model, decoder = factory(args)
    if not callable(decoder):
        raise TypeError("adapter decoder must be callable")
    return model, decoder


def _models(value: Any) -> tuple[Any, ...]:
    if isinstance(value, WeightedTeacherModels):
        return value.models
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return (value,)


def _shared_adapter_attribute(value: Any, name: str, default: str) -> str:
    observed = {str(getattr(model, name, default)) for model in _models(value)}
    if len(observed) != 1:
        raise ValueError(f"adapter models disagree on {name}: {sorted(observed)}")
    return observed.pop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="release directory")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--root-side", type=int, choices=(0, 1), default=0)
    parser.add_argument("--depth-events", type=int, default=4)
    parser.add_argument(
        "--opponent-mode", choices=("expectation", "minimax"), default="expectation"
    )
    parser.add_argument("--own-beam", type=int, default=512)
    parser.add_argument("--opponent-beam", type=int, default=12)
    parser.add_argument("--chance-beam", type=int, default=16)
    parser.add_argument("--max-nodes", type=int, default=100000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stratum", action="append", default=[])
    parser.add_argument("--per-stratum", type=int)
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--corpus-release", required=True)
    parser.add_argument("--continuation-mixture", required=True)
    parser.add_argument("--mixture-manifest", type=Path)
    parser.add_argument("--wdl-calibration", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--native-revision", required=True)
    parser.add_argument("--planner-revision", required=True)
    parser.add_argument(
        "--chance-model",
        help="explicit provenance override; normally supplied by the adapter",
    )
    parser.add_argument(
        "--information-scope",
        help="explicit provenance override; normally supplied by the adapter",
    )
    parser.add_argument(
        "--allow-budget-exhausted",
        action="store_true",
        help="record rather than reject incomplete searches (diagnostics only)",
    )
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    model, decode = _load_adapter(args.adapter, args)
    search = SearchConfig(
        depth_events=args.depth_events,
        own_beam=args.own_beam,
        opponent_beam=args.opponent_beam,
        chance_beam=args.chance_beam,
        opponent_mode=args.opponent_mode,
        max_nodes=args.max_nodes,
    )
    chance_model = args.chance_model or _shared_adapter_attribute(
        model, "chance_model", "independent-uniform-ordered-pair-v0"
    )
    information_scope = args.information_scope or _shared_adapter_attribute(
        model, "information_scope", "unspecified"
    )
    settings = ReleaseSettings(
        input_sha256=sha256_file(args.input),
        adapter=args.adapter,
        root_side=args.root_side,
        search={
            "depth_events": search.depth_events,
            "own_beam": search.own_beam,
            "opponent_beam": search.opponent_beam,
            "chance_beam": search.chance_beam,
            "opponent_mode": search.opponent_mode,
            "policy_temperature": search.policy_temperature,
            "max_nodes": search.max_nodes,
        },
        seed=args.seed,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        stratum_fields=tuple(args.stratum),
        per_stratum=args.per_stratum,
        max_states=args.max_states,
        chunk_size=args.chunk_size,
        corpus_release=args.corpus_release,
        continuation_mixture=args.continuation_mixture,
        native_revision=args.native_revision,
        planner_revision=args.planner_revision,
        mixture_manifest_sha256=(
            sha256_file(args.mixture_manifest) if args.mixture_manifest else None
        ),
        wdl_calibration_sha256=(
            sha256_file(args.wdl_calibration) if args.wdl_calibration else None
        ),
        chance_model=str(chance_model),
        information_scope=str(information_scope),
    )
    manifest = build_release(
        input_path=args.input,
        output_dir=args.output,
        adapter_spec=args.adapter,
        teacher=CounterfactualTeacher(model, config=search),
        decode=decode,
        settings=settings,
        resume=args.resume,
        reject_budget_exhausted=not args.allow_budget_exhausted,
    )
    print(manifest)


if __name__ == "__main__":
    main()
