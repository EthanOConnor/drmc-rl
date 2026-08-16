"""Generate a versioned counterfactual target release through a model adapter.

The adapter argument is ``module:function``.  The function receives parsed CLI
arguments and returns ``(model_or_models, decode_state)``.  ``decode_state``
turns each input JSON object into the backend's restorable state.  Keeping the
adapter explicit prevents this tool from silently reading hidden state or a
legacy own-board simulator.
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
from pathlib import Path
from drmc_rl.search.joint_event import SearchConfig
from drmc_rl.teachers.counterfactual import CounterfactualTeacher


def _open(path: Path, mode: str):
    return gzip.open(path, mode + "t", encoding="utf-8") if path.suffix == ".gz" else path.open(mode, encoding="utf-8")


def _load_adapter(spec: str, args: argparse.Namespace):
    if ":" not in spec:
        raise ValueError("adapter must be module:function")
    module_name, function_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    model, decoder = factory(args)
    if not callable(decoder):
        raise TypeError("adapter decoder must be callable")
    return model, decoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--root-side", type=int, choices=(0, 1), default=0)
    parser.add_argument("--depth-events", type=int, default=4)
    parser.add_argument("--opponent-mode", choices=("expectation", "minimax"), default="expectation")
    parser.add_argument("--own-beam", type=int, default=512)
    parser.add_argument("--opponent-beam", type=int, default=12)
    args, _unknown = parser.parse_known_args()
    model, decode = _load_adapter(args.adapter, args)
    teacher = CounterfactualTeacher(
        model,
        config=SearchConfig(
            depth_events=args.depth_events,
            own_beam=args.own_beam,
            opponent_beam=args.opponent_beam,
            opponent_mode=args.opponent_mode,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with _open(args.input, "r") as source, _open(args.output, "w") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            state = decode(payload)
            label = teacher.label(
                state,
                root_side=args.root_side,
                metadata={"source_line": line_number, "source_id": payload.get("id")},
            )
            target.write(json.dumps(label.to_dict(), sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
