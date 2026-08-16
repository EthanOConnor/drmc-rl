"""Run the strict earliest-lock dominance architecture gate."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner
from drmc_rl.experiments.earliest_lock import (
    EarliestLockExperiment,
    NativeForcedTimingBackend,
    load_probes,
)


def _value_adapter(spec: str | None):
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError("value adapter must be module:function")
    module_name, function_name = spec.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError("value adapter must be callable")
    return function


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL timing probes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lib-path")
    parser.add_argument("--value-epsilon", type=float, default=1e-3)
    parser.add_argument(
        "--value-adapter",
        help="optional module:function scoring (snapshot, probe) on a common value scale",
    )
    args = parser.parse_args()
    probes = load_probes(args.input)
    runner = DrMarioVsPoolRunner(num_pairs=1, lib_path=args.lib_path)
    try:
        backend = NativeForcedTimingBackend(runner, value_fn=_value_adapter(args.value_adapter))
        report = EarliestLockExperiment(backend, value_epsilon=args.value_epsilon).run(probes)
        report.write(args.output)
        summary = {
            key: value
            for key, value in report.to_dict().items()
            if key != "records"
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        runner.close()


if __name__ == "__main__":
    main()
