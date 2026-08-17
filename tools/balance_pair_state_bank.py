"""Select a deterministic level/speed/tactical quota bank from oversampled states."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from drmc_rl.search.pill_belief import CHANCE_MODEL_ID
from drmc_rl.teachers.counterfactual_release import canonical_json, sha256_file
from drmc_rl.teachers.state_bank import (
    DEFAULT_LEVELS,
    DEFAULT_SPEEDS,
    DEFAULT_TACTICAL_STRATA,
    balance_state_rows,
    bank_identity,
    cross_product_quota,
    quota_from_json,
)


def _read(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else Path.open
    kwargs = {"encoding": "utf-8"}
    with opener(path, "wt", **kwargs) as handle:
        for row in rows:
            handle.write(canonical_json(row).decode("utf-8") + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--quota-json", type=Path)
    parser.add_argument("--per-cell", type=int, default=24)
    parser.add_argument("--level", type=int, action="append", default=[])
    parser.add_argument("--speed", type=int, action="append", default=[])
    parser.add_argument("--tactical-stratum", action="append", default=[])
    parser.add_argument("--field", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--rollout-policy", required=True)
    parser.add_argument("--rollout-policy-manifest-sha256")
    parser.add_argument("--allow-shortfall", action="store_true")
    parser.add_argument("--allow-missing-reserve-belief", action="store_true")
    args = parser.parse_args()
    if args.per_cell < 1:
        parser.error("--per-cell must be positive")
    fields = tuple(args.field or ("level", "speed", "tactical_stratum"))
    if args.quota_json is not None:
        quota = quota_from_json(json.loads(args.quota_json.read_text()))
    else:
        quota = cross_product_quota(
            levels=tuple(args.level or DEFAULT_LEVELS),
            speeds=tuple(args.speed or DEFAULT_SPEEDS),
            tactical_strata=tuple(args.tactical_stratum or DEFAULT_TACTICAL_STRATA),
            per_cell=args.per_cell,
        )
    result = balance_state_rows(
        _read(args.input),
        quota=quota,
        fields=fields,
        seed=args.seed,
        require_reserve_belief=not args.allow_missing_reserve_belief,
    )
    if result.quota_shortfall and not args.allow_shortfall:
        missing = {key: value for key, value in result.shortfall.items() if value}
        raise SystemExit(
            "bank quota shortfall; oversample the missing cells or pass --allow-shortfall "
            f"for diagnostics only: {json.dumps(missing, sort_keys=True)}"
        )
    _write(args.output, result.selected)
    manifest_path = args.manifest or Path(str(args.output) + ".manifest.json")
    manifest = {
        "schema": "drmc-balanced-pair-state-bank-v1",
        "artifact": str(args.output.resolve()),
        "sha256": sha256_file(args.output),
        "bank_identity": bank_identity(result.selected),
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "states": len(result.selected),
        "fields": list(fields),
        "strata": dict(result.strata),
        "quota": dict(result.quota),
        "shortfall": dict(result.shortfall),
        "quota_shortfall": result.quota_shortfall,
        "duplicate_source_rows_ignored": result.duplicates,
        "seed": args.seed,
        "rollout_policy": args.rollout_policy,
        "rollout_policy_manifest_sha256": args.rollout_policy_manifest_sha256,
        "chance_model": (
            "missing-reserve-belief-diagnostic"
            if args.allow_missing_reserve_belief
            else CHANCE_MODEL_ID
        ),
        "diagnostic_only": bool(result.quota_shortfall or args.allow_missing_reserve_belief),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
