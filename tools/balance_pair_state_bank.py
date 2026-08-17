"""Select a deterministic level/speed/tactical quota bank from oversampled states.

Promotion banks must inherit and verify the source collector manifest. Operator-
supplied rollout labels are accepted only as cross-checks; they cannot replace
content-addressed source provenance.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Mapping

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
    with opener(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row).decode("utf-8") + "\n")


def _source_manifest(
    path: Path | None,
    *,
    input_path: Path,
    allow_unverified: bool,
) -> tuple[dict[str, Any], str | None]:
    input_hash = sha256_file(input_path)
    if path is None:
        if not allow_unverified:
            raise ValueError(
                "a source manifest is required for a promotion bank; "
                "pass --input-manifest or keep the default <input>.manifest.json"
            )
        return {}, None
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError("input manifest must be a JSON object")
    if str(payload.get("sha256", "")) != input_hash:
        raise ValueError("input manifest SHA-256 does not match input bank bytes")
    chance_model = str(payload.get("chance_model", ""))
    if chance_model != CHANCE_MODEL_ID:
        raise ValueError(
            f"input bank uses ineligible chance model {chance_model!r}; "
            f"expected {CHANCE_MODEL_ID!r}"
        )
    if not bool(payload.get("reserve_initial_board_conditioned")):
        raise ValueError(
            "input bank reserve belief is not conditioned on the public "
            "initial virus bottle"
        )
    if bool(payload.get("diagnostic_only")) and not allow_unverified:
        raise ValueError("diagnostic source banks cannot produce promotion banks")
    return dict(payload), sha256_file(path)


def _consistent_option(
    name: str,
    explicit: object,
    source: Mapping[str, Any],
    source_key: str,
) -> object:
    observed = source.get(source_key)
    if explicit is not None and observed is not None and str(explicit) != str(observed):
        raise ValueError(
            f"operator {name} {explicit!r} disagrees with source manifest "
            f"{source_key}={observed!r}"
        )
    return observed if observed is not None else explicit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--input-manifest",
        type=Path,
        help="defaults to <input>.manifest.json when that file exists",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--quota-json", type=Path)
    parser.add_argument("--per-cell", type=int, default=24)
    parser.add_argument("--level", type=int, action="append", default=[])
    parser.add_argument("--speed", type=int, action="append", default=[])
    parser.add_argument("--tactical-stratum", action="append", default=[])
    parser.add_argument("--field", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--rollout-policy")
    parser.add_argument("--rollout-policy-manifest-sha256")
    parser.add_argument("--allow-shortfall", action="store_true")
    parser.add_argument("--allow-missing-reserve-belief", action="store_true")
    parser.add_argument(
        "--allow-unverified-source",
        action="store_true",
        help="diagnostic only: allow a missing or diagnostic source manifest",
    )
    args = parser.parse_args()
    if args.per_cell < 1:
        parser.error("--per-cell must be positive")
    source_manifest_path = args.input_manifest
    if source_manifest_path is None:
        sibling = Path(str(args.input) + ".manifest.json")
        if sibling.is_file():
            source_manifest_path = sibling
    try:
        source, source_manifest_hash = _source_manifest(
            source_manifest_path,
            input_path=args.input,
            allow_unverified=args.allow_unverified_source,
        )
        rollout_policy = _consistent_option(
            "--rollout-policy",
            args.rollout_policy,
            source,
            "rollout_policy",
        )
        rollout_manifest_hash = _consistent_option(
            "--rollout-policy-manifest-sha256",
            args.rollout_policy_manifest_sha256,
            source,
            "rollout_policy_manifest_sha256",
        )
    except ValueError as error:
        parser.error(str(error))
    if not rollout_policy:
        parser.error("rollout policy is absent from both CLI and input manifest")
    if not rollout_manifest_hash and not args.allow_unverified_source:
        parser.error("promotion source manifest lacks rollout policy manifest SHA-256")

    fields = tuple(args.field or ("level", "speed", "tactical_stratum"))
    if args.quota_json is not None:
        quota = quota_from_json(json.loads(args.quota_json.read_text()))
    else:
        quota = cross_product_quota(
            levels=tuple(args.level or DEFAULT_LEVELS),
            speeds=tuple(args.speed or DEFAULT_SPEEDS),
            tactical_strata=tuple(
                args.tactical_stratum or DEFAULT_TACTICAL_STRATA
            ),
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
            "bank quota shortfall; oversample the missing cells or pass "
            "--allow-shortfall for diagnostics only: "
            f"{json.dumps(missing, sort_keys=True)}"
        )
    _write(args.output, result.selected)
    manifest_path = args.manifest or Path(str(args.output) + ".manifest.json")
    source_sampling = source.get("per_game_selection")
    source_diagnostic = bool(source.get("diagnostic_only", not bool(source)))
    diagnostic = bool(
        result.quota_shortfall
        or args.allow_missing_reserve_belief
        or args.allow_unverified_source
        or source_diagnostic
    )
    manifest = {
        "schema": "drmc-balanced-pair-state-bank-v2",
        "artifact": str(args.output.resolve()),
        "sha256": sha256_file(args.output),
        "bank_identity": bank_identity(result.selected),
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "source_manifest": (
            None
            if source_manifest_path is None
            else str(source_manifest_path.resolve())
        ),
        "source_manifest_sha256": source_manifest_hash,
        "source_bank_schema": source.get("schema"),
        "source_sampling": source_sampling,
        "source_diagnostic_only": source_diagnostic,
        "source_games_rolled": source.get("games_rolled"),
        "source_candidate_states_considered": source.get(
            "candidate_states_considered"
        ),
        "states": len(result.selected),
        "fields": list(fields),
        "strata": dict(result.strata),
        "quota": dict(result.quota),
        "shortfall": dict(result.shortfall),
        "quota_shortfall": result.quota_shortfall,
        "duplicate_source_rows_ignored": result.duplicates,
        "seed": args.seed,
        "rollout_policy": str(rollout_policy),
        "rollout_policy_manifest_sha256": rollout_manifest_hash,
        "chance_model": (
            "missing-reserve-belief-diagnostic"
            if args.allow_missing_reserve_belief
            else CHANCE_MODEL_ID
        ),
        "reserve_initial_board_conditioned": bool(
            source.get("reserve_initial_board_conditioned")
        )
        and not args.allow_missing_reserve_belief,
        "diagnostic_only": diagnostic,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
