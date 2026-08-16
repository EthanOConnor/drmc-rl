"""Inspect, validate, launch, and record the authoritative training program.

Examples
--------

    python -m tools.program status
    python -m tools.program validate --check-paths
    python -m tools.program launch g4-strong-league --dry-run
    python -m tools.program launch timing-action-gate \
        --set timing_probes=data/program/timing-probes.jsonl \
        --set timing_report=runs/program/timing-report.json
    python -m tools.program gate record pair-state-v2 --passed \
        --metric schema=drmc-pair-state-v2 --artifact runs/evidence/no-leak.json
    python -m tools.program artifact checkpoint.pt.gz --config run.yaml \
        --observation-schema drmc-public-pair-state-v2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from drmc_rl.program import ArtifactManifest, GateEvidence, ProgramSpec
from drmc_rl.program.model import format_command

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROGRAM = REPO_ROOT / "drmc_rl" / "program" / "program.yaml"


def _parse_set(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"expected NAME=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty substitution name in {item!r}")
        result[key] = value
    return result


def _parse_scalar(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _parse_metrics(values: list[str]) -> dict[str, Any]:
    return {key: _parse_scalar(value) for key, value in _parse_set(values).items()}


def _load(path: str | Path) -> ProgramSpec:
    return ProgramSpec.load(path)


def cmd_status(args: argparse.Namespace) -> int:
    spec = _load(args.program)
    payload = spec.as_dict()
    for gate_id, gate in spec.gates.items():
        evidence = spec.gate_evidence(gate_id)
        payload["gates"][gate_id]["open"] = spec.gate_is_open(gate_id)
        payload["gates"][gate_id]["evidence"] = (
            None if evidence is None else {
                "passed": evidence.passed,
                "recorded_at": evidence.recorded_at,
                "commit": evidence.commit,
            }
        )
    for recipe_id in spec.recipes:
        payload["recipes"][recipe_id]["blockers"] = spec.recipe_blockers(recipe_id)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(spec.name)
    print(f"authority: {spec.source_path.relative_to(spec.repo_root)}")
    print("\nGates")
    for gate_id, gate in sorted(spec.gates.items()):
        state = "OPEN" if spec.gate_is_open(gate_id) else gate.status.upper()
        print(f"  {state:9} {gate_id}: {gate.description}")
    print("\nRecipes")
    for recipe in sorted(spec.recipes.values(), key=lambda item: (item.stage, item.id)):
        blockers = spec.recipe_blockers(recipe.id)
        suffix = "" if not blockers else f" [blocked by {', '.join(blockers)}]"
        print(f"  stage {recipe.stage:<2} {recipe.status:7} {recipe.id}{suffix}")
        print(f"             {recipe.purpose}")
    print("\nProducts")
    for product in spec.products.values():
        open_product = all(spec.gate_is_open(gate) for gate in product.requires_gates)
        print(f"  {'READY' if open_product else 'BLOCKED':7} {product.id}: {product.description}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    spec = _load(args.program)
    spec.validate(check_paths=args.check_paths)
    print(f"valid: {spec.source_path}")
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    spec = _load(args.program)
    if args.recipe not in spec.recipes:
        raise KeyError(f"unknown recipe {args.recipe!r}")
    recipe = spec.recipes[args.recipe]
    if recipe.status == "blocked":
        raise RuntimeError(f"recipe {recipe.id!r} is blocked by program authority")
    if recipe.status == "retired":
        raise RuntimeError(f"recipe {recipe.id!r} is retired")
    if recipe.status == "staged" and not args.allow_staged:
        raise RuntimeError(
            f"recipe {recipe.id!r} is staged; pass --allow-staged after reviewing its gates"
        )
    substitutions = _parse_set(args.set)
    blockers = [
        blocker
        for blocker in spec.recipe_blockers(recipe.id)
        if not blocker.startswith("missing:{")
    ]
    # Explicit substitutions can satisfy placeholder paths.
    for path in recipe.requires_paths:
        if not (path.startswith("{") and path.endswith("}")):
            continue
        key = path[1:-1]
        value = substitutions.get(key)
        if not value or not Path(value).expanduser().exists():
            blockers.append(f"missing:{key}")
    if blockers and not args.ignore_gates:
        raise RuntimeError(f"recipe {recipe.id!r} cannot launch: {', '.join(blockers)}")
    command = list(recipe.resolved_command(spec.repo_root, substitutions))
    command.extend(args.extra)
    print(format_command(command), flush=True)
    if args.dry_run:
        return 0
    env = os.environ.copy()
    env["DRMC_PROGRAM_RECIPE"] = recipe.id
    env["DRMC_PROGRAM_VERSION"] = str(spec.version)
    completed = subprocess.run(command, cwd=spec.repo_root, env=env, check=False)
    return int(completed.returncode)


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return os.environ.get("GITHUB_SHA")


def cmd_gate_record(args: argparse.Namespace) -> int:
    spec = _load(args.program)
    if args.gate not in spec.gates:
        raise KeyError(f"unknown gate {args.gate!r}")
    gate = spec.gates[args.gate]
    evidence = GateEvidence(
        gate_id=gate.id,
        passed=bool(args.passed),
        recorded_at=datetime.now(UTC).isoformat(),
        commit=_git_commit(spec.repo_root),
        metrics=_parse_metrics(args.metric),
        artifacts=tuple(args.artifact),
        notes=tuple(args.note),
    )
    target = spec.repo_root / gate.evidence_path
    evidence.write(target)
    print(target)
    return 0


def cmd_gate_check(args: argparse.Namespace) -> int:
    spec = _load(args.program)
    gate_ids = tuple(spec.gates) if not args.gate else tuple(args.gate)
    failed = False
    for gate_id in gate_ids:
        if gate_id not in spec.gates:
            raise KeyError(f"unknown gate {gate_id!r}")
        opened = spec.gate_is_open(gate_id)
        print(f"{'OPEN' if opened else 'CLOSED'} {gate_id}")
        failed |= not opened
    return 1 if failed else 0


def _parse_json_mapping(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    return payload


def cmd_artifact(args: argparse.Namespace) -> int:
    manifest = ArtifactManifest.build(
        args.artifact,
        repo_root=REPO_ROOT,
        config=args.config,
        observation_schema=args.observation_schema,
        execution_profile=args.execution_profile,
        search=_parse_json_mapping(args.search),
        corpus_release=args.corpus_release,
        parents=args.parent,
        metadata=_parse_json_mapping(args.metadata),
    )
    target = Path(args.output) if args.output else Path(str(args.artifact) + ".manifest.json")
    manifest.write(target)
    print(target)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, default=DEFAULT_PROGRAM)
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="show stages, gates, recipes, and products")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    validate = sub.add_parser("validate", help="validate manifest structure")
    validate.add_argument("--check-paths", action="store_true")
    validate.set_defaults(func=cmd_validate)

    launch = sub.add_parser("launch", help="launch one authority-approved recipe")
    launch.add_argument("recipe")
    launch.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    launch.add_argument("--allow-staged", action="store_true")
    launch.add_argument("--ignore-gates", action="store_true")
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument(
        "extra",
        nargs="*",
        help="child-command arguments; prefix option-like arguments with --",
    )
    launch.set_defaults(func=cmd_launch)

    gate = sub.add_parser("gate", help="record or check gate evidence")
    gate_sub = gate.add_subparsers(dest="gate_command", required=True)
    record = gate_sub.add_parser("record")
    record.add_argument("gate")
    record.add_argument("--passed", action=argparse.BooleanOptionalAction, default=True)
    record.add_argument("--metric", action="append", default=[], metavar="NAME=VALUE")
    record.add_argument("--artifact", action="append", default=[])
    record.add_argument("--note", action="append", default=[])
    record.set_defaults(func=cmd_gate_record)
    check = gate_sub.add_parser("check")
    check.add_argument("gate", nargs="*")
    check.set_defaults(func=cmd_gate_check)

    artifact = sub.add_parser("artifact", help="write immutable artifact provenance")
    artifact.add_argument("artifact")
    artifact.add_argument("--output")
    artifact.add_argument("--config")
    artifact.add_argument("--observation-schema")
    artifact.add_argument("--execution-profile")
    artifact.add_argument("--search", help="JSON object")
    artifact.add_argument("--corpus-release")
    artifact.add_argument("--parent", action="append", default=[])
    artifact.add_argument("--metadata", help="JSON object")
    artifact.set_defaults(func=cmd_artifact)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
