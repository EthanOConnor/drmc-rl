"""Create provenance-bound frozen-V3 W/D/L rows for the quality gate."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import tempfile
from pathlib import Path

from drmc_rl.human.afterstate_runtime import AfterstatePolicyRuntime
from drmc_rl.teachers.counterfactual_release import canonical_json, sha256_file
from drmc_rl.teachers.v3_baseline import (
    V3_BASELINE_MANIFEST_SCHEMA,
    build_v3_baseline,
    load_source_rows,
    score_source_rows,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_gzip_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as target:
                for row in rows:
                    target.write(canonical_json(row) + b"\n")
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _git_state() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return revision, dirty


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-bank", type=Path, required=True)
    parser.add_argument("--evaluation-bank", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--calibration-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--min-calibration-games", type=int, default=192)
    parser.add_argument("--min-calibration-draw-games", type=int, default=0)
    parser.add_argument("--min-evaluation-games", type=int, default=48)
    parser.add_argument("--min-evaluation-draw-games", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.folds < 2 or args.bootstrap_samples < 1:
        parser.error("batch, fold, and bootstrap counts must be positive")
    if args.calibration_bank.resolve() == args.evaluation_bank.resolve():
        parser.error("calibration and evaluation banks must be separate artifacts")

    runtime = AfterstatePolicyRuntime(args.checkpoint, device=args.device, seed=args.seed)
    try:
        calibration_scores = score_source_rows(
            runtime, load_source_rows(args.calibration_bank), batch_size=args.batch_size
        )
        evaluation_scores = score_source_rows(
            runtime, load_source_rows(args.evaluation_bank), batch_size=args.batch_size
        )
        identity = runtime.identity
    finally:
        runtime.close()

    artifact, rows = build_v3_baseline(
        calibration_scores,
        evaluation_scores,
        seed=args.seed,
        folds=args.folds,
        bootstrap_samples=args.bootstrap_samples,
        min_calibration_games=args.min_calibration_games,
        min_calibration_draw_games=args.min_calibration_draw_games,
        min_evaluation_games=args.min_evaluation_games,
        min_evaluation_draw_games=args.min_evaluation_draw_games,
    )
    revision, dirty = _git_state()
    checkpoint_sha256 = sha256_file(args.checkpoint)
    calibration_bank_sha256 = sha256_file(args.calibration_bank)
    evaluation_bank_sha256 = sha256_file(args.evaluation_bank)
    artifact.update(
        {
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_identity": identity,
            "calibration_bank_sha256": calibration_bank_sha256,
            "evaluation_bank_sha256": evaluation_bank_sha256,
            "repository_commit": revision,
            "repository_dirty": dirty,
        }
    )
    _write_json(args.calibration_output, artifact)
    _write_gzip_jsonl(args.rows_output, rows)
    diagnostic = dirty or (
        args.min_calibration_games < 192
        or args.min_evaluation_games < 48
    )
    manifest = {
        "schema": V3_BASELINE_MANIFEST_SCHEMA,
        "rows": len(rows),
        "rows_artifact": os.path.relpath(
            args.rows_output.resolve(), args.manifest_output.parent.resolve()
        ),
        "rows_sha256": sha256_file(args.rows_output),
        "calibration_artifact": os.path.relpath(
            args.calibration_output.resolve(), args.manifest_output.parent.resolve()
        ),
        "calibration_sha256": sha256_file(args.calibration_output),
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_bank_sha256": calibration_bank_sha256,
        "evaluation_bank_sha256": evaluation_bank_sha256,
        "calibration_game_set_sha256": artifact["calibration_game_set_sha256"],
        "evaluation_game_set_sha256": artifact["evaluation_game_set_sha256"],
        "calibration_games": artifact["calibration_games"],
        "calibration_draw_games": artifact["calibration_draw_games"],
        "evaluation_games": artifact["evaluation_games"],
        "evaluation_draw_games": artifact["evaluation_draw_games"],
        "game_sets_disjoint": True,
        "repository_commit": revision,
        "repository_dirty": dirty,
        "diagnostic_only": diagnostic,
    }
    _write_json(args.manifest_output, manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
