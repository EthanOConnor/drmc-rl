"""Build a self-contained, no-Python-install human backend directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tools.build_drmario_pool import build as build_pool
from tools.build_reach_native import build as build_reach
from drmc_rl.execution.pace import PACES


def package(checkpoint: Path, output: Path, *, competitive_checkpoint: Path | None = None) -> Path:
    repo = Path(__file__).resolve().parents[1]
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if competitive_checkpoint is not None:
        competitive_checkpoint = competitive_checkpoint.expanduser().resolve()
        if not competitive_checkpoint.is_file():
            raise FileNotFoundError(competitive_checkpoint)

    reach_library = build_reach(verbose=True)
    pool_library = build_pool(verbose=True)
    package_name = "drmc-human-backend"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onedir",
            "--name",
            package_name,
            "--distpath",
            str(output),
            "--workpath",
            str(repo / "build" / "human-backend"),
            "--specpath",
            str(repo / "build" / "human-backend"),
            "--paths",
            str(repo),
            "--add-binary",
            f"{reach_library}{os.pathsep}.",
            "--add-binary",
            f"{pool_library}{os.pathsep}.",
            str(repo / "tools" / "human_backend.py"),
        ],
        cwd=repo,
        check=True,
    )
    package_dir = output / package_name
    model_dir = package_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, model_dir / "human_policy.pt.gz")
    if competitive_checkpoint is not None:
        shutil.copy2(competitive_checkpoint, model_dir / "competitive_policy.pt.gz")
    elif (model_dir / "competitive_policy.pt.gz").exists():
        raise ValueError("output contains a stale competitive model; choose a fresh output directory")
    models = {}
    for path in sorted(model_dir.glob("*.pt.gz")):
        with path.open("rb") as stream:
            models[path.name] = {"sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                                 "size_bytes": path.stat().st_size}
    (model_dir / "manifest.json").write_text(json.dumps({
        "schema": "drmc-trainer-package-v1", "models": models,
        "absolute_human_rating_calibrated": False,
        "human_execution_profile_validated": False,
        "motor_paces": [pace.to_dict() for pace in PACES],
        "unrestricted_pace_fallback": False,
        "reach_library_sha256": hashlib.sha256(reach_library.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    return package_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--competitive-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("dist"))
    args = parser.parse_args()
    print(package(args.checkpoint, args.output.resolve(), competitive_checkpoint=args.competitive_checkpoint))


if __name__ == "__main__":
    main()
