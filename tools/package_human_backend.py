"""Build a self-contained, no-Python-install human backend directory."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tools.build_drmario_pool import build as build_pool
from tools.build_reach_native import build as build_reach


def package(checkpoint: Path, output: Path) -> Path:
    repo = Path(__file__).resolve().parents[1]
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

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
    return package_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dist"))
    args = parser.parse_args()
    print(package(args.checkpoint, args.output.resolve()))


if __name__ == "__main__":
    main()
