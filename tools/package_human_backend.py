"""Build a self-contained, no-Python-install human backend directory."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tools.build_drmario_pool import build as build_pool
from tools.build_reach_native import build as build_reach
from drmc_rl.execution.pace import PACES
from drmc_rl.human.backend import PROTOCOL_SCHEMA


def verify_package(package_dir: Path, *, competitive: bool = True) -> dict:
    """Exercise the frozen process and reject packages the app cannot use."""
    executable = package_dir / ("drmc-human-backend.exe" if sys.platform == "win32" else "drmc-human-backend")
    result = subprocess.run(
        [str(executable), "--device", "cpu"],
        input=json.dumps({"schema": PROTOCOL_SCHEMA, "type": "hello"}) + "\n",
        capture_output=True, text=True, timeout=120, check=True,
    )
    hello = json.loads(result.stdout)
    caps = hello.get("capabilities", {})
    if (hello.get("schema") != PROTOCOL_SCHEMA
            or caps.get("model", {}).get("schema") != "drmc-human-afterstate-v3"
            or caps.get("scheduled_execution", {}).get("version") != 1
            or caps.get("cadence", {}).get("unrestricted_fallback") is not False
            or caps.get("cadence", {}).get("profiles", []) != [p.to_dict() for p in PACES]
            or "quality" not in caps.get("strength", {}).get("controls", [])
            or (competitive and not caps.get("strength", {}).get("competitive_ceiling"))):
        raise ValueError(f"packaged trainer is incompatible: {caps}")
    return caps


def source_identity(path: Path) -> tuple[str, bool]:
    revision = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=normal"], text=True,
    )
    return revision, bool(status.strip())


def package(checkpoint: Path, output: Path, *, competitive_checkpoint: Path | None = None) -> Path:
    repo = Path(__file__).resolve().parents[1]
    source_revision, source_dirty = source_identity(repo)
    native_revision, native_dirty = source_identity(repo / "vendor/drmario_native")
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
    capabilities = verify_package(package_dir, competitive=competitive_checkpoint is not None)
    (model_dir / "manifest.json").write_text(json.dumps({
        "schema": "drmc-trainer-package-v1", "models": models,
        "absolute_human_rating_calibrated": False,
        "human_execution_profile_validated": False,
        "motor_paces": [pace.to_dict() for pace in PACES],
        "unrestricted_pace_fallback": False,
        "reach_library_sha256": hashlib.sha256(reach_library.read_bytes()).hexdigest(),
        "pool_library_sha256": hashlib.sha256(pool_library.read_bytes()).hexdigest(),
        "source_revision": source_revision,
        "source_dirty": source_dirty,
        "native_revision": native_revision,
        "native_dirty": native_dirty,
        "python_version": sys.version,
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("torch", "numpy", "pyinstaller", "pyinstaller-hooks-contrib")},
        "verified_capabilities": capabilities,
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
