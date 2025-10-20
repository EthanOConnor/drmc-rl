#!/usr/bin/env python3
"""Create an archive with the core code and docs for review.

This script is designed to run from the directory that contains the ``drmc-rl``
repository.  It collects the tracked source, configuration, and documentation
files that are relevant for code review while deliberately skipping large
artifacts (ROMs, emulator cores, Java tooling, and similar assets).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

# Directories whose contents are not needed for a core code review.  These are
# typically binaries or third-party artifacts that would bloat the archive.
EXCLUDED_TOP_LEVEL = {
    "cores",
    "drmarioai",
    "dr-mario-disassembly",
    "roms",
}

# File suffixes we want to omit regardless of their location.  These capture
# emulator cores, ROM images, and other binary blobs that are not helpful for a
# source review.
EXCLUDED_SUFFIXES = {
    ".nes",
    ".sfc",
    ".smc",
    ".zip",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
}

DEFAULT_REPO_NAME = "drmc-rl"
DEFAULT_OUTPUT_NAME = "drmc-rl_core_review.zip"


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    """Parse the command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO_NAME,
        help=(
            "Path to the drmc-rl repository relative to the current working "
            "directory (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Name of the zip archive to create. If omitted, a sensible "
            "default is used."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be archived without creating the zip.",
    )
    return parser.parse_args(argv)


def gather_tracked_files(repo_path: Path) -> list[str]:
    """Return the git-tracked files for the repository."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "ls-files"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover
        raise SystemExit(f"Unable to list tracked files in {repo_path}: {exc}") from exc

    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return files


def should_include(path: str) -> bool:
    """Determine whether a tracked file should be included in the archive."""

    parts = path.split("/")
    if parts and parts[0] in EXCLUDED_TOP_LEVEL:
        return False

    if any(path.endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
        return False

    # Skip git-related metadata or build artifacts should they appear in the
    # tracked list (defensive; none are expected today).
    if ".git/" in path:
        return False

    return True


def build_archive(repo_path: Path, output_path: Path, dry_run: bool) -> None:
    """Create the archive with the filtered set of files."""

    tracked_files = gather_tracked_files(repo_path)
    files_to_package = [path for path in tracked_files if should_include(path)]

    if dry_run:
        for path in files_to_package:
            print(path)
        print(f"Would create archive with {len(files_to_package)} files at {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    from zipfile import ZIP_DEFLATED, ZipFile

    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        prefix = repo_path.name
        for relative_path in files_to_package:
            source = repo_path / relative_path
            arcname = f"{prefix}/{relative_path}"
            archive.write(source, arcname)

    print(f"Created {output_path} with {len(files_to_package)} files.")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    repo_path = Path(args.repo).resolve()
    if not repo_path.exists():
        raise SystemExit(f"Repository path not found: {repo_path}")

    output_path = Path(args.output) if args.output else Path(DEFAULT_OUTPUT_NAME)
    output_path = output_path.resolve()

    build_archive(repo_path, output_path, args.dry_run)


if __name__ == "__main__":
    main()
