from __future__ import annotations

"""Build the in-process Dr. Mario pool shared library.

Usage:
  python -m tools.build_drmario_pool

This produces a platform-specific shared library under `game_engine/build/`:
  - macOS: `libdrmario_pool.dylib`
  - Linux: `libdrmario_pool.so`

The pool is used by the `cpp-pool` backend in `training/envs/drmario_pool_vec.py`.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def _library_name() -> str:
    if sys.platform == "darwin":
        return "libdrmario_pool.dylib"
    if sys.platform.startswith("linux"):
        return "libdrmario_pool.so"
    if sys.platform == "win32":
        return "drmario_pool.dll"
    raise RuntimeError(f"Unsupported platform: {sys.platform!r}")


def build(*, verbose: bool = False) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    engine_dir = repo_root / "game_engine"
    if not engine_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {engine_dir}")

    cmd = ["make", "-C", str(engine_dir), "libdrmario_pool"]
    if verbose:
        print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    out = engine_dir / "build" / _library_name()
    if not out.is_file():
        raise FileNotFoundError(f"Build succeeded but library not found at {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="Print build command")
    args = parser.parse_args(argv)
    out = build(verbose=bool(args.verbose))
    print(str(out), flush=True)


if __name__ == "__main__":
    main()

