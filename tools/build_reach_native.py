from __future__ import annotations

"""Build the native reachability helper shared library.

Usage:
  python -m tools.build_reach_native

This produces a platform-specific shared library under `reach_native/build/`
that accelerates the placement reachability planner.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _library_name() -> str:
    if sys.platform == "darwin":
        return "libdrm_reach_full.dylib"
    if sys.platform.startswith("linux"):
        return "libdrm_reach_full.so"
    if sys.platform == "win32":
        return "drm_reach_full.dll"
    raise RuntimeError(f"Unsupported platform: {sys.platform!r}")


def _compile_command(src: Path, out: Path) -> list[str]:
    cflags = ["-O3", "-std=c11", "-DNDEBUG"]
    if sys.platform == "darwin":
        return ["clang", *cflags, "-dynamiclib", "-o", str(out), str(src)]
    if sys.platform.startswith("linux"):
        return ["clang", *cflags, "-shared", "-fPIC", "-o", str(out), str(src)]
    if sys.platform == "win32":
        return ["clang", *cflags, "-shared", "-o", str(out), str(src)]
    raise RuntimeError(f"Unsupported platform: {sys.platform!r}")


def build(*, verbose: bool = False) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "reach_native" / "drm_reach_full.c"
    if not src.is_file():
        raise FileNotFoundError(f"Missing source file: {src}")

    build_dir = repo_root / "reach_native" / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    out = build_dir / _library_name()

    cmd = _compile_command(src, out)
    env = dict(os.environ)
    if verbose:
        print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(repo_root), env=env)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="Print compiler invocation")
    args = parser.parse_args(argv)
    out = build(verbose=bool(args.verbose))
    print(str(out), flush=True)


if __name__ == "__main__":
    main()

