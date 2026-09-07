from __future__ import annotations

"""Build the in-process Dr. Mario pool shared library.

Usage:
  python -m tools.build_drmario_pool

This produces a platform-specific shared library under `vendor/drmario_native/build/`:
  - macOS: `libdrmario_pool.dylib`
  - Linux: `libdrmario_pool.so`

The pool is used by the `cpp-pool` backend in `drmc_rl/training/envs/drmario_pool_vec.py`.
"""

import argparse
import re
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
    engine_dir = repo_root / "vendor" / "drmario_native"
    if not engine_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {engine_dir}")

    if sys.platform == "win32":
        # The POSIX Makefile emits .so and does not export the ctypes C ABI.
        build_dir = engine_dir / "build"
        build_dir.mkdir(exist_ok=True)
        reach_obj = build_dir / "drm_reach_full.obj"
        subprocess.run(["clang", "-O3", "-std=c11", "-DNDEBUG", "-c",
                        str(engine_dir / "third_party/reach_native/drm_reach_full.c"),
                        "-o", str(reach_obj)], check=True)
        exports = sorted(set(re.findall(r"\b(drm_(?:pool|vspool)_\w+)\s*\(",
                                       (engine_dir / "drmario_pool_capi.h").read_text())))
        definition = build_dir / "drmario_pool.def"
        definition.write_text("EXPORTS\n" + "\n".join(exports) + "\n")
        cmd = ["clang++", "-O3", "-std=c++20", "-shared",
               *(str(engine_dir / name) for name in
                 ("DrMarioPool.cpp", "DrMarioVsPool.cpp", "drmario_pool_capi.cpp", "GameLogic.cpp")),
               str(reach_obj), f"-Wl,/DEF:{definition}", "-o", str(build_dir / _library_name())]
    else:
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
