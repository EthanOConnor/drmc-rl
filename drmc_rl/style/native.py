"""Optional native (C, ctypes) path for the knob features; the numpy code stays the reference.

``decision_features`` computes, for one decision, every candidate's settled
afterstate, its board + trigger features and its own immediate clear with the
same values as ``drmc_rl.style.showy_knob`` (bit-identical: the features are
integer counts, tests/test_knob_native.py checks it on recorded decisions).
It returns ``None`` when the library is absent or disabled, and callers then
use the numpy path, so a worker without the build produces the same decisions,
only slower. Pyodide never has it.

Build (once per checkout, any machine with a C compiler; the library is not
committed)::

    python -m drmc_rl.style.native build        # -> drmc_rl/style/native/libknob_features.{dylib,so}
    python -m drmc_rl.style.native status [--require]   # --require: exit 1 without it

``DRMC_KNOB_NATIVE=0`` forces the numpy path; ``DRMC_KNOB_NATIVE_LIB`` points at
a library elsewhere.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).with_name("native")
SOURCE = HERE / "knob_features.c"
ABI = 1
NBOARD = 57          # board features computed natively (log_occ is added in numpy)
NTRIG = 12
_lib = None
_tried = False


def library_path() -> Path:
    env = os.environ.get("DRMC_KNOB_NATIVE_LIB")
    if env:
        return Path(env)
    return HERE / ("libknob_features.dylib" if sys.platform == "darwin" else "libknob_features.so")


def load():
    """The loaded library, or None (absent, disabled, wrong ABI or no ctypes)."""
    global _lib, _tried
    if _tried:
        return _lib
    _tried = True
    if os.environ.get("DRMC_KNOB_NATIVE", "1") == "0":
        return None
    try:
        import ctypes
        path = library_path()
        if not path.exists():
            return None
        lib = ctypes.CDLL(str(path))
        if lib.knob_features_abi() != ABI:
            return None
        p = ctypes.c_void_p
        lib.knob_decision.argtypes = [p, ctypes.c_int, ctypes.c_int, p, ctypes.c_int, ctypes.c_int, p, p, p, p, p]
        lib.knob_decision.restype = ctypes.c_int
        _lib = lib
    except Exception:  # noqa: BLE001 - any failure means "use numpy"
        _lib = None
    return _lib


def available() -> bool:
    return load() is not None


def reset():
    """Forget the loaded library (tests toggle the environment)."""
    global _lib, _tried
    _lib, _tried = None, False


def decision_features(root, colors, actions, want_trig: bool):
    """(afterstates [n,128] uint8, board+trigger features [n,58(+12)] float32, score [n] float32,
    lines [n] int32) for one decision's candidate ``actions``, or None without the library."""
    lib = load()
    if lib is None:
        return None
    root = np.ascontiguousarray(np.asarray(root, np.uint8).reshape(128))
    acts = np.ascontiguousarray(np.asarray(actions, np.int64).reshape(-1))
    n = len(acts)
    after = np.empty((n, 128), np.uint8)
    board = np.empty((n, NBOARD), np.float32)
    trig = np.zeros((n, NTRIG), np.float32)
    score = np.empty(n, np.float32)
    lines = np.empty(n, np.int32)
    rc = lib.knob_decision(root.ctypes.data, int(colors[0]), int(colors[1]), acts.ctypes.data, n, int(bool(want_trig)),
                           after.ctypes.data, board.ctypes.data, trig.ctypes.data, score.ctypes.data, lines.ctypes.data)
    if rc == -1000000:
        return None                                   # colors outside 0..2: let the reference decide
    if rc < 0:
        raise ValueError("placement outside the empty bottle cells")
    occupied = board[:, 0].astype(np.int64)
    # log_occ exactly as board_features computes it: log1p of the int64 occupied counts, then float32
    log_occ = np.log1p(occupied).astype(np.float32)[:, None]
    parts = [board, log_occ] + ([trig] if want_trig else [])
    return after, np.concatenate(parts, axis=1), score, lines


def build(out: Path | None = None, cc: str | None = None) -> Path:
    import subprocess
    import tempfile
    out = Path(out or library_path())
    cc = cc or os.environ.get("CC", "cc")
    with tempfile.NamedTemporaryFile(dir=out.parent, suffix=out.suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run([cc, "-O2", "-std=c99", "-shared", "-fPIC", "-ffp-contract=off", "-o", str(tmp_path), str(SOURCE)],
                       check=True)
        tmp_path.replace(out)                         # atomic: concurrent workers never load a partial file
    finally:
        tmp_path.unlink(missing_ok=True)
    reset()
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="build or check the optional native knob-feature library")
    ap.add_argument("action", choices=("build", "status"))
    ap.add_argument("--cc", default=None)
    ap.add_argument("--require", action="store_true", help="exit 1 unless the library loads (deploys)")
    args = ap.parse_args(argv)
    if args.action == "build":
        print(build(cc=args.cc))
    print(f"native knob features: {'available' if available() else 'absent (numpy fallback)'} ({library_path()})")
    if args.require and not available():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
