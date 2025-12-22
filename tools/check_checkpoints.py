#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import training.utils.checkpoint_io as checkpoint_io


def _iter_checkpoints(path: Path) -> List[Path]:
    if path.is_file():
        return [path] if checkpoint_io.is_checkpoint_path(path) else []
    if path.is_dir():
        return sorted(
            {
                p
                for p in path.rglob("*")
                if p.is_file() and checkpoint_io.is_checkpoint_path(p)
            }
        )
    return []


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan checkpoints for corruption.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["runs"],
        help="Files or directories to scan (default: runs)",
    )
    parser.add_argument(
        "--delete-corrupt",
        action="store_true",
        help="Delete corrupt checkpoint files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file status output.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if checkpoint_io.torch is None:
        print("PyTorch is required to validate checkpoints.", file=sys.stderr)
        return 2

    total = 0
    ok = 0
    corrupt = 0
    missing = 0

    for raw in args.paths:
        path = Path(raw).expanduser()
        if not path.exists():
            missing += 1
            if not args.quiet:
                print(f"missing: {path}")
            continue

        checkpoints = _iter_checkpoints(path)
        if not checkpoints:
            if not args.quiet:
                label = "no checkpoints under" if path.is_dir() else "skip (not a checkpoint)"
                print(f"{label}: {path}")
            continue

        for ckpt in checkpoints:
            total += 1
            try:
                checkpoint_io.load_checkpoint(ckpt, map_location="cpu")
            except Exception as exc:
                corrupt += 1
                if not args.quiet:
                    print(f"corrupt: {ckpt} ({exc})")
                if args.delete_corrupt:
                    try:
                        ckpt.unlink()
                        if not args.quiet:
                            print(f"deleted: {ckpt}")
                    except Exception as del_exc:
                        print(f"failed to delete: {ckpt} ({del_exc})", file=sys.stderr)
            else:
                ok += 1
                if not args.quiet:
                    print(f"ok: {ckpt}")

    if not args.quiet:
        print(f"checked={total} ok={ok} corrupt={corrupt} missing={missing}")
    if corrupt > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
