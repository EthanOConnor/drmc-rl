#!/usr/bin/env python3
"""Download and install the latest QuickNES libretro core.

This repo keeps libretro cores in `cores/` (gitignored). The upstream QuickNES
nightly builds are published as a single-file zip containing the core dynamic
library.

Default target (macOS arm64):
  https://buildbot.libretro.com/nightly/apple/osx/arm64/latest/quicknes_libretro.dylib.zip
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional


DEFAULT_URL = (
    "https://buildbot.libretro.com/nightly/apple/osx/arm64/latest/quicknes_libretro.dylib.zip"
)
DEFAULT_DEST = Path("cores/quicknes_libretro.dylib")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolve_dest(dest_arg: str | None) -> Path:
    root = _repo_root()
    dest = Path(dest_arg) if dest_arg else DEFAULT_DEST
    dest = dest.expanduser()
    if not dest.is_absolute():
        dest = (root / dest).resolve()
    return dest


def _download(url: str, tmp_path: Path) -> None:
    with urllib.request.urlopen(url) as resp:  # nosec B310 (user-invoked download tool)
        if getattr(resp, "status", 200) >= 400:
            raise RuntimeError(f"Download failed: HTTP {resp.status}")
        with tmp_path.open("wb") as out:
            shutil.copyfileobj(resp, out)


def _extract_single_core(zip_path: Path, *, extract_dir: Path, suffix: str) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith(suffix) and not m.endswith("/")]
        if not members:
            raise RuntimeError(f"Zip did not contain a '{suffix}' file: {zip_path}")
        if len(members) > 1:
            raise RuntimeError(f"Zip contained multiple '{suffix}' files: {members}")
        member = members[0]
        zf.extract(member, path=extract_dir)
        extracted = extract_dir / member
        if not extracted.is_file():
            raise RuntimeError(f"Extracted file missing: {extracted}")
        final = extract_dir / Path(member).name
        if extracted != final:
            if final.exists():
                final.unlink()
            extracted.replace(final)
        return final


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the latest QuickNES libretro core")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Buildbot zip URL")
    parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help=f"Destination path (default: {DEFAULT_DEST})",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite destination if it exists")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    dest = _resolve_dest(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    suffix = dest.suffix  # .dylib / .so / .dll
    if dest.exists() and not args.force:
        sha = _sha256(dest)
        raise SystemExit(
            f"Refusing to overwrite existing core at {dest} (sha256={sha}). "
            "Pass --force to replace."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        zip_path = tmp_dir_path / "quicknes.zip"
        _download(str(args.url), zip_path)
        core_path = _extract_single_core(zip_path, extract_dir=tmp_dir_path, suffix=suffix)
        shutil.copy2(core_path, dest)

    print(f"Installed: {dest}")
    print(f"sha256: { _sha256(dest) }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
