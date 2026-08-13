"""Mount, inspect, verify, sample, and selectively cache the human corpus.

The default transport is an encrypted, key-authenticated, read-only SSHFS
mount from mombox.  It uses no production database connection and consumes no
tf3090 disk beyond filesystem metadata unless ``cache`` is explicitly used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from drmc_rl.data.human_corpus import DEFAULT_ROOT, HumanCorpus

DEFAULT_HOST = os.environ.get("DRMC_HUMAN_CORPUS_HOST", "mombox")
DEFAULT_REMOTE = os.environ.get(
    "DRMC_HUMAN_CORPUS_REMOTE", "/home/ethan/fightcadeRatings/data/corpus"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _mounted(path: Path) -> bool:
    try:
        return path.is_mount()
    except OSError:
        return False


def cmd_mount(args):
    mountpoint = Path(args.root).expanduser().resolve()
    mountpoint.mkdir(parents=True, exist_ok=True)
    if _mounted(mountpoint):
        print(f"already mounted: {mountpoint}")
        return
    command = [
        "sshfs",
        f"{args.host}:{args.remote}",
        str(mountpoint),
        "-o",
        "ro,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,auto_cache,kernel_cache",
    ]
    subprocess.run(command, check=True)
    corpus = HumanCorpus(mountpoint)
    print(f"mounted {corpus.release_id} read-only at {mountpoint}")


def cmd_unmount(args):
    mountpoint = Path(args.root).expanduser().resolve()
    if not _mounted(mountpoint):
        print(f"not mounted: {mountpoint}")
        return
    command = ["fusermount3", "-u", str(mountpoint)] if shutil.which("fusermount3") else ["umount", str(mountpoint)]
    subprocess.run(command, check=True)
    print(f"unmounted {mountpoint}")


def cmd_status(args):
    corpus = HumanCorpus(args.root, release=args.release)
    payload = {
        "root": str(corpus.root),
        "release_id": corpus.release_id,
        "schema_version": corpus.manifest["schema_version"],
        "created_at": corpus.manifest.get("created_at"),
        "stats": corpus.stats,
        "files": len(corpus.files()),
        "bytes": sum(entry.bytes for entry in corpus.files()),
        "mounted": _mounted(corpus.root),
    }
    print(json.dumps(payload, indent=2))


def cmd_verify(args):
    corpus = HumanCorpus(args.root, release=args.release)
    selected = corpus.files(args.kind, months=args.month)
    result = corpus.verify(hashes=args.hashes, files=selected)
    result.update({"release_id": corpus.release_id, "hashes": bool(args.hashes)})
    print(json.dumps(result, indent=2))


def cmd_sample(args):
    corpus = HumanCorpus(args.root, release=args.release)
    remaining = int(args.rows)
    for batch in corpus.batches(args.kind, months=args.month, batch_size=min(remaining, 1024)):
        for row in batch.to_pylist()[:remaining]:
            for key, value in list(row.items()):
                if isinstance(value, bytes):
                    row[key] = f"<{len(value)} bytes>"
            print(json.dumps(row, sort_keys=True))
        remaining -= min(remaining, batch.num_rows)
        if remaining <= 0:
            break


def cmd_cache(args):
    corpus = HumanCorpus(args.root, release=args.release)
    destination = Path(args.destination).expanduser().resolve() / corpus.release_id
    selected = corpus.files(args.kind, months=args.month)
    if not selected:
        raise SystemExit("no matching shards")
    for entry in selected:
        target = destination / entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size != entry.bytes:
            shutil.copy2(corpus.path(entry), target)
        digest = _sha256(target)
        if digest != entry.sha256:
            raise ValueError(f"cached shard hash mismatch: {target}")
        print(target)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--release", default="latest")
    sub = ap.add_subparsers(dest="command", required=True)
    mount = sub.add_parser("mount")
    mount.add_argument("--host", default=DEFAULT_HOST)
    mount.add_argument("--remote", default=DEFAULT_REMOTE)
    mount.set_defaults(fn=cmd_mount)
    unmount = sub.add_parser("unmount")
    unmount.set_defaults(fn=cmd_unmount)
    status = sub.add_parser("status")
    status.set_defaults(fn=cmd_status)
    verify = sub.add_parser("verify")
    verify.add_argument("--kind", choices=("games", "decisions", "ratings"))
    verify.add_argument("--month", action="append")
    verify.add_argument("--hashes", action="store_true")
    verify.set_defaults(fn=cmd_verify)
    sample = sub.add_parser("sample")
    sample.add_argument("--kind", default="decisions", choices=("games", "decisions", "ratings"))
    sample.add_argument("--month", action="append")
    sample.add_argument("--rows", type=int, default=3)
    sample.set_defaults(fn=cmd_sample)
    cache = sub.add_parser("cache")
    cache.add_argument("--kind", choices=("games", "decisions", "ratings"))
    cache.add_argument("--month", action="append")
    cache.add_argument("--destination", required=True)
    cache.set_defaults(fn=cmd_cache)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
