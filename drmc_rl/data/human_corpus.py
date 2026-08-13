"""Validated access to immutable fightcadeRatings human-corpus releases.

The source database never crosses this boundary.  A corpus root is either a
local directory or a read-only SSHFS mount containing ``latest/manifest.json``
and content-hashed Parquet shards.  Network/mount lifecycle lives in
``tools.human_corpus``; model and training code only sees this module.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

SUPPORTED_SCHEMAS = {"fcr-human-v1", "fcr-human-v2"}
DEFAULT_ROOT = Path(os.environ.get("DRMC_HUMAN_CORPUS_ROOT", "~/.cache/drmc-rl/human-corpus")).expanduser()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: str) -> Path:
    posix = PurePosixPath(value)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise ValueError(f"unsafe corpus file path: {value!r}")
    return Path(*posix.parts)


def decode_input_rle(data: bytes | memoryview, expected_frames: int | None = None) -> bytes:
    """Decode repeated little-endian ``(run_length:u16, buttons:u8)`` records."""

    raw = bytes(data)
    if len(raw) % 3:
        raise ValueError(f"invalid input RLE length {len(raw)}; expected a multiple of 3")
    out = bytearray()
    for offset in range(0, len(raw), 3):
        run, value = struct.unpack_from("<HB", raw, offset)
        if run == 0:
            raise ValueError("invalid zero-length input run")
        out.extend(bytes((value,)) * run)
    if expected_frames is not None and len(out) != int(expected_frames):
        raise ValueError(f"input trace has {len(out)} frames; expected {expected_frames}")
    return bytes(out)


@dataclass(frozen=True)
class CorpusFile:
    path: str
    kind: str
    rows: int
    bytes: int
    sha256: str


class HumanCorpus:
    """One immutable corpus release selected through a corpus root."""

    def __init__(self, root: Path | str = DEFAULT_ROOT, *, release: str = "latest"):
        self.root = Path(root).expanduser().resolve()
        self.release_dir = (self.root / release).resolve()
        manifest_path = self.release_dir / "manifest.json"
        try:
            self.manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"human corpus manifest not found at {manifest_path}; "
                "mount it with `python -m tools.human_corpus mount`"
            ) from exc
        schema = str(self.manifest.get("schema_version", ""))
        if schema not in SUPPORTED_SCHEMAS:
            raise ValueError(f"unsupported human corpus schema {schema!r}")
        self.release_id = str(self.manifest.get("release_id", ""))
        if not self.release_id:
            raise ValueError("corpus manifest has no release_id")
        files = self.manifest.get("files")
        if not isinstance(files, list):
            raise ValueError("corpus manifest files must be a list")
        self._files = tuple(
            CorpusFile(
                path=str(entry["path"]),
                kind=str(entry["kind"]),
                rows=int(entry["rows"]),
                bytes=int(entry["bytes"]),
                sha256=str(entry["sha256"]),
            )
            for entry in files
        )
        for entry in self._files:
            _safe_relative(entry.path)
        self._rating_index: dict[str, tuple[list[int], list[float], list[float]]] | None = None

    @property
    def stats(self) -> Mapping[str, int]:
        return self.manifest.get("stats", {})

    def files(self, kind: str | None = None, *, months: Sequence[str] | None = None) -> list[CorpusFile]:
        selected = [entry for entry in self._files if kind is None or entry.kind == kind]
        if months:
            needles = {f"year={month[:4]}/month={month[5:7]}" for month in months}
            selected = [entry for entry in selected if any(needle in entry.path for needle in needles)]
        return selected

    def path(self, entry: CorpusFile | str) -> Path:
        value = entry.path if isinstance(entry, CorpusFile) else str(entry)
        return self.release_dir / _safe_relative(value)

    def verify(self, *, hashes: bool = False, files: Iterable[CorpusFile] | None = None) -> dict[str, int]:
        checked = total_bytes = 0
        for entry in files or self._files:
            path = self.path(entry)
            stat = path.stat()
            if stat.st_size != entry.bytes:
                raise ValueError(f"size mismatch for {entry.path}: {stat.st_size} != {entry.bytes}")
            if hashes:
                actual = _sha256(path)
                if actual != entry.sha256:
                    raise ValueError(f"sha256 mismatch for {entry.path}: {actual} != {entry.sha256}")
            checked += 1
            total_bytes += stat.st_size
        return {"files": checked, "bytes": total_bytes}

    def dataset(self, kind: str, *, months: Sequence[str] | None = None):
        try:
            import pyarrow.dataset as ds
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("human corpus scans require `uv sync --extra corpus`") from exc
        paths = [str(self.path(entry)) for entry in self.files(kind, months=months)]
        if not paths:
            raise ValueError(f"no {kind!r} shards in release {self.release_id}")
        return ds.dataset(paths, format="parquet", partitioning="hive")

    def batches(
        self,
        kind: str = "decisions",
        *,
        columns: Sequence[str] | None = None,
        filter=None,
        months: Sequence[str] | None = None,
        batch_size: int = 8192,
    ) -> Iterator[Any]:
        scanner = self.dataset(kind, months=months).scanner(
            columns=None if columns is None else list(columns),
            filter=filter,
            batch_size=int(batch_size),
            use_threads=True,
        )
        yield from scanner.to_batches()

    def time_split(self, day: int) -> str:
        """Return the release-relative split without storing it in behavior shards."""

        max_day = int(self.manifest["source"]["max_day"])
        if int(day) > max_day - 180:
            return "test"
        if int(day) > max_day - 270:
            return "validation"
        return "train"

    def _load_ratings(self):
        if self._rating_index is not None:
            return
        index: dict[str, tuple[list[int], list[float], list[float]]] = {}
        for batch in self.batches(
            "ratings", columns=["player", "day", "skill_elo", "skill_sd"]
        ):
            for row in batch.to_pylist():
                days, means, sds = index.setdefault(row["player"], ([], [], []))
                days.append(int(row["day"]))
                means.append(float(row["skill_elo"]))
                sds.append(float(row["skill_sd"]))
        self._rating_index = index

    def rating_at(self, player: str, day: int) -> tuple[float | None, float | None]:
        """Linearly interpolate the release's continuous WHR-C trajectory."""

        self._load_ratings()
        row = self._rating_index.get(player) if self._rating_index else None
        if row is None:
            return None, None
        days, means, sds = row
        i = bisect.bisect_left(days, int(day))
        if i <= 0:
            return means[0], sds[0]
        if i >= len(days):
            return means[-1], sds[-1]
        lo, hi = days[i - 1], days[i]
        if hi == lo:
            return means[i], sds[i]
        weight = (int(day) - lo) / (hi - lo)
        return (
            means[i - 1] + weight * (means[i] - means[i - 1]),
            sds[i - 1] + weight * (sds[i] - sds[i - 1]),
        )
