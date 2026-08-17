"""Deterministic, resumable counterfactual pilot release construction."""

from __future__ import annotations

import gzip
import hashlib
import heapq
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from drmc_rl.game.pair_state import PAIR_STATE_SCHEMA
from drmc_rl.teachers.counterfactual import CounterfactualTeacher

RELEASE_SCHEMA = "drmc-counterfactual-release-v1"


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(payload: Mapping[str, Any]) -> str:
    explicit = payload.get("id")
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _open_input(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _stable_score(seed: int, identity: str) -> int:
    value = f"{int(seed)}\0{identity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest(), "big")


def _nested_field(payload: Mapping[str, Any], field: str) -> object:
    value: object = payload
    for component in field.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"missing required stratum field {field!r}")
        value = value[component]
    if isinstance(value, (Mapping, list, tuple)):
        return hashlib.sha256(canonical_json(value)).hexdigest()[:16]
    return value


@dataclass(frozen=True, slots=True)
class SelectedState:
    identity: str
    source_line: int
    stratum: tuple[str, ...]
    payload: dict[str, Any]


def select_states(
    path: Path,
    *,
    seed: int,
    shard_index: int,
    num_shards: int,
    stratum_fields: Sequence[str],
    per_stratum: int | None,
    max_states: int | None,
) -> list[SelectedState]:
    """Select a stable hash sample without depending on input ordering."""

    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0,num_shards)")
    if per_stratum is not None and (per_stratum < 1 or not stratum_fields):
        raise ValueError("per_stratum requires stratum fields and must be positive")
    if max_states is not None and max_states < 1:
        raise ValueError("max_states must be positive")

    seen: set[str] = set()
    heaps: dict[tuple[str, ...], list[tuple[int, str, int, dict[str, Any]]]] = {}
    with _open_input(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"source line {line_number} is not a JSON object")
            identity = source_identity(raw)
            if identity in seen:
                raise ValueError(f"duplicate source identity {identity!r}")
            seen.add(identity)
            shard = _stable_score(0, identity) % num_shards
            if shard != shard_index:
                continue
            stratum = tuple(str(_nested_field(raw, field)) for field in stratum_fields)
            score = _stable_score(seed, identity)
            heap = heaps.setdefault(stratum, [])
            limit = per_stratum or max_states
            entry = (-score, identity, line_number, raw)
            if limit is None or len(heap) < limit:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)

    chosen = [
        SelectedState(identity, line_number, stratum, payload)
        for stratum, heap in heaps.items()
        for _negative_score, identity, line_number, payload in heap
    ]
    chosen.sort(key=lambda item: (item.stratum, _stable_score(seed, item.identity), item.identity))
    if max_states is not None and len(chosen) > max_states:
        chosen = sorted(
            chosen, key=lambda item: (_stable_score(seed, item.identity), item.identity)
        )[:max_states]
        chosen.sort(key=lambda item: (item.stratum, item.identity))
    return chosen


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _gzip_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as handle:
        with gzip.GzipFile(fileobj=handle, mode="wb", filename="", mtime=0) as compressed:
            for row in rows:
                compressed.write(canonical_json(row) + b"\n")
        handle.seek(0)
        return handle.read()


def _git_revision(root: Path) -> tuple[str | None, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return commit, dirty
    except FileNotFoundError, subprocess.CalledProcessError:
        return None, False


@dataclass(frozen=True, slots=True)
class ReleaseSettings:
    input_sha256: str
    adapter: str
    root_side: int
    search: dict[str, object]
    seed: int
    shard_index: int
    num_shards: int
    stratum_fields: tuple[str, ...]
    per_stratum: int | None
    max_states: int | None
    chunk_size: int
    corpus_release: str
    continuation_mixture: str
    native_revision: str
    planner_revision: str
    mixture_manifest_sha256: str | None = None
    wdl_calibration_sha256: str | None = None

    def digest(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self))).hexdigest()


def build_release(
    *,
    input_path: Path,
    output_dir: Path,
    adapter_spec: str,
    teacher: CounterfactualTeacher[Any],
    decode: Callable[[dict[str, Any]], Any],
    settings: ReleaseSettings,
    resume: bool,
    reject_budget_exhausted: bool = True,
) -> Path:
    selected = select_states(
        input_path,
        seed=settings.seed,
        shard_index=settings.shard_index,
        num_shards=settings.num_shards,
        stratum_fields=settings.stratum_fields,
        per_stratum=settings.per_stratum,
        max_states=settings.max_states,
    )
    if not selected:
        raise ValueError("pilot selection is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_hash = settings.digest()
    parts: list[dict[str, object]] = []
    total_candidates = 0
    exhausted = 0

    for part_index, start in enumerate(range(0, len(selected), settings.chunk_size)):
        states = selected[start : start + settings.chunk_size]
        marker = output_dir / f"part-{part_index:05d}.complete.json"
        if resume and marker.is_file():
            previous = json.loads(marker.read_text())
            part_path = output_dir / str(previous["file"])
            if (
                previous.get("settings_sha256") == settings_hash
                and part_path.is_file()
                and sha256_file(part_path) == previous.get("sha256")
            ):
                parts.append(previous)
                total_candidates += int(previous["candidates"])
                exhausted += int(previous.get("budget_exhausted", 0))
                continue

        rows: list[dict[str, Any]] = []
        part_candidates = 0
        part_exhausted = 0
        for item in states:
            state = decode(item.payload)
            root_side = int(item.payload.get("root_side", settings.root_side))
            label = teacher.label(
                state,
                root_side=root_side,
                metadata={
                    "source_id": item.identity,
                    "source_line": item.source_line,
                    "stratum": list(item.stratum),
                    "pair_state_schema": PAIR_STATE_SCHEMA,
                    "corpus_release": settings.corpus_release,
                    "continuation_mixture": settings.continuation_mixture,
                },
            )
            if label.budget_exhausted:
                part_exhausted += 1
                if reject_budget_exhausted:
                    raise RuntimeError(
                        f"search budget exhausted for source {item.identity}; no release written"
                    )
            row = label.to_dict()
            rows.append(row)
            part_candidates += len(label.candidates)
        payload = _gzip_jsonl(rows)
        digest = hashlib.sha256(payload).hexdigest()
        part_name = f"part-{part_index:05d}-{digest[:16]}.jsonl.gz"
        part_path = output_dir / part_name
        _atomic_write(part_path, payload)
        record: dict[str, object] = {
            "index": part_index,
            "file": part_name,
            "sha256": digest,
            "bytes": len(payload),
            "rows": len(rows),
            "candidates": part_candidates,
            "budget_exhausted": part_exhausted,
            "settings_sha256": settings_hash,
        }
        _atomic_write(marker, json.dumps(record, indent=2, sort_keys=True).encode() + b"\n")
        parts.append(record)
        total_candidates += part_candidates
        exhausted += part_exhausted

    repo_root = Path(__file__).resolve().parents[2]
    commit, dirty = _git_revision(repo_root)
    release_digest = hashlib.sha256(canonical_json([part["sha256"] for part in parts])).hexdigest()
    manifest = {
        "schema": RELEASE_SCHEMA,
        "release_sha256": release_digest,
        "settings_sha256": settings_hash,
        "settings": asdict(settings),
        "adapter": adapter_spec,
        "pair_state_schema": PAIR_STATE_SCHEMA,
        "repository_commit": commit,
        "repository_dirty": dirty,
        "selected_states": len(selected),
        "candidate_labels": total_candidates,
        "budget_exhausted": exhausted,
        "parts": parts,
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_write(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    return manifest_path


__all__ = [
    "RELEASE_SCHEMA",
    "ReleaseSettings",
    "SelectedState",
    "build_release",
    "canonical_json",
    "select_states",
    "sha256_file",
    "source_identity",
]
