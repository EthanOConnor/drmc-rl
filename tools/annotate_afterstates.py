"""Materialize exact post-placement states for the sampled human corpus.

The output is ragged: one 128-byte native bottle per legal candidate, plus a
small target file.  Candidate offsets preserve the source shard's packed slot
order.  This avoids the roughly 5x storage and compute waste of padding every
decision to 128 candidates.

Example on tf3090::

    .venv/bin/python -m tools.annotate_afterstates \
        --dataset data/human_vs/human_policy_v2 \
        --output data/human_vs/human_afterstates_v3 --num-envs 4096
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from drmc_rl.human.afterstate_model import HUMAN_AFTERSTATE_SCHEMA
from drmc_rl.human.afterstate_sim import NativeAfterstateSimulator, encode_sparse_deltas


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "data" / "human_vs" / "human_policy_v2"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "human_vs" / "human_afterstates_v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _collect_reports(output: Path, source: Path) -> dict:
    """Merge legacy manifest rows with authoritative per-shard reports."""

    manifest_path = output / "manifest.json"
    by_source = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        by_source.update({row["source"]: row for row in manifest.get("shards", ())})
    for path in sorted(output.glob("*.report.json")):
        report = json.loads(path.read_text())
        by_source[report["source"]] = report
    return {
        "schema": HUMAN_AFTERSTATE_SCHEMA,
        "source": str(source),
        "shards": [by_source[key] for key in sorted(by_source)],
    }


def annotate_shard(
    source: Path,
    output: Path,
    simulator: NativeAfterstateSimulator,
    *,
    row_batch: int,
    max_rows: int | None = None,
) -> dict[str, int | float | str]:
    with np.load(source, allow_pickle=False) as data:
        keys = (
            "field",
            "pill",
            "preview",
            "candidate_actions",
            "candidate_costs",
            "candidate_count",
            "speed",
            "speed_ups",
        )
        # NpzFile.__getitem__ inflates a compressed member. Keep each source
        # array once instead of decompressing it again for every row batch.
        arrays = {key: data[key] for key in keys}
        source_rows = len(arrays["candidate_count"])
        rows = source_rows
        if max_rows is not None:
            rows = min(rows, int(max_rows))
        counts = arrays["candidate_count"][:rows].astype(np.int64)
        offsets = np.empty(rows + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        candidates = int(offsets[-1])

        stem = source.stem
        delta_offsets_path = output / f"{stem}.delta_offsets.npy"
        delta_cells_path = output / f"{stem}.delta_cells.bin"
        delta_values_path = output / f"{stem}.delta_values.bin"
        targets_path = output / f"{stem}.targets.npz"
        temp_delta_offsets = output / f".{stem}.delta_offsets.tmp.npy"
        temp_delta_cells = output / f".{stem}.delta_cells.tmp.bin"
        temp_delta_values = output / f".{stem}.delta_values.tmp.bin"
        temp_targets = output / f".{stem}.targets.tmp.npz"
        delta_offsets = np.empty(candidates + 1, dtype=np.uint32)
        delta_offsets[0] = 0
        terminal = np.empty(candidates, dtype=np.uint8)
        invalid = np.empty(candidates, dtype=np.bool_)
        tau = np.empty(candidates, dtype=np.uint32)
        remaining = np.empty(candidates, dtype=np.uint16)
        viruses = np.empty(candidates, dtype=np.uint16)
        nonviruses = np.empty(candidates, dtype=np.uint16)
        events = np.empty(candidates, dtype=np.uint16)

        started = time.perf_counter()
        delta_cursor = 0
        with (
            temp_delta_cells.open("wb") as cell_stream,
            temp_delta_values.open("wb") as value_stream,
        ):
            for row_start in range(0, rows, int(row_batch)):
                row_stop = min(row_start + int(row_batch), rows)
                out_start, out_stop = int(offsets[row_start]), int(offsets[row_stop])
                batch_counts = counts[row_start:row_stop]
                result = simulator.simulate_packed(
                    fields=arrays["field"][row_start:row_stop],
                    pills=arrays["pill"][row_start:row_stop],
                    previews=arrays["preview"][row_start:row_stop],
                    candidate_actions=arrays["candidate_actions"][row_start:row_stop],
                    candidate_costs=arrays["candidate_costs"][row_start:row_stop],
                    candidate_count=batch_counts,
                    speed=arrays["speed"][row_start:row_stop],
                    speed_ups=arrays["speed_ups"][row_start:row_stop],
                )
                local_offsets, cells, values = encode_sparse_deltas(
                    arrays["field"][row_start:row_stop], batch_counts, result.fields
                )
                cell_stream.write(cells.tobytes())
                value_stream.write(values.tobytes())
                delta_offsets[out_start + 1 : out_stop + 1] = delta_cursor + local_offsets[1:]
                delta_cursor += len(cells)
                terminal[out_start:out_stop] = result.terminal_reason
                invalid[out_start:out_stop] = result.invalid
                tau[out_start:out_stop] = result.tau_frames
                remaining[out_start:out_stop] = result.viruses_remaining
                viruses[out_start:out_stop] = result.viruses_cleared
                nonviruses[out_start:out_stop] = result.nonviruses_cleared
                events[out_start:out_stop] = result.clear_events
                done = out_stop
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"{source.name}: rows={row_stop:,}/{rows:,} "
                    f"candidates={done:,}/{candidates:,} rate={done / elapsed:,.0f}/s",
                    flush=True,
                )
        np.save(temp_delta_offsets, delta_offsets)
        np.savez(
            temp_targets,
            candidate_offsets=offsets,
            terminal_reason=terminal,
            invalid=invalid,
            tau_frames=tau,
            viruses_remaining=remaining,
            viruses_cleared=viruses,
            nonviruses_cleared=nonviruses,
            clear_events=events,
        )
        os.replace(temp_delta_offsets, delta_offsets_path)
        os.replace(temp_delta_cells, delta_cells_path)
        os.replace(temp_delta_values, delta_values_path)
        os.replace(temp_targets, targets_path)
        elapsed = time.perf_counter() - started
        return {
            "source": source.name,
            "source_sha256": _sha256(source),
            "rows": rows,
            "source_rows": source_rows,
            "complete": rows == source_rows,
            "candidates": candidates,
            "changed_cells": int(delta_cursor),
            "invalid": int(invalid.sum()),
            "seconds": elapsed,
            "candidates_per_second": candidates / max(elapsed, 1e-9),
            "delta_offsets": delta_offsets_path.name,
            "delta_cells": delta_cells_path.name,
            "delta_values": delta_values_path.name,
            "targets": targets_path.name,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--row-batch", type=int, default=4096)
    parser.add_argument("--max-rows-per-shard", type=int)
    parser.add_argument("--shard", action="append", help="source shard stem; repeatable")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="atomically rebuild the aggregate manifest from per-shard reports and exit",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if args.manifest_only:
        manifest = _collect_reports(args.output, args.dataset)
        _write_json_atomic(manifest_path, manifest)
        print(json.dumps(manifest, indent=2), flush=True)
        return
    selected = set(args.shard or ())
    sources = [
        path for path in sorted(args.dataset.glob("*.npz")) if not selected or path.stem in selected
    ]
    if not sources:
        raise SystemExit(f"no source shards found in {args.dataset}")
    manifest = _collect_reports(args.output, args.dataset)
    by_source = {row["source"]: row for row in manifest["shards"]}
    with NativeAfterstateSimulator(num_envs=args.num_envs) as simulator:
        for source in sources:
            existing = by_source.get(source.name)
            if (
                existing
                and existing.get("complete")
                and existing.get("source_sha256") == _sha256(source)
                and not args.overwrite
            ):
                print(f"skip complete {source.name}", flush=True)
                continue
            report = annotate_shard(
                source,
                args.output,
                simulator,
                row_batch=args.row_batch,
                max_rows=args.max_rows_per_shard,
            )
            report_path = args.output / f"{source.stem}.report.json"
            _write_json_atomic(report_path, report)
            # Aggregate metadata is a convenience view. Per-shard reports are
            # the concurrency-safe completion authority for distributed workers.
            manifest = _collect_reports(args.output, args.dataset)
            _write_json_atomic(manifest_path, manifest)
            by_source = {row["source"]: row for row in manifest["shards"]}
            print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
