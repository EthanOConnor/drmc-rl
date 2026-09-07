"""Replay recorded human controller windows through the exact frame stepper.

This validates a recording/planner boundary contract. It is not a ROM replay
certificate and does not fit or promote a human operation profile.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from drmc_rl.data.human_corpus import HumanCorpus, decode_input_rle
from drmc_rl.planning.fast_reach import FrameState, HoldDir, Rotation, compute_speed_threshold, simulate_frame

FIELDS = (
    "decision_id", "game_id", "field", "held_at_spawn", "horizontal_velocity",
    "speed_counter", "frame_counter", "speed", "speed_ups", "lock_x", "lock_y_top",
    "lock_rotation", "lock_repaired", "input_frames", "input_rle_u16_u8", "tau_frames",
)


def replay_row(row, *, parity_xor=0, input_delay_frames=0):
    if parity_xor not in (0, 1):
        raise ValueError("parity_xor must be 0 or 1")
    if input_delay_frames not in (0, 1):
        raise ValueError("input_delay_frames must be 0 or 1")
    if row["lock_repaired"]:
        return {"status": "excluded", "reason": "repaired_lock"}
    raw = decode_input_rle(row["input_rle_u16_u8"], row["input_frames"])
    if len(raw) != int(row["tau_frames"]) + 1 or not raw:
        raise ValueError("input_window_mismatch")
    if raw[0] != row["held_at_spawn"]:
        raise ValueError("spawn_held_mismatch")
    if any(mask & 0x30 for mask in raw):
        return {"status": "excluded", "reason": "menu_input"}
    if any(mask & 3 == 3 for mask in raw):
        return {"status": "excluded", "reason": "opposing_horizontal_buttons"}
    if any(mask & 0xC0 == 0xC0 for mask in raw):
        return {"status": "excluded", "reason": "both_rotation_buttons"}

    def direction(mask):
        return 1 if mask & 2 else 2 if mask & 1 else 0

    def rotation(mask):
        return 1 if mask & 0x80 else 2 if mask & 0x40 else 0

    board = np.frombuffer(bytes(row["field"]), dtype=np.uint8).reshape(16, 8)
    columns = np.zeros(8, dtype=np.uint16)
    for y in range(16):
        columns |= (board[y] != 0xFF).astype(np.uint16) << y
    prior = row.get("held_before_spawn") if input_delay_frames else raw[0]
    initial = int(raw[0] if prior is None else prior)
    state = FrameState(x=3, y=0, rot=0, speed_counter=int(row["speed_counter"]),
        hor_velocity=int(row["horizontal_velocity"]) & 15,
        hold_dir=HoldDir(direction(initial)), rot_hold=Rotation(rotation(initial)),
        frame_parity=(int(row["frame_counter"]) & 1) ^ parity_xor)
    threshold = compute_speed_threshold(int(row["speed"]), int(row["speed_ups"]))
    lock_at = None
    # The FBNeo recording boundary can expose the next NMI's processed input
    # before the corresponding movement update. Probe that convention
    # explicitly; it must not change the live host/planner contract.
    window = raw[:-1] if input_delay_frames else raw[1:]
    for frame, mask in enumerate(window, 1):
        action = direction(mask) * 6 + (3 if mask & 4 else 0) + rotation(mask)
        state = simulate_frame(columns, state, action, speed_threshold=threshold)
        if state.locked:
            lock_at = frame
            break
    actual = (state.x, state.y, state.rot)
    expected = (int(row["lock_x"]), int(row["lock_y_top"]), int(row["lock_rotation"]))
    pose_match = state.locked and actual == expected
    time_match = lock_at == int(row["tau_frames"])
    return {"status": "match" if pose_match and time_match else "mismatch",
            "pose_match": bool(pose_match), "time_match": time_match,
            "actual_pose": actual, "expected_pose": expected,
            "actual_tau": lock_at, "expected_tau": int(row["tau_frames"]),
            "initial_buttons": initial, "initial_horizontal_velocity": int(row["horizontal_velocity"]),
            "prior_held_verified": prior is not None,
            "all_zero_input": not any(raw),
            "speed": int(row["speed"]), "speed_ups": int(row["speed_ups"])}


def main():
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--month", default="2026-08")
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--parity-xor", type=int, choices=(0, 1), default=0,
                        help="explicitly probe the recording's frame-counter convention")
    parser.add_argument("--input-delay-frames", type=int, choices=(0, 1), default=0,
                        help="probe processed input being visible one frame before its movement update")
    parser.add_argument("--sample-modulus", type=int, default=1,
                        help="retain decision ids whose stable hash is divisible by this value")
    args = parser.parse_args()
    if args.release == "latest" or args.max_rows < 1 or args.sample_modulus < 1:
        parser.error("use an immutable release and a positive row bound")
    if args.output.exists():
        parser.error("audit output exists; use a new artifact identity")
    corpus = HumanCorpus(args.root, release=args.release)
    counts, records = Counter(), []
    fields = list(FIELDS)
    if corpus.manifest.get("contracts", {}).get("held_before_spawn") == "previous-recorded-frame-or-null-v1":
        fields.append("held_before_spawn")
    for batch in corpus.batches("decisions", columns=fields, months=[args.month], batch_size=256):
        for row in batch.to_pylist():
            key = hashlib.blake2b(row["decision_id"].encode(), digest_size=8).digest()
            if int.from_bytes(key, "little") % args.sample_modulus:
                continue
            try:
                result = replay_row(row, parity_xor=args.parity_xor,
                                    input_delay_frames=args.input_delay_frames)
            except (ValueError, TypeError) as error:
                result = {"status": "excluded", "reason": str(error)}
            counts[result["status"]] += 1
            if result["status"] == "excluded":
                counts["excluded/"+result["reason"]] += 1
            else:
                counts["pose_match"] += int(result["pose_match"])
                counts["time_match"] += int(result["time_match"])
            records.append({"decision_id": row["decision_id"], "game_id": row["game_id"], **result})
            if len(records) >= args.max_rows:
                break
        if len(records) >= args.max_rows:
            break
    report = {"schema": "drmc-execution-replay-audit-v1", "diagnostic_only": True,
        "rom_replay_verified": False, "product_gates_passed": False,
        "corpus_release": corpus.release_id,
        "corpus_manifest_sha256": hashlib.sha256((corpus.release_dir/"manifest.json").read_bytes()).hexdigest(),
        "sampling": "bounded stable-hash recording-boundary probe; not a population estimate",
        "month": args.month, "parity_xor": args.parity_xor,
        "input_delay_frames": args.input_delay_frames, "sample_modulus": args.sample_modulus,
        "source_input_coverage_verified": corpus.manifest.get("contracts", {}).get("input_coverage") == "complete-spawn-lock-window-v1",
        "rows": len(records), "games": len({r["game_id"] for r in records}),
        "counts": dict(counts), "records": records,
        "source_code_sha256": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in (
            "tools/audit_execution_replay.py", "drmc_rl/planning/fast_reach.py", "drmc_rl/data/human_corpus.py")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
