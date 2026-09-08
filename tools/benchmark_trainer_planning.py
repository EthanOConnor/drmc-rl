"""Measure the installed live policy and nine conditional previews on public roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from drmc_rl.execution.pace import BY_ID
from drmc_rl.game.observation import board_bytes_to_semantic_planes
from drmc_rl.human.anticipation import score_preview_branches, NextTurnPreparer
from drmc_rl.human.backend import HumanBackend, PROTOCOL_SCHEMA


def root_request(row):
    root, spawn = row["public_root"], row["public_root"]["spawn"]
    return {"schema": PROTOCOL_SCHEMA, "type": "decide", "strength_control": "quality",
        "target_rating": 2400, "temperature": 0, "pace": "frame_perfect", "deadline_ms": 10000,
        "execution_delay_frames": 0, "request_id": 1, "frame_id": 1,
        "state": {"board_planes": board_bytes_to_semantic_planes(root["board"]).tolist(),
            "opponent_board_planes": board_bytes_to_semantic_planes(root["opponent"]).tolist(),
            "pill": root["pill"], "preview": root["preview"], "opponent_pill": root["opponent_pill"],
            "speed": row["speed"], "speed_ups": row["speed_ups"], "level": row["level"],
            "pill_counter_total": root.get("pill_counter_total", 1),
            "falling": {"x": spawn["x"], "y": spawn["y"], "rotation": spawn["rot"],
                "speed_counter": spawn["speed_counter"], "horizontal_velocity": spawn["hor_velocity"],
                "hold_dir": spawn["hold_dir"], "rotation_hold": spawn["rot_hold"], "frame_parity": spawn["frame_parity"]}}}


def quantiles(values):
    return {name: float(np.quantile(values, q)) for name, q in (("p50_ms", .5), ("p95_ms", .95), ("p99_ms", .99), ("max_ms", 1))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--human-checkpoint", type=Path, required=True)
    ap.add_argument("--roots", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--states", type=int, default=46)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows = json.loads(args.roots.read_text())["rows"][:args.states]
    backend = HumanBackend(str(args.human_checkpoint), device=args.device,
        competitive_checkpoint=str(args.checkpoint))
    preparer = NextTurnPreparer(backend.competitive, backend.planner)
    stage_times, records = {}, []
    originals = []
    def timed(owner, name, label):
        original = getattr(owner, name)
        originals.append((owner, name, original))
        def wrapped(*a, **kw):
            start = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                stage_times[label] = stage_times.get(label, 0) + 1000*(time.perf_counter()-start)
        setattr(owner, name, wrapped)
    timed(backend, "_candidates", "candidates_ms")
    timed(backend.competitive, "score", "policy_ms")
    timed(backend.runtime, "timing_prediction", "cadence_ms")
    try:
        # Warm the actual B=9/B=3 shapes before reporting steady-state costs.
        for row in rows:
            request = root_request(row)
            own, other, pill, _, _, _, _, _, _, costs = backend._candidates(request["state"], 0, BY_ID["frame_perfect"])
            for batch in (3, 9):
                score_preview_branches(backend.competitive, own, other, pill,
                    request["state"]["opponent_pill"], costs, batch_size=batch)
        print(json.dumps({"warmup_complete": len(rows), "device": args.device}), flush=True)
        for row in rows:
            request = root_request(row)
            result = backend._infer(request, remaining_ms=10000)
            preparer.prepare(request["state"], result, BY_ID["frame_perfect"])
        for index, row in enumerate(rows):
            request = root_request(row)
            stage_times.clear()
            start = time.perf_counter()
            response = backend._infer(request, remaining_ms=10000)
            record = {"root": row["root_sha256"], "level": row["level"],
                "total_ms": 1000*(time.perf_counter()-start), **stage_times,
                "candidates": response["candidate_count"], "action": response["placement"]["action"]}
            own, other, pill, _, _, _, _, _, _, costs = backend._candidates(request["state"], 0, BY_ID["frame_perfect"])
            choices = []
            for batch in (1, 3, 9):
                start = time.perf_counter()
                scores = score_preview_branches(backend.competitive, own, other, pill,
                    request["state"]["opponent_pill"], costs, batch_size=batch)
                record[f"preview_batch{batch}_ms"] = 1000*(time.perf_counter()-start)
                choices.append(scores.argmax(axis=1).tolist())
            record["preview_batch_action_agreement"] = choices[0] == choices[1] == choices[2]
            start = time.perf_counter()
            prepared = preparer.prepare(request["state"], response, BY_ID["frame_perfect"])
            if prepared is not None:
                record["full_preparation_ms"] = 1000*(time.perf_counter()-start)
            records.append(record)
            if index % 8 == 0:
                print(json.dumps({"completed": index+1, **record}), flush=True)
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)
        backend.close()
        preparer.close()
    labels = [key for key in records[0] if key.endswith("_ms")]
    report = {"device": args.device, "torch": torch.__version__, "states": len(records),
        "scope": "Warm in-process Mac timings; transport and emulation contention measured separately.",
        "summary": {label: quantiles([row[label] for row in records if label in row]) for label in labels},
        "all_preview_batches_agree": all(row["preview_batch_action_agreement"] for row in records), "rows": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
