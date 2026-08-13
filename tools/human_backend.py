"""JSONL process boundary for the human player and coach backend.

Professor Pills should supervise this process off its emulation/render/audio
threads, retain only the latest frame's result, and treat backend death as a
recoverable loss of AI/coach service rather than a gameplay failure.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from drmc_rl.human.backend import HumanBackend, PROTOCOL_SCHEMA


def serve(backend: HumanBackend) -> None:
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                response = {
                    "schema": PROTOCOL_SCHEMA,
                    "type": "error",
                    "request_id": -1,
                    "frame_id": -1,
                    "error": {"kind": type(exc).__name__, "message": str(exc)},
                }
            else:
                if request.get("type") == "shutdown":
                    response = {
                        "schema": PROTOCOL_SCHEMA,
                        "type": "shutdown",
                        "request_id": int(request.get("request_id", -1)),
                        "frame_id": int(request.get("frame_id", -1)),
                    }
                    print(json.dumps(response, separators=(",", ":")), flush=True)
                    return
                response = backend.handle(request)
            print(json.dumps(response, separators=(",", ":")), flush=True)
    finally:
        backend.close()


def benchmark(backend: HumanBackend, iterations: int) -> None:
    planes = np.zeros((8, 16, 8), dtype=np.float32).tolist()
    for request_id in range(1, int(iterations) + 2):
        response = backend.handle(
            {
                "schema": PROTOCOL_SCHEMA,
                "type": "decide",
                "request_id": request_id,
                "frame_id": request_id,
                "deadline_ms": 10_000,
                "target_rating": backend.runtime.condition.mean,
                "temperature": 0,
                "state": {
                    "board_planes": planes,
                    "opponent_board_planes": planes,
                    "opponent_state_age_frames": 0,
                    "pill": [0, 1],
                    "preview": [2, 0],
                    "speed": 2,
                    "speed_ups": 0,
                    "falling": {"x": 3, "y": 0, "rotation": 0, "frame_parity": 0},
                },
            }
        )
        if response["type"] != "result":
            raise RuntimeError(response)
        if request_id == 1:
            backend.latencies_ms.clear()  # exclude CUDA/model warmup
    print(json.dumps(backend.health(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bench", type=int, default=0, help="benchmark N warmed-up decisions")
    args = parser.parse_args()
    backend = HumanBackend(args.checkpoint, device=args.device, seed=args.seed)
    if args.bench:
        benchmark(backend, args.bench)
    else:
        serve(backend)


if __name__ == "__main__":
    main()
