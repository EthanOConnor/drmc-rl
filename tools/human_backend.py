"""JSONL process boundary for the human player and coach backend.

Professor Pills should supervise this process off its emulation/render/audio
threads, retain only the latest frame's result, and treat backend death as a
recoverable loss of AI/coach service rather than a gameplay failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from drmc_rl.human.backend import HumanBackend, PROTOCOL_SCHEMA

DEFAULT_CHECKPOINTS = ("human_policy.pt.gz", "human_policy_v2.pt.gz")


def resolve_checkpoint(path: str | None) -> Path:
    """Find the requested model or the model shipped with a frozen backend."""

    if path:
        candidates = [Path(path).expanduser()]
    elif model_path := os.environ.get("DRMC_HUMAN_MODEL"):
        candidates = [Path(model_path).expanduser()]
    else:
        roots: list[Path] = []
        if frozen_root := getattr(sys, "_MEIPASS", None):
            roots.append(Path(frozen_root))
        roots.append(Path(sys.executable).resolve().parent)
        roots.append(Path(__file__).resolve().parents[1])
        candidates = [root / "models" / name for root in roots for name in DEFAULT_CHECKPOINTS]
        candidates.extend(
            roots[-1] / "runs" / "human_policy" / name for name in DEFAULT_CHECKPOINTS
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Human policy checkpoint not found; searched: {searched}")


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
    parser.add_argument(
        "--checkpoint",
        help="model path; defaults to DRMC_HUMAN_MODEL or the packaged model",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--realtime-profile",
        choices=("auto", "fast", "balanced", "deep"),
        default="auto",
    )
    parser.add_argument("--bench", type=int, default=0, help="benchmark N warmed-up decisions")
    args = parser.parse_args()
    backend = HumanBackend(
        resolve_checkpoint(args.checkpoint),
        device=args.device,
        seed=args.seed,
        realtime_profile=args.realtime_profile,
    )
    if args.bench:
        benchmark(backend, args.bench)
    else:
        serve(backend)


if __name__ == "__main__":
    main()
