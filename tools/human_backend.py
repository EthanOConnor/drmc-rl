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


def resolve_device(requested: str) -> str:
    """Use available acceleration for live deadlines; honor explicit devices."""
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def resolve_competitive_checkpoint(path: str | None, checkpoint: Path) -> Path | None:
    """Use an explicit ceiling or the companion model in a trainer package."""
    candidate = Path(path).expanduser() if path else checkpoint.parent / "competitive_policy.pt.gz"
    if candidate.is_file():
        return candidate.resolve()
    if path:
        raise FileNotFoundError(candidate)
    return None


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


def benchmark(backend: HumanBackend, iterations: int, *, strength_control: str = "regret") -> None:
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
                "strength_control": strength_control,
                "state": {
                    "board_planes": planes,
                    "opponent_board_planes": planes,
                    "opponent_state_age_frames": 0,
                    "pill": [0, 1],
                    "opponent_pill": [0, 1],
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
    parser.add_argument("--device", default="auto", help="auto selects CUDA, Metal, then CPU")
    parser.add_argument("--threads", type=int, default=1, help="inference CPU threads; default 1 keeps gameplay responsive")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--competitive-checkpoint", help="optional public V5 ceiling for quality mode")
    parser.add_argument("--bench-strength", choices=("regret", "quality"), default="regret")
    parser.add_argument(
        "--realtime-profile",
        choices=("auto", "fast", "balanced", "deep"),
        default="auto",
    )
    parser.add_argument("--bench", type=int, default=0, help="benchmark N warmed-up decisions")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    import torch
    torch.set_num_threads(args.threads)
    checkpoint = resolve_checkpoint(args.checkpoint)
    backend = HumanBackend(
        checkpoint,
        device=resolve_device(args.device),
        seed=args.seed,
        realtime_profile=args.realtime_profile,
        competitive_checkpoint=resolve_competitive_checkpoint(args.competitive_checkpoint, checkpoint),
    )
    if args.bench:
        benchmark(backend, args.bench, strength_control=args.bench_strength)
    else:
        serve(backend)


if __name__ == "__main__":
    main()
