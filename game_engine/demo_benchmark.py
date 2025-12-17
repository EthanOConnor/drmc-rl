"""
Benchmark / demo playback using the original NES demo script.

This drives the C++ engine via shared memory with the recorded demo inputs from
the disassembly and reports a simple board hash plus throughput.
"""

import argparse
import hashlib
import ctypes
import os
import subprocess
import time
from pathlib import Path
from typing import Iterable, List, Tuple

from game_engine.engine_shm import SHM_SIZE, DrMarioStatePy, open_shared_memory, repo_root

# NES button bits (matches drmario constants)
BTN_RIGHT = 0x01
BTN_LEFT = 0x02
BTN_DOWN = 0x04
BTN_UP = 0x08
BTN_START = 0x10
BTN_SELECT = 0x20
BTN_B = 0x40
BTN_A = 0x80

DEMO_INPUTS_PATH = (
    repo_root() / "dr-mario-disassembly" / "data" / "drmario_data_demo_inputs.asm"
)


def parse_demo_instruction_set(path: Path = DEMO_INPUTS_PATH) -> List[Tuple[int, int]]:
    """
    Parse the first demo_instructionSet block (non-EU) as pairs of (buttons, frames).
    """
    lines = path.read_text().splitlines()
    collecting = False
    instructions: List[Tuple[int, int]] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("demo_instructionSet"):
            collecting = True
            continue
        if collecting and stripped.startswith("else"):
            break
        if not collecting:
            continue
        if ".db" not in stripped:
            continue
        data = stripped.split(".db", 1)[1]
        parts = [p.strip() for p in data.split(",") if p.strip()]
        while len(parts) >= 2:
            b0 = parts.pop(0)
            b1 = parts.pop(0)
            try:
                btn = int(b0.replace("$", ""), 16)
                delay = int(b1.replace("$", ""), 16)
            except ValueError:
                continue
            instructions.append((btn, delay))
    return instructions


def drive_demo(state: DrMarioStatePy, instructions: Iterable[Tuple[int, int]]) -> None:
    """
    Stream the demo inputs into shared memory; engine advances one frame per
    step bit (manual stepping).
    """
    state.buttons = 0
    # Start gate + manual stepping
    state.control_flags = 0x03

    # Expand instructions to per-frame buttons
    per_frame_inputs = []
    for buttons, frames in instructions:
        per_frame_inputs.extend([buttons] * frames)

    for btn in per_frame_inputs:
        state.buttons = btn
        state.control_flags |= 0x04  # request single step
        while state.control_flags & 0x04:
            pass
        if state.level_fail or state.stage_clear:
            break


def board_hash(state: DrMarioStatePy) -> str:
    h = hashlib.md5(bytes(state.board)).hexdigest()
    return h


def run_benchmark(engine_path: Path, shm_file: Path | None) -> None:
    engine_path = engine_path.resolve()
    env = os.environ.copy()
    if shm_file:
        env["DRMARIO_SHM_FILE"] = str(shm_file)
        os.environ["DRMARIO_SHM_FILE"] = str(shm_file)
    env["DRMARIO_MODE"] = "demo"

    proc = subprocess.Popen(
        [str(engine_path), "--demo", "--no-sleep", "--wait-start", "--manual-step"],
        cwd=engine_path.parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    time.sleep(0.05)
    mm, state = open_shared_memory()
    try:
        instructions = parse_demo_instruction_set()
        expected_frames = sum(delay for _, delay in instructions)
        state.frame_budget = 0
        state.frame_count = 0
        start_frame = state.frame_count
        start_time = time.perf_counter()
        drive_demo(state, instructions)
        state.control_flags |= 0x08  # request stop
        end_time = time.perf_counter()
        end_frame = state.frame_count

        frames_played = (ctypes.c_uint32(end_frame - start_frame).value)
        fps = frames_played / (end_time - start_time + 1e-9)
        print("Frames played:", frames_played)
        print("Frames expected:", expected_frames)
        print("Wall time (s):", end_time - start_time)
        print("Approx FPS:", round(fps, 2))
        print("Viruses remaining:", state.viruses_remaining)
        print("Board MD5:", board_hash(state))
    finally:
        del state
        mm.close()
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run demo benchmark via shared memory.")
    parser.add_argument(
        "--engine",
        type=Path,
        default=repo_root() / "game_engine" / "drmario_engine",
        help="Path to compiled drmario_engine binary.",
    )
    parser.add_argument(
        "--shm-file",
        type=Path,
        default=None,
        help="File path for mmap (use when POSIX shm is unavailable).",
    )
    args = parser.parse_args()

    engine_exec = args.engine.resolve()
    if not engine_exec.exists():
        raise SystemExit(f"Engine binary not found: {engine_exec}")

    run_benchmark(engine_exec, args.shm_file)


if __name__ == "__main__":
    main()
