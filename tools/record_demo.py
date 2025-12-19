"""Record demo playback from the C++ engine to a `GameTranscript`.

Important: The C++ engine replays the retail ROM demo input stream internally
(mirroring `getInputs_checkMode` in the disassembly). For parity with the NES
ground-truth transcript (`data/nes_demo.json`), this recorder:

- Does **not** feed any controller inputs from Python (always writes 0).
- Records `FrameState.buttons = 0` (the NES recorder also stores 0 in demo mode).
- Stops when the engine exits demo mode, *before* appending that final frame,
  matching the NES recorder’s frame list semantics.

Usage:
  cd game_engine && make
  python tools/record_demo.py --output data/cpp_demo.json
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.game_transcript import (
    GameTranscript,
    FrameState,
    save_transcript,
    BOARD_SIZE,
    TILE_EMPTY,
)


def record_cpp_demo(
    engine_path: Path,
    max_frames: int = 10000,
    verbose: bool = False,
) -> GameTranscript:
    """Record demo playback from C++ engine.
    
    Args:
        engine_path: Path to compiled `drmario_engine` binary.
        max_frames: Maximum frames to step after recording starts.
        verbose: Print progress.
    
    Returns:
        `GameTranscript` with frame-by-frame recording.
    """
    import tempfile
    
    # Create temp file for shared memory - set env var BEFORE importing shm module
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        shm_file = Path(f.name)
        # Pre-allocate to correct size
        from game_engine.engine_shm import SHM_SIZE
        f.write(b"\x00" * SHM_SIZE)
    
    # Set env var for both engine and our Python code (restore on exit so tests
    # and other tools are not affected).
    prev_shm_env = os.environ.get("DRMARIO_SHM_FILE")
    os.environ["DRMARIO_SHM_FILE"] = str(shm_file)
    
    env = os.environ.copy()
    
    # Import shared memory module AFTER setting env var
    from game_engine.engine_shm import open_shared_memory
    
    # Start engine in demo mode with manual-step for synchronized stepping
    # --manual-step: engine only advances when control_flags bit 0x04 is set
    proc = subprocess.Popen(
        [str(engine_path), "--demo", "--wait-start", "--manual-step"],
        cwd=engine_path.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    
    # Give it time to initialize
    time.sleep(0.1)
    
    mm = None
    state = None
    try:
        mm, state = open_shared_memory()
        
        # Create transcript
        transcript = GameTranscript(
            source="cpp_engine",
            timestamp=datetime.now().isoformat(),
            level=10,  # Demo is level 10
            speed=1,   # Medium speed
        )
        
        # Wait for the engine to finish its startup memset/arg parsing before
        # releasing the start gate. If we set the gate too early it can be
        # wiped and the engine will block forever in the --wait-start loop.
        t0 = time.perf_counter()
        while (int(state.control_flags) & 0x02) == 0:
            if time.perf_counter() - t0 > 1.0:
                break
            time.sleep(0.001)

        # Release wait-start gate and ensure no external inputs are applied.
        state.control_flags |= 0x01
        state.buttons = 0

        # Engine performs its initial `reset()` after the gate is released.
        # Wait briefly for level setup to populate the board.
        t0 = time.perf_counter()
        while int(state.board[0]) == 0:
            if time.perf_counter() - t0 > 1.0:
                break
            time.sleep(0.001)
        
        # Capture initial state
        transcript.initial_board = bytes(state.board)
        if verbose:
            viruses = sum(1 for b in transcript.initial_board if b != TILE_EMPTY and (b & 0xF0) == 0xD0)
            print(f"Initial board captured: {viruses} viruses")
        
        # State tracking for delta encoding (mirrors NES recorder).
        prev_board = bytearray(state.board)
        prev_pill_row = state.falling_pill_row
        prev_pill_col = state.falling_pill_col
        prev_pill_orient = state.falling_pill_orient
        prev_viruses = state.viruses_remaining
        
        frame_num = 0
        while frame_num < max_frames:
            # Demo mode: engine replays the ROM inputs internally, so we always
            # provide 0 from the outside (matching the NES recorder).
            state.buttons = 0

            expected_frame = state.frame_count + 1
            state.control_flags |= 0x04
            
            # Wait for engine to process frame
            timeout = 20000
            while state.frame_count < expected_frame and timeout > 0:
                if proc.poll() is not None:
                    out, err = proc.communicate(timeout=1)
                    raise RuntimeError(
                        "Engine exited unexpectedly while recording.\n"
                        f"stdout:\n{out.decode(errors='replace')}\n"
                        f"stderr:\n{err.decode(errors='replace')}\n"
                    )
                time.sleep(0.0001)
                timeout -= 1
            
            if timeout == 0:
                if verbose:
                    print(f"Warning: timeout at frame {state.frame_count}")
                break
            
            # Use a deterministic frame counter matching `tools/record_nes_demo.py`
            # (not the ROM `frameCounter`), so transcripts compare cleanly even
            # when `frameCounter` is seeded for parity.
            frame_num += 1

            # Stop conditions are checked *before* appending a FrameState, to
            # mirror the NES ground-truth recorder.
            if state.mode != 0x00:  # MODE_DEMO
                if verbose:
                    print(f"Demo ended at frame {frame_num}, mode=0x{state.mode:02x}")
                break
            if state.stage_clear:
                transcript.outcome = "clear"
                if verbose:
                    print(f"Stage cleared at frame {frame_num}")
                break
            if state.level_fail:
                transcript.outcome = "topout"
                if verbose:
                    print(f"Top-out at frame {frame_num}")
                break
            
            # Build frame state
            fs = FrameState(frame=frame_num, buttons=0)
            
            # Check pill position changes
            if state.falling_pill_row != prev_pill_row:
                fs.pill_row = state.falling_pill_row
                prev_pill_row = state.falling_pill_row
            if state.falling_pill_col != prev_pill_col:
                fs.pill_col = state.falling_pill_col
                prev_pill_col = state.falling_pill_col
            if state.falling_pill_orient != prev_pill_orient:
                fs.pill_orient = state.falling_pill_orient
                prev_pill_orient = state.falling_pill_orient
            
            # Check board changes
            board_changes: List[Tuple[int, int, int]] = []
            current_board = bytearray(state.board)
            for i in range(BOARD_SIZE):
                if current_board[i] != prev_board[i]:
                    board_changes.append((i, prev_board[i], current_board[i]))
                    prev_board[i] = current_board[i]
            
            if board_changes:
                fs.board_changes = board_changes
                # Detect pill lock (new tiles appear)
                new_tiles = sum(1 for _, old, new in board_changes 
                               if old == TILE_EMPTY and new != TILE_EMPTY and (new & 0xF0) != 0xD0)
                if new_tiles > 0:
                    fs.pill_locked = True
            
            # Check virus clears
            if state.viruses_remaining < prev_viruses:
                fs.viruses_cleared = prev_viruses - state.viruses_remaining
                prev_viruses = state.viruses_remaining
            
            transcript.frames.append(fs)
            
            if verbose and frame_num % 500 == 0:
                print(f"Frame {frame_num}: viruses={state.viruses_remaining}, pills={state.pill_counter_total}")
        
        if transcript.outcome not in {"clear", "topout"}:
            transcript.outcome = "ongoing"

        transcript.total_frames = frame_num
        
        if verbose:
            print(f"Recording complete: {frame_num} frames, outcome={transcript.outcome}")
        
        return transcript
        
    finally:
        if state is not None:
            del state
        if mm is not None:
            mm.close()

        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        
        # Cleanup temp file
        try:
            shm_file.unlink()
        except Exception:
            pass

        if prev_shm_env is None:
            os.environ.pop("DRMARIO_SHM_FILE", None)
        else:
            os.environ["DRMARIO_SHM_FILE"] = prev_shm_env


def main():
    parser = argparse.ArgumentParser(description="Record C++ engine demo to GameTranscript")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/cpp_demo.json"),
        help="Output file (.json or .msgpack)",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=Path("game_engine/drmario_engine"),
        help="Path to engine binary",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10000,
        help="Maximum frames to record",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    repo_root = Path(__file__).parent.parent
    engine_path = repo_root / args.engine
    output = repo_root / args.output
    
    if not engine_path.exists():
        print(f"Error: Engine not found at {engine_path}")
        print("Build it with: cd game_engine && make")
        return 1
    
    # Record
    transcript = record_cpp_demo(
        engine_path=engine_path,
        max_frames=args.max_frames,
        verbose=args.verbose,
    )
    
    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    save_transcript(transcript, output)
    print(f"Saved transcript to {output}")
    print(f"  Frames: {transcript.total_frames}")
    print(f"  Outcome: {transcript.outcome}")
    print(f"  Pills locked: {sum(1 for f in transcript.frames if f.pill_locked)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
