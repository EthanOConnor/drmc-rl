"""Record demo playback from C++ engine to GameTranscript.

This feeds the demo inputs to the C++ engine frame-by-frame and
records the full state changes for parity verification.

Usage:
    # Build engine first
    cd game_engine && make
    
    # Record demo
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
from typing import List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.game_transcript import (
    GameTranscript,
    FrameState,
    parse_demo_inputs,
    expand_inputs,
    save_transcript,
    BOARD_SIZE,
    TILE_EMPTY,
)


def record_cpp_demo(
    engine_path: Path,
    demo_inputs_path: Path,
    max_frames: int = 10000,
    verbose: bool = False,
) -> GameTranscript:
    """Record demo playback from C++ engine.
    
    Args:
        engine_path: Path to compiled drmario_engine binary
        demo_inputs_path: Path to demo_inputs.asm
        max_frames: Maximum frames to record
        verbose: Print progress
    
    Returns:
        GameTranscript with frame-by-frame recording
    """
    import tempfile
    
    # Parse demo inputs
    rle_inputs = parse_demo_inputs(demo_inputs_path)
    frame_inputs = expand_inputs(rle_inputs)
    
    if verbose:
        print(f"Parsed {len(rle_inputs)} input pairs → {len(frame_inputs)} frames")
    
    # Create temp file for shared memory - set env var BEFORE importing shm module
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        shm_file = Path(f.name)
        # Pre-allocate to correct size
        from game_engine.engine_shm import SHM_SIZE
        f.write(b"\x00" * SHM_SIZE)
    
    # Set env var for both engine and our Python code
    os.environ["DRMARIO_SHM_FILE"] = str(shm_file)
    
    env = os.environ.copy()
    
    # Import shared memory module AFTER setting env var
    from game_engine.engine_shm import DrMarioStatePy, open_shared_memory
    
    # Start engine in demo mode with manual-step for synchronized stepping
    # --manual-step: engine only advances when control_flags bit 0x04 is set
    proc = subprocess.Popen(
        [str(engine_path), "--demo", "--manual-step"],
        cwd=engine_path.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    
    # Give it time to initialize
    time.sleep(0.1)
    
    try:
        mm, state = open_shared_memory()
        
        # Create transcript
        transcript = GameTranscript(
            source="cpp_engine",
            timestamp=datetime.now().isoformat(),
            level=10,  # Demo is level 10
            speed=1,   # Medium speed
        )
        
        # Start the engine: set bit 0x01 to release any wait gate
        state.control_flags |= 0x01
        state.buttons = 0
        
        # Trigger first step to initialize
        state.control_flags |= 0x04  # Step trigger
        time.sleep(0.01)
        
        # Wait for first frame
        timeout = 100
        while state.frame_count == 0 and timeout > 0:
            state.control_flags |= 0x04
            time.sleep(0.001)
            timeout -= 1
        
        if timeout == 0:
            raise RuntimeError("Engine failed to start")
        
        # Capture initial state
        transcript.initial_board = bytes(state.board)
        if verbose:
            viruses = sum(1 for b in transcript.initial_board if b != TILE_EMPTY and (b & 0xF0) == 0xD0)
            print(f"Initial board captured: {viruses} viruses")
        
        # State tracking for delta encoding
        prev_board = bytearray(state.board)
        prev_pill_row = state.falling_pill_row
        prev_pill_col = state.falling_pill_col
        prev_pill_orient = state.falling_pill_orient
        prev_viruses = state.viruses_remaining
        
        # Use frame_count as canonical frame number (engine starts at 1)
        start_frame = state.frame_count
        
        while state.frame_count < start_frame + max_frames and not state.stage_clear and not state.level_fail:
            # Get button input for this frame - use frame_count as index
            # NES demo inputs are 0-indexed from game start
            input_idx = state.frame_count - 1  # frame_count 1 = input[0]
            buttons = frame_inputs[input_idx] if input_idx < len(frame_inputs) else 0
            
            # Send input
            state.buttons = buttons
            
            # Trigger step (manual-step mode: set bit 0x04)
            expected_frame = state.frame_count + 1
            state.control_flags |= 0x04
            
            # Wait for engine to process frame
            timeout = 1000
            while state.frame_count < expected_frame and timeout > 0:
                time.sleep(0.0001)
                timeout -= 1
            
            if timeout == 0:
                if verbose:
                    print(f"Warning: timeout at frame {state.frame_count}")
                break
            
            frame_num = state.frame_count
            
            # Build frame state
            fs = FrameState(frame=frame_num, buttons=buttons)
            
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
        
        # Record outcome
        if state.stage_clear:
            transcript.outcome = "clear"
        elif state.level_fail:
            transcript.outcome = "topout"
        else:
            transcript.outcome = "ongoing"
        
        transcript.total_frames = frame_num
        
        if verbose:
            print(f"Recording complete: {frame_num} frames, outcome={transcript.outcome}")
        
        return transcript
        
    finally:
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
    demo_inputs = repo_root / "dr-mario-disassembly" / "data" / "drmario_data_demo_inputs.asm"
    output = repo_root / args.output
    
    if not engine_path.exists():
        print(f"Error: Engine not found at {engine_path}")
        print("Build it with: cd game_engine && make")
        return 1
    
    if not demo_inputs.exists():
        print(f"Error: Demo inputs not found at {demo_inputs}")
        return 1
    
    # Record
    transcript = record_cpp_demo(
        engine_path=engine_path,
        demo_inputs_path=demo_inputs,
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
