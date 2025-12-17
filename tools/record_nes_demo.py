"""Record NES emulator demo playback to GameTranscript as ground truth.

This loads the actual Dr. Mario ROM in the libretro emulator, lets
the demo mode run, and captures frame-by-frame state for comparison.

Usage:
    # Set environment
    export DRMARIO_CORE_PATH=/path/to/mesen_libretro.dylib
    export DRMARIO_ROM_PATH=/path/to/DrMario.nes
    
    # Record demo
    python tools/record_nes_demo.py --output data/nes_demo.json -v
"""
from __future__ import annotations

import argparse
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
    save_transcript,
    BOARD_SIZE,
    TILE_EMPTY,
)


# NES RAM addresses (from ram_offsets.json and AGENTS.md)
ADDR_BOARD = 0x0400
ADDR_PILL_COL = 0x0305
ADDR_PILL_ROW = 0x0306
ADDR_PILL_ORIENT = 0x0325
ADDR_PILL_COUNTER = 0x0310
ADDR_VIRUSES = 0x0324
ADDR_MODE = 0x0046
ADDR_STAGE_CLEAR = 0x0055
ADDR_LEVEL = 0x0316
ADDR_SPEED = 0x030B
ADDR_FRAME_COUNTER = 0x0043

# Mode values
MODE_DEMO = 0x04  # Demo and gameplay both use this when active
STAGE_CLEARED = 0x01


def record_nes_demo(
    max_frames: int = 10000,
    wait_demo_start: int = 180,  # ~3 seconds at 60fps
    verbose: bool = False,
) -> GameTranscript:
    """Record NES demo playback from libretro emulator.
    
    Args:
        max_frames: Maximum frames to record after demo starts
        wait_demo_start: Frames to wait for demo to automatically start
        verbose: Print progress
    
    Returns:
        GameTranscript with frame-by-frame recording
    """
    from envs.backends.libretro_backend import LibretroBackend
    
    # Initialize backend
    backend = LibretroBackend()
    backend.load()
    
    if verbose:
        print("Backend loaded, waiting for demo to start...")
    
    try:
        # Wait for demo mode to activate
        # On NES, demo starts after ~3 seconds on title screen
        demo_started = False
        
        for _ in range(wait_demo_start):
            backend.step([0] * 9)  # 9 buttons, all released
            ram = backend.get_ram()
            if ram is None:
                continue
            
            # Check if we're in gameplay mode with viruses
            mode = int(ram[ADDR_MODE])
            viruses = int(ram[ADDR_VIRUSES])
            
            if mode == MODE_DEMO and viruses > 0:
                demo_started = True
                if verbose:
                    print(f"Demo started! Mode=0x{mode:02x}, Viruses={viruses}")
                break
        
        if not demo_started:
            raise RuntimeError("Demo mode did not start within timeout")
        
        # Create transcript
        ram = backend.get_ram()
        assert ram is not None
        
        transcript = GameTranscript(
            source="nes_emulator",
            timestamp=datetime.now().isoformat(),
            level=int(ram[ADDR_LEVEL]),
            speed=int(ram[ADDR_SPEED]),
        )
        
        # Capture initial board
        initial_board = bytes(ram[ADDR_BOARD:ADDR_BOARD + BOARD_SIZE])
        transcript.initial_board = initial_board
        
        viruses_initial = sum(1 for b in initial_board 
                              if b != TILE_EMPTY and (b & 0xF0) == 0xD0)
        if verbose:
            print(f"Initial board captured: {viruses_initial} viruses")
        
        # State tracking
        prev_board = bytearray(initial_board)
        prev_pill_row = ram[ADDR_PILL_ROW]
        prev_pill_col = ram[ADDR_PILL_COL]
        prev_pill_orient = ram[ADDR_PILL_ORIENT]
        prev_viruses = ram[ADDR_VIRUSES]
        
        frame_num = 0
        
        while frame_num < max_frames:
            # Step emulator (no input - it's demo mode, CPU plays)
            backend.step([0] * 9)
            frame_num += 1
            
            ram = backend.get_ram()
            if ram is None:
                continue
            
            # Check for demo end
            mode = int(ram[ADDR_MODE])
            stage_clear = int(ram[ADDR_STAGE_CLEAR])
            
            if mode != MODE_DEMO:
                if verbose:
                    print(f"Mode changed to 0x{mode:02x} at frame {frame_num}")
                break
            
            if stage_clear == STAGE_CLEARED:
                if verbose:
                    print(f"Stage cleared at frame {frame_num}")
                transcript.outcome = "clear"
                break
            
            # Build frame state
            fs = FrameState(frame=frame_num, buttons=0)
            
            # Check pill position changes
            curr_row = ram[ADDR_PILL_ROW]
            curr_col = ram[ADDR_PILL_COL]
            curr_orient = ram[ADDR_PILL_ORIENT]
            
            if curr_row != prev_pill_row:
                fs.pill_row = curr_row
                prev_pill_row = curr_row
            if curr_col != prev_pill_col:
                fs.pill_col = curr_col
                prev_pill_col = curr_col
            if curr_orient != prev_pill_orient:
                fs.pill_orient = curr_orient
                prev_pill_orient = curr_orient
            
            # Check board changes
            current_board = ram[ADDR_BOARD:ADDR_BOARD + BOARD_SIZE]
            board_changes: List[Tuple[int, int, int]] = []
            
            for i in range(BOARD_SIZE):
                if current_board[i] != prev_board[i]:
                    board_changes.append((i, prev_board[i], current_board[i]))
                    prev_board[i] = current_board[i]
            
            if board_changes:
                fs.board_changes = board_changes
                # Detect pill lock (new non-virus tiles appear)
                new_tiles = sum(1 for _, old, new in board_changes 
                               if old == TILE_EMPTY and new != TILE_EMPTY 
                               and (new & 0xF0) != 0xD0)
                if new_tiles > 0:
                    fs.pill_locked = True
            
            # Check virus clears
            curr_viruses = ram[ADDR_VIRUSES]
            if curr_viruses < prev_viruses:
                fs.viruses_cleared = prev_viruses - curr_viruses
                prev_viruses = curr_viruses
            
            transcript.frames.append(fs)
            
            if verbose and frame_num % 500 == 0:
                print(f"Frame {frame_num}: viruses={curr_viruses}")
        
        if transcript.outcome != "clear":
            transcript.outcome = "ongoing"
        
        transcript.total_frames = frame_num
        
        pills_locked = sum(1 for f in transcript.frames if f.pill_locked)
        if verbose:
            print(f"Recording complete: {frame_num} frames, {pills_locked} pills")
        
        return transcript
        
    finally:
        backend.close()


def main():
    parser = argparse.ArgumentParser(
        description="Record NES emulator demo to GameTranscript (ground truth)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/nes_demo.json"),
        help="Output file (.json or .msgpack)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10000,
        help="Maximum frames to record",
    )
    parser.add_argument(
        "--wait-demo",
        type=int,
        default=180,
        help="Frames to wait for demo to start (~3 sec)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress",
    )
    
    args = parser.parse_args()
    
    # Record
    transcript = record_nes_demo(
        max_frames=args.max_frames,
        wait_demo_start=args.wait_demo,
        verbose=args.verbose,
    )
    
    # Resolve output path
    repo_root = Path(__file__).parent.parent
    output = repo_root / args.output
    
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
