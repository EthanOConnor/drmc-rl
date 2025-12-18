"""Game Transcript — Frame-accurate recording format for Dr. Mario games.

Supports both JSON (human-readable) and MessagePack (compact binary) formats
with utilities to convert between them.

Design decisions:
- Frame-by-frame: Precise divergence detection matters more than file size
- Delta encoding: Only store changes to reduce redundancy
- MD5 hashes: Quick board comparison without full diff
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    msgpack = None  # type: ignore
    MSGPACK_AVAILABLE = False


# === Constants ===

BOARD_WIDTH = 8
BOARD_HEIGHT = 16
BOARD_SIZE = BOARD_WIDTH * BOARD_HEIGHT

# Tile encoding (matches NES RAM)
TILE_EMPTY = 0xFF
TILE_VIRUS_MASK = 0xD0
COLOR_MASK = 0x03


# === Data Structures ===

@dataclass
class FrameState:
    """State delta for a single frame.
    
    Only non-None fields are recorded (delta encoding).
    """
    frame: int
    
    # Controller input (always recorded)
    buttons: int = 0
    
    # Falling pill position (only if changed)
    pill_row: Optional[int] = None
    pill_col: Optional[int] = None
    pill_orient: Optional[int] = None
    
    # Board changes: list of (index, old_value, new_value)
    board_changes: Optional[List[Tuple[int, int, int]]] = None
    
    # Events
    pill_locked: bool = False
    viruses_cleared: int = 0
    chain_triggered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, omitting None values for compactness."""
        d: Dict[str, Any] = {"f": self.frame, "b": self.buttons}
        if self.pill_row is not None:
            d["pr"] = self.pill_row
        if self.pill_col is not None:
            d["pc"] = self.pill_col
        if self.pill_orient is not None:
            d["po"] = self.pill_orient
        if self.board_changes:
            d["bc"] = self.board_changes
        if self.pill_locked:
            d["pl"] = 1
        if self.viruses_cleared > 0:
            d["vc"] = self.viruses_cleared
        if self.chain_triggered:
            d["ch"] = 1
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FrameState":
        bc_raw = d.get("bc")
        bc: Optional[List[Tuple[int, int, int]]] = None
        if bc_raw:
            # JSON decodes tuples as lists; normalize to a stable internal type so that
            # comparisons and reconstruction behave identically across formats.
            bc = [(int(i), int(old), int(new)) for i, old, new in bc_raw]
        return cls(
            frame=d["f"],
            buttons=d.get("b", 0),
            pill_row=d.get("pr"),
            pill_col=d.get("pc"),
            pill_orient=d.get("po"),
            board_changes=bc,
            pill_locked=bool(d.get("pl", 0)),
            viruses_cleared=d.get("vc", 0),
            chain_triggered=bool(d.get("ch", 0)),
        )


@dataclass
class GameTranscript:
    """Complete frame-by-frame record of a Dr. Mario game."""
    
    # Metadata
    version: str = "1.0"
    source: str = "unknown"  # "nes_demo", "cpp_engine", "emulator", "human"
    timestamp: str = ""
    
    # Game parameters
    level: int = 0
    speed: int = 1  # 0=low, 1=med, 2=hi
    seed: Optional[int] = None
    
    # Initial state (full board, not delta)
    initial_board: bytes = field(default_factory=lambda: bytes([TILE_EMPTY] * BOARD_SIZE))
    pill_sequence: List[int] = field(default_factory=list)  # Color combo indices 0-8
    
    # Frame-by-frame recording
    frames: List[FrameState] = field(default_factory=list)
    
    # Outcome
    outcome: str = "ongoing"  # "clear", "topout", "timeout", "ongoing"
    total_frames: int = 0

    def board_hash(self, board: bytes) -> str:
        """MD5 hash of board state for quick comparison."""
        return hashlib.md5(board).hexdigest()[:16]

    def initial_board_hash(self) -> str:
        return self.board_hash(self.initial_board)

    def reconstruct_board(self, up_to_frame: int) -> bytes:
        """Reconstruct board state at given frame by applying deltas."""
        board = bytearray(self.initial_board)
        for fs in self.frames:
            if fs.frame > up_to_frame:
                break
            if fs.board_changes:
                for idx, _old, new in fs.board_changes:
                    board[idx] = new
        return bytes(board)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "version": self.version,
            "source": self.source,
            "timestamp": self.timestamp,
            "level": self.level,
            "speed": self.speed,
            "seed": self.seed,
            "initial_board": self.initial_board.hex(),
            "pill_sequence": self.pill_sequence,
            "frames": [f.to_dict() for f in self.frames],
            "outcome": self.outcome,
            "total_frames": self.total_frames,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GameTranscript":
        return cls(
            version=d.get("version", "1.0"),
            source=d.get("source", "unknown"),
            timestamp=d.get("timestamp", ""),
            level=d.get("level", 0),
            speed=d.get("speed", 1),
            seed=d.get("seed"),
            initial_board=bytes.fromhex(d.get("initial_board", "ff" * BOARD_SIZE)),
            pill_sequence=d.get("pill_sequence", []),
            frames=[FrameState.from_dict(f) for f in d.get("frames", [])],
            outcome=d.get("outcome", "ongoing"),
            total_frames=d.get("total_frames", 0),
        )


# === Serialization ===

def save_json(transcript: GameTranscript, path: Path) -> None:
    """Save transcript to human-readable JSON."""
    import numpy as np
    
    class NumpyEncoder(json.JSONEncoder):
        """JSON encoder that handles numpy types."""
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating)):
                return int(obj) if isinstance(obj, np.integer) else float(obj)
            return super().default(obj)
    
    path = Path(path)
    with open(path, "w") as f:
        json.dump(transcript.to_dict(), f, indent=2, cls=NumpyEncoder)


def load_json(path: Path) -> GameTranscript:
    """Load transcript from JSON file."""
    path = Path(path)
    with open(path) as f:
        return GameTranscript.from_dict(json.load(f))


def save_binary(transcript: GameTranscript, path: Path) -> None:
    """Save transcript to compact MessagePack binary."""
    if not MSGPACK_AVAILABLE:
        raise ImportError("msgpack required for binary format: pip install msgpack")
    path = Path(path)
    with open(path, "wb") as f:
        # Use raw bytes for initial_board (more compact than hex string)
        d = transcript.to_dict()
        d["initial_board"] = transcript.initial_board  # bytes, not hex
        msgpack.pack(d, f, use_bin_type=True)


def load_binary(path: Path) -> GameTranscript:
    """Load transcript from MessagePack binary."""
    if not MSGPACK_AVAILABLE:
        raise ImportError("msgpack required for binary format: pip install msgpack")
    path = Path(path)
    with open(path, "rb") as f:
        d = msgpack.unpack(f, raw=False)
        # Convert bytes back to hex for from_dict
        if isinstance(d.get("initial_board"), bytes):
            d["initial_board"] = d["initial_board"].hex()
        return GameTranscript.from_dict(d)


def save_transcript(transcript: GameTranscript, path: Path) -> None:
    """Auto-detect format from extension and save."""
    path = Path(path)
    if path.suffix == ".json":
        save_json(transcript, path)
    elif path.suffix in (".msgpack", ".bin", ".mpk"):
        save_binary(transcript, path)
    else:
        raise ValueError(f"Unknown format for {path}, use .json or .msgpack")


def load_transcript(path: Path) -> GameTranscript:
    """Auto-detect format from extension and load."""
    path = Path(path)
    if path.suffix == ".json":
        return load_json(path)
    elif path.suffix in (".msgpack", ".bin", ".mpk"):
        return load_binary(path)
    else:
        raise ValueError(f"Unknown format for {path}, use .json or .msgpack")


def convert(src: Path, dst: Path) -> None:
    """Convert between JSON and binary formats."""
    transcript = load_transcript(src)
    save_transcript(transcript, dst)


# === Comparison ===

@dataclass
class Divergence:
    """Record of where two transcripts diverged."""
    frame: int
    field: str
    expected: Any
    actual: Any
    message: str = ""


def compare_transcripts(
    expected: GameTranscript,
    actual: GameTranscript,
    stop_on_first: bool = True,
) -> List[Divergence]:
    """Compare two transcripts frame-by-frame.
    
    Returns list of divergences (empty = perfect match).
    """
    divergences: List[Divergence] = []

    # Check initial board
    if expected.initial_board != actual.initial_board:
        divergences.append(Divergence(
            frame=0,
            field="initial_board",
            expected=expected.initial_board_hash(),
            actual=actual.initial_board_hash(),
            message="Initial board state differs",
        ))
        if stop_on_first:
            return divergences

    # Build frame lookup for expected
    exp_frames = {f.frame: f for f in expected.frames}
    act_frames = {f.frame: f for f in actual.frames}
    
    all_frame_nums = sorted(set(exp_frames.keys()) | set(act_frames.keys()))
    
    for frame_num in all_frame_nums:
        exp_f = exp_frames.get(frame_num)
        act_f = act_frames.get(frame_num)
        
        if exp_f is None:
            divergences.append(Divergence(
                frame=frame_num,
                field="frame_extra",
                expected="missing",
                actual="present",
                message=f"Unexpected frame {frame_num} in actual",
            ))
        elif act_f is None:
            divergences.append(Divergence(
                frame=frame_num,
                field="frame_missing",
                expected="present",
                actual="missing",
                message=f"Frame {frame_num} missing in actual",
            ))
        else:
            # Compare frame contents
            for field_name in ["buttons", "pill_row", "pill_col", "pill_orient",
                               "pill_locked", "viruses_cleared", "chain_triggered"]:
                exp_val = getattr(exp_f, field_name)
                act_val = getattr(act_f, field_name)
                if exp_val != act_val:
                    divergences.append(Divergence(
                        frame=frame_num,
                        field=field_name,
                        expected=exp_val,
                        actual=act_val,
                        message=f"Frame {frame_num}: {field_name} differs",
                    ))
                    if stop_on_first:
                        return divergences
            
            # Compare board changes
            exp_bc = exp_f.board_changes or []
            act_bc = act_f.board_changes or []
            # Defensive normalization: ensure stable type even for callers that
            # construct FrameState objects directly.
            exp_bc_norm = [tuple(map(int, bc)) for bc in exp_bc]
            act_bc_norm = [tuple(map(int, bc)) for bc in act_bc]
            if exp_bc_norm != act_bc_norm:
                divergences.append(Divergence(
                    frame=frame_num,
                    field="board_changes",
                    expected=exp_bc_norm,
                    actual=act_bc_norm,
                    message=f"Frame {frame_num}: board changes differ",
                ))
                if stop_on_first:
                    return divergences

    return divergences


# === Demo Input Parsing ===

# NES controller button masks
BTN_RIGHT = 0x01
BTN_LEFT = 0x02
BTN_DOWN = 0x04
BTN_UP = 0x08
BTN_START = 0x10
BTN_SELECT = 0x20
BTN_B = 0x40
BTN_A = 0x80


def parse_demo_inputs(asm_path: Path) -> List[Tuple[int, int]]:
    """Parse demo_instructionSet from asm file.
    
    Returns list of (buttons, duration_frames) pairs.
    The file format is run-length encoded: button byte, then frame count.
    """
    import re
    
    path = Path(asm_path)
    content = path.read_text()
    
    # Find demo_instructionSet block (use NTSC version, skip EU)
    in_block = False
    in_eu = False
    instructions: List[Tuple[int, int]] = []
    
    for line in content.splitlines():
        if "demo_instructionSet:" in line:
            in_block = True
            continue
        if "if !ver_EU" in line:
            in_eu = False
            continue
        if "else" in line and in_block:
            in_eu = True
            continue
        if "endif" in line and in_block:
            break
        
        if not in_block or in_eu:
            continue
        
        # Parse .db lines
        matches = re.findall(r"\$([0-9A-Fa-f]{2})", line)
        for i in range(0, len(matches) - 1, 2):
            btn = int(matches[i], 16)
            dur = int(matches[i + 1], 16)
            instructions.append((btn, dur))
    
    return instructions


def expand_inputs(rle_inputs: List[Tuple[int, int]]) -> List[int]:
    """Expand demo inputs to per-frame button states.

    The retail ROM stores demo instructions as (buttons, duration) pairs, where
    `duration` is loaded into a countdown that is decremented each frame.

    Because the new (buttons, duration) pair is applied immediately on the load
    frame (when the countdown is 0), each pair lasts for `duration + 1` frames.
    """
    frames = []
    for btn, duration in rle_inputs:
        frames.extend([btn] * (duration + 1))
    return frames


# === CLI ===

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Game transcript utilities")
    sub = parser.add_subparsers(dest="cmd")
    
    conv = sub.add_parser("convert", help="Convert between JSON and binary")
    conv.add_argument("src", type=Path)
    conv.add_argument("dst", type=Path)
    
    cmp = sub.add_parser("compare", help="Compare two transcripts")
    cmp.add_argument("a", type=Path)
    cmp.add_argument("b", type=Path)
    cmp.add_argument("--all", action="store_true", help="Show all divergences")
    
    args = parser.parse_args()
    
    if args.cmd == "convert":
        convert(args.src, args.dst)
        print(f"Converted {args.src} → {args.dst}")
    elif args.cmd == "compare":
        a = load_transcript(args.a)
        b = load_transcript(args.b)
        divs = compare_transcripts(a, b, stop_on_first=not args.all)
        if divs:
            print(f"Found {len(divs)} divergence(s):")
            for d in divs[:10]:
                print(f"  Frame {d.frame}: {d.field} - {d.message}")
                print(f"    Expected: {d.expected}")
                print(f"    Actual:   {d.actual}")
        else:
            print("Perfect match!")


if __name__ == "__main__":
    main()
