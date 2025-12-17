"""Board state viewer for Rich TUI.

Renders Dr. Mario board state with colored Unicode block characters.
Supports both raw board bytes and state tensor observations.

Features:
- Color-coded tiles (red, yellow, blue)
- Visual distinction for viruses vs pills
- Falling pill highlighting
- Preview pill display
- Single-step mode support
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from rich.console import Console, Group, RenderableType
    from rich.panel import Panel
    from rich.style import Style
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# Board dimensions
BOARD_HEIGHT = 16
BOARD_WIDTH = 8

# Tile type masks (from NES encoding)
TILE_EMPTY = 0xFF
TILE_VIRUS = 0xD0
TILE_TOP = 0x40
TILE_BOTTOM = 0x50
TILE_LEFT = 0x60
TILE_RIGHT = 0x70
TILE_SINGLE = 0x80
MASK_TYPE = 0xF0
MASK_COLOR = 0x03

# Color indices
COLOR_YELLOW = 0
COLOR_RED = 1
COLOR_BLUE = 2

# Unicode block characters for rendering
# Using half blocks for compact display
CHAR_EMPTY = "  "
CHAR_VIRUS = "☣ "  # Biohazard for virus
CHAR_PILL_HALF = "██"  # Full block for pill
CHAR_PILL_SINGLE = "▓▓"  # Medium shade for single
CHAR_FALLING = "▒▒"  # Light shade for falling pill

# Rich colors for Dr. Mario
COLORS = {
    COLOR_YELLOW: "bright_yellow",
    COLOR_RED: "bright_red", 
    COLOR_BLUE: "bright_blue",
}

# Background colors for contrast
BG_EMPTY = "grey11"
BG_VIRUS = "grey23"


@dataclass
class BoardState:
    """Parsed board state with overlay information."""
    
    # 16x8 array of tile bytes
    board: np.ndarray
    
    # Falling pill info (optional)
    falling_row: Optional[int] = None
    falling_col: Optional[int] = None
    falling_orient: Optional[int] = None  # 0=vertical, 1=horizontal
    falling_color_l: Optional[int] = None
    falling_color_r: Optional[int] = None
    
    # Preview pill (optional)
    preview_color_l: Optional[int] = None
    preview_color_r: Optional[int] = None
    
    # Game state
    viruses_remaining: int = 0
    frame_count: int = 0
    pill_count: int = 0
    level: int = 0


def parse_board_bytes(board_bytes: Union[bytes, np.ndarray, List[int]]) -> np.ndarray:
    """Parse 128-byte board data into 16x8 array."""
    arr = np.asarray(board_bytes, dtype=np.uint8)
    if arr.size != 128:
        raise ValueError(f"Board must be 128 bytes, got {arr.size}")
    return arr.reshape((BOARD_HEIGHT, BOARD_WIDTH))


def decode_tile(tile: int) -> Tuple[str, Optional[int], str]:
    """Decode a tile byte into (char, color_idx, tile_type).
    
    Returns:
        (display_char, color_index_or_None, type_name)
    """
    if tile == TILE_EMPTY:
        return CHAR_EMPTY, None, "empty"
    
    tile_type = tile & MASK_TYPE
    color = tile & MASK_COLOR
    
    if tile_type == TILE_VIRUS:
        return CHAR_VIRUS, color, "virus"
    elif tile_type == TILE_TOP:
        return "▀▀", color, "top"  # Upper half block
    elif tile_type == TILE_BOTTOM:
        return "▄▄", color, "bottom"  # Lower half block
    elif tile_type == TILE_LEFT:
        return "█▌", color, "left"  # Block + left half
    elif tile_type == TILE_RIGHT:
        return "▐█", color, "right"  # Right half + block
    elif tile_type == TILE_SINGLE:
        return CHAR_PILL_SINGLE, color, "single"
    else:
        # Unknown tile type
        return "??", color, f"unk_{tile_type:02x}"


def render_board_text(
    state: BoardState,
    show_preview: bool = True,
    show_stats: bool = True,
    compact: bool = False,
) -> Text:
    """Render board state as Rich Text.
    
    Args:
        state: Parsed board state
        show_preview: Show preview pill above board
        show_stats: Show game stats below board
        compact: Use single characters instead of double-width
    
    Returns:
        Rich Text object with colored board
    """
    text = Text()
    
    # Top border
    border_style = Style(color="white", dim=True)
    text.append("╔" + "══" * BOARD_WIDTH + "╗\n", style=border_style)
    
    # Preview area (2 rows above board)
    if show_preview and state.preview_color_l is not None:
        text.append("║", style=border_style)
        for col in range(BOARD_WIDTH):
            if col in (3, 4):  # Preview in center columns
                if col == 3 and state.preview_color_l is not None:
                    color = COLORS.get(state.preview_color_l, "white")
                    text.append("██", style=Style(color=color))
                elif col == 4 and state.preview_color_r is not None:
                    color = COLORS.get(state.preview_color_r, "white")
                    text.append("██", style=Style(color=color))
                else:
                    text.append("  ", style=Style(bgcolor=BG_EMPTY))
            else:
                text.append("  ", style=Style(bgcolor=BG_EMPTY))
        text.append("║\n", style=border_style)
        text.append("╠" + "══" * BOARD_WIDTH + "╣\n", style=border_style)
    
    # Board rows
    for row in range(BOARD_HEIGHT):
        text.append("║", style=border_style)
        
        for col in range(BOARD_WIDTH):
            tile = state.board[row, col]
            char, color_idx, tile_type = decode_tile(tile)
            
            # Check if this is the falling pill position
            is_falling = False
            if state.falling_row is not None and state.falling_col is not None:
                # Falling pill can span two cells
                if state.falling_orient == 0:  # Vertical
                    if row == state.falling_row and col == state.falling_col:
                        is_falling = True
                        color_idx = state.falling_color_l
                        char = CHAR_FALLING
                    elif row == state.falling_row - 1 and col == state.falling_col:
                        is_falling = True
                        color_idx = state.falling_color_r
                        char = CHAR_FALLING
                else:  # Horizontal
                    if row == state.falling_row and col == state.falling_col:
                        is_falling = True
                        color_idx = state.falling_color_l
                        char = CHAR_FALLING
                    elif row == state.falling_row and col == state.falling_col + 1:
                        is_falling = True
                        color_idx = state.falling_color_r
                        char = CHAR_FALLING
            
            # Style based on tile content
            if color_idx is not None:
                color = COLORS.get(color_idx, "white")
                if tile_type == "virus":
                    style = Style(color=color, bold=True, bgcolor=BG_VIRUS)
                elif is_falling:
                    style = Style(color=color, blink=True)
                else:
                    style = Style(color=color)
            else:
                style = Style(color="grey30", bgcolor=BG_EMPTY)
            
            text.append(char, style=style)
        
        text.append("║\n", style=border_style)
    
    # Bottom border
    text.append("╚" + "══" * BOARD_WIDTH + "╝", style=border_style)
    
    # Stats line
    if show_stats:
        stats_style = Style(color="cyan", dim=True)
        text.append(f"\n L:{state.level} V:{state.viruses_remaining} "
                   f"P:{state.pill_count} F:{state.frame_count}", style=stats_style)
    
    return text


def render_board_panel(
    state: BoardState,
    title: str = "Dr. Mario",
    **kwargs,
) -> Panel:
    """Render board state as a Rich Panel.
    
    Args:
        state: Parsed board state
        title: Panel title
        **kwargs: Passed to render_board_text
    
    Returns:
        Rich Panel containing the rendered board
    """
    text = render_board_text(state, **kwargs)
    
    # Compose subtitle with game info
    subtitle = []
    if state.viruses_remaining > 0:
        subtitle.append(f"🦠 {state.viruses_remaining}")
    if state.level > 0:
        subtitle.append(f"Lv.{state.level}")
    subtitle_str = " | ".join(subtitle) if subtitle else None
    
    return Panel(
        text,
        title=f"[bold cyan]{title}[/bold cyan]",
        subtitle=subtitle_str,
        border_style="blue",
    )


def board_from_env_info(info: Dict[str, Any]) -> BoardState:
    """Create BoardState from environment info dict.
    
    Supports different info formats from various env wrappers.
    """
    board = np.full((BOARD_HEIGHT, BOARD_WIDTH), TILE_EMPTY, dtype=np.uint8)
    
    # Try to get board from various keys
    if "board" in info:
        board_data = info["board"]
        if hasattr(board_data, "__len__") and len(board_data) == 128:
            board = parse_board_bytes(board_data)
    elif "raw_ram" in info:
        # Extract from RAM at offset 0x400
        raw_ram = info["raw_ram"]
        if len(raw_ram) >= 0x480:
            board = parse_board_bytes(raw_ram[0x400:0x480])
    
    return BoardState(
        board=board,
        falling_row=info.get("falling_pill_row"),
        falling_col=info.get("falling_pill_col"),
        falling_orient=info.get("falling_pill_orient"),
        falling_color_l=info.get("falling_color_l"),
        falling_color_r=info.get("falling_color_r"),
        preview_color_l=info.get("preview_color_l"),
        preview_color_r=info.get("preview_color_r"),
        viruses_remaining=info.get("viruses_remaining", 0),
        frame_count=info.get("frame_count", 0),
        pill_count=info.get("pill_count", 0),
        level=info.get("level", 0),
    )


def demo_board() -> BoardState:
    """Create a demo board for testing."""
    board = np.full((BOARD_HEIGHT, BOARD_WIDTH), TILE_EMPTY, dtype=np.uint8)
    
    # Add some viruses
    board[14, 2] = TILE_VIRUS | COLOR_RED
    board[14, 5] = TILE_VIRUS | COLOR_BLUE
    board[13, 3] = TILE_VIRUS | COLOR_YELLOW
    board[12, 6] = TILE_VIRUS | COLOR_RED
    board[11, 1] = TILE_VIRUS | COLOR_BLUE
    board[10, 4] = TILE_VIRUS | COLOR_YELLOW
    
    # Add some pill pieces
    board[15, 0] = TILE_LEFT | COLOR_RED
    board[15, 1] = TILE_RIGHT | COLOR_BLUE
    board[15, 3] = TILE_SINGLE | COLOR_YELLOW
    board[14, 0] = TILE_BOTTOM | COLOR_YELLOW
    board[13, 0] = TILE_TOP | COLOR_YELLOW
    
    return BoardState(
        board=board,
        falling_row=4,
        falling_col=3,
        falling_orient=1,  # Horizontal
        falling_color_l=COLOR_RED,
        falling_color_r=COLOR_BLUE,
        preview_color_l=COLOR_YELLOW,
        preview_color_r=COLOR_RED,
        viruses_remaining=6,
        frame_count=1234,
        pill_count=5,
        level=5,
    )


def demo():
    """Demo the board viewer."""
    if not RICH_AVAILABLE:
        print("Rich not available. Install with: pip install rich")
        return
    
    console = Console()
    state = demo_board()
    panel = render_board_panel(state, title="Dr. Mario Board Viewer")
    console.print(panel)
    
    # Also show a legend
    legend = Table(title="Legend", show_header=False, box=None)
    legend.add_column("Symbol", width=4)
    legend.add_column("Meaning", width=20)
    legend.add_row("☣ ", "Virus")
    legend.add_row("██", "Pill half")
    legend.add_row("▒▒", "Falling pill")
    legend.add_row("▓▓", "Single piece")
    legend.add_row("  ", "Empty")
    console.print(legend)


if __name__ == "__main__":
    demo()
