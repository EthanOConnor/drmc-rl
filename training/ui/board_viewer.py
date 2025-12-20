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
    preview_rotation: Optional[int] = None
    
    # Game state
    viruses_remaining: int = 0
    frame_count: int = 0
    pill_count: int = 0
    level: int = 0


def parse_board_bytes(
    board_bytes: Union[bytes, bytearray, memoryview, np.ndarray, List[int]]
) -> np.ndarray:
    """Parse 128-byte board data into a 16x8 array.

    Accepts:
      - `bytes`/`bytearray`/`memoryview` (raw NES/RAM bytes)
      - `np.ndarray` or `List[int]` (already decoded)
    """
    if isinstance(board_bytes, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(board_bytes, dtype=np.uint8)
    else:
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
    if compact:
        horiz = "─"
        corner_tl, corner_tr = "┌", "┐"
        corner_bl, corner_br = "└", "┘"
        tee_l, tee_r = "├", "┤"
        vert = "│"
        cell_empty = " "
        cell_fill = " "
        border_row = horiz * BOARD_WIDTH
    else:
        horiz = "═"
        corner_tl, corner_tr = "╔", "╗"
        corner_bl, corner_br = "╚", "╝"
        tee_l, tee_r = "╠", "╣"
        vert = "║"
        cell_empty = "  "
        cell_fill = "██"
        border_row = horiz * (BOARD_WIDTH * 2)
    text.append(f"{corner_tl}{border_row}{corner_tr}\n", style=border_style)
    
    # Preview area (2 rows above board)
    if show_preview and state.preview_color_l is not None:
        preview_rows = 2
        center_col = BOARD_WIDTH // 2
        center_row = preview_rows - 1
        rot = int(state.preview_rotation or 0) & 0x03
        placements = {
            0: ((center_row, center_col - 1), (center_row, center_col)),
            1: ((center_row, center_col), (center_row - 1, center_col)),
            2: ((center_row, center_col), (center_row, center_col - 1)),
            3: ((center_row - 1, center_col), (center_row, center_col)),
        }
        c1 = None if state.preview_color_l is None else int(state.preview_color_l) & 0x03
        c2 = None if state.preview_color_r is None else int(state.preview_color_r) & 0x03
        coords = placements.get(rot, placements[0])
        preview_cells: Dict[Tuple[int, int], int] = {}
        if c1 is not None:
            preview_cells[coords[0]] = c1
        if c2 is not None:
            preview_cells[coords[1]] = c2

        for rr in range(preview_rows):
            text.append(vert, style=border_style)
            for cc in range(BOARD_WIDTH):
                color_idx = preview_cells.get((rr, cc))
                if color_idx is None:
                    text.append(cell_empty, style=Style(bgcolor=BG_EMPTY))
                    continue
                color = COLORS.get(int(color_idx), "white")
                # Fill the cell to avoid gaps from terminal glyph rendering.
                text.append(cell_fill, style=Style(color=color, bgcolor=color))
            text.append(f"{vert}\n", style=border_style)
        text.append(f"{tee_l}{border_row}{tee_r}\n", style=border_style)
    
    # Board rows
    for row in range(BOARD_HEIGHT):
        text.append(vert, style=border_style)

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
                    elif row == state.falling_row - 1 and col == state.falling_col:
                        is_falling = True
                        color_idx = state.falling_color_r
                else:  # Horizontal
                    if row == state.falling_row and col == state.falling_col:
                        is_falling = True
                        color_idx = state.falling_color_l
                    elif row == state.falling_row and col == state.falling_col + 1:
                        is_falling = True
                        color_idx = state.falling_color_r

            if compact:
                if color_idx is not None:
                    color = COLORS.get(int(color_idx), "white")
                    if tile_type == "virus":
                        style = Style(color=color, bgcolor=BG_VIRUS)
                        cell_char = "•"
                    else:
                        style = Style(color=color, bgcolor=color, dim=is_falling)
                        cell_char = cell_fill
                else:
                    style = Style(color="grey30", bgcolor=BG_EMPTY)
                    cell_char = cell_empty
                text.append(cell_char, style=style)
                continue

            # Style based on tile content (non-compact)
            if color_idx is not None:
                color = COLORS.get(color_idx, "white")
                if tile_type == "virus":
                    style = Style(color=color, bold=True, bgcolor=BG_VIRUS)
                elif is_falling:
                    # Use background fill to avoid glyph “seams” when adjacent
                    # pill halves have different colors.
                    style = Style(color=color, bgcolor=color, bold=True)
                    char = CHAR_FALLING
                else:
                    # Fill the full cell with background color. This avoids a
                    # common terminal/font artifact where half-block glyphs
                    # (▌▐▀▄) leave “gaps” that show the terminal background.
                    style = Style(color=color, bgcolor=color)
            else:
                style = Style(color="grey30", bgcolor=BG_EMPTY)

            text.append(char, style=style)

        text.append(f"{vert}\n", style=border_style)
    
    # Bottom border
    text.append(f"{corner_bl}{border_row}{corner_br}", style=border_style)
    
    # Stats line
    if show_stats:
        stats_style = Style(color="cyan", dim=True)
        text.append(
            f"\n L:{state.level} V:{state.viruses_remaining} "
            f"P:{state.pill_count} F:{state.frame_count}",
            style=stats_style,
        )
    
    return text


def render_board_panel(
    state: BoardState,
    title: str = "Dr. Mario",
    show_subtitle: bool = True,
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
    if show_subtitle:
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
        raw_ram = info.get("raw_ram")
        if raw_ram is not None and hasattr(raw_ram, "__len__") and len(raw_ram) >= 0x480:
            board = parse_board_bytes(raw_ram[0x400:0x480])
    
    preview_color_l = info.get("preview_color_l")
    preview_color_r = info.get("preview_color_r")
    preview_rotation = info.get("preview_rotation")
    preview = info.get("preview_pill")
    if (preview_color_l is None or preview_color_r is None) and preview is not None:
        if isinstance(preview, dict):
            preview_color_l = preview.get("first_color", preview_color_l)
            preview_color_r = preview.get("second_color", preview_color_r)
            preview_rotation = preview.get("rotation", preview_rotation)
        elif isinstance(preview, (list, tuple)) and len(preview) >= 2:
            try:
                preview_color_l = preview[0] if preview_color_l is None else preview_color_l
                preview_color_r = preview[1] if preview_color_r is None else preview_color_r
                if len(preview) >= 3 and preview_rotation is None:
                    preview_rotation = preview[2]
            except Exception:
                pass
    if (preview_color_l is None or preview_color_r is None) and "raw_ram" in info:
        raw_ram = info.get("raw_ram")
        try:
            if isinstance(raw_ram, (bytes, bytearray, memoryview)):
                preview_color_l = raw_ram[0x031A] if preview_color_l is None else preview_color_l
                preview_color_r = raw_ram[0x031B] if preview_color_r is None else preview_color_r
                preview_rotation = raw_ram[0x0322] if preview_rotation is None else preview_rotation
            elif isinstance(raw_ram, (list, tuple)) and len(raw_ram) > 0x0322:
                preview_color_l = raw_ram[0x031A] if preview_color_l is None else preview_color_l
                preview_color_r = raw_ram[0x031B] if preview_color_r is None else preview_color_r
                preview_rotation = raw_ram[0x0322] if preview_rotation is None else preview_rotation
        except Exception:
            pass

    return BoardState(
        board=board,
        falling_row=info.get("falling_pill_row"),
        falling_col=info.get("falling_pill_col"),
        falling_orient=info.get("falling_pill_orient"),
        falling_color_l=info.get("falling_color_l"),
        falling_color_r=info.get("falling_color_r"),
        preview_color_l=None if preview_color_l is None else int(preview_color_l) & 0x03,
        preview_color_r=None if preview_color_r is None else int(preview_color_r) & 0x03,
        preview_rotation=None if preview_rotation is None else int(preview_rotation) & 0x03,
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
