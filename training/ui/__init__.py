"""Training UI module.

Provides terminal-based visualization for training runs.
"""
from __future__ import annotations

from .tui import TrainingTUI, TrainingMetrics, RICH_AVAILABLE
from .board_viewer import (
    BoardState,
    render_board_panel,
    render_board_text,
    board_from_env_info,
    demo_board,
)
from .debug_viewer import DebugViewer

__all__ = [
    "TrainingTUI",
    "TrainingMetrics",
    "RICH_AVAILABLE",
    "BoardState",
    "render_board_panel",
    "render_board_text",
    "board_from_env_info",
    "demo_board",
    "DebugViewer",
]


