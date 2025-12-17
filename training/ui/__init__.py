"""Training UI module.

Provides terminal-based visualization for training runs.

Direct execution of submodules:
    python -m training.ui.board_viewer
    python -m training.ui.debug_viewer
    python -m training.ui.tui
"""
from __future__ import annotations

# Core exports - always available
from .tui import TrainingTUI, TrainingMetrics, RICH_AVAILABLE

# Lazy imports for optional components to avoid circular import warnings
# when running submodules directly (python -m training.ui.board_viewer)
def __getattr__(name: str):
    """Lazy loading for submodule components."""
    if name in ("BoardState", "render_board_panel", "render_board_text", 
                "board_from_env_info", "demo_board"):
        from . import board_viewer
        return getattr(board_viewer, name)
    if name == "DebugViewer":
        from . import debug_viewer
        return debug_viewer.DebugViewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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



