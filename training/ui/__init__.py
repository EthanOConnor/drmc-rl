"""Training UI module.

Provides terminal-based visualization for training runs.
"""
from __future__ import annotations

from .tui import TrainingTUI, TrainingMetrics, RICH_AVAILABLE

__all__ = ["TrainingTUI", "TrainingMetrics", "RICH_AVAILABLE"]
