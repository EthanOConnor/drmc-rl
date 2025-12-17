"""Simple logging utilities for training."""
from __future__ import annotations

from typing import Any


class ConsoleLogger:
    """Basic console logger for training metrics."""
    
    def __init__(self):
        self._step = 0
        
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        self._step = step
        # In a real implementation, this would write to TensorBoard/WandB
        # For now, just store in memory
        
    def flush(self) -> None:
        """Flush any buffered logs."""
        pass
