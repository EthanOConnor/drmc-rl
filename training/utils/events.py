"""Simple event bus for training callbacks."""
from __future__ import annotations

from typing import Any, Callable, Dict, List


class EventBus:
    """Simple publish-subscribe event bus."""
    
    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
        
    def on(self, event: str) -> Callable:
        """Decorator to register event handler."""
        def decorator(func: Callable) -> Callable:
            if event not in self._handlers:
                self._handlers[event] = []
            self._handlers[event].append(func)
            return func
        return decorator
        
    def emit(self, event: str, **kwargs: Any) -> None:
        """Emit an event with keyword arguments."""
        if event in self._handlers:
            for handler in self._handlers[event]:
                try:
                    handler(**kwargs)
                except Exception as e:
                    print(f"Error in event handler for '{event}': {e}")
