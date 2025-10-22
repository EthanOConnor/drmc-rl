from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, List


class EventBus:
    """Minimal publish/subscribe helper used by adapters and diagnostics."""

    def __init__(self) -> None:
        self._subs: DefaultDict[str, List[Callable[[Dict[str, Any]], None]]] = defaultdict(list)

    def on(self, event_type: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._subs[event_type].append(callback)

    def emit(self, event_type: str, **payload: Any) -> None:
        for callback in list(self._subs[event_type]):
            callback({"type": event_type, **payload})
