"""TUI event handler for Rich-based training visualization.

Bridges the training EventBus to the Rich TUI display.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..diagnostics.event_bus import EventBus

try:
    from .tui import TrainingTUI, RICH_AVAILABLE
except ImportError:
    RICH_AVAILABLE = False
    TrainingTUI = None  # type: ignore


class TUIEventHandler:
    """Bridges EventBus to TrainingTUI.
    
    Subscribes to training events and updates TUI display.
    
    Usage:
        tui = TrainingTUI(experiment_name="DrMC-RL")
        handler = TUIEventHandler(event_bus, tui)
        
        with tui:
            adapter.train_forever()
    """
    
    def __init__(self, event_bus: EventBus, tui: "TrainingTUI") -> None:
        self.event_bus = event_bus
        self.tui = tui
        
        # Track metrics for TUI updates
        self._episode_count = 0
        self._total_steps = 0
        self._last_episode_reward = 0.0
        self._last_episode_length = 0
        
        # Subscribe to events
        event_bus.on("episode_end", self._on_episode_end)
        event_bus.on("update_end", self._on_update_end)
        event_bus.on("rollout_end", self._on_rollout_end)
        event_bus.on("checkpoint", self._on_checkpoint)
    
    def _on_episode_end(self, payload: Dict[str, Any]) -> None:
        """Handle episode completion."""
        step = payload.get("step", 0)
        ret = payload.get("ret", 0.0)
        length = payload.get("len", 0)
        
        self._episode_count += 1
        self._last_episode_reward = ret
        self._last_episode_length = length
        self._total_steps = step
        
        # Update TUI
        self.tui.update(
            episode_reward=ret,
            episode_length=length,
            total_steps=step,
        )
    
    def _on_update_end(self, payload: Dict[str, Any]) -> None:
        """Handle policy update completion."""
        step = payload.get("step", 0)
        self._total_steps = step
        
        # Extract key metrics
        policy_loss = payload.get("loss/policy")
        value_loss = payload.get("loss/value")
        entropy = payload.get("policy/entropy")
        sps = payload.get("perf/sps")
        
        self.tui.update(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
            steps_per_second=sps,
            total_steps=step,
        )
    
    def _on_rollout_end(self, payload: Dict[str, Any]) -> None:
        """Handle rollout completion."""
        step = payload.get("step", 0)
        self._total_steps = step
        # Rollout timing could be displayed if needed
    
    def _on_checkpoint(self, payload: Dict[str, Any]) -> None:
        """Handle checkpoint save."""
        path = payload.get("path", "")
        self.tui.set_status(f"Checkpoint saved: {path}")


def create_tui_handler(
    event_bus: EventBus,
    experiment_name: str = "DrMC-RL Training",
    hyperparams: Optional[Dict[str, Any]] = None,
) -> Optional["TUIEventHandler"]:
    """Create TUI and handler if Rich is available.
    
    Args:
        event_bus: EventBus to subscribe to
        experiment_name: Display name for the experiment
        hyperparams: Hyperparameters to display in TUI
    
    Returns:
        TUIEventHandler or None if Rich not available
    """
    if not RICH_AVAILABLE or TrainingTUI is None:
        return None
    
    tui = TrainingTUI(experiment_name=experiment_name)
    if hyperparams:
        tui.set_hyperparams(hyperparams)
    
    handler = TUIEventHandler(event_bus, tui)
    return handler
