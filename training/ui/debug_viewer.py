"""Interactive debug viewer for Dr. Mario episodes.

Combines board visualization with step controls for debugging and demos.
Uses Rich Live display for real-time updates.

Features:
- Board state visualization
- Step-by-step execution
- Action display
- Episode stats
- Single-step mode
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .board_viewer import BoardState, board_from_env_info, render_board_panel


# Action names for display
ACTION_NAMES = {
    0: "NOOP",
    1: "←",
    2: "→",
    3: "↓",
    4: "A (CW)",
    5: "B (CCW)",
    6: "←←",
    7: "→→",
    8: "↓↓",
    9: "A+B",
}


@dataclass
class DebugViewerState:
    """State for the debug viewer."""
    
    # Current observation
    obs: Optional[np.ndarray] = None
    
    # Current info dict from env
    info: Dict[str, Any] = field(default_factory=dict)
    
    # Episode history
    episode_reward: float = 0.0
    episode_length: int = 0
    total_episodes: int = 0
    
    # Action history (last N)
    action_history: List[int] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)
    
    # Control state
    paused: bool = True
    single_step: bool = False
    step_delay: float = 0.1  # seconds between steps
    
    # Status message
    status: str = "Ready"


class DebugViewer:
    """Interactive debug viewer for Dr. Mario.
    
    Usage:
        viewer = DebugViewer()
        env = gym.make("DrMario-v0")
        obs, info = env.reset()
        viewer.update(obs, info)
        
        with viewer:
            for step in range(1000):
                if viewer.paused:
                    continue
                action = policy(obs)
                obs, reward, done, truncated, info = env.step(action)
                viewer.update(obs, info, action=action, reward=reward)
                if done:
                    obs, info = env.reset()
    """
    
    def __init__(
        self,
        title: str = "Dr. Mario Debug Viewer",
        refresh_rate: float = 10.0,
        action_history_size: int = 10,
    ):
        if not RICH_AVAILABLE:
            raise ImportError("Rich is required. Install: pip install rich")
        
        self.title = title
        self.refresh_rate = refresh_rate
        self.action_history_size = action_history_size
        self.console = Console()
        self.state = DebugViewerState()
        self._live: Optional[Live] = None
    
    @property
    def paused(self) -> bool:
        return self.state.paused
    
    def toggle_pause(self) -> None:
        """Toggle pause state."""
        self.state.paused = not self.state.paused
        self.state.status = "Paused" if self.state.paused else "Running"
    
    def single_step_mode(self) -> None:
        """Request a single step."""
        self.state.single_step = True
        self.state.paused = True
    
    def set_delay(self, delay: float) -> None:
        """Set step delay in seconds."""
        self.state.step_delay = max(0.0, delay)
    
    def update(
        self,
        obs: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
        action: Optional[int] = None,
        reward: Optional[float] = None,
        done: bool = False,
    ) -> None:
        """Update viewer state."""
        s = self.state
        
        if obs is not None:
            s.obs = obs
        if info is not None:
            s.info = info
        
        if action is not None:
            s.action_history.append(action)
            if len(s.action_history) > self.action_history_size:
                s.action_history.pop(0)
        
        if reward is not None:
            s.episode_reward += reward
            s.reward_history.append(reward)
            if len(s.reward_history) > self.action_history_size:
                s.reward_history.pop(0)
        
        s.episode_length += 1
        
        if done:
            s.total_episodes += 1
            s.episode_reward = 0.0
            s.episode_length = 0
            s.status = "Episode done"
        
        # Update live display
        if self._live:
            self._live.update(self._render())
    
    def __enter__(self) -> "DebugViewer":
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=self.refresh_rate,
            screen=False,
        )
        self._live.__enter__()
        return self
    
    def __exit__(self, *args) -> None:
        if self._live:
            self._live.__exit__(*args)
            self._live = None
    
    def _render(self) -> Group:
        """Render the debug viewer layout."""
        s = self.state
        
        # Board panel
        board_state = board_from_env_info(s.info)
        board_panel = render_board_panel(board_state, title="Board")
        
        # Stats panel
        stats_panel = self._render_stats()
        
        # Action history panel  
        action_panel = self._render_actions()
        
        # Controls help
        controls = self._render_controls()
        
        return Group(board_panel, stats_panel, action_panel, controls)
    
    def _render_stats(self) -> Panel:
        """Render episode statistics."""
        s = self.state
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="dim")
        table.add_column("Value")
        
        table.add_row("Episode", f"{s.total_episodes + 1}")
        table.add_row("Step", f"{s.episode_length}")
        table.add_row("Reward", f"{s.episode_reward:.1f}")
        table.add_row("Status", s.status)
        
        # Info fields
        if s.info:
            for key in ["viruses_remaining", "frame_count", "pill_count"]:
                if key in s.info:
                    table.add_row(key, f"{s.info[key]}")
        
        return Panel(table, title="[bold]Stats[/bold]", border_style="green")
    
    def _render_actions(self) -> Panel:
        """Render action history."""
        s = self.state
        
        if not s.action_history:
            content = Text("[dim]No actions yet[/dim]")
        else:
            lines = []
            for i, (action, reward) in enumerate(zip(
                reversed(s.action_history),
                reversed(s.reward_history) if s.reward_history else [0.0] * len(s.action_history),
            )):
                action_name = ACTION_NAMES.get(action, f"#{action}")
                reward_str = f"{reward:+.1f}" if reward != 0 else ""
                lines.append(f"{action_name:8} {reward_str}")
            content = Text("\n".join(lines[:5]))  # Show last 5
        
        return Panel(content, title="[bold]Actions[/bold]", border_style="yellow")
    
    def _render_controls(self) -> Text:
        """Render control hints."""
        status = "⏸ PAUSED" if self.state.paused else "▶ RUNNING"
        controls = f"{status} | [Space] Pause | [N] Step | [+/-] Speed"
        return Text(controls, style="dim")


def demo():
    """Demo the debug viewer with random actions."""
    import random
    
    viewer = DebugViewer()
    
    # Simulate environment
    from .board_viewer import demo_board
    
    with viewer:
        board_state = demo_board()
        info = {
            "board": board_state.board.flatten().tolist(),
            "viruses_remaining": board_state.viruses_remaining,
            "frame_count": 0,
            "pill_count": 0,
            "level": 5,
        }
        
        viewer.update(info=info)
        viewer.state.paused = False
        viewer.state.status = "Demo mode"
        
        for step in range(100):
            time.sleep(0.2)
            
            action = random.randint(0, 9)
            reward = random.gauss(0, 0.5)
            info["frame_count"] = step
            
            viewer.update(
                info=info,
                action=action,
                reward=reward,
                done=(step > 0 and step % 30 == 0),
            )


if __name__ == "__main__":
    demo()
