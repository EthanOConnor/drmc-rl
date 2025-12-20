"""Rich-based TUI for training visualization.

Uses Rich's Live display with panels and tables for real-time training metrics.
Simpler and more robust than Tkinter, runs in the terminal without requiring X11.

Key features:
- Real-time reward/loss metrics with sparklines
- Episode statistics table  
- Hyperparameter display
- Keyboard shortcuts via threading (save, quit)
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None  # type: ignore
    Live = None  # type: ignore


# Sparkline characters (block height representation)
SPARKLINE_CHARS = " ▁▂▃▄▅▆▇█"


def _curriculum_goal(level: int) -> str:
    """Human-readable label for a synthetic curriculum level."""

    lvl = int(level)
    if lvl <= -4:
        # Match-count stages: `envs/retro/drmario_env.py` maps `level -> match_target`
        # as `max(1, 16 + level)` for `level -15..-4`.
        matches = int(max(1, 16 + lvl))  # lvl=-15 -> 1 match, lvl=-4 -> 12 matches
        plural = "" if matches == 1 else "es"
        return f"0 viruses, clear {matches} match{plural}"
    if lvl <= 0:
        viruses = int(4 + lvl)  # lvl=0 -> 4 viruses, lvl=-3 -> 1 virus
        plural = "" if viruses == 1 else "es"
        return f"{viruses} virus{plural}"
    return f"level {lvl}"


def _format_env_level_counts(counts: Dict[Any, Any], *, max_items: int = 8) -> str:
    items: List[Tuple[int, int]] = []
    for key, value in counts.items():
        try:
            lvl = int(key)
            cnt = int(value)
        except Exception:
            continue
        items.append((lvl, cnt))
    items.sort(key=lambda kv: kv[0])
    if not items:
        return ""
    parts = [f"{lvl}x{cnt}" for lvl, cnt in items[:max_items]]
    if len(items) > max_items:
        parts.append("...")
    return " ".join(parts)


def _sparkline(values: List[float], width: int = 20) -> str:
    """Generate a text-based sparkline from values."""
    if not values:
        return " " * width
    
    # Take last `width` values
    recent = values[-width:]
    if len(recent) < width:
        recent = [0.0] * (width - len(recent)) + recent
    
    # Normalize to 0-8 range for character selection
    min_val, max_val = min(recent), max(recent)
    if max_val - min_val < 1e-6:
        return SPARKLINE_CHARS[4] * width  # flat line
    
    normalized = [(v - min_val) / (max_val - min_val) for v in recent]
    chars = [SPARKLINE_CHARS[int(n * 8)] for n in normalized]
    return "".join(chars)


@dataclass
class TrainingMetrics:
    """Container for training metrics history."""
    
    # Episode-level metrics
    episode_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    episode_lengths: Deque[int] = field(default_factory=lambda: deque(maxlen=1000))
    
    # Training metrics
    policy_losses: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    value_losses: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    entropies: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    
    # Reward components (last episode)
    reward_components: Dict[str, float] = field(default_factory=dict)
    
    # Timing
    steps_per_second: float = 0.0
    total_steps: int = 0
    total_episodes: int = 0
    
    # Hyperparameters (for display)
    hyperparams: Dict[str, Any] = field(default_factory=dict)

    # Curriculum (optional)
    curriculum_level: Optional[int] = None
    curriculum_goal: str = ""
    curriculum_rate: float = 0.0
    curriculum_window_n: int = 0
    curriculum_window_size: int = 0
    curriculum_window_successes: Optional[int] = None
    curriculum_confidence_sigmas: Optional[float] = None
    curriculum_confidence_lower_bound: Optional[float] = None
    curriculum_episodes_total: int = 0
    curriculum_start_level: Optional[int] = None
    curriculum_max_level: Optional[int] = None
    curriculum_success_threshold: Optional[float] = None
    curriculum_env_levels: str = ""
    curriculum_advanced: str = ""
    curriculum_mode: str = ""
    curriculum_stage_index: Optional[int] = None
    curriculum_stage_count: Optional[int] = None
    curriculum_probe_threshold: Optional[float] = None
    curriculum_time_budget_frames: Optional[int] = None
    curriculum_time_mean_frames: Optional[float] = None
    curriculum_time_mad_frames: Optional[float] = None
    
    def avg_reward(self, last_n: int = 100) -> float:
        """Average reward over last N episodes."""
        recent = list(self.episode_rewards)[-last_n:]
        return sum(recent) / len(recent) if recent else 0.0
    
    def avg_length(self, last_n: int = 100) -> int:
        """Average episode length over last N episodes."""
        recent = list(self.episode_lengths)[-last_n:]
        return int(sum(recent) / len(recent)) if recent else 0


class TrainingTUI:
    """Rich-based terminal UI for training visualization.
    
    Usage:
        tui = TrainingTUI()
        with tui:
            for batch in training_loop():
                tui.update(metrics)
                if tui.should_stop:
                    break
    """
    
    def __init__(
        self,
        experiment_name: str = "DrMC-RL Training",
        refresh_rate: float = 4.0,
    ):
        if not RICH_AVAILABLE:
            raise ImportError("Rich is required for TUI. Install with: pip install rich")
        
        self.experiment_name = experiment_name
        self.refresh_rate = refresh_rate
        self.metrics = TrainingMetrics()
        self.console = Console()
        self._live: Optional[Live] = None
        self._should_stop = threading.Event()
        self._save_requested = threading.Event()
        self._status_message = ""
        
    @property
    def should_stop(self) -> bool:
        """Check if user requested stop (Ctrl+C or 'q')."""
        return self._should_stop.is_set()
    
    @property
    def save_requested(self) -> bool:
        """Check if user requested checkpoint save."""
        if self._save_requested.is_set():
            self._save_requested.clear()
            return True
        return False
    
    def set_status(self, message: str) -> None:
        """Set a temporary status message."""
        self._status_message = message
    
    def __enter__(self) -> "TrainingTUI":
        """Start the live display."""
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=self.refresh_rate,
            screen=False,  # Don't use alternate screen
        )
        self._live.__enter__()
        return self
    
    def __exit__(self, *args) -> None:
        """Stop the live display."""
        if self._live:
            self._live.__exit__(*args)
            self._live = None
    
    def update(
        self,
        episode_reward: Optional[float] = None,
        episode_length: Optional[int] = None,
        policy_loss: Optional[float] = None,
        value_loss: Optional[float] = None,
        entropy: Optional[float] = None,
        reward_components: Optional[Dict[str, float]] = None,
        steps_per_second: Optional[float] = None,
        total_steps: Optional[int] = None,
        curriculum: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Update metrics and refresh display."""
        m = self.metrics
        
        if episode_reward is not None:
            m.episode_rewards.append(episode_reward)
            m.total_episodes += 1
        if episode_length is not None:
            m.episode_lengths.append(episode_length)
        if policy_loss is not None:
            m.policy_losses.append(policy_loss)
        if value_loss is not None:
            m.value_losses.append(value_loss)
        if entropy is not None:
            m.entropies.append(entropy)
        if reward_components is not None:
            m.reward_components = reward_components
        if steps_per_second is not None:
            m.steps_per_second = steps_per_second
        if total_steps is not None:
            m.total_steps = total_steps

        if curriculum is not None:
            self._update_curriculum(curriculum)
        
        # Update live display
        if self._live:
            self._live.update(self._render())

    def _update_curriculum(self, curriculum: Dict[str, Any]) -> None:
        m = self.metrics

        def _to_int(value: Any) -> Optional[int]:
            if value is None:
                return None
            try:
                return int(value)
            except Exception:
                return None

        def _to_float(value: Any) -> Optional[float]:
            if value is None:
                return None
            try:
                return float(value)
            except Exception:
                return None

        lvl = _to_int(curriculum.get("current_level", curriculum.get("curriculum/current_level")))
        if lvl is not None:
            m.curriculum_level = int(lvl)
            m.curriculum_goal = _curriculum_goal(int(lvl))

        rate = _to_float(curriculum.get("rate_current", curriculum.get("curriculum/rate_current")))
        if rate is not None:
            m.curriculum_rate = float(rate)

        window_n = _to_int(curriculum.get("window_n", curriculum.get("curriculum/window_n")))
        if window_n is not None:
            m.curriculum_window_n = int(window_n)

        window_size = _to_int(curriculum.get("window_size", curriculum.get("curriculum/window_size")))
        if window_size is not None:
            m.curriculum_window_size = int(window_size)

        window_successes = _to_int(curriculum.get("window_successes", curriculum.get("curriculum/window_successes")))
        if window_successes is not None:
            m.curriculum_window_successes = int(window_successes)

        conf_sigmas = _to_float(curriculum.get("confidence_sigmas", curriculum.get("curriculum/confidence_sigmas")))
        if conf_sigmas is not None:
            m.curriculum_confidence_sigmas = float(conf_sigmas)

        conf_lb = _to_float(
            curriculum.get("confidence_lower_bound", curriculum.get("curriculum/confidence_lower_bound"))
        )
        if conf_lb is not None:
            m.curriculum_confidence_lower_bound = float(conf_lb)

        total_eps = _to_int(
            curriculum.get("episodes_current_total", curriculum.get("curriculum/episodes_current_total"))
        )
        if total_eps is not None:
            m.curriculum_episodes_total = int(total_eps)

        start_level = _to_int(curriculum.get("start_level", curriculum.get("curriculum/start_level")))
        if start_level is not None:
            m.curriculum_start_level = int(start_level)

        max_level = _to_int(curriculum.get("max_level", curriculum.get("curriculum/max_level")))
        if max_level is not None:
            m.curriculum_max_level = int(max_level)

        threshold = _to_float(curriculum.get("success_threshold", curriculum.get("curriculum/success_threshold")))
        if threshold is not None:
            m.curriculum_success_threshold = float(threshold)

        mode = curriculum.get("mode", curriculum.get("curriculum/mode"))
        if isinstance(mode, str):
            m.curriculum_mode = str(mode)

        stage_index = _to_int(curriculum.get("stage_index", curriculum.get("curriculum/stage_index")))
        if stage_index is not None:
            m.curriculum_stage_index = int(stage_index)

        stage_count = _to_int(curriculum.get("stage_count", curriculum.get("curriculum/stage_count")))
        if stage_count is not None:
            m.curriculum_stage_count = int(stage_count)

        probe_threshold = _to_float(
            curriculum.get("probe_threshold", curriculum.get("curriculum/probe_threshold"))
        )
        if probe_threshold is not None and float(probe_threshold) > 0.0:
            m.curriculum_probe_threshold = float(probe_threshold)

        counts = curriculum.get("env_level_counts", curriculum.get("curriculum/env_level_counts"))
        if isinstance(counts, dict):
            m.curriculum_env_levels = _format_env_level_counts(counts)

        time_budget = _to_int(curriculum.get("time_budget_frames", curriculum.get("curriculum/time_budget_frames")))
        if time_budget is not None:
            m.curriculum_time_budget_frames = int(time_budget)

        time_mean = _to_float(curriculum.get("time_mean_frames", curriculum.get("curriculum/time_mean_frames")))
        if time_mean is not None:
            m.curriculum_time_mean_frames = float(time_mean)

        time_mad = _to_float(curriculum.get("time_mad_frames", curriculum.get("curriculum/time_mad_frames")))
        if time_mad is not None:
            m.curriculum_time_mad_frames = float(time_mad)

        adv_to = _to_int(curriculum.get("advanced_to", curriculum.get("curriculum/advanced_to")))
        if adv_to is not None:
            adv_from = _to_int(curriculum.get("advanced_from", curriculum.get("curriculum/advanced_from")))
            if adv_from is not None:
                m.curriculum_advanced = f"{adv_from} → {adv_to}"
            else:
                m.curriculum_advanced = f"→ {adv_to}"
    
    def set_hyperparams(self, hyperparams: Dict[str, Any]) -> None:
        """Set hyperparameters for display."""
        self.metrics.hyperparams = hyperparams
    
    def _render(self) -> Group:
        """Render the full TUI layout."""
        m = self.metrics
        
        # Header with experiment info
        header = self._render_header()
        
        # Main metrics panel
        metrics_panel = self._render_metrics()
        
        # Training stats table
        stats_table = self._render_stats()
        
        # Hyperparameters
        hparams_panel = self._render_hyperparams()
        
        # Status/footer
        footer = self._render_footer()
        
        return Group(header, metrics_panel, stats_table, hparams_panel, footer)
    
    def _render_header(self) -> Panel:
        """Render header with experiment name and progress."""
        m = self.metrics
        title = Text(self.experiment_name, style="bold cyan")
        
        # Progress if we know target
        target = m.hyperparams.get("total_steps", 0)
        if target:
            pct = min(100, m.total_steps / target * 100)
            progress_text = f"Steps: {m.total_steps:,} / {target:,} ({pct:.1f}%)"
        else:
            progress_text = f"Steps: {m.total_steps:,}"
        
        return Panel(
            f"{title}\n{progress_text}",
            title="[bold]DrMC-RL[/bold]",
            border_style="blue",
        )
    
    def _render_metrics(self) -> Panel:
        """Render main metrics with sparklines."""
        m = self.metrics
        
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Metric", width=15)
        table.add_column("Value", width=12, justify="right")
        table.add_column("Avg100", width=12, justify="right")
        table.add_column("Trend", width=22)
        
        # Episode reward
        last_reward = m.episode_rewards[-1] if m.episode_rewards else 0.0
        avg_reward = m.avg_reward(100)
        reward_spark = _sparkline(list(m.episode_rewards), 20)
        table.add_row(
            "Episode Reward",
            f"{last_reward:.1f}",
            f"{avg_reward:.1f}",
            reward_spark,
        )
        
        # Policy loss
        last_ploss = m.policy_losses[-1] if m.policy_losses else 0.0
        ploss_spark = _sparkline(list(m.policy_losses), 20)
        table.add_row(
            "Policy Loss",
            f"{last_ploss:.4f}",
            "",
            ploss_spark,
        )
        
        # Value loss
        last_vloss = m.value_losses[-1] if m.value_losses else 0.0
        vloss_spark = _sparkline(list(m.value_losses), 20)
        table.add_row(
            "Value Loss",
            f"{last_vloss:.4f}",
            "",
            vloss_spark,
        )
        
        # Entropy
        last_ent = m.entropies[-1] if m.entropies else 0.0
        ent_spark = _sparkline(list(m.entropies), 20)
        table.add_row(
            "Entropy",
            f"{last_ent:.4f}",
            "",
            ent_spark,
        )
        
        return Panel(table, title="[bold]Training Metrics[/bold]", border_style="green")
    
    def _render_stats(self) -> Panel:
        """Render training statistics."""
        m = self.metrics
        
        table = Table(show_header=False, box=None)
        table.add_column("Key", width=15, style="dim")
        table.add_column("Value", width=15)
        table.add_column("Key", width=15, style="dim")
        table.add_column("Value", width=15)
        
        table.add_row(
            "Episodes", f"{m.total_episodes:,}",
            "Steps/sec", f"{m.steps_per_second:.0f}",
        )
        table.add_row(
            "Avg Length", f"{m.avg_length(100)}",
            "Total Steps", f"{m.total_steps:,}",
        )

        if m.curriculum_level is not None:
            level_label = str(int(m.curriculum_level))
            if m.curriculum_start_level is not None and m.curriculum_max_level is not None:
                level_label = (
                    f"{int(m.curriculum_level)} "
                    f"({int(m.curriculum_start_level)}→{int(m.curriculum_max_level)})"
                )

            stage_label = ""
            if m.curriculum_stage_index is not None and m.curriculum_stage_count is not None:
                stage_label = f"{int(m.curriculum_stage_index) + 1}/{int(m.curriculum_stage_count)}"
            table.add_row("Curriculum", level_label, "Stage", stage_label)
            table.add_row("Goal", m.curriculum_goal, "Mode", m.curriculum_mode)

            rate_pct = float(m.curriculum_rate) * 100.0
            threshold = m.curriculum_success_threshold
            rate_label = f"{rate_pct:.1f}%"
            if threshold is not None and float(threshold) > 0.0:
                rate_label = f"{rate_pct:.1f}% / {float(threshold) * 100.0:.1f}%"
            window_label = (
                f"{int(m.curriculum_window_n)}/{int(m.curriculum_window_size)}"
                if int(m.curriculum_window_size) > 0
                else f"{int(m.curriculum_window_n)}"
            )
            table.add_row("Success", rate_label, "Window", window_label)

            if m.curriculum_confidence_lower_bound is not None and m.curriculum_confidence_sigmas is not None:
                lb_pct = float(m.curriculum_confidence_lower_bound) * 100.0
                sig = float(m.curriculum_confidence_sigmas)
                table.add_row("Wilson LB", f"{lb_pct:.1f}%", "Sigmas", f"{sig:.1f}")

            if (
                m.curriculum_time_budget_frames is not None
                or m.curriculum_time_mean_frames is not None
                or m.curriculum_time_mad_frames is not None
            ):
                budget_label = (
                    f"{int(m.curriculum_time_budget_frames)}"
                    if m.curriculum_time_budget_frames is not None
                    else "-"
                )
                mean_label = (
                    f"{float(m.curriculum_time_mean_frames):.0f}"
                    if m.curriculum_time_mean_frames is not None
                    else "-"
                )
                mad_label = (
                    f"{float(m.curriculum_time_mad_frames):.0f}"
                    if m.curriculum_time_mad_frames is not None
                    else "-"
                )
                table.add_row("Time Budget", budget_label, "Mean±MAD", f"{mean_label}±{mad_label}")

            env_levels = m.curriculum_env_levels or ""
            table.add_row("Env Levels", env_levels, "Seen", f"{int(m.curriculum_episodes_total)}")
            if m.curriculum_advanced:
                table.add_row("Advanced", m.curriculum_advanced, "", "")
        
        # Reward components if available
        if m.reward_components:
            for key, val in list(m.reward_components.items())[:4]:
                table.add_row(key, f"{val:+.2f}", "", "")
        
        return Panel(table, title="[bold]Statistics[/bold]", border_style="yellow")
    
    def _render_hyperparams(self) -> Panel:
        """Render hyperparameters."""
        hp = self.metrics.hyperparams
        if not hp:
            return Panel("[dim]No hyperparameters set[/dim]", title="Hyperparameters")
        
        # Format as compact key=value pairs
        items = [f"{k}={v}" for k, v in hp.items() if k != "total_steps"]
        text = "  ".join(items[:8])  # Limit to 8 items
        
        return Panel(text, title="[bold]Hyperparameters[/bold]", border_style="dim")
    
    def _render_footer(self) -> Text:
        """Render footer with controls and status."""
        if self._status_message:
            return Text(f"Status: {self._status_message}", style="yellow")
        return Text("Press Ctrl+C to stop", style="dim")


def demo():
    """Demo the TUI with fake data."""
    import random
    
    tui = TrainingTUI(experiment_name="Demo Experiment")
    tui.set_hyperparams({
        "algo": "ppo",
        "lr": 3e-4,
        "gamma": 0.99,
        "total_steps": 1_000_000,
    })
    
    with tui:
        for i in range(1000):
            # Simulate training
            time.sleep(0.05)
            
            # Generate fake metrics
            reward = 100 + i * 0.1 + random.gauss(0, 10)
            tui.update(
                episode_reward=reward,
                episode_length=random.randint(100, 500),
                policy_loss=max(0, 0.5 - i * 0.0005 + random.gauss(0, 0.05)),
                value_loss=max(0, 1.0 - i * 0.001 + random.gauss(0, 0.1)),
                entropy=max(0, 2.0 - i * 0.002 + random.gauss(0, 0.1)),
                steps_per_second=800 + random.gauss(0, 50),
                total_steps=i * 1024,
            )
            
            if tui.should_stop:
                break


if __name__ == "__main__":
    demo()
