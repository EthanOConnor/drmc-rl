from __future__ import annotations

"""Interactive Rich TUI for `training.run` with board visualization + speed control.

This is the "new runner" replacement for the old Tk-based viewer loops used by
`envs/retro/demo.py` and `training/speedrun_experiment.py`.

Design goals:
  - Works in a terminal (Rich Live).
  - Does not require editing training algorithms: speed control is applied via
    `training.envs.interactive.RateLimitedVecEnv`.
  - Provides pause/single-step and an FPS multiplier (`x` against base FPS).
"""

import select
import sys
import termios
import threading
import time
import tty
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

import numpy as np

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    RICH_AVAILABLE = False

from training.envs.interactive import PlaybackControl, RateLimitedVecEnv
from training.ui.board_viewer import board_from_env_info, render_board_panel


class _RawTerminal:
    def __init__(self) -> None:
        self._enabled = bool(sys.stdin.isatty())
        self._fd = None
        self._old = None

    def __enter__(self) -> "_RawTerminal":
        if not self._enabled:
            return self
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._enabled and self._fd is not None and self._old is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            except Exception:
                pass

    @property
    def enabled(self) -> bool:
        return self._enabled

    def poll_key(self) -> Optional[str]:
        if not self._enabled or self._fd is None:
            return None
        try:
            r, _, _ = select.select([self._fd], [], [], 0.0)
        except Exception:
            return None
        if not r:
            return None
        try:
            data = sys.stdin.read(1)
        except Exception:
            return None
        return data or None


@dataclass
class _MetricState:
    steps: int = 0
    episodes: int = 0
    last_return: float = 0.0
    last_length: int = 0
    sps: float = 0.0
    last_update: Dict[str, float] = field(default_factory=dict)
    recent_returns: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def mean_return_100(self) -> float:
        if not self.recent_returns:
            return 0.0
        return float(sum(self.recent_returns) / float(len(self.recent_returns)))

    def median_return_16(self) -> float:
        if not self.recent_returns:
            return 0.0
        window = list(self.recent_returns)[-16:]
        if not window:
            return 0.0
        window_sorted = sorted(float(x) for x in window)
        mid = len(window_sorted) // 2
        if len(window_sorted) % 2 == 1:
            return float(window_sorted[mid])
        return 0.5 * float(window_sorted[mid - 1] + window_sorted[mid])


class RunnerDebugTUI:
    def __init__(
        self,
        *,
        env: RateLimitedVecEnv,
        control: PlaybackControl,
        event_bus: Any,
        title: str = "DrMC-RL",
        refresh_hz: float = 30.0,
    ) -> None:
        if not RICH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("Rich is required for RunnerDebugTUI (pip install rich).")
        self.env = env
        self.control = control
        self._title = str(title)
        self._refresh_hz = float(max(1.0, refresh_hz))
        self._metrics = _MetricState()
        self._metrics_lock = threading.Lock()
        self._status: str = ""
        self._show_help = True

        # Subscribe to training events (emitted from the training thread).
        event_bus.on("episode_end", self._on_episode_end)
        event_bus.on("update_end", self._on_update_end)

    # ---------------------------- event handlers (training thread)
    def _on_episode_end(self, payload: Dict[str, Any]) -> None:
        with self._metrics_lock:
            self._metrics.steps = int(payload.get("step", self._metrics.steps))
            self._metrics.episodes += 1
            ret = float(payload.get("ret", 0.0))
            length = int(payload.get("len", 0))
            self._metrics.last_return = ret
            self._metrics.last_length = length
            self._metrics.recent_returns.append(ret)

    def _on_update_end(self, payload: Dict[str, Any]) -> None:
        with self._metrics_lock:
            self._metrics.steps = int(payload.get("step", self._metrics.steps))
            try:
                self._metrics.sps = float(payload.get("perf/sps", self._metrics.sps))
            except Exception:
                pass
            # Keep a compact metric snapshot.
            keep = ("loss/policy", "loss/value", "policy/entropy", "loss/kl", "vf/explained_var", "perf/sps")
            self._metrics.last_update = {
                k: float(payload[k]) for k in keep if k in payload and isinstance(payload[k], (int, float))
            }

    # ---------------------------- rendering
    def _render_perf_panel(self) -> Panel:
        ctrl = self.control.snapshot()
        perf = self.env.perf_snapshot()

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="dim")
        table.add_column("Value")

        table.add_row("paused", "yes" if ctrl["paused"] else "no")
        table.add_row("speed", "MAX" if ctrl["max_speed"] else f"{ctrl['speed_x']:.2f}x")
        table.add_row("target_hz", "-" if ctrl["max_speed"] else f"{ctrl['target_hz']:.1f}")
        table.add_row("rng", "on" if ctrl.get("rng_randomize") else "off")
        table.add_row("ui_hz", f"{self._refresh_hz:.1f}")
        table.add_row("emu_fps(step)", f"{perf.get('step_fps', 0.0):.1f}")
        table.add_row("emu_fps(total)", f"{perf.get('emu_fps', 0.0):.1f}")
        table.add_row("tau_max", f"{perf.get('tau_max', 1)}")
        table.add_row("step_ms/frame", f"{perf.get('step_ms_per_frame', 0.0):.4f}")

        infer_calls = int(perf.get("inference_calls", 0) or 0)
        if infer_calls > 0:
            table.add_row("", "")
            table.add_row("infer_calls", f"{infer_calls:,}")
            table.add_row(
                "infer_ms/frame",
                f"{perf.get('inference_ms_per_frame', 0.0):.4f} "
                f"(avg {perf.get('inference_ms_per_call', 0.0):.3f}, last {perf.get('last_inference_ms', 0.0):.3f})",
            )

        planner_calls = int(perf.get("planner_calls", 0) or 0)
        if planner_calls > 0:
            table.add_row("", "")
            table.add_row("planner_calls", f"{planner_calls:,}")
            table.add_row(
                "planner_ms/frame",
                f"{perf.get('planner_ms_per_frame', 0.0):.4f} (avg {perf.get('planner_ms_per_call', 0.0):.3f})",
            )
            table.add_row(
                "planner_build",
                f"{int(perf.get('planner_build_calls', 0) or 0):,} calls  "
                f"avg {perf.get('planner_build_ms_per_call', 0.0):.3f}  "
                f"last {perf.get('last_planner_build_ms', 0.0):.3f}",
            )
            table.add_row(
                "planner_plan",
                f"{int(perf.get('planner_plan_calls', 0) or 0):,} calls  "
                f"avg {perf.get('planner_plan_ms_per_call', 0.0):.3f}  "
                f"last {perf.get('last_planner_plan_ms', 0.0):.3f}",
            )

        update_calls = int(perf.get("update_calls", 0) or 0)
        if update_calls > 0:
            table.add_row("", "")
            table.add_row(
                "update_last",
                f"{perf.get('update_sec_last', 0.0):.3f}s "
                f"({perf.get('update_ms_per_frame', 0.0):.4f} ms/frame)",
            )
            table.add_row("update_ms/frame", f"{perf.get('update_ms_per_frame_avg', 0.0):.4f} (avg)")
        return Panel(table, title="[bold]Perf[/bold]", border_style="green")

    def _render_learning_panel(self) -> Panel:
        perf = self.env.perf_snapshot()
        info0 = self.env.latest_info(0)

        with self._metrics_lock:
            metrics = self._metrics
            last_update = dict(metrics.last_update)

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="dim")
        table.add_column("Value")

        table.add_row("steps", f"{metrics.steps:,}")
        table.add_row("episodes", f"{metrics.episodes:,}")
        table.add_row("ret(curr)", f"{perf.get('ep_return_curr', 0.0):.3f}")
        table.add_row("ret(last)", f"{metrics.last_return:.3f}")
        table.add_row("ret(med16)", f"{metrics.median_return_16():.3f}")
        table.add_row("ret(mean100)", f"{metrics.mean_return_100():.3f}")
        table.add_row(
            "len(curr)",
            f"{int(perf.get('ep_decisions_curr', 0) or 0)} dec / {int(perf.get('ep_frames_curr', 0) or 0)} f",
        )
        table.add_row("len(last)", f"{metrics.last_length} f")
        table.add_row("sps", f"{metrics.sps:.0f}")

        spawn_id = info0.get("placements/spawn_id")
        if spawn_id is not None:
            try:
                table.add_row("spawn_id", f"{int(spawn_id)}")
            except Exception:
                pass

        # Curriculum (optional)
        curr_env_level = info0.get("curriculum/env_level", info0.get("curriculum_level"))
        if curr_env_level is not None:
            try:
                table.add_row("curriculum", f"{int(curr_env_level)}")
            except Exception:
                pass
        curr_stage = info0.get("curriculum/current_level")
        if curr_stage is not None:
            try:
                rate = float(info0.get("curriculum/rate_current", 0.0) or 0.0)
                window_n = int(info0.get("curriculum/window_n", 0) or 0)
                window_size = int(info0.get("curriculum/window_size", 0) or 0)
                total_eps = int(info0.get("curriculum/episodes_current_total", 0) or 0)
                suffix = (
                    f"{rate*100:.1f}% last{window_n}/{window_size}"
                    if window_n > 0 and window_size > 0
                    else f"{rate*100:.1f}%"
                )
                if total_eps > 0:
                    suffix = f"{suffix} (tot {total_eps})"
                table.add_row("curr_stage", f"{int(curr_stage)} ({suffix})")
            except Exception:
                pass
        task_mode = info0.get("task_mode")
        if task_mode is not None:
            table.add_row("task", str(task_mode))

        level = info0.get("level")
        if level is not None:
            try:
                table.add_row("level", f"{int(level)}")
            except Exception:
                pass
        viruses_remaining = info0.get("viruses_remaining")
        if viruses_remaining is not None:
            try:
                table.add_row("viruses", f"{int(viruses_remaining)}")
            except Exception:
                pass
        options = info0.get("placements/options")
        if options is not None:
            try:
                table.add_row("options", f"{int(options)}")
            except Exception:
                pass
        backend = info0.get("placements/reach_backend")
        if backend is not None:
            table.add_row("planner", str(backend))

        if last_update:
            table.add_row("", "")
            for k, v in sorted(last_update.items()):
                if k == "perf/sps":
                    continue
                table.add_row(k, f"{v:.4f}")

        return Panel(table, title="[bold]Learning[/bold]", border_style="yellow")

    def _render_reward_panel(self) -> Panel:
        perf = self.env.perf_snapshot()
        show_last = bool(
            int(perf.get("ep_decisions_curr", 0) or 0) == 0 and int(perf.get("ep_decisions_last", 0) or 0) > 0
        )
        reward = (
            perf.get("ep_reward_breakdown_last", {}) if show_last else perf.get("ep_reward_breakdown_curr", {})
        ) or {}
        counts = (
            perf.get("ep_reward_counts_last", {}) if show_last else perf.get("ep_reward_counts_curr", {})
        ) or {}
        r_total = float(perf.get("ep_return_last" if show_last else "ep_return_curr", 0.0) or 0.0)

        def _f(x: Any) -> float:
            try:
                return float(x)
            except Exception:
                return 0.0

        def _i(x: Any) -> int:
            try:
                return int(x)
            except Exception:
                return 0

        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("Term", style="dim")
        table.add_column("Count", justify="right")
        table.add_column("Reward", justify="right")

        pill_locks = _i(counts.get("pill_locks"))
        delta_v = _i(counts.get("delta_v"))
        tiles_nv = _i(counts.get("tiles_cleared_non_virus"))
        action_events = _i(counts.get("action_events"))
        clear_events = _i(counts.get("clear_events"))
        clearing_max = _i(counts.get("tiles_clearing_max"))

        table.add_row("pill_lock", f"{pill_locks}", f"{_f(reward.get('pill_bonus_adjusted')):+.4f}")
        table.add_row("virus_clear", f"{delta_v}", f"{_f(reward.get('virus_clear_reward')):+.4f}")
        table.add_row("nonvirus_clear", f"{tiles_nv}", f"{_f(reward.get('non_virus_bonus')):+.4f}")
        table.add_row("adjacency", "-", f"{_f(reward.get('adjacency_bonus')):+.4f}")
        table.add_row("height", "-", f"{_f(reward.get('height_penalty_delta')):+.4f}")
        table.add_row("action_pen", f"{action_events}", f"{-_f(reward.get('action_penalty')):+.4f}")
        table.add_row("terminal", "-", f"{_f(reward.get('terminal_bonus')):+.4f}")
        table.add_row("topout", "-", f"{_f(reward.get('topout_penalty')):+.4f}")
        table.add_row("time", "-", f"{_f(reward.get('time_reward')):+.4f}")

        table.add_row("", "", "")
        table.add_row("clear_ev", f"{clear_events}", f"max {clearing_max}")
        table.add_row("r_env", "-", f"{_f(reward.get('r_env')):+.4f}")
        table.add_row("r_shape", "-", f"{_f(reward.get('r_shape')):+.4f}")
        table.add_row("r_total", "-", f"{r_total:+.4f}")

        title = "[bold]Reward (last)[/bold]" if show_last else "[bold]Reward (curr)[/bold]"
        return Panel(table, title=title, border_style="magenta")

    def _render_footer(self, interactive: bool) -> Panel:
        lines = []
        if self._status:
            lines.append(self._status)
        if not interactive:
            lines.append("stdin is not a TTY: controls disabled (rendering only)")
        if self._show_help:
            lines.append("Controls: Space pause  n step  f+60  +/- speed  0 max  1/2/4 presets  r rng  h help  q quit")
        return Panel(Text("\n".join(lines) if lines else ""), border_style="blue")

    def _render_layout(self, interactive: bool) -> Layout:
        layout = Layout()
        layout.split_column(Layout(name="main", ratio=1), Layout(name="footer", size=3))
        layout["main"].split_row(
            Layout(name="board", ratio=2),
            Layout(name="perf", ratio=1),
            Layout(name="learning", ratio=1),
            Layout(name="reward", ratio=1),
        )

        info0 = self.env.latest_info(0)
        board_state = board_from_env_info(info0)
        layout["board"].update(render_board_panel(board_state, title=self._title))
        layout["perf"].update(self._render_perf_panel())
        layout["learning"].update(self._render_learning_panel())
        layout["reward"].update(self._render_reward_panel())
        layout["footer"].update(self._render_footer(interactive))
        return layout

    # ---------------------------- main loop (UI thread)
    def run(self, training_thread: threading.Thread) -> None:
        console = Console()
        raw = _RawTerminal()
        interactive = raw.enabled
        if not interactive:
            self._show_help = False

        with raw:
            with Live(
                self._render_layout(interactive),
                console=console,
                refresh_per_second=self._refresh_hz,
                screen=False,
            ) as live:
                try:
                    while training_thread.is_alive():
                        key = raw.poll_key()
                        if key is not None:
                            if key in {"q", "Q"}:
                                self._status = "Stopping…"
                                self.control.request_stop()
                                break
                            if key == " ":
                                self.control.toggle_pause()
                            elif key in {"n", "N"}:
                                self.control.step_once(1)
                            elif key in {"f", "F"}:
                                self.control.step_once(60)
                            elif key in {"+", "="}:
                                self.control.faster()
                            elif key == "-":
                                self.control.slower()
                            elif key == "0":
                                self.control.set_max_speed(True)
                            elif key == "1":
                                self.control.set_speed_x(1.0)
                            elif key == "2":
                                self.control.set_speed_x(2.0)
                            elif key == "4":
                                self.control.set_speed_x(4.0)
                            elif key in {"r", "R"}:
                                self.control.toggle_rng_randomize()
                            elif key in {"h", "H", "?"}:
                                self._show_help = not self._show_help
                        live.update(self._render_layout(interactive))
                        time.sleep(1.0 / self._refresh_hz)
                except KeyboardInterrupt:
                    self.control.request_stop()
                finally:
                    # One last render with stop status.
                    live.update(self._render_layout(interactive))
