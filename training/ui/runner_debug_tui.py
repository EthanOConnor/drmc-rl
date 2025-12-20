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

import envs.specs.ram_to_state as ram_specs

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
        if not data:
            return None
        if data == "\x1b":
            # Attempt to decode escape sequences (e.g., Shift+Tab = ESC [ Z).
            seq = data
            try:
                for _ in range(2):
                    r, _, _ = select.select([self._fd], [], [], 0.0001)
                    if not r:
                        break
                    seq += sys.stdin.read(1)
            except Exception:
                return data
            if seq == "\x1b[Z":
                return "SHIFT_TAB"
            return seq
        return data


@dataclass
class _MetricState:
    steps: int = 0
    episodes: int = 0
    last_return: float = 0.0
    last_length: int = 0
    sps: float = 0.0
    dps: float = 0.0
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
        self._status_until: float = 0.0
        self._show_help = True
        self._show_planes = False
        self._selected_env = 0
        self._last_env = 0
        self._fps_baseline: Optional[float] = None
        self._ui_hz_target: float = float(self._refresh_hz)
        self._env_count_debounce_sec: float = 0.5
        self._pending_env_restart: Optional[int] = None
        self._pending_env_restart_at: float = 0.0
        self._env_count_entry_active: bool = False
        self._env_count_entry: str = ""

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
            try:
                dps_val = payload.get("perf/dps_decisions_total", payload.get("perf/dps"))
                if dps_val is not None:
                    self._metrics.dps = float(dps_val)
            except Exception:
                pass
            # Keep a compact metric snapshot.
            keep = (
                "loss/policy",
                "loss/value",
                "policy/entropy",
                "loss/kl",
                "vf/explained_var",
                "perf/sps",
                "perf/sps_frames_total",
                "perf/dps_decisions_total",
            )
            self._metrics.last_update = {
                k: float(payload[k])
                for k in keep
                if k in payload and isinstance(payload[k], (int, float))
            }

    def set_env(self, env: RateLimitedVecEnv) -> None:
        """Update the active environment after a controlled restart."""
        self.env = env
        self._clamp_selection()

    def _clamp_selection(self) -> None:
        num_envs = int(getattr(self.env, "num_envs", 1))
        if num_envs <= 0:
            self._selected_env = 0
            self._last_env = 0
            return
        if self._last_env >= num_envs:
            self._last_env = max(0, num_envs - 1)
        if self._selected_env >= num_envs:
            self._selected_env = max(0, num_envs - 1)

    def _set_selected_env(self, env_index: int) -> None:
        num_envs = int(getattr(self.env, "num_envs", 1))
        if num_envs <= 0:
            return
        idx = max(0, min(int(env_index), num_envs - 1))
        self._selected_env = idx
        self._last_env = idx

    def _select_summary(self) -> None:
        self._selected_env = -1

    def _toggle_summary(self) -> None:
        if self._selected_env == -1:
            self._selected_env = int(self._last_env)
        else:
            self._last_env = int(self._selected_env)
            self._selected_env = -1

    def _set_status(self, message: str, *, ttl_sec: float = 2.0) -> None:
        self._status = str(message)
        if ttl_sec and ttl_sec > 0.0:
            self._status_until = time.monotonic() + float(ttl_sec)
        else:
            self._status_until = 0.0

    def _status_text(self) -> str:
        if not self._status:
            return ""
        if self._status_until > 0.0 and time.monotonic() >= self._status_until:
            self._status = ""
            self._status_until = 0.0
            return ""
        return self._status

    def _current_num_envs(self) -> int:
        try:
            return int(getattr(self.env, "num_envs", 1))
        except Exception:
            return 1

    def _queue_env_restart(self, num_envs: int, *, debounce: bool = True) -> None:
        target = max(1, int(num_envs))
        current = self._current_num_envs()
        if target == current:
            self._pending_env_restart = None
            self._pending_env_restart_at = 0.0
            return
        self._pending_env_restart = target
        delay = float(self._env_count_debounce_sec if debounce else 0.0)
        self._pending_env_restart_at = time.monotonic() + delay

    def _cancel_pending_restart(self) -> None:
        self._pending_env_restart = None
        self._pending_env_restart_at = 0.0

    def _pending_env_line(self) -> str:
        if self._env_count_entry_active:
            val = self._env_count_entry if self._env_count_entry else ""
            suffix = "Enter apply • Esc cancel • Backspace edit"
            return f"Set envs: {val}{'_' if not val else ''}  ({suffix})"
        if self._pending_env_restart is None:
            return ""
        remaining = max(0.0, float(self._pending_env_restart_at - time.monotonic()))
        current = self._current_num_envs()
        target = int(self._pending_env_restart)
        return (
            f"Pending restart: num_envs {current} → {target}  "
            f"(apply in {remaining:.1f}s, Enter now, Esc cancel)"
        )

    def _target_ui_hz(self) -> float:
        base = float(self._refresh_hz)
        try:
            num_envs = int(getattr(self.env, "num_envs", 1))
        except Exception:
            num_envs = 1
        summary = bool(self._selected_env == -1)

        if summary:
            # Grid view: keep CPU overhead roughly bounded as env count grows.
            target = base * 4.0 / float(max(4, num_envs))
        else:
            target = base

        # Clamp to keep the UI responsive.
        return float(max(5.0, min(base, target)))

    def _format_ms(self, value: Any) -> str:
        try:
            return f"{float(value):.4f}"
        except Exception:
            return "-"

    def _timing_rows(
        self, *, perf: Dict[str, Any], sps: float
    ) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        total_ms: Optional[float] = None
        if sps > 0.0:
            total_ms = float(1000.0 / max(sps, 1e-9))
            rows.append(("total", self._format_ms(total_ms)))
        else:
            rows.append(("total", "-"))

        step_ms = float(perf.get("step_ms_per_frame", 0.0) or 0.0)
        infer_ms = float(perf.get("inference_ms_per_frame", 0.0) or 0.0)
        update_ms = float(perf.get("update_ms_per_frame_avg", 0.0) or 0.0)
        rows.append(("  env.step", self._format_ms(step_ms)))
        rows.append(
            (
                "    env",
                self._format_ms(perf.get("env_step_ms_per_frame", 0.0) or 0.0),
            )
        )
        rows.append(
            (
                "    planner",
                self._format_ms(perf.get("planner_ms_per_frame", 0.0) or 0.0),
            )
        )
        rows.append(
            (
                "    macro_other",
                self._format_ms(perf.get("macro_other_ms_per_frame", 0.0) or 0.0),
            )
        )
        if float(perf.get("env_step_ms_per_frame", 0.0) or 0.0) > 0.0:
            rows.append(
                (
                    "      backend",
                    self._format_ms(perf.get("env_backend_ms_per_frame", 0.0) or 0.0),
                )
            )
            rows.append(
                (
                    "      get_ram",
                    self._format_ms(perf.get("env_get_ram_ms_per_frame", 0.0) or 0.0),
                )
            )
            rows.append(
                (
                    "      ram_bytes",
                    self._format_ms(perf.get("env_ram_bytes_ms_per_frame", 0.0) or 0.0),
                )
            )
            rows.append(
                (
                    "      build_state",
                    self._format_ms(
                        perf.get("env_build_state_ms_per_frame", 0.0) or 0.0
                    ),
                )
            )
            rows.append(
                (
                    "      observe",
                    self._format_ms(perf.get("env_observe_ms_per_frame", 0.0) or 0.0),
                )
            )
            rows.append(
                (
                    "      reward",
                    self._format_ms(perf.get("env_reward_ms_per_frame", 0.0) or 0.0),
                )
            )
            rows.append(
                (
                    "      info",
                    self._format_ms(perf.get("env_info_ms_per_frame", 0.0) or 0.0),
                )
            )
            frame_ms = float(perf.get("env_get_frame_ms_per_frame", 0.0) or 0.0)
            pix_ms = float(perf.get("env_pixel_stack_ms_per_frame", 0.0) or 0.0)
            if frame_ms > 0.0 or pix_ms > 0.0:
                rows.append(("      get_frame", self._format_ms(frame_ms)))
                rows.append(("      pix_stack", self._format_ms(pix_ms)))

        rows.append(("  inference", self._format_ms(infer_ms)))
        rows.append(("  update(avg)", self._format_ms(update_ms)))

        if total_ms is not None:
            accounted_ms = step_ms + infer_ms + update_ms
            unaccounted_ms = max(0.0, float(total_ms) - float(accounted_ms))
            rows.append(("  accounted", self._format_ms(accounted_ms)))
            rows.append(("  unaccounted", self._format_ms(unaccounted_ms)))

        return rows

    def _render_timing_breakdown(self, *, perf: Dict[str, Any], sps: float) -> Table:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="dim")
        table.add_column("ms/frame")
        for k, v in self._timing_rows(perf=perf, sps=sps):
            table.add_row(k, v)
        return table

    def _cycle_env(self, direction: int) -> None:
        num_envs = int(getattr(self.env, "num_envs", 1))
        if num_envs <= 0:
            return
        order = [-1] + list(range(num_envs))
        try:
            pos = order.index(self._selected_env)
        except ValueError:
            pos = 0
        pos = (pos + int(direction)) % len(order)
        self._selected_env = int(order[pos])
        if self._selected_env >= 0:
            self._last_env = int(self._selected_env)

    # ---------------------------- rendering
    def _render_perf_panel(self, env_index: int = 0) -> Panel:
        ctrl = self.control.snapshot()
        perf = self.env.perf_snapshot(env_index)
        info0 = self.env.latest_info(env_index)
        num_envs = int(perf.get("num_envs", getattr(self.env, "num_envs", 1)))
        with self._metrics_lock:
            sps = float(self._metrics.sps)

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="dim")
        table.add_column("Value")

        table.add_row("paused", "yes" if ctrl["paused"] else "no")
        table.add_row("speed", "MAX" if ctrl["max_speed"] else f"{ctrl['speed_x']:.2f}x")
        table.add_row("target_hz", "-" if ctrl["max_speed"] else f"{ctrl['target_hz']:.1f}")
        table.add_row("rng", "on" if ctrl.get("rng_randomize") else "off")
        table.add_row("ui_hz", f"{self._ui_hz_target:.1f}")
        if num_envs > 1:
            table.add_row("env", f"{env_index + 1}/{num_envs}")
        table.add_row("emu_fps(step)", f"{perf.get('step_fps', 0.0):.1f}")
        table.add_row("emu_fps(total)", f"{perf.get('emu_fps', 0.0):.1f}")
        if num_envs > 1:
            emu_fps_env = float(perf.get("emu_fps", 0.0)) / max(1, num_envs)
            table.add_row("emu_fps/env", f"{emu_fps_env:.1f}")
        table.add_row("tau_max", f"{perf.get('tau_max', 1)}")
        table.add_row("tau_total", f"{perf.get('tau_total', perf.get('tau_max', 1))}")
        table.add_row("spawns_total", f"{int(perf.get('spawns_total', 0) or 0):,}")

        if sps > 0.0:
            table.add_row("sps(frames/sec)", f"{sps:.0f}")
            if num_envs > 1:
                table.add_row("sps/env", f"{float(sps) / max(1, num_envs):.1f}")
        table.add_row("", "")
        for k, v in self._timing_rows(perf=perf, sps=sps):
            table.add_row(k, v)

        # ----------------------------------------------------------------- policy inputs
        try:
            state_repr = str(ram_specs.get_state_representation())
        except Exception:
            state_repr = ""

        obs_any = self.env.latest_obs()
        obs_arr = None
        if obs_any is not None:
            try:
                if isinstance(obs_any, dict) and "obs" in obs_any:
                    obs_any = obs_any["obs"]
                obs_arr = np.asarray(obs_any)
            except Exception:
                obs_arr = None

        obs0_shape = None
        obs_dtype = None
        if obs_arr is not None and obs_arr.size > 0:
            obs_dtype = getattr(obs_arr, "dtype", None)
            try:
                # Vector envs: obs is typically [N, ...]. For N==1, obs[0] is still correct.
                obs0_shape = tuple(np.asarray(obs_arr[0]).shape)
            except Exception:
                obs0_shape = tuple(getattr(obs_arr, "shape", ()))

        if state_repr or obs0_shape is not None:
            table.add_row("", "")
            if state_repr:
                table.add_row("state_repr", state_repr)
            if obs0_shape is not None:
                dtype_s = str(obs_dtype) if obs_dtype is not None else "?"
                table.add_row(f"obs(env{env_index + 1})", f"{obs0_shape} {dtype_s}")

            try:
                names = list(ram_specs.get_plane_names())
            except Exception:
                names = []

            if names:
                if self._show_planes:
                    # Display a compact index→name map, grouped to avoid an overly tall panel.
                    def _chunk(items, n: int):
                        for i in range(0, len(items), n):
                            yield items[i : i + n]

                    lines = []
                    for group in _chunk(list(enumerate(names)), 4):
                        lines.append("  ".join(f"{i}:{nm}" for i, nm in group))
                    table.add_row("planes", "\n".join(lines))
                else:
                    table.add_row("planes", f"{len(names)} channels (press p)")

            colors = info0.get("next_pill_colors")
            if colors is not None:
                try:
                    c = np.asarray(colors, dtype=np.int64).reshape(-1)
                    if c.size >= 2:
                        idx0, idx1 = int(c[0]), int(c[1])
                        lut = {0: "R", 1: "Y", 2: "B"}
                        table.add_row(
                            "pill",
                            f"[{idx0},{idx1}] ({lut.get(idx0,'?')}{lut.get(idx1,'?')})",
                        )
                except Exception:
                    pass

            # Preview pill colors (decoded from raw RAM when available).
            raw_left = None
            raw_right = None
            raw_ram = info0.get("raw_ram")
            try:
                if isinstance(raw_ram, (bytes, bytearray, memoryview)) and len(raw_ram) > 0x031B:
                    raw_left = int(raw_ram[0x031A]) & 0x03
                    raw_right = int(raw_ram[0x031B]) & 0x03
            except Exception:
                raw_left = None
                raw_right = None
            if raw_left is None or raw_right is None:
                preview = info0.get("preview_pill")
                if isinstance(preview, dict):
                    try:
                        raw_left = int(preview.get("first_color", 0)) & 0x03
                        raw_right = int(preview.get("second_color", 0)) & 0x03
                    except Exception:
                        raw_left = None
                        raw_right = None
            if raw_left is not None and raw_right is not None:
                try:
                    idx0 = int(ram_specs.COLOR_VALUE_TO_INDEX.get(int(raw_left), 0))
                    idx1 = int(ram_specs.COLOR_VALUE_TO_INDEX.get(int(raw_right), 0))
                    lut = {0: "R", 1: "Y", 2: "B"}
                    table.add_row(
                        "preview",
                        f"[{idx0},{idx1}] ({lut.get(idx0,'?')}{lut.get(idx1,'?')})",
                    )
                except Exception:
                    pass

            mask = None
            for key in ("placements/feasible_mask", "placements/legal_mask", "mask"):
                if key in info0:
                    mask = info0.get(key)
                    break
            if mask is not None:
                try:
                    m = np.asarray(mask)
                    if m.shape == (4, 16, 8):
                        table.add_row("mask", f"{m.shape} (true={int(m.sum())})")
                        table.add_row("orient", "0:H+  1:V+  2:H-  3:V-")
                except Exception:
                    pass

        infer_calls = int(perf.get("inference_calls", 0) or 0)
        if infer_calls > 0:
            table.add_row("", "")
            table.add_row("infer_calls", f"{infer_calls:,}")
            table.add_row(
                "infer/spawn",
                f"{float(perf.get('inference_per_spawn', 0.0) or 0.0):.3f}",
            )
            table.add_row(
                "infer_ms/frame",
                f"{perf.get('inference_ms_per_frame', 0.0):.4f} "
                f"(avg {perf.get('inference_ms_per_call', 0.0):.3f}, "
                f"last {perf.get('last_inference_ms', 0.0):.3f})",
            )

        planner_calls = int(perf.get("planner_calls", 0) or 0)
        if planner_calls > 0:
            table.add_row("", "")
            table.add_row("planner_calls", f"{planner_calls:,}")
            table.add_row(
                "planner/spawn",
                f"{float(perf.get('planner_per_spawn', 0.0) or 0.0):.3f}",
            )
            table.add_row(
                "planner_ms/frame",
                f"{perf.get('planner_ms_per_frame', 0.0):.4f} "
                f"(avg {perf.get('planner_ms_per_call', 0.0):.3f})",
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
            table.add_row(
                "update_ms/frame",
                f"{perf.get('update_ms_per_frame_avg', 0.0):.4f} (avg)",
            )
        return Panel(table, title="[bold]Perf[/bold]", border_style="green")

    def _render_learning_panel(self, env_index: int = 0, *, summary: bool = False) -> Panel:
        perf = self.env.perf_snapshot(env_index)
        info0 = self.env.latest_info(env_index)
        num_envs = int(perf.get("num_envs", getattr(self.env, "num_envs", 1)))

        with self._metrics_lock:
            metrics = self._metrics
            last_update = dict(metrics.last_update)

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="dim")
        table.add_column("Value")

        table.add_row("steps", f"{metrics.steps:,}")
        table.add_row("episodes", f"{metrics.episodes:,}")
        if summary:
            curr_returns = perf.get("ep_return_curr_all", []) or []
            last_returns = perf.get("ep_return_last_all", []) or []
            if curr_returns:
                table.add_row("ret(curr)", f"{float(np.mean(curr_returns)):.3f}")
            else:
                table.add_row("ret(curr)", f"{0.0:.3f}")
            if last_returns:
                table.add_row("ret(last)", f"{float(np.mean(last_returns)):.3f}")
            else:
                table.add_row("ret(last)", f"{0.0:.3f}")
        else:
            table.add_row("ret(curr)", f"{perf.get('ep_return_curr', 0.0):.3f}")
            table.add_row("ret(last)", f"{perf.get('ep_return_last', 0.0):.3f}")
        table.add_row("ret(med16)", f"{metrics.median_return_16():.3f}")
        table.add_row("ret(mean100)", f"{metrics.mean_return_100():.3f}")
        table.add_row(
            "len(curr)",
            f"{int(perf.get('ep_decisions_curr', 0) or 0)} dec / "
            f"{int(perf.get('ep_frames_curr', 0) or 0)} f",
        )
        table.add_row("len(last)", f"{int(perf.get('ep_frames_last', 0) or 0)} f")
        table.add_row("sps", f"{metrics.sps:.0f}")
        if num_envs > 1:
            table.add_row("sps/env", f"{float(metrics.sps) / max(1, num_envs):.1f}")
        if metrics.dps > 0.0:
            table.add_row("dps", f"{metrics.dps:.0f}")
            if num_envs > 1:
                table.add_row("dps/env", f"{float(metrics.dps) / max(1, num_envs):.1f}")
        terminal_last = str(perf.get("terminal_reason_last", "") or "")
        if terminal_last:
            table.add_row("terminal(last)", terminal_last)

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

        conf_lb = info0.get("curriculum/confidence_lower_bound")
        conf_sigmas = info0.get("curriculum/confidence_sigmas")
        if conf_lb is not None and conf_sigmas is not None:
            try:
                table.add_row("curr_lb", f"{float(conf_lb):.4f} ({float(conf_sigmas):.1f}σ)")
            except Exception:
                pass

        time_budget = info0.get("curriculum/time_budget_frames")
        if time_budget is not None:
            mean = info0.get("curriculum/time_mean_frames")
            mad = info0.get("curriculum/time_mad_frames")
            try:
                extra = ""
                if mean is not None and mad is not None:
                    extra = f" mean±mad {float(mean):.0f}±{float(mad):.0f}"
                table.add_row("time_budget", f"{int(time_budget)}f{extra}")
            except Exception:
                pass
        task_mode = info0.get("task_mode")
        if task_mode is not None:
            table.add_row("task", str(task_mode))

        reward_cfg_loaded = info0.get("reward_config_loaded")
        if reward_cfg_loaded is not None:
            try:
                label = "loaded" if bool(reward_cfg_loaded) else "default"
            except Exception:
                label = str(reward_cfg_loaded)
            reward_cfg_path = str(info0.get("reward_config_path", "") or "")
            if reward_cfg_path:
                table.add_row("reward_cfg", f"{label} ({reward_cfg_path})")
            else:
                table.add_row("reward_cfg", str(label))

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

        pose_ok = info0.get("placements/pose_ok")
        if pose_ok is not None:
            table.add_row("pose_ok", "yes" if bool(pose_ok) else "no")
            tgt = info0.get("placements/target_pose")
            if tgt is not None:
                table.add_row("target_pose", str(tgt))
            lock_pose = info0.get("placements/lock_pose")
            if lock_pose is not None:
                table.add_row("lock_pose", str(lock_pose))
            lock_reason = info0.get("placements/lock_reason")
            if lock_reason is not None:
                table.add_row("lock_reason", str(lock_reason))
            if not bool(pose_ok):
                dx = info0.get("placements/pose_dx")
                dy = info0.get("placements/pose_dy")
                drot = info0.get("placements/pose_drot")
                if dx is not None or dy is not None or drot is not None:
                    table.add_row("pose_err", f"dx={dx} dy={dy} rot={drot}")
        mismatch_count = info0.get("placements/pose_mismatch_count")
        if mismatch_count is not None:
            try:
                table.add_row("pose_mismatches", f"{int(mismatch_count):,}")
            except Exception:
                pass
        mismatch_last = info0.get("placements/pose_mismatch_last")
        if mismatch_last:
            mismatch_id = info0.get("placements/pose_mismatch_id")
            if mismatch_id is not None:
                table.add_row("mismatch_last", f"yes (#{int(mismatch_id)})")
            else:
                table.add_row("mismatch_last", "yes")

        if last_update:
            table.add_row("", "")
            for k, v in sorted(last_update.items()):
                if k == "perf/sps":
                    continue
                table.add_row(k, f"{v:.4f}")

        return Panel(table, title="[bold]Learning[/bold]", border_style="yellow")

    def _render_reward_panel(self, env_index: int = 0) -> Panel:
        perf = self.env.perf_snapshot(env_index)
        show_last = bool(
            int(perf.get("ep_decisions_curr", 0) or 0) == 0
            and int(perf.get("ep_decisions_last", 0) or 0) > 0
        )
        reward = (
            perf.get("ep_reward_breakdown_last", {})
            if show_last
            else perf.get("ep_reward_breakdown_curr", {})
        ) or {}
        counts = (
            perf.get("ep_reward_counts_last", {})
            if show_last
            else perf.get("ep_reward_counts_curr", {})
        ) or {}
        r_total = float(perf.get("ep_return_last" if show_last else "ep_return_curr", 0.0) or 0.0)
        reward_last = perf.get("ep_reward_breakdown_last", {}) or {}
        r_total_last = float(perf.get("ep_return_last", 0.0) or 0.0)

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
        table.add_row("virus_adj", "-", f"{_f(reward.get('virus_adjacency_bonus')):+.4f}")
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

        # Always include last-episode summary lines so top-out penalties are
        # visible even when a new episode has already started.
        table.add_row("", "", "")
        table.add_row("r_total(last)", "-", f"{r_total_last:+.4f}")
        table.add_row("topout(last)", "-", f"{_f(reward_last.get('topout_penalty')):+.4f}")

        title = "[bold]Reward (last)[/bold]" if show_last else "[bold]Reward (curr)[/bold]"
        return Panel(table, title=title, border_style="magenta")

    def _render_summary_panel(self) -> Panel:
        perf = self.env.perf_snapshot(0)
        num_envs = int(perf.get("num_envs", getattr(self.env, "num_envs", 1)))
        fps_total = float(perf.get("emu_fps", 0.0) or 0.0)
        fps_per_env = fps_total / max(1, num_envs)

        if num_envs == 1 and fps_total > 0.0:
            self._fps_baseline = fps_total

        speedup = None
        efficiency = None
        if self._fps_baseline and num_envs > 0:
            speedup = fps_total / max(self._fps_baseline, 1e-9)
            efficiency = 100.0 * speedup / float(num_envs)

        with self._metrics_lock:
            metrics = self._metrics

        left = Table(show_header=False, box=None, padding=(0, 1))
        left.add_column("Key", style="dim")
        left.add_column("Value")

        left.add_row("envs", f"{num_envs}")
        left.add_row("ui_hz", f"{self._ui_hz_target:.1f}")
        left.add_row("fps(total)", f"{fps_total:.1f}")
        left.add_row("fps/env", f"{fps_per_env:.1f}")
        if metrics.sps > 0.0:
            left.add_row("sps", f"{metrics.sps:.0f}")
            if num_envs > 1:
                left.add_row("sps/env", f"{float(metrics.sps) / max(1, num_envs):.1f}")
        if metrics.dps > 0.0:
            left.add_row("dps", f"{metrics.dps:.1f}")
            left.add_row("dps/env", f"{metrics.dps / max(1, num_envs):.2f}")
        if speedup is not None and efficiency is not None:
            left.add_row("speedup", f"{speedup:.2f}x")
            left.add_row("efficiency", f"{efficiency:.1f}%")
        else:
            left.add_row("speedup", "n/a")
            left.add_row("efficiency", "n/a")
        left.add_row("episodes", f"{metrics.episodes:,}")
        left.add_row("ret(med16)", f"{metrics.median_return_16():.3f}")
        left.add_row("ret(mean100)", f"{metrics.mean_return_100():.3f}")

        right = self._render_timing_breakdown(perf=perf, sps=float(metrics.sps))

        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=2)
        grid.add_row(left, right)
        return Panel(grid, title="[bold]Summary[/bold]", border_style="cyan")

    def _render_grid_panel(self) -> Panel:
        num_envs = int(getattr(self.env, "num_envs", 1))
        cols = min(4, max(1, num_envs))
        grid = Table.grid(expand=True)
        for _ in range(cols):
            grid.add_column(justify="center")

        panels = []
        for i in range(num_envs):
            info_i = self.env.latest_info(i)
            board_state = board_from_env_info(info_i)
            panels.append(
                render_board_panel(
                    board_state,
                    title=f"Env {i + 1}",
                    show_stats=False,
                    compact=True,
                    show_preview=False,
                    show_subtitle=False,
                )
            )

        if not panels:
            panels.append(Text("No envs"))

        for start in range(0, len(panels), cols):
            row = panels[start : start + cols]
            if len(row) < cols:
                row.extend("" for _ in range(cols - len(row)))
            grid.add_row(*row)

        title = f"[bold]All Boards ({num_envs})[/bold]"
        return Panel(grid, title=title, border_style="blue")

    def _render_footer(self, interactive: bool) -> Panel:
        lines = []
        pending = self._pending_env_line()
        if pending:
            lines.append(pending)
        status = self._status_text()
        if status:
            lines.append(status)
        if not interactive:
            lines.append("stdin is not a TTY: controls disabled (rendering only)")
        if self._show_help:
            lines.append(
                "Controls: Space pause  n step  f+60  +/- speed  m max  0 summary  "
                "tab/shift+tab env  1-9 jump  g grid  [/] envs  e set_envs  "
                "r rng  p planes  h help  q quit"
            )
        return Panel(Text("\n".join(lines) if lines else ""), border_style="blue")

    def _render_layout(self, interactive: bool) -> Layout:
        layout = Layout()
        layout.split_column(Layout(name="main", ratio=1), Layout(name="footer", size=4))
        summary = self._selected_env == -1
        if summary:
            layout["main"].split_row(
                Layout(name="board", ratio=2),
                Layout(name="perf", ratio=2),
                Layout(name="learning", ratio=1),
            )
        else:
            layout["main"].split_row(
                Layout(name="board", ratio=2),
                Layout(name="perf", ratio=1),
                Layout(name="learning", ratio=1),
                Layout(name="reward", ratio=1),
            )
        focus_env = int(self._last_env if summary else self._selected_env)
        focus_env = max(0, min(focus_env, int(getattr(self.env, "num_envs", 1)) - 1))

        if summary:
            layout["board"].update(self._render_grid_panel())
            layout["perf"].update(self._render_summary_panel())
            layout["learning"].update(self._render_learning_panel(focus_env, summary=True))
        else:
            info_focus = self.env.latest_info(focus_env)
            board_state = board_from_env_info(info_focus)
            layout["board"].update(
                render_board_panel(
                    board_state,
                    title=f"{self._title} (Env {focus_env + 1}/{getattr(self.env, 'num_envs', 1)})",
                )
            )
            layout["perf"].update(self._render_perf_panel(focus_env))
            layout["learning"].update(self._render_learning_panel(focus_env, summary=False))
            layout["reward"].update(self._render_reward_panel(focus_env))
        layout["footer"].update(self._render_footer(interactive))
        return layout

    # ---------------------------- main loop (UI thread)
    def run(self, session: Any) -> None:
        console = Console()
        raw = _RawTerminal()
        interactive = raw.enabled
        if not interactive:
            self._show_help = False

        with raw:
            with Live(
                self._render_layout(interactive),
                console=console,
                screen=True,
                transient=True,
                auto_refresh=False,
            ) as live:
                try:
                    next_render = 0.0
                    force_render = True
                    while session.training_thread.is_alive():
                        self._ui_hz_target = self._target_ui_hz()
                        key = raw.poll_key()
                        if key is not None:
                            force_render = True
                            if self._env_count_entry_active:
                                if key.isdigit():
                                    if len(self._env_count_entry) < 4:
                                        self._env_count_entry += key
                                elif key in {"\x7f", "\b"}:
                                    self._env_count_entry = self._env_count_entry[:-1]
                                elif key in {"\r", "\n"}:
                                    if self._env_count_entry:
                                        self._queue_env_restart(
                                            int(self._env_count_entry), debounce=False
                                        )
                                    self._env_count_entry_active = False
                                    self._env_count_entry = ""
                                elif key == "\x1b":
                                    self._env_count_entry_active = False
                                    self._env_count_entry = ""
                                continue
                            if key in {"q", "Q"}:
                                self._set_status("Stopping…", ttl_sec=0.0)
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
                            elif key in {"m", "M"}:
                                self.control.set_max_speed(True)
                            elif key == "\t":
                                self._cycle_env(1)
                            elif key == "SHIFT_TAB":
                                self._cycle_env(-1)
                            elif key == "0":
                                self._select_summary()
                            elif key in {"g", "G"}:
                                self._toggle_summary()
                            elif key in {"[", "]"}:
                                delta = -1 if key == "[" else 1
                                base = (
                                    self._pending_env_restart
                                    if self._pending_env_restart is not None
                                    else self._current_num_envs()
                                )
                                new_num_envs = max(1, int(base) + delta)
                                self._queue_env_restart(new_num_envs, debounce=True)
                            elif key.isdigit() and key != "0":
                                self._set_selected_env(int(key) - 1)
                            elif key in {"r", "R"}:
                                self.control.toggle_rng_randomize()
                            elif key in {"e", "E"}:
                                self._env_count_entry_active = True
                                self._env_count_entry = ""
                            elif key in {"p", "P"}:
                                self._show_planes = not self._show_planes
                            elif key in {"h", "H", "?"}:
                                self._show_help = not self._show_help
                            elif key in {"\r", "\n"} and self._pending_env_restart is not None:
                                self._pending_env_restart_at = time.monotonic()
                            elif key == "\x1b":
                                self._cancel_pending_restart()

                        now = time.monotonic()
                        if (
                            self._pending_env_restart is not None
                            and not self._env_count_entry_active
                            and now >= float(self._pending_env_restart_at)
                        ):
                            new_num_envs = int(self._pending_env_restart)
                            self._cancel_pending_restart()
                            self._set_status(
                                f"Restarting with num_envs={new_num_envs}…", ttl_sec=0.0
                            )
                            live.update(self._render_layout(interactive), refresh=True)
                            try:
                                session.restart(new_num_envs)
                                self.set_env(session.env)
                                self._set_status(
                                    f"Restarted with num_envs={new_num_envs}", ttl_sec=2.0
                                )
                            except Exception as exc:  # noqa: BLE001
                                self._set_status(f"Restart failed: {exc}", ttl_sec=0.0)
                                raise
                            force_render = True
                        if force_render or now >= next_render:
                            self._ui_hz_target = self._target_ui_hz()
                            period = 1.0 / float(max(self._ui_hz_target, 1e-6))
                            live.update(self._render_layout(interactive), refresh=True)
                            next_render = time.monotonic() + period
                            force_render = False

                        sleep_sec = max(0.0, next_render - time.monotonic())
                        time.sleep(min(0.02, sleep_sec))
                except KeyboardInterrupt:
                    self.control.request_stop()
                finally:
                    # One last render with stop status.
                    live.update(self._render_layout(interactive), refresh=True)
