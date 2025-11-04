"""Smoke test for DrMarioRetroEnv with configurable emulator backend.

Usage:
    python envs/retro/demo.py --mode pixel --level 0 --steps 1000 --risk-tau 0.5 --backend libretro
"""
import argparse
import multiprocessing as mp
import queue
import sys
import time
from collections import deque
from pathlib import Path

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

import envs.specs.ram_to_state as ram_specs

try:
    from multiprocessing import shared_memory
except Exception:  # pragma: no cover - optional on some platforms
    shared_memory = None  # type: ignore[assignment]

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None
    ImageTk = None
else:
    try:
        from PIL import ImageTk
    except Exception:  # pragma: no cover - optional dependency
        ImageTk = None

# Ensure repo root is on sys.path when running as a script
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from envs.retro.register_env import register_env_id
from envs.retro.state_viz import state_to_rgb
from gymnasium import make
_state_to_rgb = state_to_rgb


def _viewer_worker(
    title: str,
    scale: float,
    with_stats: bool,
    frame_queue: "mp.Queue",
    stop_event: "mp.Event",
    control_queue: "mp.Queue",
    mode: str,
    refresh_hz: float,
    state_config: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        import tkinter as tk
        from PIL import Image, ImageTk  # type: ignore
    except Exception:
        stop_event.set()
        return

    interval_ms = 0
    if refresh_hz > 0:
        interval_ms = max(1, int(round(1000.0 / refresh_hz)))

    root = tk.Tk()
    root.title(title)
    img_label = tk.Label(root)
    img_label.pack()
    text_var: Optional[tk.StringVar] = tk.StringVar(value="") if with_stats else None
    default_fg: Optional[str] = None
    redecision_fg = "#ff6b6b"
    text_label: Optional[tk.Label] = None
    if text_var is not None:
        text_label = tk.Label(root, textvariable=text_var, anchor="w", justify="left")
        text_label.pack(fill="x", padx=4, pady=2)
        try:
            default_fg = str(text_label.cget("fg"))
        except Exception:
            default_fg = None
    holder: Dict[str, Any] = {"img": None}
    scale = max(0.5, float(scale))

    if mode != "state":
        state_config = None

    state_shm = None
    state_view: Optional[np.ndarray] = None
    state_lock = None
    state_generation = None
    rendered_generation = None
    last_generation = -1
    latest_stats: Optional[Dict[str, Any]] = None

    if state_config is not None:
        if shared_memory is None:
            stop_event.set()
            return
        state_repr = state_config.get("state_repr") if isinstance(state_config, dict) else None
        if state_repr:
            try:
                ram_specs.set_state_representation(str(state_repr))
            except Exception:
                pass
        try:
            state_shape = tuple(int(v) for v in state_config["shape"])
            state_dtype = np.dtype(state_config["dtype"])
            state_shm = shared_memory.SharedMemory(name=state_config["name"])
            state_view = np.ndarray(state_shape, dtype=state_dtype, buffer=state_shm.buf)
            state_lock = state_config.get("lock")
            state_generation = state_config.get("generation")
            rendered_generation = state_config.get("rendered_generation")
        except Exception:
            stop_event.set()
            if state_shm is not None:
                try:
                    state_shm.close()
                except Exception:
                    pass
            return

    def on_close() -> None:
        if not stop_event.is_set():
            stop_event.set()
        try:
            root.destroy()
        except Exception:
            pass

    root.protocol("WM_DELETE_WINDOW", on_close)

    viz_times: deque[float] = deque(maxlen=120)
    viz_fps = 0.0
    spawn_decision_counts: Dict[int, int] = {}
    last_spawn_id_seen: Optional[int] = None

    def record_viz_update() -> None:
        nonlocal viz_fps
        if text_var is None:
            return
        now = time.perf_counter()
        viz_times.append(now)
        if len(viz_times) >= 2:
            span = viz_times[-1] - viz_times[0]
            if span > 0:
                viz_fps = float(len(viz_times) - 1) / span

    def send_control(payload: Dict[str, Any]) -> None:
        try:
            control_queue.put_nowait(payload)
        except queue.Full:
            pass
        except Exception:
            pass

    def on_key(event: Any) -> None:
        if control_queue is None:
            return
        sym = getattr(event, "keysym", "")
        sym_lower = sym.lower()
        if sym_lower in {"space", "p"}:
            send_control({"type": "pause_toggle"})
            return
        if sym_lower in {"n", "period"}:
            send_control({"type": "step_once", "frames": 1})
            return
        if sym_lower in {"plus", "equal", "+"}:
            send_control({"type": "ratio", "mode": "faster"})
        elif sym_lower in {"minus", "underscore", "-"}:
            send_control({"type": "ratio", "mode": "slower"})
        elif sym == "0":
            send_control({"type": "ratio", "mode": "match", "viz_fps": viz_fps})

    if control_queue is not None:
        root.bind("<Key>", on_key)

    def update_text(stats: Optional[Dict[str, Any]]) -> None:
        nonlocal last_spawn_id_seen
        if text_var is None or stats is None:
            return
        info = stats.get("info", {}) if isinstance(stats, dict) else {}
        level_val = info.get("level_state") if isinstance(info, dict) else None
        if level_val is None and isinstance(info, dict):
            level_val = info.get("level")
        level_text = level_val if level_val is not None else "?"
        def fmt_component(value: float) -> str:
            if abs(value) < 1e-6:
                return "0.00"
            return f"{value:+.2f}"

        needs_action = bool(info.get("placements/needs_action", False))
        spawn_id_val = info.get("placements/spawn_id")
        spawn_id_int: Optional[int] = None
        try:
            if spawn_id_val is not None:
                spawn_id_int = int(spawn_id_val)
        except (TypeError, ValueError):
            spawn_id_int = None
        highlight_spawn = False
        if spawn_id_int is not None:
            if last_spawn_id_seen is None or spawn_id_int > last_spawn_id_seen:
                last_spawn_id_seen = spawn_id_int
            if needs_action:
                current_count = spawn_decision_counts.get(spawn_id_int, 0) + 1
                spawn_decision_counts[spawn_id_int] = current_count
                highlight_spawn = current_count > 1
            else:
                spawn_decision_counts.setdefault(spawn_id_int, 0)
            # prune stale entries to keep memory bounded
            stale_keys = [key for key in spawn_decision_counts.keys() if key < spawn_id_int - 8]
            for key in stale_keys:
                spawn_decision_counts.pop(key, None)

        lines = [
            f"Step {stats.get('step', 0)}  Total {stats.get('cumulative', 0.0):.1f}",
            f"Last reward {stats.get('reward', 0.0):.2f} (env {info.get('r_env', 0.0):.2f})",
        ]
        placement_bonus = float(info.get("pill_bonus", 0.0))
        nonvirus_bonus = float(info.get("non_virus_bonus", 0.0))
        adjacency_bonus = float(info.get("adjacency_bonus", 0.0))
        time_reward = float(info.get("time_reward", 0.0))
        if any(abs(val) > 1e-6 for val in (placement_bonus, nonvirus_bonus, adjacency_bonus, time_reward)):
            lines.append("Components:")
            if abs(placement_bonus) > 1e-6:
                lines.append(f"  placement {fmt_component(placement_bonus)}")
            if abs(nonvirus_bonus) > 1e-6:
                lines.append(f"  non-virus {fmt_component(nonvirus_bonus)}")
            if abs(adjacency_bonus) > 1e-6:
                lines.append(f"  adjacency {fmt_component(adjacency_bonus)}")
            if abs(time_reward) > 1e-6:
                lines.append(f"  time {fmt_component(time_reward)}")
        lines.extend(
            [
                f"Viruses {info.get('viruses_remaining', '?')}  Level {level_text}",
                f"Topout {info.get('topout', False)}  Cleared {info.get('cleared', False)}",
            ]
        )
        term = info.get("terminal_reason")
        if term:
            lines.append(f"Terminal: {term}")
        action = stats.get("action")
        if action is not None:
            lines.append(f"Action: {action}")
        
        # Extract perf_stats early so we can use it in spawn line and speedup
        perf_stats = stats.get("perf") if isinstance(stats, dict) else None
        
        if spawn_id_int is not None:
            spawn_line = f"Spawn {spawn_id_int}"
            # Display inference count inline with spawn
            if isinstance(perf_stats, dict):
                call_count_val = perf_stats.get("inference_calls")
                if call_count_val is not None:
                    inference_count = int(call_count_val)
                    spawn_line += f"  Inference {inference_count}"
                    # Flag if inference calls lagging behind spawns
                    if inference_count < (spawn_id_int - 1):
                        spawn_line += " (LOW!)"
            if needs_action:
                spawn_line += " (needs action)"
                if highlight_spawn:
                    spawn_line += " – re-decide"
            lines.append(spawn_line)
        
        # Calculate speedups: this run and total execution
        step_count = stats.get("step", 0)
        wall_s = perf_stats.get("wall_s") if isinstance(perf_stats, dict) else None
        total_wall_s = perf_stats.get("total_wall_s") if isinstance(perf_stats, dict) else None
        total_steps = perf_stats.get("total_steps") if isinstance(perf_stats, dict) else None
        
        # Speedup for this run
        if step_count is not None and wall_s is not None and wall_s > 0:
            game_seconds = float(step_count) / 60.0  # NES runs at 60 FPS
            speedup_run = game_seconds / float(wall_s)
            lines.append(f"Speedup (run) {speedup_run:.2f}x ({step_count} steps, {float(wall_s):.1f}s)")
        
        # Speedup for total execution
        if total_steps is not None and total_wall_s is not None and total_wall_s > 0:
            total_game_seconds = float(total_steps) / 60.0
            speedup_total = total_game_seconds / float(total_wall_s)
            lines.append(f"Speedup (total) {speedup_total:.2f}x ({total_steps} steps, {float(total_wall_s):.1f}s)")
        
        # Fallback to old emu_fps display if wall times not available
        if wall_s is None or (step_count is None or step_count == 0):
            emu_fps = stats.get("emu_fps")
            target_hz = stats.get("target_hz")
            if emu_fps is not None:
                target_text = "free" if not target_hz else f"target {target_hz:.1f}"
                realtime_multiplier = emu_fps / 60.0
                lines.append(
                    f"Emu FPS {emu_fps:.1f} ({target_text}, {realtime_multiplier:.2f}x)"
                )
        lines.append(f"Viz FPS {viz_fps:.1f}")
        ratio = stats.get("emu_vis_ratio")
        if ratio:
            lines[-1] += f"  ratio×{ratio:.2f}"
        else:
            lines[-1] += "  ratio×free"

        planner_totals = stats.get("planner") if isinstance(stats, dict) else None
        if isinstance(planner_totals, dict):
            calls_total = float(planner_totals.get("calls_total", 0.0) or 0.0)
            latency_avg = float(planner_totals.get("latency_ms_avg", 0.0) or 0.0)
            latency_max = float(planner_totals.get("latency_ms_max", 0.0) or 0.0)
            options_avg = float(planner_totals.get("options_avg", 0.0) or 0.0)
            options_last = float(planner_totals.get("options_last", 0.0) or 0.0)
            replans = float(planner_totals.get("replan_attempts", 0.0) or 0.0)
            replan_failures = float(planner_totals.get("replan_failures", 0.0) or 0.0)
            if (
                calls_total > 0.0
                or replans > 0.0
                or replan_failures > 0.0
                or latency_avg > 0.0
            ):
                lines.append(
                    "Planner totals: calls "
                    f"{calls_total:.0f}  avg {latency_avg:.2f} ms  max {latency_max:.2f} ms"
                )
                lines.append(
                    "  options avg "
                    f"{options_avg:.2f}  last {options_last:.2f}"
                    f"  replans {replans:.0f} (fail {replan_failures:.0f})"
                )

        planner_debug = stats.get("planner_debug") if isinstance(stats, dict) else None
        if isinstance(planner_debug, dict) and planner_debug:
            legal = planner_debug.get("legal_count")
            feasible = planner_debug.get("feasible_count")
            selected = planner_debug.get("selected_action")
            plan_cost = planner_debug.get("plan_cost")
            lines.append(
                "Planner debug: legal "
                f"{int(legal) if legal is not None else '?'}  "
                f"feasible {int(feasible) if feasible is not None else '?'}"
            )
            lines.append(
                "  selected "
                f"{selected if selected is not None else '-'}  "
                f"cost {int(plan_cost) if plan_cost is not None else '-'}"
            )
            exec_step = planner_debug.get("exec_step")
            exec_total = planner_debug.get("exec_total")
            ctrl_l = planner_debug.get("ctrl_left")
            ctrl_r = planner_debug.get("ctrl_right")
            ctrl_d = planner_debug.get("ctrl_down")
            if exec_step is not None and exec_total is not None:
                lines.append(
                    f"  exec {int(exec_step)}/{int(exec_total)}  holds L:{int(ctrl_l or 0)} R:{int(ctrl_r or 0)} D:{int(ctrl_d or 0)}"
                )

        plan_calls = info.get("placements/plan_calls") if isinstance(info, dict) else None
        plan_latency_total = (
            info.get("placements/plan_latency_ms_total") if isinstance(info, dict) else None
        )
        plan_latency_max = (
            info.get("placements/plan_latency_ms_max") if isinstance(info, dict) else None
        )
        plan_options = info.get("placements/plan_count_last") if isinstance(info, dict) else None
        plan_replans = info.get("placements/replan_attempts") if isinstance(info, dict) else None
        if plan_calls:
            try:
                call_count = int(plan_calls)
            except Exception:
                call_count = 0
            if call_count > 0:
                latency_total = float(plan_latency_total or 0.0)
                latency_max = float(plan_latency_max or 0.0)
                options_last = float(plan_options or 0.0)
                replans = float(plan_replans or 0.0)
                lines.append(
                    "Planner step: "
                    f"calls {call_count}  latency {latency_total:.2f} ms (max {latency_max:.2f})"
                )
                lines.append(f"  options {options_last:.2f}  replans {replans:.0f}")

        perf_stats = stats.get("perf") if isinstance(stats, dict) else None
        if isinstance(perf_stats, dict) and perf_stats:
            # Display wait time percentages
            inference_pct = perf_stats.get("inference_pct_wall")
            planner_pct = perf_stats.get("planner_pct_wall")
            emu_pct = perf_stats.get("emu_pct_wall")
            
            wait_parts = []
            if inference_pct is not None:
                wait_parts.append(f"inference {float(inference_pct):.1f}%")
            if planner_pct is not None:
                wait_parts.append(f"planner {float(planner_pct):.1f}%")
            if emu_pct is not None:
                wait_parts.append(f"emu {float(emu_pct):.1f}%")
            
            if wait_parts:
                lines.append("Wall time: " + "  ".join(wait_parts))
            
            # Keep old inference wait display for compute percentage
            compute_pct = perf_stats.get("inference_pct_compute")
            if inference_pct is not None and compute_pct is not None:
                lines.append(
                    f"Inference wait {float(inference_pct):.1f}% wall ({float(compute_pct):.1f}% loop)"
                )
            last_infer = perf_stats.get("last_inference_ms")
            last_step = perf_stats.get("last_step_ms")
            if last_infer is not None or last_step is not None:
                infer_text = f"{float(last_infer):.2f}" if last_infer is not None else "?"
                step_text = f"{float(last_step):.2f}" if last_step is not None else "?"
                lines.append(f"Last inference {infer_text} ms  step {step_text} ms")
            episode_ms_last = perf_stats.get("episode_update_last_ms")
            batch_ms_last = perf_stats.get("batch_update_last_ms")
            if episode_ms_last is not None or batch_ms_last is not None:
                episode_text = (
                    f"{float(episode_ms_last):.2f}"
                    if episode_ms_last is not None
                    else "?"
                )
                batch_text = (
                    f"{float(batch_ms_last):.2f}"
                    if batch_ms_last is not None
                    else "?"
                )
                lines.append(f"Update ms episode {episode_text}  batch {batch_text}")
            episode_avg = perf_stats.get("episode_update_avg_ms")
            batch_avg = perf_stats.get("batch_update_avg_ms")
            if episode_avg is not None or batch_avg is not None:
                episode_avg_text = (
                    f"{float(episode_avg):.2f}"
                    if episode_avg is not None
                    else "?"
                )
                batch_avg_text = (
                    f"{float(batch_avg):.2f}"
                    if batch_avg is not None
                    else "?"
                )
                lines.append(f"Avg update ms episode {episode_avg_text}  batch {batch_avg_text}")
        lines.append("Controls: 0=emu=viz  +=faster  -=slower")
        text_var.set("\n".join(lines))
        if default_fg is not None and text_label is not None:
            try:
                if highlight_spawn and needs_action:
                    text_label.configure(fg=redecision_fg)
                else:
                    text_label.configure(fg=default_fg)
            except Exception:
                pass

    def pump() -> None:
        nonlocal latest_stats, last_generation
        try:
            while True:
                item = frame_queue.get_nowait()
                if item is None:
                    on_close()
                    return
                if state_view is None:
                    frame, stats = item
                    latest_stats = stats if isinstance(stats, dict) else None
                    try:
                        image = Image.fromarray(frame)
                        if scale != 1.0:
                            w, h = image.size
                            image = image.resize((int(w * scale), int(h * scale)), Image.NEAREST)
                        holder["img"] = ImageTk.PhotoImage(image)
                        img_label.configure(image=holder["img"])
                        record_viz_update()
                        update_text(latest_stats)
                    except Exception:
                        continue
                else:
                    latest_stats = item if isinstance(item, dict) else None
        except queue.Empty:
            pass

        if state_view is not None and state_lock is not None and state_generation is not None:
            copied: Optional[np.ndarray] = None
            current_gen = last_generation
            try:
                with state_lock:
                    current_gen = state_generation.value
                    if current_gen != last_generation and state_view is not None:
                        copied = np.array(state_view, copy=True)
            except Exception:
                copied = None
            if copied is not None and current_gen != last_generation:
                info = latest_stats.get("info") if isinstance(latest_stats, dict) else None
                try:
                    frame = _state_to_rgb(copied, info)
                except Exception:
                    frame = None
                if frame is not None:
                    try:
                        image = Image.fromarray(frame)
                        if scale != 1.0:
                            w, h = image.size
                            image = image.resize((int(w * scale), int(h * scale)), Image.NEAREST)
                        holder["img"] = ImageTk.PhotoImage(image)
                        img_label.configure(image=holder["img"])
                        record_viz_update()
                    except Exception:
                        frame = None
                last_generation = current_gen
                if rendered_generation is not None:
                    try:
                        rendered_generation.value = current_gen
                    except Exception:
                        pass
            update_text(latest_stats)

        if not stop_event.is_set():
            if interval_ms > 0:
                root.after(interval_ms, pump)
            else:
                root.after_idle(pump)

    pump()
    try:
        root.mainloop()
    finally:
        on_close()
        if state_shm is not None:
            try:
                state_shm.close()
            except Exception:
                pass


class _ProcessViewer:
    def __init__(
        self,
        title: str,
        scale: float,
        with_stats: bool,
        *,
        mode: str = "pixel",
        refresh_hz: float = 60.0,
        state_shape: Optional[Sequence[int]] = None,
        state_dtype: str | np.dtype = "float32",
        sync_to_viewer: bool = False,
    ) -> None:
        self._ctx = mp.get_context("spawn")
        self._mode = mode
        self._refresh_hz = max(0.0, float(refresh_hz))
        self._queue: mp.Queue = self._ctx.Queue(maxsize=8 if mode == "state" else 2)
        self._stop = self._ctx.Event()
        self._closed = False
        self._sync_to_viewer = bool(sync_to_viewer) and mode == "state"
        self._control_queue: mp.Queue = self._ctx.Queue(maxsize=4)

        self._shared_mem = None
        self._state_view: Optional[np.ndarray] = None
        self._state_lock = None
        self._state_generation = None
        self._rendered_generation = None
        state_config: Optional[Dict[str, Any]] = None

        if mode == "state":
            if shared_memory is None:
                raise RuntimeError("Python shared_memory module unavailable; cannot spawn state viewer")
            if state_shape is None:
                raise ValueError("state_shape must be provided for state mode viewer")
            state_dtype_np = np.dtype(state_dtype)
            shape_tuple: Tuple[int, ...] = tuple(int(v) for v in state_shape)
            nbytes = int(np.prod(shape_tuple)) * state_dtype_np.itemsize
            self._shared_mem = shared_memory.SharedMemory(create=True, size=nbytes)
            self._state_view = np.ndarray(shape_tuple, dtype=state_dtype_np, buffer=self._shared_mem.buf)
            self._state_lock = self._ctx.Lock()
            self._state_generation = self._ctx.Value("Q", 0)
            self._rendered_generation = self._ctx.Value("Q", 0)
            state_config = {
                "name": self._shared_mem.name,
                "shape": shape_tuple,
                "dtype": state_dtype_np.str,
                "lock": self._state_lock,
                "generation": self._state_generation,
                "rendered_generation": self._rendered_generation,
                "state_repr": ram_specs.get_state_representation(),
            }

        self._proc = self._ctx.Process(
            target=_viewer_worker,
            args=(
                title,
                scale,
                with_stats,
                self._queue,
                self._stop,
                self._control_queue,
                mode,
                self._refresh_hz,
                state_config,
            ),
            daemon=True,
        )
        self._proc.start()

    def push(self, payload: np.ndarray, stats: Optional[Dict[str, Any]]) -> bool:
        if self._closed or self._stop.is_set():
            return False
        if self._mode == "state":
            if self._state_view is None or self._state_lock is None or self._state_generation is None:
                return False
            if self._sync_to_viewer and self._rendered_generation is not None:
                while (
                    not self._stop.is_set()
                    and self._state_generation.value != self._rendered_generation.value
                ):
                    time.sleep(0.001)
            try:
                data = np.asarray(payload, dtype=self._state_view.dtype)
                if data.shape != self._state_view.shape:
                    return False
                with self._state_lock:
                    np.copyto(self._state_view, data)
                    self._state_generation.value += 1
            except Exception:
                return False
            if stats is not None:
                try:
                    self._queue.put_nowait(stats)
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                        self._queue.put_nowait(stats)
                    except queue.Empty:
                        pass
                    except queue.Full:
                        pass
                except Exception:
                    return False
            return not self._stop.is_set()

        try:
            frame = np.asarray(payload, dtype=np.uint8).copy()
            self._queue.put_nowait((frame, stats))
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait((frame, stats))
            except queue.Empty:
                pass
            except queue.Full:
                pass
        except Exception:
            return False
        return not self._stop.is_set()

    def poll_control(self) -> Optional[Dict[str, Any]]:
        try:
            return self._control_queue.get_nowait()
        except queue.Empty:
            return None
        except Exception:
            return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stop.set()
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            except Exception:
                pass
            try:
                self._control_queue.put_nowait({"type": "shutdown"})
            except queue.Full:
                pass
            except Exception:
                pass
            self._proc.join(timeout=1.0)
        finally:
            if self._proc.is_alive():
                self._proc.terminate()
        if self._shared_mem is not None:
            try:
                self._shared_mem.close()
                self._shared_mem.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['pixel', 'state'], default='pixel')
    ap.add_argument('--level', type=int, default=0)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--risk-tau', type=float, default=0.5)
    ap.add_argument('--dump-state', action='store_true', help='Print state-plane stats (state mode)')
    ap.add_argument('--frame-offset', type=int, default=0, help='Advance N frames with NOOP at reset (influences ROM RNG)')
    ap.add_argument(
        '--randomize-rng',
        action='store_true',
        help='Randomize the in-game RNG state at each reset (writes new rng0/rng1 bytes)',
    )
    ap.add_argument('--start-presses', type=int, default=None, help='Override auto start presses at reset (default: 2 on first boot, 1 afterwards)')
    ap.add_argument('--start-hold-frames', type=int, default=None, help='Frames to hold START during auto-start (default 6)')
    ap.add_argument('--start-gap-frames', type=int, default=None, help='Frames between START taps (default 40)')
    ap.add_argument('--start-level-taps', type=int, default=None, help='Number of LEFT taps on level select before starting (default 12)')
    ap.add_argument('--start-settle-frames', type=int, default=None, help='Frames to run NOOP after auto-start before gameplay (default 180)')
    ap.add_argument('--start-wait-frames', type=int, default=None, help='Additional NOOP frames while waiting for virus count to become positive (default 600)')
    ap.add_argument('--backend', type=str, default=None, help='libretro (default), stable-retro, or mock')
    ap.add_argument('--core-path', type=str, default=None, help='Override DRMARIO_CORE_PATH for libretro backend')
    ap.add_argument('--rom-path', type=str, default=None, help='Override DRMARIO_ROM_PATH for libretro backend')
    ap.add_argument('--no-auto-start', action='store_true', help='Disable automatic START presses at reset')
    ap.add_argument('--state-repr', choices=['extended', 'bitplane'], default='extended', help='State tensor layout when --mode state is used')
    ap.add_argument('--save-frames', type=str, default=None, help='Directory to dump rendered frames (PNG if Pillow installed, else .npy)')
    ap.add_argument('--show-window', action='store_true', help='Open an OpenCV window showing live frames (press q to exit early)')
    ap.add_argument('--display-scale', type=float, default=2.0, help='Scale factor for the preview window (requires --show-window)')
    ap.add_argument('--viz-refresh-hz', type=float, default=60.0, help='Max refresh rate for the preview window (0 disables throttling)')
    ap.add_argument(
        '--emu-target-hz',
        type=float,
        default=0.0,
        help='Throttle environment stepping to this rate (0 keeps emulator free-running)',
    )
    ap.add_argument(
        '--viz-sync',
        action='store_true',
        help='In state mode, wait for the preview to draw the latest state before advancing',
    )
    args = ap.parse_args()

    ram_specs.set_state_representation(args.state_repr)

    register_env_id()
    env_kwargs = {
        "obs_mode": args.mode,
        "level": args.level,
        "risk_tau": args.risk_tau,
        "render_mode": "rgb_array",
        "state_repr": args.state_repr,
    }
    if args.backend:
        env_kwargs["backend"] = args.backend
    if args.core_path:
        env_kwargs["core_path"] = Path(args.core_path).expanduser()
    if args.rom_path:
        env_kwargs["rom_path"] = Path(args.rom_path).expanduser()
    if args.no_auto_start:
        env_kwargs["auto_start"] = False
    env = make("DrMarioRetroEnv-v0", **env_kwargs)
    reset_options: Dict[str, Any] = {'frame_offset': args.frame_offset}
    if args.randomize_rng:
        reset_options['randomize_rng'] = True
    if args.start_presses is not None:
        reset_options['start_presses'] = int(args.start_presses)
    if args.start_hold_frames is not None:
        reset_options['start_hold_frames'] = int(args.start_hold_frames)
    if args.start_gap_frames is not None:
        reset_options['start_gap_frames'] = int(args.start_gap_frames)
    if args.start_level_taps is not None:
        reset_options['start_level_taps'] = int(args.start_level_taps)
    if args.start_settle_frames is not None:
        reset_options['start_settle_frames'] = int(args.start_settle_frames)
    if args.start_wait_frames is not None:
        reset_options['start_wait_viruses'] = int(args.start_wait_frames)
    obs, info = env.reset(options=reset_options)
    t0 = time.time()
    steps, reward_sum = 0, 0.0
    save_dir = None
    if args.save_frames:
        save_dir = Path(args.save_frames).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
        if Image is None:
            print("Pillow not installed; frames will be saved as .npy arrays.")

    def extract_state_stack(observation: Any) -> np.ndarray:
        core = observation["obs"] if isinstance(observation, dict) else observation
        return np.asarray(core)

    state_shape: Optional[Tuple[int, ...]] = None
    state_dtype: Optional[np.dtype] = None
    if args.mode == "state":
        initial_stack = extract_state_stack(obs)
        state_shape = initial_stack.shape
        state_dtype = initial_stack.dtype

    show_window = bool(args.show_window)
    viewer: Optional[_ProcessViewer] = None
    if show_window:
        if Image is None or ImageTk is None:
            print("Pillow (with ImageTk) is required for the live window; skipping preview.")
            show_window = False
        else:
            try:
                viewer_kwargs: Dict[str, Any] = {
                    "mode": args.mode,
                    "refresh_hz": args.viz_refresh_hz,
                    "sync_to_viewer": args.viz_sync,
                }
                if args.mode == "state":
                    if state_shape is None or state_dtype is None:
                        raise RuntimeError("State shape unavailable for state-mode viewer")
                    viewer_kwargs["state_shape"] = state_shape
                    viewer_kwargs["state_dtype"] = state_dtype
                viewer = _ProcessViewer(
                    "Dr. Mario",
                    float(args.display_scale),
                    with_stats=(args.mode == "state"),
                    **viewer_kwargs,
                )
            except Exception as exc:
                print(f"Viewer process unavailable ({exc}); skipping live window.")
                show_window = False
                viewer = None

    def save_frame(frame: np.ndarray, index: int) -> None:
        if save_dir is None:
            return
        fname = save_dir / f"frame_{index:05d}"
        if Image is not None:
            Image.fromarray(frame).save(fname.with_suffix(".png"))
        else:
            np.save(fname.with_suffix(".npy"), frame)

    def publish_frame(
        observation: Any,
        info_payload: Dict[str, Any],
        action_val: Optional[int],
        reward_val: float,
        step_idx: int,
        cumulative_val: float,
    ) -> None:
        nonlocal viewer, show_window, frame_index, state_shape, state_dtype
        if not (show_window or save_dir):
            return
        stats = annotate_stats(
            {
                "info": info_payload,
                "step": step_idx,
                "reward": reward_val,
                "cumulative": cumulative_val,
                "action": None if action_val is None else int(action_val),
            }
        )
        if args.mode == "state":
            state_stack = extract_state_stack(observation)
            if state_shape is None or state_dtype is None:
                state_shape = state_stack.shape
                state_dtype = state_stack.dtype
            if viewer is not None and not viewer.push(state_stack, stats):
                viewer = None
                show_window = False
            if save_dir:
                frame = _state_to_rgb(np.asarray(state_stack), info_payload)
                save_frame(frame, frame_index)
        else:
            frame = env.render()
            if viewer is not None and not viewer.push(frame, stats):
                viewer = None
                show_window = False
            if save_dir:
                save_frame(frame, frame_index)
        frame_index += 1

    frame_index = 0
    cumulative_reward = 0.0
    step_times: deque[float] = deque(maxlen=240)

    target_hz = max(0.0, float(args.emu_target_hz))
    viz_baseline_hz = args.viz_refresh_hz if args.viz_refresh_hz > 0 else 60.0
    emu_vis_ratio: Optional[float]
    if target_hz > 0 and viz_baseline_hz > 0:
        emu_vis_ratio = target_hz / viz_baseline_hz
    else:
        emu_vis_ratio = None
    ratio_step = 1.25
    min_ratio, max_ratio = 0.1, 8.0
    emu_fps = 0.0

    def annotate_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
        stats = dict(stats)
        stats["emu_fps"] = emu_fps
        stats["target_hz"] = target_hz if target_hz > 0 else None
        stats["emu_vis_ratio"] = emu_vis_ratio
        return stats

    publish_frame(obs, info, None, 0.0, steps, cumulative_reward)

    def apply_ratio(new_ratio: Optional[float], anchor_hz: Optional[float] = None) -> None:
        nonlocal target_hz, step_period, emu_vis_ratio, viz_baseline_hz
        base = anchor_hz if anchor_hz and anchor_hz > 0 else viz_baseline_hz
        if not base or base <= 0:
            base = 60.0
        viz_baseline_hz = base
        if new_ratio is None:
            emu_vis_ratio = None
            target_hz = 0.0
        else:
            clamped = min(max(new_ratio, min_ratio), max_ratio)
            emu_vis_ratio = clamped
            target_hz = base * clamped
        step_period = 1.0 / target_hz if target_hz > 0 else 0.0

    step_period = 1.0 / target_hz if target_hz > 0 else 0.0
    next_step_time = time.perf_counter()

    episodes = 0
    last_info = info
    interrupted = False
    paused = False
    step_requests = 0
    try:
        while steps < args.steps:
            if viewer is not None:
                while True:
                    command = viewer.poll_control()
                    if not command:
                        break
                    if command.get("type") == "ratio":
                        mode = command.get("mode")
                        if mode == "match":
                            viz_fps = command.get("viz_fps")
                            anchor = float(viz_fps) if viz_fps else viz_baseline_hz
                            apply_ratio(1.0, anchor_hz=anchor)
                        elif mode == "faster":
                            if emu_vis_ratio is None:
                                apply_ratio(ratio_step)
                            else:
                                apply_ratio(emu_vis_ratio * ratio_step)
                        elif mode == "slower":
                            if emu_vis_ratio is None:
                                apply_ratio(1.0 / ratio_step)
                            else:
                                apply_ratio(emu_vis_ratio / ratio_step)
                    elif command.get("type") == "pause_toggle":
                        paused = not paused
                    elif command.get("type") == "step_once":
                        frames = int(command.get("frames", 1) or 1)
                        step_requests += max(1, frames)
            if step_period > 0:
                now = time.perf_counter()
                if now < next_step_time:
                    time.sleep(next_step_time - now)
                    now = next_step_time
                else:
                    now = time.perf_counter()
                next_step_time = now + step_period
            if paused and step_requests <= 0:
                time.sleep(0.01)
                continue
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            step_times.append(time.perf_counter())
            if len(step_times) >= 2:
                span = step_times[-1] - step_times[0]
                if span > 0:
                    emu_fps = float(len(step_times) - 1) / span
            if args.dump_state and args.mode == 'state':
                core = extract_state_stack(obs)
                latest = core[-1]
                nz = [int(latest[c].sum()) for c in range(latest.shape[0])]
                print('nz per channel:', nz)
            cumulative_reward += r
            reward_sum += r
            steps += 1
            publish_frame(obs, info, int(a), r, steps, cumulative_reward)
            last_info = info
            if paused and step_requests > 0:
                step_requests = max(0, step_requests - 1)
            if term or trunc:
                episodes += 1
                if steps >= args.steps:
                    break
                cumulative_reward = 0.0
                step_times.clear()
                obs, info = env.reset(options=reset_options)
                publish_frame(obs, info, None, 0.0, steps, cumulative_reward)
                last_info = info
                continue
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted by user; shutting down demo.", file=sys.stderr)
    finally:
        dt = time.time() - t0
        fps = steps / max(dt, 1e-6)
        backend_name = getattr(env.unwrapped, "backend_name", "unknown")
        summary_info = last_info if isinstance(last_info, dict) else {}
        suffix = " (interrupted)" if interrupted else ""
        summary = (
            f"Ran {steps} steps, reward={reward_sum:.1f}, cleared={summary_info.get('cleared', False)}, "
            f"FPS≈{fps:.1f}, backend={backend_name}, active={summary_info.get('backend_active', False)}, episodes={episodes}"
        )
        print(summary + suffix)
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == '__main__':
    main()
