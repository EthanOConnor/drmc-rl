"""Single-environment training experiment for level-0 speedrun attempts.

This runner mirrors the demo interface (viewer window, state rendering) while
adding a lightweight monitor process that tracks hyperparameters and episode
scores. It is intentionally simple: a single environment, sequential episodes,
and a pluggable heuristic/epsilon-greedy strategy. Future work can swap in a
real learner; the scaffolding (monitor, viewer, reward logging) remains useful.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from envs.retro.demo import _ProcessViewer
from envs.retro.register_env import register_env_id
from envs.retro.drmario_env import Action
from gymnasium import make


def _monitor_worker(
    title: str,
    init_payload: Dict[str, Any],
    score_queue: "mp.Queue",
    stop_event: "mp.Event",
) -> None:
    try:
        import tkinter as tk
    except Exception:
        stop_event.set()
        return

    root = tk.Tk()
    root.title(title)

    strategy = str(init_payload.get("strategy", "unknown"))
    hyperparams = init_payload.get("hyperparams", {})

    def format_hyperparams(hparams: Dict[str, Any]) -> str:
        lines = [f"Strategy: {strategy}"]
        for key in sorted(hparams):
            lines.append(f"{key}: {hparams[key]}")
        return "\n".join(lines)

    hyper_label = tk.Label(root, text=format_hyperparams(hyperparams), anchor="w", justify="left")
    hyper_label.pack(fill="x", padx=8, pady=4)

    status_var = tk.StringVar(value="Waiting for runs...")
    status_label = tk.Label(root, textvariable=status_var, anchor="w", justify="left")
    status_label.pack(fill="x", padx=8, pady=2)

    canvas_width, canvas_height = 480, 260
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
    canvas.pack(fill="both", expand=True, padx=8, pady=8)

    scores: list[Tuple[int, float]] = []

    def render_plot() -> None:
        canvas.delete("plot")
        if not scores:
            return
        runs = [run for run, _ in scores]
        values = [val for _, val in scores]
        min_run, max_run = min(runs), max(runs)
        min_val, max_val = min(values), max(values)
        if max_run == min_run:
            max_run += 1
        if max_val == min_val:
            max_val += 1
        margin = 32
        plot_w = canvas_width - 2 * margin
        plot_h = canvas_height - 2 * margin

        # Axes
        canvas.create_line(margin, margin, margin, margin + plot_h, tags="plot", fill="#444")
        canvas.create_line(margin, margin + plot_h, margin + plot_w, margin + plot_h, tags="plot", fill="#444")

        # Plot polyline
        points: list[float] = []
        for run, value in scores:
            x_norm = (run - min_run) / (max_run - min_run)
            y_norm = (value - min_val) / (max_val - min_val)
            x = margin + x_norm * plot_w
            y = margin + plot_h - y_norm * plot_h
            points.extend([x, y])
        if len(points) >= 4:
            canvas.create_line(*points, tags="plot", fill="#0077cc", width=2.0)

        # Latest point marker
        last_x, last_y = points[-2], points[-1]
        canvas.create_oval(
            last_x - 3,
            last_y - 3,
            last_x + 3,
            last_y + 3,
            tags="plot",
            fill="#ff6600",
            outline="",
        )

        canvas.create_text(
            margin,
            margin - 10,
            anchor="w",
            tags="plot",
            text=f"Runs {min_run}–{max_run} | Score {values[-1]:.1f}",
            fill="#222",
        )

    def pump_queue() -> None:
        updated = False
        try:
            while True:
                message = score_queue.get_nowait()
                if message is None:
                    stop_event.set()
                    break
                if not isinstance(message, dict):
                    continue
                mtype = message.get("type")
                if mtype == "score":
                    run_idx = int(message.get("run", len(scores)))
                    score_val = float(message.get("score", 0.0))
                    scores.append((run_idx, score_val))
                    status_var.set(f"Run {run_idx}: score {score_val:.2f}")
                    updated = True
                elif mtype == "status":
                    text = message.get("text")
                    if text is not None:
                        status_var.set(str(text))
                elif mtype == "hyperparams":
                    payload = message.get("hyperparams")
                    if isinstance(payload, dict):
                        hyper_label.configure(text=format_hyperparams(payload))
        except queue.Empty:
            pass

        if updated:
            render_plot()

        if stop_event.is_set():
            try:
                root.destroy()
            except Exception:
                pass
            return
        root.after(200, pump_queue)

    root.after(200, pump_queue)
    try:
        root.mainloop()
    finally:
        stop_event.set()


class ExperimentMonitor:
    def __init__(self, hyperparams: Dict[str, Any], strategy: str) -> None:
        self._ctx = mp.get_context("spawn")
        self._queue: mp.Queue = self._ctx.Queue(maxsize=64)
        self._stop = self._ctx.Event()
        self._proc = self._ctx.Process(
            target=_monitor_worker,
            args=("Speedrun Monitor", {"hyperparams": hyperparams, "strategy": strategy}, self._queue, self._stop),
        )
        self._proc.daemon = True
        self._proc.start()
        self._closed = False

    def publish_score(self, run_idx: int, score: float) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait({"type": "score", "run": run_idx, "score": score})
        except queue.Full:
            pass
        except Exception:
            pass

    def publish_status(self, text: str) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait({"type": "status", "text": text})
        except queue.Full:
            pass
        except Exception:
            pass

    def update_hyperparams(self, hyperparams: Dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait({"type": "hyperparams", "hyperparams": hyperparams})
        except queue.Full:
            pass
        except Exception:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        self._stop.set()
        try:
            self._proc.join(timeout=1.0)
        finally:
            if self._proc.is_alive():
                self._proc.terminate()


class SpeedrunAgent:
    def __init__(
        self,
        action_space,
        strategy: str,
        epsilon: float,
        epsilon_decay: float,
        epsilon_min: float,
    ) -> None:
        self._action_space = action_space
        self._strategy = strategy
        self.epsilon = float(epsilon)
        self._epsilon_decay = float(epsilon_decay)
        self._epsilon_min = float(epsilon_min)
        self._rng = np.random.default_rng()

    def begin_episode(self) -> None:
        pass

    def select_action(self, _obs: Any, _info: Dict[str, Any]) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._action_space.sample())
        if self._strategy == "down":
            return int(Action.DOWN)
        if self._strategy == "down_hold":
            return int(Action.DOWN_HOLD)
        if self._strategy == "noop":
            return int(Action.NOOP)
        return int(self._action_space.sample())

    def end_episode(self, _score: float) -> None:
        if self.epsilon > self._epsilon_min:
            self.epsilon = max(self._epsilon_min, self.epsilon * self._epsilon_decay)


def _extract_state_stack(observation: Any) -> np.ndarray:
    core = observation["obs"] if isinstance(observation, dict) else observation
    return np.asarray(core)


def _format_hyperparams_for_monitor(args: argparse.Namespace, agent: SpeedrunAgent) -> Dict[str, Any]:
    return {
        "runs": args.runs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "epsilon": round(agent.epsilon, 4),
        "epsilon_decay": args.epsilon_decay,
        "epsilon_min": args.epsilon_min,
        "randomize_rng": bool(args.randomize_rng),
        "frame_offset": args.frame_offset,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Level-0 speedrun experiment harness")
    ap.add_argument("--mode", choices=["pixel", "state"], default="state")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--runs", type=int, default=10, help="Number of training runs (episodes)")
    ap.add_argument("--max-steps", type=int, default=2000, help="Max environment steps per run")
    ap.add_argument("--risk-tau", type=float, default=0.5)
    ap.add_argument("--frame-offset", type=int, default=0)
    ap.add_argument("--randomize-rng", action="store_true")
    ap.add_argument("--backend", type=str, default=None)
    ap.add_argument("--core-path", type=str, default=None)
    ap.add_argument("--rom-path", type=str, default=None)
    ap.add_argument("--no-auto-start", action="store_true")
    ap.add_argument("--start-presses", type=int, default=None)
    ap.add_argument("--start-hold-frames", type=int, default=None)
    ap.add_argument("--start-gap-frames", type=int, default=None)
    ap.add_argument("--start-level-taps", type=int, default=None)
    ap.add_argument("--start-settle-frames", type=int, default=None)
    ap.add_argument("--start-wait-frames", type=int, default=None)
    ap.add_argument("--emu-target-hz", type=float, default=0.0)
    ap.add_argument("--viz-sync", action="store_true")
    ap.add_argument("--display-scale", type=float, default=2.0)
    ap.add_argument("--viz-refresh-hz", type=float, default=60.0)
    ap.add_argument("--no-show-window", action="store_true")
    ap.add_argument("--no-monitor", action="store_true")
    ap.add_argument("--strategy", choices=["random", "down", "down_hold", "noop"], default="random")
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--epsilon-decay", type=float, default=0.97)
    ap.add_argument("--epsilon-min", type=float, default=0.05)

    args = ap.parse_args()

    register_env_id()

    env_kwargs: Dict[str, Any] = {
        "obs_mode": args.mode,
        "level": args.level,
        "risk_tau": args.risk_tau,
        "render_mode": "rgb_array",
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

    reset_options: Dict[str, Any] = {"frame_offset": args.frame_offset}
    if args.randomize_rng:
        reset_options["randomize_rng"] = True
    if args.start_presses is not None:
        reset_options["start_presses"] = int(args.start_presses)
    if args.start_hold_frames is not None:
        reset_options["start_hold_frames"] = int(args.start_hold_frames)
    if args.start_gap_frames is not None:
        reset_options["start_gap_frames"] = int(args.start_gap_frames)
    if args.start_level_taps is not None:
        reset_options["start_level_taps"] = int(args.start_level_taps)
    if args.start_settle_frames is not None:
        reset_options["start_settle_frames"] = int(args.start_settle_frames)
    if args.start_wait_frames is not None:
        reset_options["start_wait_viruses"] = int(args.start_wait_frames)

    obs, info = env.reset(options=reset_options)

    agent = SpeedrunAgent(
        env.action_space,
        strategy=args.strategy,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
    )

    monitor: Optional[ExperimentMonitor]
    if args.no_monitor:
        monitor = None
    else:
        monitor = ExperimentMonitor(_format_hyperparams_for_monitor(args, agent), args.strategy)

    show_window = not args.no_show_window
    viewer: Optional[_ProcessViewer] = None
    state_shape: Optional[Tuple[int, ...]] = None
    state_dtype: Optional[np.dtype] = None
    if args.mode == "state":
        stack = _extract_state_stack(obs)
        state_shape = stack.shape
        state_dtype = stack.dtype
    if show_window:
        try:
            viewer_kwargs: Dict[str, Any] = {
                "mode": args.mode,
                "refresh_hz": args.viz_refresh_hz,
                "sync_to_viewer": args.viz_sync,
            }
            if args.mode == "state":
                if state_shape is None or state_dtype is None:
                    raise RuntimeError("State viewer requires shape information")
                viewer_kwargs["state_shape"] = state_shape
                viewer_kwargs["state_dtype"] = state_dtype
            viewer = _ProcessViewer(
                "Dr. Mario Trainer",
                float(args.display_scale),
                with_stats=(args.mode == "state"),
                **viewer_kwargs,
            )
        except Exception as exc:
            print(f"Viewer unavailable: {exc}", file=sys.stderr)
            viewer = None
            show_window = False

    frame_index = 0
    total_steps = 0
    step_times: deque[float] = deque(maxlen=240)
    target_hz = max(0.0, float(args.emu_target_hz))
    step_period = 1.0 / target_hz if target_hz > 0 else 0.0
    next_step_time = time.perf_counter()
    emu_fps = 0.0

    def publish_frame(
        observation: Any,
        info_payload: Dict[str, Any],
        action_val: Optional[int],
        reward_val: float,
        step_idx: int,
        run_idx: int,
        episode_reward: float,
    ) -> None:
        nonlocal viewer, show_window, frame_index, state_shape, state_dtype
        if viewer is None:
            return
        stats = {
            "info": info_payload,
            "step": step_idx,
            "reward": reward_val,
            "cumulative": episode_reward,
            "action": None if action_val is None else int(action_val),
            "run": run_idx,
            "emu_fps": emu_fps,
            "target_hz": target_hz if target_hz > 0 else None,
            "emu_vis_ratio": None,
        }
        if args.mode == "state":
            stack = _extract_state_stack(observation)
            if state_shape is None or state_dtype is None:
                state_shape = stack.shape
                state_dtype = stack.dtype
            if not viewer.push(stack, stats):
                viewer = None
                show_window = False
        else:
            frame = env.render()
            if not viewer.push(frame, stats):
                viewer = None
                show_window = False
        frame_index += 1

    results: list[float] = []
    cumulative_reward = 0.0

    publish_frame(obs, info, None, 0.0, total_steps, 0, cumulative_reward)

    try:
        for run_idx in range(args.runs):
            agent.begin_episode()
            run_reward = 0.0
            episode_steps = 0
            done = False
            if monitor is not None:
                monitor.publish_status(f"Run {run_idx + 1} / {args.runs}")

            while episode_steps < args.max_steps and not done:
                if target_hz > 0:
                    now = time.perf_counter()
                    if now < next_step_time:
                        time.sleep(max(0.0, next_step_time - now))
                        now = next_step_time
                    else:
                        now = time.perf_counter()
                    next_step_time = now + step_period

                action = agent.select_action(obs, info)
                obs, reward, term, trunc, info = env.step(action)
                run_reward += reward
                cumulative_reward += reward
                episode_steps += 1
                total_steps += 1

                step_times.append(time.perf_counter())
                if len(step_times) >= 2:
                    span = step_times[-1] - step_times[0]
                    if span > 0:
                        emu_fps = float(len(step_times) - 1) / span

                publish_frame(obs, info, int(action), reward, total_steps, run_idx, run_reward)

                if term or trunc:
                    done = True

            results.append(run_reward)
            agent.end_episode(run_reward)
            if monitor is not None:
                monitor.publish_score(run_idx, run_reward)
                monitor.update_hyperparams(_format_hyperparams_for_monitor(args, agent))

            print(
                f"Run {run_idx + 1}/{args.runs}: reward={run_reward:.2f} steps={episode_steps} "
                f"epsilon={agent.epsilon:.3f}"
            )

            obs, info = env.reset(options=reset_options)
            cumulative_reward = 0.0
            publish_frame(obs, info, None, 0.0, total_steps, run_idx + 1, cumulative_reward)

    except KeyboardInterrupt:
        print("Interrupted by user; stopping experiment.", file=sys.stderr)
    finally:
        if monitor is not None:
            monitor.close()
        if viewer is not None:
            viewer.close()
        env.close()
        if results:
            mean_reward = float(np.mean(results))
            best_reward = float(np.max(results))
            print(f"Completed {len(results)} runs. Mean reward={mean_reward:.2f}, best={best_reward:.2f}")


if __name__ == "__main__":
    main()

