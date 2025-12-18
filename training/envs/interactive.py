from __future__ import annotations

"""Interactive wrappers for vector environments (pause/step/speed control).

These utilities are used by `training.run` when running with the interactive
debug TUI (board rendering + playback controls). The goal is to let the UI
control *wall-clock playback* without changing the training algorithms.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np


class StopTraining(RuntimeError):
    """Raised to interrupt training from an interactive UI."""


@dataclass
class PlaybackControl:
    """Thread-safe playback controls shared between the UI and env wrapper.

    Semantics:
      - If `max_speed` is True, do not sleep to enforce a target FPS.
      - Otherwise, aim for `target_hz = base_fps * speed_x` frames/sec where
        "frames" are derived from `placements/tau` when present (macro env),
        else assumed to be 1 per env step (controller env).
      - If paused, block env.step until either unpaused or `pending_steps > 0`.
    """

    base_fps: float = 60.098813897  # NES NTSC (approx)
    speed_x: float = 1.0
    max_speed: bool = True
    paused: bool = False
    pending_steps: int = 0
    stop_requested: bool = False
    rng_randomize: bool = False

    def __post_init__(self) -> None:
        self._cond = threading.Condition()
        self._timing_reset = False
        self._rng_dirty = True

    # ---------------------------- UI actions (thread-safe)
    def request_stop(self) -> None:
        with self._cond:
            self.stop_requested = True
            self._cond.notify_all()

    def toggle_pause(self) -> None:
        with self._cond:
            self.paused = not self.paused
            self._timing_reset = True
            self._cond.notify_all()

    def step_once(self, frames: int = 1) -> None:
        with self._cond:
            self.pending_steps += max(1, int(frames))
            self._timing_reset = True
            self._cond.notify_all()

    def set_max_speed(self, enabled: bool) -> None:
        with self._cond:
            self.max_speed = bool(enabled)
            self._timing_reset = True
            self._cond.notify_all()

    def set_speed_x(self, speed_x: float) -> None:
        with self._cond:
            self.speed_x = float(max(0.05, min(speed_x, 512.0)))
            self.max_speed = False
            self._timing_reset = True
            self._cond.notify_all()

    def faster(self, factor: float = 1.25) -> None:
        with self._cond:
            self.speed_x = float(min(self.speed_x * float(factor), 512.0))
            self.max_speed = False
            self._timing_reset = True
            self._cond.notify_all()

    def slower(self, factor: float = 1.25) -> None:
        with self._cond:
            self.speed_x = float(max(self.speed_x / float(factor), 0.05))
            self.max_speed = False
            self._timing_reset = True
            self._cond.notify_all()

    def toggle_rng_randomize(self) -> None:
        with self._cond:
            self.rng_randomize = not bool(self.rng_randomize)
            self._rng_dirty = True
            self._cond.notify_all()

    def set_rng_randomize(self, enabled: bool) -> None:
        with self._cond:
            self.rng_randomize = bool(enabled)
            self._rng_dirty = True
            self._cond.notify_all()

    # ---------------------------- env-side helpers (thread-safe)
    def target_hz(self) -> float:
        with self._cond:
            if self.max_speed:
                return 0.0
            return float(max(0.0, self.base_fps * self.speed_x))

    def consume_timing_reset(self) -> bool:
        with self._cond:
            val = bool(self._timing_reset)
            self._timing_reset = False
            return val

    def consume_rng_update(self) -> Optional[bool]:
        with self._cond:
            if not self._rng_dirty:
                return None
            self._rng_dirty = False
            return bool(self.rng_randomize)

    def wait_until_step_allowed(self) -> None:
        with self._cond:
            while True:
                if self.stop_requested:
                    raise StopTraining("Stop requested by interactive UI")
                if not self.paused:
                    return
                if self.pending_steps > 0:
                    return
                self._cond.wait(timeout=0.05)

    def note_step_consumed(self) -> None:
        with self._cond:
            if self.paused and self.pending_steps > 0:
                self.pending_steps = max(0, int(self.pending_steps) - 1)

    def snapshot(self) -> Dict[str, Any]:
        with self._cond:
            return {
                "base_fps": float(self.base_fps),
                "speed_x": float(self.speed_x),
                "max_speed": bool(self.max_speed),
                "paused": bool(self.paused),
                "pending_steps": int(self.pending_steps),
                "target_hz": 0.0 if self.max_speed else float(self.base_fps * self.speed_x),
                "rng_randomize": bool(self.rng_randomize),
            }


def _extract_tau(info: Any) -> int:
    if not isinstance(info, dict):
        return 1
    tau = info.get("placements/tau", 1)
    if isinstance(tau, np.ndarray):
        try:
            return int(tau.item())
        except Exception:
            return 1
    try:
        return int(tau) if tau else 1
    except Exception:
        return 1


def _extract_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        try:
            value = value.item()
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


class RateLimitedVecEnv:
    """Vector env wrapper that implements pause/step and frame-rate limiting."""

    def __init__(self, env: Any, control: PlaybackControl) -> None:
        self.env = env
        self.control = control
        self.num_envs = int(getattr(env, "num_envs", 1))
        self._next_step_time = time.perf_counter()
        self._frames_total = 0.0
        self._fps_times: Deque[Tuple[float, float]] = deque(maxlen=240)
        self._lock = threading.Lock()
        self._last_infos: Optional[Sequence[Dict[str, Any]]] = None
        self._last_obs: Any = None
        self._last_tau_max: int = 1
        self._last_emu_fps: float = 0.0
        self._last_step_fps: float = 0.0
        self._step_calls_total: int = 0
        self._step_sec_total: float = 0.0
        self._last_rewards: Optional[np.ndarray] = None

        # Per-episode (level) live stats for the debug UI. This is independent of
        # the training algorithm and uses env-returned rewards + SMDP frame
        # counts (`placements/tau` when present).
        self._ep_return = np.zeros(self.num_envs, dtype=np.float64)
        self._ep_frames = np.zeros(self.num_envs, dtype=np.int64)
        self._ep_decisions = np.zeros(self.num_envs, dtype=np.int64)
        self._ep_return_last = np.zeros(self.num_envs, dtype=np.float64)
        self._ep_frames_last = np.zeros(self.num_envs, dtype=np.int64)
        self._ep_decisions_last = np.zeros(self.num_envs, dtype=np.int64)

        self._planner_build_calls_total: int = 0
        self._planner_build_sec_total: float = 0.0
        self._planner_plan_calls_total: int = 0
        self._planner_plan_sec_total: float = 0.0
        self._last_planner_build_sec: float = 0.0
        self._last_planner_plan_sec: float = 0.0

        self._inference_calls_total: int = 0
        self._inference_sec_total: float = 0.0
        self._last_inference_sec: float = 0.0

        self._update_calls_total: int = 0
        self._update_sec_total: float = 0.0
        self._update_frames_total: float = 0.0
        self._last_update_sec: float = 0.0
        self._last_update_frames: int = 0

    # ---------------------------- snapshot access (thread-safe)
    def latest_info(self, env_index: int = 0) -> Dict[str, Any]:
        with self._lock:
            if not self._last_infos:
                return {}
            if env_index < 0 or env_index >= len(self._last_infos):
                return dict(self._last_infos[0])
            return dict(self._last_infos[env_index])

    def latest_obs(self) -> Any:
        with self._lock:
            return self._last_obs

    def perf_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            frames_total = float(self._frames_total)

            inference_calls = int(self._inference_calls_total)
            inference_sec = float(self._inference_sec_total)
            planner_build_calls = int(self._planner_build_calls_total)
            planner_build_sec = float(self._planner_build_sec_total)
            planner_plan_calls = int(self._planner_plan_calls_total)
            planner_plan_sec = float(self._planner_plan_sec_total)
            planner_calls = int(planner_build_calls + planner_plan_calls)
            planner_sec = float(planner_build_sec + planner_plan_sec)

            def _ms_per_call(sec_total: float, calls: int) -> float:
                if calls <= 0:
                    return 0.0
                return float(sec_total) * 1000.0 / float(calls)

            def _ms_per_frame(sec_total: float) -> float:
                if frames_total <= 0.0:
                    return 0.0
                return float(sec_total) * 1000.0 / frames_total

            return {
                "frames_total": frames_total,
                "tau_max": int(self._last_tau_max),
                "emu_fps": float(self._last_emu_fps),
                "step_fps": float(self._last_step_fps),
                "step_calls_total": int(self._step_calls_total),
                "step_ms_per_frame": _ms_per_frame(float(self._step_sec_total)),
                "reward_last": float(self._last_rewards[0]) if self._last_rewards is not None else 0.0,
                "ep_return_curr": float(self._ep_return[0]) if self._ep_return.size else 0.0,
                "ep_frames_curr": int(self._ep_frames[0]) if self._ep_frames.size else 0,
                "ep_decisions_curr": int(self._ep_decisions[0]) if self._ep_decisions.size else 0,
                "ep_return_last": float(self._ep_return_last[0]) if self._ep_return_last.size else 0.0,
                "ep_frames_last": int(self._ep_frames_last[0]) if self._ep_frames_last.size else 0,
                "ep_decisions_last": int(self._ep_decisions_last[0]) if self._ep_decisions_last.size else 0,

                "inference_calls": inference_calls,
                "inference_ms_per_call": _ms_per_call(inference_sec, inference_calls),
                "inference_ms_per_frame": _ms_per_frame(inference_sec),
                "last_inference_ms": float(self._last_inference_sec) * 1000.0,

                "planner_calls": planner_calls,
                "planner_ms_per_call": _ms_per_call(planner_sec, planner_calls),
                "planner_ms_per_frame": _ms_per_frame(planner_sec),
                "planner_build_calls": planner_build_calls,
                "planner_build_ms_per_call": _ms_per_call(planner_build_sec, planner_build_calls),
                "planner_build_ms_per_frame": _ms_per_frame(planner_build_sec),
                "planner_plan_calls": planner_plan_calls,
                "planner_plan_ms_per_call": _ms_per_call(planner_plan_sec, planner_plan_calls),
                "planner_plan_ms_per_frame": _ms_per_frame(planner_plan_sec),
                "last_planner_build_ms": float(self._last_planner_build_sec) * 1000.0,
                "last_planner_plan_ms": float(self._last_planner_plan_sec) * 1000.0,

                "update_calls": int(self._update_calls_total),
                "update_sec_last": float(self._last_update_sec),
                "update_frames_last": int(self._last_update_frames),
                "update_ms_per_frame": (
                    float(self._last_update_sec) * 1000.0 / float(self._last_update_frames)
                    if int(self._last_update_frames) > 0
                    else 0.0
                ),
                "update_ms_per_frame_avg": (
                    float(self._update_sec_total) * 1000.0 / float(self._update_frames_total)
                    if self._update_frames_total > 0.0
                    else 0.0
                ),
            }

    # ---------------------------- external perf hooks (training thread)
    def record_inference(self, duration_sec: float) -> None:
        dur = float(max(0.0, duration_sec))
        with self._lock:
            self._inference_calls_total += 1
            self._inference_sec_total += dur
            self._last_inference_sec = dur

    def record_update(self, duration_sec: float, *, frames: int = 0) -> None:
        dur = float(max(0.0, duration_sec))
        frame_count = int(max(0, frames))
        with self._lock:
            self._update_calls_total += 1
            self._update_sec_total += dur
            self._update_frames_total += float(frame_count)
            self._last_update_sec = dur
            self._last_update_frames = frame_count

    # ---------------------------- vector api
    def reset(self, *args: Any, **kwargs: Any):
        obs, infos = self.env.reset(*args, **kwargs)
        with self._lock:
            self._last_obs = obs
            self._last_infos = infos if isinstance(infos, (list, tuple)) else [infos]
            self._last_tau_max = 1
            self._last_emu_fps = 0.0
            self._last_step_fps = 0.0
            self._frames_total = 0.0
            self._fps_times.clear()
            self._step_calls_total = 0
            self._step_sec_total = 0.0
            self._last_rewards = None
            self._ep_return.fill(0.0)
            self._ep_frames.fill(0)
            self._ep_decisions.fill(0)
            self._ep_return_last.fill(0.0)
            self._ep_frames_last.fill(0)
            self._ep_decisions_last.fill(0)

            self._planner_build_calls_total = 0
            self._planner_build_sec_total = 0.0
            self._planner_plan_calls_total = 0
            self._planner_plan_sec_total = 0.0
            self._last_planner_build_sec = 0.0
            self._last_planner_plan_sec = 0.0

            self._inference_calls_total = 0
            self._inference_sec_total = 0.0
            self._last_inference_sec = 0.0

            self._update_calls_total = 0
            self._update_sec_total = 0.0
            self._update_frames_total = 0.0
            self._last_update_sec = 0.0
            self._last_update_frames = 0
        self._next_step_time = time.perf_counter()
        return obs, infos

    def step(self, actions: Any):
        self.control.wait_until_step_allowed()

        rng_update = self.control.consume_rng_update()
        if rng_update is not None:
            try:
                if hasattr(self.env, "set_attr"):
                    self.env.set_attr("rng_randomize", bool(rng_update))
                else:
                    setattr(self.env, "rng_randomize", bool(rng_update))
            except Exception:
                pass

        if self.control.consume_timing_reset():
            self._next_step_time = time.perf_counter()

        target_hz = float(self.control.target_hz())
        if target_hz > 0.0:
            now = time.perf_counter()
            if now < self._next_step_time:
                time.sleep(self._next_step_time - now)

        step_start = time.perf_counter()
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        step_end = time.perf_counter()

        infos_seq: Sequence[Dict[str, Any]]
        if isinstance(infos, (list, tuple)):
            infos_seq = infos
        else:
            infos_seq = [infos]

        tau_max = 1
        try:
            tau_max = max(1, max(_extract_tau(info) for info in infos_seq))
        except Exception:
            tau_max = 1

        if target_hz > 0.0:
            # Schedule next step based on the number of emulated frames this step consumed.
            step_period = float(tau_max) / float(target_hz)
            self._next_step_time = step_start + step_period

        # Update FPS estimate (emulated frames / wall time).
        with self._lock:
            self._last_obs = obs
            self._last_infos = infos_seq
            try:
                self._last_rewards = np.asarray(rewards, dtype=np.float64).reshape(self.num_envs)
            except Exception:
                self._last_rewards = None
            self._last_tau_max = int(tau_max)
            self._step_calls_total += 1
            self._frames_total += float(tau_max)
            dt = float(step_end - step_start)
            self._step_sec_total += dt
            if dt > 1e-9:
                self._last_step_fps = float(tau_max) / dt
            self._fps_times.append((step_end, self._frames_total))
            if len(self._fps_times) >= 2:
                t0, f0 = self._fps_times[0]
                t1, f1 = self._fps_times[-1]
                span = float(t1 - t0)
                if span > 1e-6:
                    self._last_emu_fps = float(f1 - f0) / span

            for info in infos_seq:
                if not isinstance(info, dict):
                    continue
                build_sec = _extract_float(info.get("perf/planner_build_sec"))
                if build_sec is not None:
                    self._planner_build_calls_total += 1
                    self._planner_build_sec_total += float(max(0.0, build_sec))
                    self._last_planner_build_sec = float(max(0.0, build_sec))
                plan_sec = _extract_float(info.get("perf/planner_plan_sec"))
                if plan_sec is not None:
                    self._planner_plan_calls_total += 1
                    self._planner_plan_sec_total += float(max(0.0, plan_sec))
                    self._last_planner_plan_sec = float(max(0.0, plan_sec))

            # Live per-episode stats for env0 (and friends).
            try:
                terminated_arr = np.asarray(terminated, dtype=bool).reshape(self.num_envs)
                truncated_arr = np.asarray(truncated, dtype=bool).reshape(self.num_envs)
                rewards_arr = (
                    np.asarray(rewards, dtype=np.float64).reshape(self.num_envs)
                    if self._last_rewards is not None
                    else np.zeros((self.num_envs,), dtype=np.float64)
                )
            except Exception:
                terminated_arr = np.zeros((self.num_envs,), dtype=bool)
                truncated_arr = np.zeros((self.num_envs,), dtype=bool)
                rewards_arr = np.zeros((self.num_envs,), dtype=np.float64)

            for i in range(self.num_envs):
                info_i = infos_seq[i] if i < len(infos_seq) and isinstance(infos_seq[i], dict) else {}
                tau_i = _extract_tau(info_i)
                self._ep_return[i] += float(rewards_arr[i])
                self._ep_frames[i] += int(tau_i)
                self._ep_decisions[i] += 1
                if bool(terminated_arr[i] or truncated_arr[i]):
                    self._ep_return_last[i] = float(self._ep_return[i])
                    self._ep_frames_last[i] = int(self._ep_frames[i])
                    self._ep_decisions_last[i] = int(self._ep_decisions[i])
                    self._ep_return[i] = 0.0
                    self._ep_frames[i] = 0
                    self._ep_decisions[i] = 0

        self.control.note_step_consumed()
        return obs, rewards, terminated, truncated, infos_seq

    def render(self, *args: Any, **kwargs: Any) -> Any:
        frame = None
        if hasattr(self.env, "render"):
            frame = self.env.render(*args, **kwargs)
        # Gymnasium VectorEnvs sometimes return a list of frames.
        if isinstance(frame, (list, tuple)) and frame:
            return frame[0]
        return frame

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()

    # ---------------------------- attribute forwarding
    def __getattr__(self, name: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self.env, name)
