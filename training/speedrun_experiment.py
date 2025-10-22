#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-environment training experiment for level-0 speedrun attempts.

This runner mirrors the demo interface (viewer window, state rendering) while
adding a lightweight monitor process that tracks hyperparameters and episode
scores. It is intentionally simple: a single environment, sequential episodes,
and a pluggable heuristic/epsilon-greedy strategy. Future work can swap in a
real learner; the scaffolding (monitor, viewer, reward logging) remains useful.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import numbers
import os
import pickle
import random
import queue
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from contextlib import nullcontext

import numpy as np

import envs.specs.ram_to_state as ram_specs
from envs.retro.demo import _ProcessViewer
from envs.retro.drmario_env import Action
from envs.retro.intent_wrapper import DrMarioIntentEnv
from envs.retro.register_env import register_env_id, register_intent_env_id
from gymnasium import make
from models.policy.networks import DrMarioStatePolicyNet, DrMarioPixelUNetPolicyNet
from training.discounting import discounted_returns_mlx, discounted_returns_torch
try:
    from models.policy.mlx_networks import DrMarioStatePolicyMLX
except Exception:  # pragma: no cover - optional dependency
    DrMarioStatePolicyMLX = None

try:
    import torch
    from torch import nn, optim
    from torch.distributions import Categorical
except Exception:  # pragma: no cover - torch is optional for this experiment
    torch = None
    nn = None
    optim = None
    Categorical = None

try:  # pragma: no cover - optional dependency
    import mlx.core as mx
    import mlx.nn as nn_mlx
    import mlx.optimizers as optim_mlx
except Exception:  # pragma: no cover - MLX is optional
    mx = None
    nn_mlx = None
    optim_mlx = None


def _mlx_log_softmax(tensor: "mx.array", axis: int = -1) -> "mx.array":
    """Compute a numerically stable log-softmax with MLX primitives."""

    if mx is None:
        raise RuntimeError("MLX backend is unavailable")
    if hasattr(mx, "log_softmax"):
        return mx.log_softmax(tensor, axis=axis)

    axis_normalized = axis if axis >= 0 else tensor.ndim + axis
    max_logits = mx.max(tensor, axis=axis_normalized, keepdims=True)
    shifted = tensor - max_logits
    exp_shifted = mx.exp(shifted)
    sum_exp = mx.sum(exp_shifted, axis=axis_normalized, keepdims=True)
    logsumexp = mx.log(sum_exp)
    return shifted - logsumexp


def _mlx_flip(tensor: "mx.array", axis: int = 0) -> "mx.array":
    """Flip an MLX tensor along the provided axis with compatibility fallbacks."""

    if mx is None:
        raise RuntimeError("MLX backend is unavailable")

    if hasattr(mx, "flip"):
        return mx.flip(tensor, axis=axis)

    if hasattr(mx, "reverse"):
        return mx.reverse(tensor, axis=axis)

    axis_normalized = axis if axis >= 0 else tensor.ndim + axis
    np_view = np.asarray(tensor)
    flipped = np.flip(np_view, axis=axis_normalized)
    return mx.array(flipped, dtype=tensor.dtype)


@dataclass
class _MLXDeviceInfo:
    """Runtime description of an MLX compute device."""

    ordinal: int
    identifier: str
    kind: str
    name: str
    hw_index: Optional[int]
    handle: Any
    is_default: bool = False
    memory_bytes: Optional[int] = None

    def summary(self) -> str:
        label_kind = self.kind.upper() if self.kind else "UNKNOWN"
        label = f"[{self.ordinal}] {label_kind}"
        if self.hw_index is not None:
            label += f":{self.hw_index}"
        if self.name:
            label += f" - {self.name}"
        if self.memory_bytes:
            label += f" ({_format_bytes(self.memory_bytes)})"
        if self.is_default:
            label += " (default)"
        return label

    def aliases(self) -> List[str]:
        tokens: set[str] = set()
        tokens.add(str(self.ordinal))
        if self.identifier:
            tokens.add(self.identifier.lower())
        if self.hw_index is not None:
            tokens.add(str(self.hw_index))
            if self.kind:
                tokens.add(f"{self.kind.lower()}:{self.hw_index}")
        if self.kind:
            tokens.add(self.kind.lower())
        if self.name:
            tokens.add(self.name.lower())
        return [token for token in tokens if token]


def _format_bytes(num_bytes: int) -> str:
    """Return a human readable representation of a byte count."""

    suffixes = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    value = float(max(num_bytes, 0))
    for suffix in suffixes:
        if value < 1024.0 or suffix == suffixes[-1]:
            if suffix == "B":
                return f"{int(value)} {suffix}"
            return f"{value:.1f} {suffix}"
        value /= 1024.0
    return f"{num_bytes} B"


def _safe_getattr(obj: Any, attr: str) -> Any:
    try:
        return getattr(obj, attr)
    except Exception:
        return None


def _mlx_get_default_device_handle() -> Any:
    if mx is None:
        return None
    default_candidate = _safe_getattr(mx, "default_device")
    if callable(default_candidate):
        try:
            return default_candidate()
        except Exception:
            return None
    return default_candidate


def _normalize_device_kind(kind_value: Any) -> str:
    if kind_value is None:
        return ""
    candidate = getattr(kind_value, "name", kind_value)
    return str(candidate).strip()


def _normalize_device_name(name_value: Any) -> str:
    if name_value is None:
        return ""
    return str(name_value).strip()


def _normalize_hw_index(index_value: Any) -> Optional[int]:
    if isinstance(index_value, numbers.Integral):
        return int(index_value)
    if isinstance(index_value, str) and index_value.isdigit():
        return int(index_value)
    return None


def _normalize_memory_size(memory_value: Any) -> Optional[int]:
    if isinstance(memory_value, numbers.Integral):
        return int(memory_value)
    if isinstance(memory_value, (tuple, list)) and memory_value:
        first = memory_value[0]
        if isinstance(first, numbers.Integral):
            return int(first)
    return None


def _mlx_device_tokens(device: Any) -> set[str]:
    tokens: set[str] = set()
    if device is None:
        return tokens
    try:
        tokens.add(str(device).lower())
    except Exception:
        pass
    for attr in ("name", "description", "label"):
        value = _safe_getattr(device, attr)
        if isinstance(value, str):
            tokens.add(value.lower())
    kind_value = _safe_getattr(device, "type") or _safe_getattr(device, "kind")
    if kind_value is not None:
        tokens.add(_normalize_device_kind(kind_value).lower())
    index_value = (
        _safe_getattr(device, "index")
        or _safe_getattr(device, "device_id")
        or _safe_getattr(device, "id")
        or _safe_getattr(device, "ordinal")
    )
    hw_index = _normalize_hw_index(index_value)
    if hw_index is not None:
        tokens.add(str(hw_index))
    return {token for token in tokens if token}


def _mlx_device_equal(lhs: Any, rhs: Any) -> bool:
    if lhs is rhs:
        return True
    if lhs is None or rhs is None:
        return False
    if isinstance(lhs, str) and isinstance(rhs, str):
        return lhs == rhs
    if isinstance(lhs, str):
        return lhs.lower() in _mlx_device_tokens(rhs)
    if isinstance(rhs, str):
        return rhs.lower() in _mlx_device_tokens(lhs)
    try:
        if lhs == rhs:
            return True
    except Exception:
        pass
    return bool(_mlx_device_tokens(lhs) & _mlx_device_tokens(rhs))


def _mlx_describe_device(ordinal: int, device: Any, default_handle: Any) -> _MLXDeviceInfo:
    kind_value = _safe_getattr(device, "type") or _safe_getattr(device, "kind")
    name_value = (
        _safe_getattr(device, "name")
        or _safe_getattr(device, "description")
        or _safe_getattr(device, "label")
    )
    index_value = (
        _safe_getattr(device, "index")
        or _safe_getattr(device, "device_id")
        or _safe_getattr(device, "id")
        or _safe_getattr(device, "ordinal")
    )
    memory_value = (
        _safe_getattr(device, "memory")
        or _safe_getattr(device, "memory_size")
        or _safe_getattr(device, "total_memory")
        or _safe_getattr(device, "capacity")
    )
    kind = _normalize_device_kind(kind_value)
    name = _normalize_device_name(name_value)
    hw_index = _normalize_hw_index(index_value)
    memory_bytes = _normalize_memory_size(memory_value)
    identifier_parts: List[str] = []
    if kind:
        identifier_parts.append(kind.lower())
    if hw_index is not None:
        identifier_parts.append(str(hw_index))
    identifier = ":".join(identifier_parts) if identifier_parts else str(ordinal)
    is_default = _mlx_device_equal(device, default_handle)
    return _MLXDeviceInfo(
        ordinal=ordinal,
        identifier=identifier,
        kind=kind,
        name=name,
        hw_index=hw_index,
        handle=device,
        is_default=is_default,
        memory_bytes=memory_bytes,
    )


def _mlx_collect_devices() -> List[_MLXDeviceInfo]:
    if mx is None:
        return []
    devices: Sequence[Any] = ()
    for attr in ("devices", "list_devices"):
        candidate = _safe_getattr(mx, attr)
        if candidate is None:
            continue
        if callable(candidate):
            try:
                result = candidate()
            except Exception:
                continue
        else:
            result = candidate
        if isinstance(result, dict):
            result = list(result.values())
        if isinstance(result, (list, tuple)):
            devices = list(result)
            break
    if not devices:
        default_handle = _mlx_get_default_device_handle()
        if default_handle is not None:
            devices = (default_handle,)
    default_handle = _mlx_get_default_device_handle()
    info_list: List[_MLXDeviceInfo] = []
    for ordinal, device in enumerate(devices):
        info_list.append(_mlx_describe_device(ordinal, device, default_handle))
    return info_list


def _mlx_resolve_device(spec: str, devices: Optional[Sequence[_MLXDeviceInfo]] = None) -> _MLXDeviceInfo:
    if mx is None:
        raise RuntimeError("MLX backend is unavailable.")
    normalized = spec.strip().lower()
    if devices is None:
        devices = _mlx_collect_devices()
    if not devices:
        raise RuntimeError("No MLX devices detected.")
    if normalized in {"", "default"}:
        for info in devices:
            if info.is_default:
                return info
        return devices[0]
    for info in devices:
        aliases = info.aliases()
        if normalized in aliases:
            return info
    for info in devices:
        aliases = info.aliases()
        if any(normalized in alias for alias in aliases):
            return info
    raise ValueError(f"Unknown MLX device spec '{spec}'.")


def _mlx_set_default_device(device: Any) -> None:
    if mx is None:
        raise RuntimeError("MLX backend is unavailable.")
    setters = [
        _safe_getattr(mx, "set_default_device"),
        _safe_getattr(mx, "set_device"),
    ]
    for setter in setters:
        if callable(setter):
            setter(device)
            return
    default_attr = _safe_getattr(mx, "default_device")
    if default_attr is not None and not callable(default_attr):
        try:
            setattr(mx, "default_device", device)
            return
        except Exception:
            pass
    raise RuntimeError(
        "Unable to set MLX default device: no supported setter found in current mlx version."
    )


def _mlx_configure_device(spec: Optional[str]) -> Optional[_MLXDeviceInfo]:
    if mx is None:
        return None
    devices = _mlx_collect_devices()
    if not devices:
        return None
    if spec is None:
        for info in devices:
            if info.is_default:
                return info
        return devices[0]
    info = _mlx_resolve_device(spec, devices)
    _mlx_set_default_device(info.handle)
    return info


def _mlx_print_available_devices() -> int:
    if mx is None:
        print("MLX backend is unavailable; please install mlx to inspect devices.", file=sys.stderr)
        return 1
    devices = _mlx_collect_devices()
    if not devices:
        print("No MLX devices detected by the current mlx installation.", file=sys.stderr)
        return 1
    print("Available MLX devices:")
    for info in devices:
        line = f"  {info.summary()}"
        alias = info.identifier.lower()
        if alias and alias != str(info.ordinal):
            line += f"  (alias: {alias})"
        print(line)
    return 0


COMPONENT_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("episode_reward", "Total"),
    ("pill_bonus_adjusted", "Pill"),
    ("action_penalty", "Action"),
    ("non_virus_bonus", "Non-virus"),
    ("adjacency_bonus", "Adjacency"),
    ("height_penalty_delta", "Height"),
    ("time_reward", "Time"),
)


def _monitor_worker(
    title: str,
    init_payload: Dict[str, Any],
    score_queue: "mp.Queue",
    command_queue: "mp.Queue",
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

    hyper_label = tk.Label(
        root, text=format_hyperparams(hyperparams), anchor="w", justify="left"
    )
    hyper_label.pack(fill="x", padx=8, pady=4)

    rng_enabled = bool(hyperparams.get("randomize_rng", False))

    status_var = tk.StringVar(value="Waiting for runs...")
    status_label = tk.Label(root, textvariable=status_var, anchor="w", justify="left")
    status_label.pack(fill="x", padx=8, pady=2)

    rng_state_var = tk.StringVar()
    seed_state_var = tk.StringVar()

    def set_rng_label() -> None:
        rng_state_var.set(f"RNG randomize: {'ON' if rng_enabled else 'OFF'}  (Ctrl+R)")

    current_seed_index = int(hyperparams.get("seed_index", 0))

    def set_seed_label() -> None:
        seed_state_var.set(f"Seed index: {current_seed_index}  (+/-)")

    set_rng_label()
    rng_label = tk.Label(root, textvariable=rng_state_var, anchor="w", justify="left")
    rng_label.pack(fill="x", padx=8, pady=2)

    diag_frame = tk.LabelFrame(root, text="Diagnostics")
    diag_frame.pack(fill="x", padx=8, pady=6)
    diagnostics_var = tk.StringVar(value="No diagnostics yet.")
    diagnostics_label = tk.Label(
        diag_frame,
        textvariable=diagnostics_var,
        anchor="w",
        justify="left",
        font=("Courier", 10),
    )
    diagnostics_label.pack(fill="x", padx=4, pady=4)

    canvas_width, canvas_height = 480, 260
    try:
        plot_window = tk.Toplevel(root)
        plot_window.title("Training Plots")
        reward_canvas = tk.Canvas(
            plot_window, width=canvas_width, height=canvas_height, bg="white"
        )
        reward_canvas.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        policy_canvas = tk.Canvas(
            plot_window, width=canvas_width, height=canvas_height, bg="white"
        )
        policy_canvas.pack(fill="both", expand=True, padx=8, pady=4)
        value_canvas = tk.Canvas(
            plot_window, width=canvas_width, height=canvas_height, bg="white"
        )
        value_canvas.pack(fill="both", expand=True, padx=8, pady=(4, 8))
    except Exception:
        plot_window = None
        reward_canvas = None
        policy_canvas = None
        value_canvas = None

    component_fields: list[Tuple[str, str]] = list(COMPONENT_FIELDS)
    component_columns: list[Dict[str, Any]] = []
    ticker_max_cols = 12
    ticker_history_limit = 120
    ticker_col_width = 120
    ticker_row_height = 18
    ticker_margin = 10
    ticker_canvas: Optional["tk.Canvas"]
    try:
        ticker_window = tk.Toplevel(root)
        ticker_window.title("Reward Components")
        ticker_canvas = tk.Canvas(
            ticker_window,
            width=ticker_col_width * ticker_max_cols + ticker_margin * 2,
            height=ticker_row_height * (len(component_fields) + 2) + ticker_margin * 2,
            bg="white",
        )
        ticker_canvas.pack(fill="both", expand=True)
    except Exception:
        ticker_canvas = None
        ticker_window = None

    def send_command(payload: Dict[str, Any]) -> None:
        if command_queue is None:
            return
        try:
            command_queue.put_nowait(payload)
        except queue.Full:
            try:
                command_queue.get_nowait()
                command_queue.put_nowait(payload)
            except Exception:
                pass
        except Exception:
            pass

    def choose_checkpoint_path(save: bool) -> Optional[str]:
        nonlocal last_checkpoint_path
        try:
            from tkinter import filedialog
        except Exception:
            return last_checkpoint_path
        initialfile = None
        initialdir = None
        if last_checkpoint_path:
            try:
                initial_path = Path(last_checkpoint_path)
                if initial_path.is_dir():
                    initialdir = str(initial_path)
                else:
                    initialdir = str(initial_path.parent)
                    initialfile = str(initial_path.name)
            except Exception:
                initialfile = None
                initialdir = None
        try:
            if save:
                filename = filedialog.asksaveasfilename(
                    title="Save checkpoint",
                    defaultextension=".pkl",
                    filetypes=[("Checkpoint", "*.pkl"), ("All files", "*.*")],
                    initialdir=initialdir,
                    initialfile=initialfile,
                )
            else:
                filename = filedialog.askopenfilename(
                    title="Load checkpoint",
                    filetypes=[("Checkpoint", "*.pkl"), ("All files", "*.*")],
                    initialdir=initialdir,
                )
        except Exception:
            filename = ""
        if not filename:
            return None
        path_obj = Path(filename).expanduser()
        if save and not path_obj.suffix:
            path_obj = path_obj.with_suffix(".pkl")
        last_checkpoint_path = str(path_obj)
        return last_checkpoint_path

    def request_checkpoint_save(_event=None) -> None:
        path = choose_checkpoint_path(save=True)
        if not path:
            return
        status_var.set(f"Saving checkpoint to {path}")
        send_command({"type": "checkpoint_save", "path": path})

    def request_checkpoint_load(_event=None) -> None:
        path = choose_checkpoint_path(save=False)
        if not path:
            return
        status_var.set(f"Loading checkpoint from {path}")
        send_command({"type": "checkpoint_load", "path": path})

    def toggle_rng(_event=None) -> None:
        nonlocal rng_enabled
        rng_enabled = not rng_enabled
        set_rng_label()
        send_command({"type": "rng_toggle", "enabled": rng_enabled})
        try:
            hyperparams["randomize_rng"] = rng_enabled
            hyper_label.configure(text=format_hyperparams(hyperparams))
        except Exception:
            pass
        try:
            status_var.set(
                f"Randomize RNG {'enabled' if rng_enabled else 'disabled'} (Ctrl+R to toggle)"
            )
        except Exception:
            pass

    def adjust_seed(delta: int) -> None:
        nonlocal current_seed_index
        current_seed_index = (current_seed_index + delta) % 1000000
        set_seed_label()
        send_command({"type": "seed_adjust", "seed_index": current_seed_index})
        try:
            hyperparams["seed_index"] = current_seed_index
            hyper_label.configure(text=format_hyperparams(hyperparams))
        except Exception:
            pass

    def inc_seed(_event=None) -> None:
        adjust_seed(1)

    def dec_seed(_event=None) -> None:
        adjust_seed(-1)

    def bind_shortcuts(widget: Optional[Any]) -> None:
        if widget is None:
            return
        try:
            widget.bind("<Control-r>", toggle_rng)
            widget.bind("<Control-R>", toggle_rng)
            widget.bind("<plus>", inc_seed)
            widget.bind("<KP_Add>", inc_seed)
            widget.bind("<minus>", dec_seed)
            widget.bind("<KP_Subtract>", dec_seed)
            widget.bind("<Control-s>", request_checkpoint_save)
            widget.bind("<Control-S>", request_checkpoint_save)
            widget.bind("<Control-o>", request_checkpoint_load)
            widget.bind("<Control-O>", request_checkpoint_load)
            widget.bind("<Control-l>", request_checkpoint_load)
            widget.bind("<Control-L>", request_checkpoint_load)
        except Exception:
            pass

    bind_shortcuts(root)
    bind_shortcuts(ticker_window if 'ticker_window' in locals() else None)
    bind_shortcuts(plot_window)

    scores: list[Tuple[int, float]] = []
    score_by_run: Dict[int, float] = {}
    batch_medians: Dict[int, float] = {}
    runs_total = int(hyperparams.get("runs", 0) or 0)
    parallel_envs_var = int(hyperparams.get("parallel_envs", 1) or 1)
    try:
        batch_runs_var = int(hyperparams.get("batch_runs", 1))
    except Exception:
        batch_runs_var = 1
    batch_runs_var = max(0, batch_runs_var)
    group_size_var = max(1, parallel_envs_var, batch_runs_var)
    latest_diagnostics: Dict[str, Any] = {}
    diagnostic_history: Dict[str, deque[float]] = {}
    diagnostics_window = 20
    policy_loss_series: list[Tuple[float, float]] = []
    value_loss_series: list[Tuple[float, float]] = []
    loss_history_limit = 200
    last_checkpoint_path: Optional[str] = None
    if isinstance(hyperparams, dict):
        checkpoint_hint = hyperparams.get("checkpoint_path")
        if isinstance(checkpoint_hint, str) and checkpoint_hint:
            last_checkpoint_path = checkpoint_hint

    def format_diagnostics() -> str:
        if not latest_diagnostics:
            return "No diagnostics yet."
        lines: list[str] = []
        grouped: Dict[str, list[Tuple[str, Any, str]]] = {}
        for key, value in latest_diagnostics.items():
            if "/" in key:
                group, metric = key.split("/", 1)
            else:
                group, metric = "", key
            grouped.setdefault(group, []).append((metric, value, key))

        for group in sorted(grouped):
            entries = sorted(grouped[group], key=lambda kv: kv[0])
            prefix = ""
            if group:
                lines.append(f"{group}:")
                prefix = "  "
            for metric, value, full_key in entries:
                if isinstance(value, numbers.Number):
                    history = diagnostic_history.get(full_key)
                    if history and len(history) > 1:
                        avg = float(sum(history) / len(history))
                        trend = history[-1] - history[0]
                        lines.append(
                            f"{prefix}{metric}: {float(value):.4f} "
                            f"(avg {avg:.4f}, diff {trend:+.4f})"
                        )
                    else:
                        lines.append(f"{prefix}{metric}: {float(value):.4f}")
                else:
                    lines.append(f"{prefix}{metric}: {value}")
        return "\n".join(lines)

    def append_series_point(series: list[Tuple[float, float]], x_val: float, y_val: float) -> None:
        point = (float(x_val), float(y_val))
        if series and abs(series[-1][0] - point[0]) < 1e-6:
            series[-1] = point
        else:
            series.append(point)
        if len(series) > loss_history_limit:
            del series[: len(series) - loss_history_limit]

    def render_line_plot(
        canvas_obj: Optional["tk.Canvas"],
        series: list[Tuple[float, float]],
        title: str,
        color: str,
        subtitle: Optional[str] = None,
    ) -> None:
        if canvas_obj is None:
            return
        try:
            canvas_obj.delete("plot")
        except Exception:
            return
        try:
            width = max(int(canvas_obj.winfo_width()), 10)
            height = max(int(canvas_obj.winfo_height()), 10)
        except Exception:
            width = canvas_width
            height = canvas_height
        left_margin = 40
        right_margin = 32
        bottom_margin = 40
        top_margin = 55  # leave room for titles
        plot_w = max(1, width - left_margin - right_margin)
        plot_h = max(1, height - top_margin - bottom_margin)
        axis_left = left_margin
        axis_top = top_margin
        axis_right = left_margin + plot_w
        axis_bottom = top_margin + plot_h

        canvas_obj.create_text(
            left_margin,
            8,
            anchor="nw",
            tags="plot",
            text=title,
            fill="#222",
        )
        if subtitle:
            canvas_obj.create_text(
                left_margin,
                24,
                anchor="nw",
                tags="plot",
                text=subtitle,
                fill="#444",
            )

        canvas_obj.create_line(
            axis_left,
            axis_top,
            axis_left,
            axis_bottom,
            tags="plot",
            fill="#444",
        )
        canvas_obj.create_line(
            axis_left,
            axis_bottom,
            axis_right,
            axis_bottom,
            tags="plot",
            fill="#444",
        )
        if not series:
            canvas_obj.create_text(
                width // 2,
                height // 2,
                anchor="center",
                tags="plot",
                text="No data yet",
                fill="#666",
            )
            return

        xs = [float(item[0]) for item in series]
        ys = [float(item[1]) for item in series]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if abs(max_x - min_x) < 1e-9:
            max_x = min_x + 1.0
        if abs(max_y - min_y) < 1e-9:
            max_y = min_y + 1.0

        points: list[float] = []
        for x_val, y_val in series:
            x_norm = (x_val - min_x) / (max_x - min_x)
            y_norm = (y_val - min_y) / (max_y - min_y)
            x = axis_left + x_norm * plot_w
            y = axis_top + plot_h - y_norm * plot_h
            points.extend([x, y])
        if len(points) >= 4:
            canvas_obj.create_line(*points, tags="plot", fill=color, width=2.0)
            last_x, last_y = points[-2], points[-1]
            canvas_obj.create_oval(
                last_x - 3,
                last_y - 3,
                last_x + 3,
                last_y + 3,
                tags="plot",
                fill="#ff6600",
                outline="",
            )

        # Zero reference line if range crosses zero
        if min_y < 0.0 < max_y:
            zero_norm = (0.0 - min_y) / (max_y - min_y)
            y_zero = axis_top + plot_h - zero_norm * plot_h
            canvas_obj.create_line(
                axis_left,
                y_zero,
                axis_right,
                y_zero,
                tags="plot",
                fill="#bbbbbb",
                dash=(3, 4),
            )

    def render_reward_plot() -> None:
        if not scores:
            render_line_plot(reward_canvas, [], "Batch Median Reward", "#0077cc")
            return
        runs = [float(run) for run, _ in scores]
        values = [float(val) for _, val in scores]
        subtitle = (
            f"Batches {int(min(runs))}–{int(max(runs))} | Median {values[-1]:.1f}"
            if runs
            else None
        )
        render_line_plot(reward_canvas, [(float(r), float(v)) for r, v in scores], "Batch Median Reward", "#0077cc", subtitle)

    def render_policy_plot() -> None:
        subtitle = None
        if policy_loss_series:
            latest = policy_loss_series[-1]
            subtitle = f"Latest {latest[1]:.4f} @ episode {int(latest[0])}"
        render_line_plot(policy_canvas, policy_loss_series, "Policy Loss", "#9c27b0", subtitle)

    def render_value_plot() -> None:
        subtitle = None
        if value_loss_series:
            latest = value_loss_series[-1]
            subtitle = f"Latest {latest[1]:.4f} @ episode {int(latest[0])}"
        render_line_plot(value_canvas, value_loss_series, "Value Loss", "#ff5722", subtitle)

    def render_all_plots() -> None:
        render_reward_plot()
        render_policy_plot()
        render_value_plot()

    render_all_plots()

    def recompute_series(group_size: int) -> None:
        scores.clear()
        batch_medians.clear()
        if group_size <= 0:
            return
        if not score_by_run:
            return
        max_run_idx = max(score_by_run)
        if max_run_idx < 0:
            return
        batches = (max_run_idx + group_size) // group_size
        for batch_idx in range(batches):
            batch_start = batch_idx * group_size
            expected = group_size
            if runs_total > 0:
                remaining = runs_total - batch_start
                if remaining < expected:
                    expected = max(1, remaining)
            collected: list[float] = []
            for r_idx in range(batch_start, batch_start + expected):
                if r_idx in score_by_run:
                    collected.append(score_by_run[r_idx])
            if collected and len(collected) == expected:
                median_val = float(np.median(collected))
                batch_medians[batch_idx] = median_val
        for b_idx in sorted(batch_medians):
            scores.append((b_idx + 1, batch_medians[b_idx]))

    def render_ticker() -> None:
        if ticker_canvas is None:
            return
        ticker_canvas.delete("all")
        if not component_columns:
            return
        columns = component_columns[-ticker_max_cols:]

        label_texts = [f"{label}:" for _, label in component_fields]
        label_width = max(len(text) for text in label_texts)
        value_width = 6
        sep = " |  "
        char_px = 8

        def format_value(raw: Optional[float]) -> str:
            if raw is None:
                return "-"
            try:
                return f"{float(raw):.0f}"
            except Exception:
                return "-"

        def format_row(values: list[Optional[float]]) -> str:
            parts: list[str] = []
            for idx, val in enumerate(values):
                text = format_value(val)
                text = text.rjust(value_width)
                if idx == 0:
                    parts.append(text)
                else:
                    parts.append(f"{sep}{text}")
            return "".join(parts)

        header_values: list[Optional[float]] = []
        for column in columns:
            run_label = column.get("run")
            try:
                header_values.append(float(run_label))
            except Exception:
                header_values.append(None)

        header_string = " " * (label_width + 1) + " " + format_row(header_values)
        row_strings: list[str] = []
        for idx, (field_name, label) in enumerate(component_fields):
            label_text = f"{label:<{label_width}}:"
            row_values = [columns[col_idx].get(field_name) for col_idx in range(len(columns))]
            row_strings.append(f"{label_text} {format_row(row_values)}")

        all_lines = [header_string] + row_strings
        max_line_len = max(len(line) for line in all_lines)
        height = ticker_row_height * len(all_lines) + ticker_margin * 2
        width = max_line_len * char_px + ticker_margin * 2
        ticker_canvas.configure(width=width, height=height)

        for line_idx, line in enumerate(all_lines):
            y = ticker_margin + line_idx * ticker_row_height
            ticker_canvas.create_text(
                ticker_margin,
                y,
                anchor="nw",
                text=line,
                font=("Courier", 10 if line_idx == 0 else 9),
                fill="#222" if line_idx == 0 else "#333",
            )

    def pump_queue() -> None:
        nonlocal runs_total, parallel_envs_var, batch_runs_var, group_size_var, component_columns
        plots_dirty = False
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
                    score_by_run[run_idx] = score_val
                    group_size_local = max(1, group_size_var)
                    batch_idx = run_idx // group_size_local
                    batch_start = batch_idx * group_size_local
                    expected = group_size_local
                    if runs_total > 0:
                        remaining = runs_total - batch_start
                        if remaining < group_size_local:
                            expected = max(1, remaining)
                    collected_scores: list[float] = []
                    for r_idx in range(batch_start, batch_start + expected):
                        if r_idx in score_by_run:
                            collected_scores.append(score_by_run[r_idx])
                    median_val: Optional[float] = None
                    if collected_scores and len(collected_scores) == expected:
                        median_val = float(np.median(collected_scores))
                        batch_medians[batch_idx] = median_val
                        scores.clear()
                        for b_idx in sorted(batch_medians):
                            scores.append((b_idx + 1, batch_medians[b_idx]))
                        plots_dirty = True
                    if median_val is not None:
                        status_var.set(
                            f"Run {run_idx}: score {score_val:.2f} | "
                            f"batch {batch_idx + 1} median {median_val:.2f} "
                            f"(group {group_size_local})"
                        )
                    else:
                        status_var.set(f"Run {run_idx}: score {score_val:.2f}")
                elif mtype == "status":
                    text = message.get("text")
                    if text is not None:
                        status_var.set(str(text))
                elif mtype == "hyperparams":
                    payload = message.get("hyperparams")
                    if isinstance(payload, dict):
                        hyper_label.configure(text=format_hyperparams(payload))
                        try:
                            runs_total = int(payload.get("runs", runs_total) or runs_total or 0)
                        except Exception:
                            pass
                        try:
                            parallel_envs_var = int(
                                payload.get("parallel_envs", parallel_envs_var) or parallel_envs_var or 1
                            )
                        except Exception:
                            pass
                        if "batch_runs" in payload:
                            try:
                                batch_runs_candidate = int(payload.get("batch_runs", batch_runs_var))
                                batch_runs_var = max(0, batch_runs_candidate)
                            except Exception:
                                pass
                        if "randomize_rng" in payload:
                            try:
                                rng_enabled = bool(payload.get("randomize_rng", rng_enabled))
                                set_rng_label()
                            except Exception:
                                pass
                        if "seed_index" in payload:
                            try:
                                current_seed_index = int(payload.get("seed_index", current_seed_index))
                                set_seed_label()
                            except Exception:
                                pass
                        if "checkpoint_path" in payload:
                            cp_value = payload.get("checkpoint_path")
                            if isinstance(cp_value, str) and cp_value:
                                last_checkpoint_path = cp_value
                        group_size_new = max(1, parallel_envs_var, batch_runs_var)
                        if group_size_new != group_size_var:
                            group_size_var = group_size_new
                            recompute_series(group_size_var)
                            plots_dirty = True
                elif mtype == "score_reset":
                    score_by_run.clear()
                    batch_medians.clear()
                    scores.clear()
                    component_columns.clear()
                    policy_loss_series.clear()
                    value_loss_series.clear()
                    if ticker_canvas is not None:
                        ticker_canvas.delete("all")
                    payload_scores = message.get("scores")
                    if isinstance(payload_scores, (list, tuple)):
                        for entry in payload_scores:
                            if not entry:
                                continue
                            try:
                                run_idx, score_val = entry
                                run_idx = int(run_idx)
                                score_val = float(score_val)
                            except Exception:
                                continue
                            score_by_run[run_idx] = score_val
                    recompute_series(group_size_var)
                    render_ticker()
                    status_var.set("Scores reloaded from checkpoint.")
                    plots_dirty = True
                elif mtype == "diagnostics":
                    payload = message.get("metrics")
                    if isinstance(payload, dict):
                        for key, value in payload.items():
                            if isinstance(value, numbers.Number):
                                value_f = float(value)
                                latest_diagnostics[key] = value_f
                                history = diagnostic_history.setdefault(
                                    key, deque(maxlen=diagnostics_window)
                                )
                                history.append(value_f)
                            else:
                                latest_diagnostics[key] = str(value)
                                diagnostic_history.pop(key, None)
                        diagnostics_var.set(format_diagnostics())
                        run_number: Optional[int]
                        try:
                            run_number = int(payload.get("episodes_completed", 0))
                        except Exception:
                            run_number = None
                        if run_number is not None and run_number >= 1:
                            component_entry: Dict[str, Any] = {"run": run_number}
                            for field_name, _label in component_fields:
                                raw_val = payload.get(field_name)
                                if isinstance(raw_val, numbers.Number):
                                    component_entry[field_name] = float(raw_val)
                                else:
                                    try:
                                        component_entry[field_name] = float(raw_val)
                                    except Exception:
                                        component_entry[field_name] = None
                            updated_column = False
                            for idx, existing in enumerate(component_columns):
                                if existing.get("run") == run_number:
                                    component_columns[idx] = component_entry
                                    updated_column = True
                                    break
                            if not updated_column:
                                component_columns.append(component_entry)
                            component_columns.sort(key=lambda item: item.get("run", 0))
                            if len(component_columns) > ticker_history_limit:
                                component_columns[:] = component_columns[-ticker_history_limit:]
                            render_ticker()
                        x_coord: Optional[float]
                        if run_number is not None and run_number >= 1:
                            x_coord = float(run_number)
                        else:
                            updates_val = payload.get("learner/updates")
                            x_coord = (
                                float(updates_val)
                                if isinstance(updates_val, numbers.Number)
                                else None
                            )
                        if x_coord is None:
                            x_coord = float(len(policy_loss_series) + 1)
                        policy_val = payload.get("learner/policy_loss")
                        if isinstance(policy_val, numbers.Number):
                            append_series_point(policy_loss_series, x_coord, float(policy_val))
                            plots_dirty = True
                        value_val = payload.get("learner/value_loss")
                        if isinstance(value_val, numbers.Number):
                            append_series_point(value_loss_series, x_coord, float(value_val))
                            plots_dirty = True
        except queue.Empty:
            pass

        if plots_dirty:
            render_all_plots()

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


class NetworkDiagnosticsTracker:
    """Collect lightweight diagnostics from policy networks and optimizers."""

    def __init__(self, agent: Any):
        self._torch = torch
        self._nn = nn
        self._optim = optim
        self._mx = mx
        self._nn_mlx = nn_mlx
        self._optim_mlx = optim_mlx
        self._agent = agent
        self._modules: Dict[str, Any] = {}
        self._optimizers: Dict[str, Any] = {}
        self._prev_params: Dict[str, "torch.Tensor"] = {}
        self._mlx_modules: Dict[str, Any] = {}
        self._mlx_optimizers: Dict[str, Any] = {}
        self._prev_params_mlx: Dict[str, np.ndarray] = {}
        self._discover_modules(agent)
        self._snapshot_initial_params()

    def _discover_modules(self, agent: Any) -> None:
        attr_names = dir(agent)
        for name in attr_names:
            # getattr may raise for properties; guard accordingly.
            try:
                value = getattr(agent, name)
            except Exception:
                continue
            if self._torch is not None and self._nn is not None and isinstance(value, self._nn.Module):
                existing = self._modules.get(name)
                if existing is not value:
                    self._modules[name] = value
                    flat = self._flatten_parameters(value)
                    if flat is not None:
                        self._prev_params[name] = flat
            if self._optim is not None and self._torch is not None and isinstance(value, self._optim.Optimizer):
                self._optimizers[name] = value
            if self._nn_mlx is not None and isinstance(value, self._nn_mlx.Module):
                existing = self._mlx_modules.get(name)
                if existing is not value:
                    self._mlx_modules[name] = value
                    flat_mlx = self._flatten_parameters_mlx(value)
                    if flat_mlx is not None:
                        self._prev_params_mlx[name] = flat_mlx
            if self._is_mlx_optimizer(value):
                self._mlx_optimizers[name] = value

    def _snapshot_initial_params(self) -> None:
        if self._torch is not None:
            for name, module in self._modules.items():
                flat = self._flatten_parameters(module)
                if flat is not None:
                    self._prev_params[name] = flat
        if self._mx is not None and self._nn_mlx is not None:
            for name, module in self._mlx_modules.items():
                flat_mlx = self._flatten_parameters_mlx(module)
                if flat_mlx is not None:
                    self._prev_params_mlx[name] = flat_mlx

    def _flatten_parameters(self, module: Any):
        if self._torch is None:
            return None
        chunks = []
        for param in module.parameters(recurse=True):
            if not param.requires_grad:
                continue
            chunks.append(param.detach().float().cpu().view(-1))
        if not chunks:
            return None
        return self._torch.cat(chunks)

    def _flatten_gradients(self, module: Any):
        if self._torch is None:
            return None
        chunks = []
        for param in module.parameters(recurse=True):
            grad = param.grad
            if grad is None:
                continue
            chunks.append(grad.detach().float().cpu().view(-1))
        if not chunks:
            return None
        return self._torch.cat(chunks)

    def _flatten_parameters_mlx(self, module: Any) -> Optional[np.ndarray]:
        if self._mx is None or self._nn_mlx is None:
            return None
        try:
            params = module.parameters()
        except Exception:
            return None
        leaves: list[np.ndarray] = []

        def _gather(tree: Any) -> None:
            if tree is None:
                return
            if isinstance(tree, dict):
                for key in sorted(tree.keys()):
                    _gather(tree[key])
                return
            if isinstance(tree, (list, tuple)):
                for item in tree:
                    _gather(item)
                return
            arr = tree
            try:
                if self._mx is not None:
                    self._mx.eval(arr)
            except Exception:
                pass
            try:
                arr_np = np.asarray(arr, dtype=np.float32).reshape(-1)
            except Exception:
                arr_np = np.array(arr, dtype=np.float32).reshape(-1)
            if arr_np.size > 0:
                leaves.append(arr_np.copy())

        _gather(params)
        if not leaves:
            return None
        return np.concatenate(leaves)

    def _is_mlx_optimizer(self, value: Any) -> bool:
        if self._optim_mlx is None:
            return False
        module_name = getattr(getattr(value, "__class__", None), "__module__", "")
        if not module_name:
            return False
        if not module_name.startswith("mlx.optimizers"):
            return False
        return hasattr(value, "update")

    def collect(self) -> Dict[str, Any]:
        """Return a dictionary of diagnostic scalars (and status strings)."""
        # Refresh in case new modules/optimizers were attached since the last call.
        self._discover_modules(self._agent)
        metrics: Dict[str, Any] = {}
        if not self._modules and not self._mlx_modules:
            if self._torch is None:
                metrics["networks/status"] = "torch unavailable"
            else:
                metrics["networks/status"] = "no modules detected"
            return metrics

        status_parts: list[str] = []
        if self._modules:
            status_parts.append(f"{len(self._modules)} torch module(s)")
        if self._mlx_modules:
            status_parts.append(f"{len(self._mlx_modules)} mlx module(s)")
        metrics["networks/status"] = ", ".join(status_parts) + " tracked"

        for name, module in self._modules.items():
            flat_params = self._flatten_parameters(module)
            if flat_params is None or flat_params.numel() == 0:
                continue
            param_norm = flat_params.norm().item()
            metrics[f"{name}/param_norm"] = param_norm
            metrics[f"{name}/num_params"] = float(flat_params.numel())

            flat_grads = self._flatten_gradients(module)
            if flat_grads is not None and flat_grads.numel() > 0:
                grad_norm = flat_grads.norm().item()
                metrics[f"{name}/grad_norm"] = grad_norm
                if param_norm > 1e-8:
                    metrics[f"{name}/grad_to_param"] = grad_norm / param_norm

            prev = self._prev_params.get(name)
            if prev is not None and prev.numel() == flat_params.numel():
                delta = (flat_params - prev).norm().item()
                metrics[f"{name}/param_delta"] = delta
                if param_norm > 1e-8:
                    metrics[f"{name}/delta_ratio"] = delta / param_norm
            self._prev_params[name] = flat_params

        for name, optimizer in self._optimizers.items():
            group_lrs = [float(group.get("lr", 0.0)) for group in optimizer.param_groups if "lr" in group]
            if group_lrs:
                metrics[f"{name}/lr_mean"] = float(np.mean(group_lrs))
                metrics[f"{name}/lr_min"] = float(np.min(group_lrs))
                metrics[f"{name}/lr_max"] = float(np.max(group_lrs))
            max_step = 0.0
            for state in optimizer.state.values():
                step_val = state.get("step")
                if isinstance(step_val, (int, float)):
                    max_step = max(max_step, float(step_val))
            metrics[f"{name}/step"] = max_step

        for name, module in self._mlx_modules.items():
            flat_params = self._flatten_parameters_mlx(module)
            if flat_params is None or flat_params.size == 0:
                continue
            param_norm = float(np.linalg.norm(flat_params))
            metrics[f"{name}/param_norm"] = param_norm
            metrics[f"{name}/num_params"] = float(flat_params.size)

            prev = self._prev_params_mlx.get(name)
            if prev is not None and prev.size == flat_params.size:
                delta = float(np.linalg.norm(flat_params - prev))
                metrics[f"{name}/param_delta"] = delta
                if param_norm > 1e-8:
                    metrics[f"{name}/delta_ratio"] = delta / param_norm
            self._prev_params_mlx[name] = flat_params

        for name, optimizer in self._mlx_optimizers.items():
            lr_value = getattr(optimizer, "learning_rate", None)
            if lr_value is not None:
                try:
                    lr_float = float(np.asarray(lr_value).reshape(()))
                    metrics[f"{name}/lr"] = lr_float
                except Exception:
                    try:
                        metrics[f"{name}/lr"] = float(lr_value)
                    except Exception:
                        pass

        return metrics


class ExperimentMonitor:
    def __init__(self, hyperparams: Dict[str, Any], strategy: str) -> None:
        self._ctx = mp.get_context("spawn")
        self._queue: mp.Queue = self._ctx.Queue(maxsize=64)
        self._commands: mp.Queue = self._ctx.Queue(maxsize=32)
        self._stop = self._ctx.Event()
        self._proc = self._ctx.Process(
            target=_monitor_worker,
            args=(
                "Speedrun Monitor",
                {"hyperparams": hyperparams, "strategy": strategy},
                self._queue,
                self._commands,
                self._stop,
            ),
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

    def reset_scores(self, run_scores: Optional[List[Tuple[int, float]]] = None) -> None:
        if self._closed:
            return
        payload: Dict[str, Any] = {"type": "score_reset"}
        if run_scores:
            payload["scores"] = [(int(run), float(score)) for run, score in run_scores]
        try:
            self._queue.put_nowait(payload)
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

    def publish_diagnostics(self, metrics: Dict[str, Any]) -> None:
        if self._closed or not metrics:
            return
        try:
            self._queue.put_nowait({"type": "diagnostics", "metrics": metrics})
        except queue.Full:
            pass
        except Exception:
            pass

    def poll_commands(self) -> list[Dict[str, Any]]:
        commands: list[Dict[str, Any]] = []
        if self._closed:
            return commands
        try:
            while True:
                cmd = self._commands.get_nowait()
                if cmd is None:
                    break
                if isinstance(cmd, dict):
                    commands.append(cmd)
        except queue.Empty:
            pass
        except Exception:
            pass
        return commands

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        try:
            self._commands.put_nowait(None)
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

    def begin_episode(self, context_id: Optional[int] = None) -> None:
        pass

    def observe_step(
        self,
        _reward: float,
        _done: bool,
        _info: Optional[Dict[str, Any]] = None,
        context_id: Optional[int] = None,
    ) -> None:
        pass

    def select_action(self, _obs: Any, _info: Dict[str, Any], context_id: Optional[int] = None) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._action_space.sample())
        if self._strategy == "down":
            return int(Action.DOWN)
        if self._strategy == "down_hold":
            return int(Action.DOWN_HOLD)
        if self._strategy == "noop":
            return int(Action.NOOP)
        return int(self._action_space.sample())

    def end_episode(self, _score: float, context_id: Optional[int] = None) -> None:
        if self.epsilon > self._epsilon_min:
            self.epsilon = max(self._epsilon_min, self.epsilon * self._epsilon_decay)

    def timing_snapshot(self) -> Dict[str, float]:
        return {}

    def get_state(self) -> Dict[str, Any]:
        return {
            "epsilon": float(self.epsilon),
            "epsilon_decay": float(self._epsilon_decay),
            "epsilon_min": float(self._epsilon_min),
            "strategy": self._strategy,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        try:
            if "epsilon" in state:
                self.epsilon = float(state["epsilon"])
        except Exception:
            pass


class SimpleStateActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        hidden = 256
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base = self.encoder(x)
        logits = self.policy_head(base)
        value = self.value_head(base).squeeze(-1)
        return logits, value


@dataclass
class _EpisodeContext:
    observations: List[Any] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    reset_flags: List[bool] = field(default_factory=list)
    recurrent_state: Optional[Any] = None
    last_spawn_id: Optional[int] = None


@dataclass
class _EnvSlot:
    index: int
    env: Any
    viewer: Optional[_ProcessViewer]
    context_id: int
    obs: Any = None
    info: Dict[str, Any] = field(default_factory=dict)
    run_idx: Optional[int] = None
    episode_reward: float = 0.0
    episode_steps: int = 0
    frame_index: int = 0
    emu_fps: float = 0.0
    step_times: deque = field(default_factory=lambda: deque(maxlen=240))
    next_step_time: float = field(default_factory=time.perf_counter)
    active: bool = False
    component_sums: Dict[str, float] = field(default_factory=dict)
    inference_time: float = 0.0
    step_compute_time: float = 0.0
    inference_calls: int = 0
    run_start_time: float = 0.0
    last_inference_duration: float = 0.0
    last_step_duration: float = 0.0


class PolicyGradientAgent:
    def __init__(
        self,
        action_space,
        prototype_obs: np.ndarray,
        gamma: float,
        learning_rate: float,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: Optional[str] = None,
        policy_arch: str = "mlp",
        color_repr: str = "none",
        obs_mode: str = "state",
        batch_runs: int = 1,
        batch_steps: Optional[int] = None,
        lr_schedule: str = "constant",
        lr_warmup_steps: int = 0,
        lr_cosine_steps: int = 100000,
        lr_min_scale: float = 0.0,
        torch_amp: bool = False,
        torch_compile: bool = False,
        torch_matmul_precision: Optional[str] = None,
    ) -> None:
        if torch is None or nn is None or optim is None or Categorical is None:
            raise RuntimeError("PyTorch is required for the policy gradient agent.")
        if prototype_obs.ndim < 1:
            raise ValueError("Prototype observation must have at least 1 dimension.")
        self._action_space = action_space
        self._gamma = float(gamma)
        self._entropy_coef = float(entropy_coef)
        self._value_coef = float(value_coef)
        self._max_grad_norm = float(max_grad_norm)
        self._device = torch.device(device) if device else torch.device("cpu")
        self._policy_arch = policy_arch
        self._color_repr = color_repr
        self._obs_mode = obs_mode
        self._batch_runs = max(0, int(batch_runs))
        self._batch_steps = max(0, int(batch_steps)) if batch_steps is not None else 0
        self._base_learning_rate = float(learning_rate)
        self._lr_schedule = str(lr_schedule)
        self._lr_warmup_steps = max(0, int(lr_warmup_steps))
        self._lr_cosine_steps = max(1, int(lr_cosine_steps))
        self._lr_min_scale = max(0.0, float(lr_min_scale))
        self._current_learning_rate = self._base_learning_rate
        self._total_env_steps = 0
        self._pixel_memory_format = (
            torch.channels_last
            if obs_mode == "pixel" and self._device.type == "cuda"
            else torch.contiguous_format
        )
        self._episodes_in_batch = 0
        self._steps_in_batch = 0
        self._batch_accum_policy_loss = 0.0
        self._batch_accum_value_loss = 0.0
        self._batch_accum_entropy = 0.0
        self._batch_accum_critic_ev = 0.0
        self._latest_critic_ev = 0.0
        self._contexts: Dict[int, _EpisodeContext] = {}
        augmented_stack = _apply_color_representation(np.asarray(prototype_obs, dtype=np.float32), color_repr)
        self._stack_depth = augmented_stack.shape[0] if augmented_stack.ndim >= 3 else 1
        self._input_dim = int(np.prod(augmented_stack.shape))
        self._nan_rewards_seen = 0
        self._nan_reward_episodes = 0
        self._total_reward_steps = 0
        self._last_episode_nan_repaired = False
        if policy_arch == "mlp":
            self.model = SimpleStateActorCritic(self._input_dim, int(action_space.n)).to(self._device)
        elif policy_arch == "drmario_cnn":
            in_channels = (
                augmented_stack.shape[1] if augmented_stack.ndim >= 3 else ram_specs.STATE_CHANNELS
            )
            self.model = DrMarioStatePolicyNet(int(action_space.n), in_channels=in_channels).to(self._device)
        elif policy_arch == "drmario_color_cnn":
            default_color_channels = ram_specs.STATE_CHANNELS + (
                0 if ram_specs.STATE_USE_BITPLANES else 3
            )
            in_channels = (
                augmented_stack.shape[1]
                if augmented_stack.ndim >= 3
                else default_color_channels
            )
            self.model = DrMarioStatePolicyNet(int(action_space.n), in_channels=in_channels).to(self._device)
        elif policy_arch == "drmario_unet_pixel":
            if obs_mode != "pixel":
                raise ValueError("drmario_unet_pixel architecture requires pixel observations")
            pixel_stack = np.asarray(prototype_obs, dtype=np.float32)
            if pixel_stack.ndim != 4:
                raise ValueError("Pixel observations must be 4D (stack, H, W, C)")
            in_channels = int(pixel_stack.shape[-1])
            self.model = DrMarioPixelUNetPolicyNet(int(action_space.n), in_channels=in_channels).to(self._device)
        else:
            raise ValueError(f"Unknown policy architecture '{policy_arch}'")
        if obs_mode == "pixel" and self._device.type == "cuda":
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
            except Exception:
                pass
        if torch_matmul_precision and hasattr(torch, "set_float32_matmul_precision"):
            if self._device.type != "mps":
                try:
                    torch.set_float32_matmul_precision(str(torch_matmul_precision))
                except Exception:
                    pass
        # Optional torch.compile for supported devices/backends
        self._use_compile = bool(torch_compile)
        if self._use_compile and hasattr(torch, "compile"):
            # torch.compile is not consistently stable on MPS; prefer CPU/CUDA
            try:
                if self._device.type != "mps":
                    self.model = torch.compile(self.model)  # type: ignore[attr-defined]
            except Exception:
                pass

        self.optimizer = optim.Adam(self.model.parameters(), lr=self._base_learning_rate)
        self._apply_lr_schedule(force=True)

        # AMP controls
        self._use_amp = bool(torch_amp) and self._device.type in ("cuda", "mps")
        self._grad_scaler: Optional[Any] = None
        if self._use_amp and self._device.type == "cuda":
            scaler_cls = getattr(getattr(torch.cuda, "amp", None), "GradScaler", None)
            if scaler_cls is not None:
                try:
                    self._grad_scaler = scaler_cls()
                except Exception:
                    self._grad_scaler = None

        self._updates = 0
        self._last_metrics: Dict[str, float] = {}
        self._last_episode_update_time = 0.0
        self._episode_update_time_accum = 0.0
        self._episode_update_count = 0
        self._last_batch_update_time = 0.0
        self._batch_update_time_accum = 0.0
        self._batch_update_count = 0

    def _compute_scheduled_lr(self) -> float:
        base_lr = self._base_learning_rate
        if self._lr_schedule != "cosine":
            return base_lr
        min_lr = base_lr * self._lr_min_scale
        steps = self._total_env_steps
        warmup = self._lr_warmup_steps
        if warmup > 0 and steps < warmup:
            progress = float(steps) / float(max(1, warmup))
            return min_lr + (base_lr - min_lr) * progress
        decay_steps = max(0, steps - warmup)
        cosine_progress = min(1.0, float(decay_steps) / float(self._lr_cosine_steps))
        cosine_value = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
        return min_lr + (base_lr - min_lr) * cosine_value

    def _apply_lr_schedule(self, force: bool = False) -> None:
        if self._lr_schedule == "constant" and not force:
            return
        new_lr = self._compute_scheduled_lr()
        if not force and abs(new_lr - self._current_learning_rate) < 1e-12:
            return
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
        self._current_learning_rate = new_lr

    def _prepare_obs(self, obs: Any) -> torch.Tensor:
        stack = _extract_state_stack(obs)
        stack_np = stack.astype(np.float32, copy=False)
        if self._obs_mode == "state":
            stack_np = _apply_color_representation(stack_np, self._color_repr)
        if self._policy_arch == "mlp":
            flat = torch.from_numpy(stack_np).view(-1)
            return flat.to(self._device)
        if self._obs_mode == "pixel":
            tensor = (
                torch.from_numpy(stack_np)
                .permute(0, 3, 1, 2)
                .contiguous(memory_format=self._pixel_memory_format)
            )
        else:
            tensor = torch.from_numpy(stack_np).contiguous()
        return tensor.to(self._device)

    def _context_key(self, context_id: Optional[int]) -> int:
        if context_id is None:
            return 0
        return int(context_id)

    def _get_context(self, context_id: Optional[int]) -> _EpisodeContext:
        key = self._context_key(context_id)
        ctx = self._contexts.get(key)
        if ctx is None:
            ctx = _EpisodeContext()
            self._contexts[key] = ctx
        return ctx

    def _reset_context(self, context_id: Optional[int]) -> _EpisodeContext:
        ctx = self._get_context(context_id)
        ctx.observations.clear()
        ctx.actions.clear()
        ctx.rewards.clear()
        ctx.dones.clear()
        ctx.reset_flags.clear()
        ctx.recurrent_state = None
        ctx.last_spawn_id = None
        return ctx

    def _zero_recurrent_state(self, ctx: _EpisodeContext) -> None:
        hx = ctx.recurrent_state
        if hx is None:
            return
        if isinstance(hx, torch.Tensor):
            ctx.recurrent_state = torch.zeros_like(hx.detach())
        elif isinstance(hx, tuple):
            ctx.recurrent_state = tuple(torch.zeros_like(t.detach()) for t in hx)
        else:
            ctx.recurrent_state = None

    def _detach_recurrent(self, hx: Optional[Any]) -> Optional[Any]:
        if hx is None:
            return None
        if isinstance(hx, torch.Tensor):
            return hx.detach()
        if isinstance(hx, tuple):
            return tuple(self._detach_recurrent(t) for t in hx)
        return None

    def _zero_like_recurrent(self, hx: Optional[Any]) -> Optional[Any]:
        if hx is None:
            return None
        if isinstance(hx, torch.Tensor):
            return torch.zeros_like(hx)
        if isinstance(hx, tuple):
            return tuple(self._zero_like_recurrent(t) for t in hx)
        return None

    def begin_episode(self, context_id: Optional[int] = None) -> None:
        self.model.train(True)
        ctx = self._reset_context(context_id)
        if self._policy_arch == "mlp":
            ctx.recurrent_state = None

    def select_action(self, obs: Any, _info: Dict[str, Any], context_id: Optional[int] = None) -> int:
        ctx = self._get_context(context_id)
        state_tensor = self._prepare_obs(obs)
        ctx.observations.append(state_tensor.detach().to("cpu").contiguous())
        autocast_ctx = (
            torch.autocast(device_type=self._device.type, dtype=torch.float16)
            if self._use_amp
            else nullcontext()
        )
        with autocast_ctx:
            if self._policy_arch == "mlp":
                logits, _ = self.model(state_tensor.unsqueeze(0))
                dist = Categorical(logits=logits.squeeze(0))
                action = dist.sample()
            else:
                if state_tensor.dim() == 3:
                    input_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
                elif state_tensor.dim() == 4:
                    input_tensor = state_tensor.unsqueeze(0)
                else:
                    raise RuntimeError("Unexpected observation tensor shape")
                logits, _, hx = self.model(input_tensor, ctx.recurrent_state)
                if logits.dim() == 3:
                    logits_slice = logits[:, -1, :]
                else:
                    logits_slice = logits
                dist = Categorical(logits=logits_slice.squeeze(0))
                action = dist.sample()
                if isinstance(hx, torch.Tensor):
                    ctx.recurrent_state = hx.detach()
                elif isinstance(hx, tuple):
                    ctx.recurrent_state = tuple(t.detach() for t in hx)
                else:
                    ctx.recurrent_state = None

        ctx.actions.append(int(action.item()))

        return int(action.item())

    def observe_step(
        self,
        reward: float,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
        context_id: Optional[int] = None,
    ) -> None:
        ctx = self._get_context(context_id)
        ctx.rewards.append(float(reward))
        ctx.dones.append(bool(done))

        spawn_id: Optional[int] = None
        if info is not None:
            raw_spawn = info.get("spawn_id")
            if isinstance(raw_spawn, numbers.Integral):
                spawn_id = int(raw_spawn)
            elif isinstance(raw_spawn, numbers.Number):
                spawn_id = int(raw_spawn)
        reset_required = False
        if spawn_id is not None:
            if ctx.last_spawn_id is None:
                ctx.last_spawn_id = spawn_id
            elif spawn_id != ctx.last_spawn_id:
                self._zero_recurrent_state(ctx)
                ctx.last_spawn_id = spawn_id
                reset_required = True
        if done:
            self._zero_recurrent_state(ctx)
            ctx.last_spawn_id = None
            reset_required = True
        ctx.reset_flags.append(bool(reset_required))

    def _record_episode_update_time(self, duration: float) -> None:
        self._last_episode_update_time = float(duration)
        self._episode_update_time_accum += float(duration)
        self._episode_update_count += 1

    def _record_batch_update_time(self, duration: float) -> None:
        self._last_batch_update_time = float(duration)
        self._batch_update_time_accum += float(duration)
        self._batch_update_count += 1

    def timing_snapshot(self) -> Dict[str, float]:
        snapshot: Dict[str, float] = {
            "episode_update_last_s": float(self._last_episode_update_time),
            "batch_update_last_s": float(self._last_batch_update_time),
            "episode_update_count": float(self._episode_update_count),
            "batch_update_count": float(self._batch_update_count),
        }
        if self._episode_update_count > 0:
            snapshot["episode_update_avg_s"] = (
                self._episode_update_time_accum / float(self._episode_update_count)
            )
        else:
            snapshot["episode_update_avg_s"] = 0.0
        if self._batch_update_count > 0:
            snapshot["batch_update_avg_s"] = (
                self._batch_update_time_accum / float(self._batch_update_count)
            )
        else:
            snapshot["batch_update_avg_s"] = 0.0
        return snapshot

    def end_episode(self, _score: float, context_id: Optional[int] = None) -> None:
        start_time = time.perf_counter()
        try:
            ctx = self._get_context(context_id)
            self._last_episode_nan_repaired = False
            if not ctx.actions:
                return

            episode_len = min(
                len(ctx.rewards),
                len(ctx.actions),
                len(ctx.observations),
                len(ctx.reset_flags),
                len(ctx.dones),
            )
            if episode_len <= 0:
                self._reset_context(context_id)
                return

            rewards_np = np.asarray(ctx.rewards[:episode_len], dtype=np.float32)
            nan_mask = np.isnan(rewards_np)
            nan_count = int(nan_mask.sum())
            self._total_reward_steps += episode_len
            if nan_count:
                self._nan_rewards_seen += nan_count
                self._nan_reward_episodes += 1
                self._last_episode_nan_repaired = True
                rewards_np = rewards_np.copy()
                rewards_np[nan_mask] = 0.0

            rewards_tensor = torch.as_tensor(
                rewards_np, dtype=torch.float32, device=self._device
            )
            if rewards_tensor.dim() == 0:
                rewards_tensor = rewards_tensor.unsqueeze(0)

            dones_tensor = torch.as_tensor(
                ctx.dones[:episode_len], dtype=torch.bool, device=self._device
            )
            if dones_tensor.dim() == 0:
                dones_tensor = dones_tensor.unsqueeze(0)

            obs_sequence = ctx.observations[:episode_len]
            action_sequence = ctx.actions[:episode_len]
            reset_flags = ctx.reset_flags[:episode_len]

            obs_batch = torch.stack(obs_sequence, dim=0)
            obs_batch = obs_batch.to(self._device, non_blocking=True)
            actions_tensor = torch.as_tensor(
                action_sequence, dtype=torch.long, device=self._device
            )

            log_prob_list: list[torch.Tensor] = []
            value_list: list[torch.Tensor] = []
            entropy_list: list[torch.Tensor] = []
            recurrent = None
            autocast_ctx = (
                torch.autocast(device_type=self._device.type, dtype=torch.float16)
                if self._use_amp
                else nullcontext()
            )
            with autocast_ctx:
                for step_idx, reset_flag in enumerate(reset_flags):
                    obs_tensor = obs_batch[step_idx]
                    if self._obs_mode == "pixel" and obs_tensor.dim() >= 4:
                        obs_tensor = obs_tensor.contiguous(memory_format=self._pixel_memory_format)
                    action_tensor = actions_tensor[step_idx]

                    if self._policy_arch == "mlp":
                        logits, values_seq = self.model(obs_tensor.unsqueeze(0))
                        dist = Categorical(logits=logits.squeeze(0))
                        value_tensor = values_seq.squeeze(0)
                    else:
                        if obs_tensor.dim() == 3:
                            input_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)
                        elif obs_tensor.dim() == 4:
                            input_tensor = obs_tensor.unsqueeze(0)
                        else:
                            raise RuntimeError("Unexpected observation tensor shape")
                        logits, values_seq, hx = self.model(input_tensor, recurrent)
                        if logits.dim() == 3:
                            logits_slice = logits[:, -1, :]
                        else:
                            logits_slice = logits
                        if values_seq.dim() == 2:
                            value_tensor = values_seq[:, -1]
                        else:
                            value_tensor = values_seq
                        dist = Categorical(logits=logits_slice.squeeze(0))
                        value_tensor = value_tensor.squeeze(0)
                        recurrent = self._detach_recurrent(hx)
                        if reset_flag:
                            recurrent = self._zero_like_recurrent(recurrent)

                    log_prob_list.append(dist.log_prob(action_tensor))
                    value_list.append(value_tensor)
                    entropy_list.append(dist.entropy())

            if not log_prob_list:
                self._reset_context(context_id)
                return

            log_probs = torch.stack(log_prob_list).to(dtype=torch.float32)
            values = torch.stack(value_list).to(dtype=torch.float32)
            entropies = torch.stack(entropy_list).to(dtype=torch.float32)

            bootstrap_value = None
            if not bool(dones_tensor[-1].item()):
                bootstrap_value = values[-1].detach()

            returns_tensor = discounted_returns_torch(
                rewards_tensor, self._gamma, dones=dones_tensor, bootstrap=bootstrap_value
            )

            if self._episodes_in_batch == 0:
                self.optimizer.zero_grad(set_to_none=True)
                self._batch_accum_policy_loss = 0.0
                self._batch_accum_value_loss = 0.0
                self._batch_accum_entropy = 0.0
                self._batch_accum_critic_ev = 0.0

            if returns_tensor.shape[0] != log_probs.shape[0]:
                min_len = min(returns_tensor.shape[0], log_probs.shape[0])
                returns_tensor = returns_tensor[:min_len]
                log_probs = log_probs[:min_len]
                values = values[:min_len]
                entropies = entropies[:min_len]

            advantages = returns_tensor - values.detach()
            adv_var, adv_mean = torch.var_mean(advantages, dim=0, unbiased=False)
            adv_std = torch.sqrt(adv_var).clamp_min(1e-6)
            advantages = (advantages - adv_mean) / adv_std
            policy_loss = -(log_probs * advantages).mean()
            value_errors = returns_tensor - values
            value_sq = value_errors.pow(2)
            value_mse = value_sq.mean()
            value_loss = 0.5 * torch.clamp(value_sq, max=10.0).mean()
            entropy_term = entropies.mean()
            loss = policy_loss + self._value_coef * value_loss - self._entropy_coef * entropy_term

            value_err_float = float(value_mse.item())
            value_var_float = float(returns_tensor.var(unbiased=False).item())
            if value_var_float <= 1e-8:
                ev_value = 0.0
            else:
                ev_value = 1.0 - (value_err_float / (value_var_float + 1e-8))
            ev_value = max(-1.0, min(1.0, ev_value))

            if self._grad_scaler is not None:
                self._grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            self._batch_accum_policy_loss += float(policy_loss.item())
            self._batch_accum_value_loss += float(value_loss.item())
            self._batch_accum_entropy += float(entropy_term.item())
            self._batch_accum_critic_ev += ev_value
            self._latest_critic_ev = ev_value

            episode_steps = int(returns_tensor.shape[0])
            self._episodes_in_batch += 1
            self._steps_in_batch += episode_steps
            self._total_env_steps += episode_steps

            ctx.observations.clear()
            ctx.actions.clear()
            ctx.rewards.clear()
            ctx.dones.clear()
            ctx.reset_flags.clear()
            ctx.recurrent_state = None
            ctx.last_spawn_id = None

            should_update = (
                (self._batch_runs > 0 and self._episodes_in_batch >= self._batch_runs)
                or (self._batch_steps > 0 and self._steps_in_batch >= self._batch_steps)
            )
            if should_update:
                self._update_policy()
        finally:
            duration = time.perf_counter() - start_time
            self._record_episode_update_time(duration)

    def latest_metrics(self) -> Dict[str, float]:
        metrics = dict(self._last_metrics)
        metrics["learner/pending_runs"] = float(self._episodes_in_batch)
        metrics["learner/pending_steps"] = float(self._steps_in_batch)
        metrics["learner/env_steps_total"] = float(self._total_env_steps)
        metrics["optimizer/lr"] = float(self._current_learning_rate)
        metrics["critic/latest_ev"] = float(self._latest_critic_ev)
        metrics["perf/episode_update_ms_last"] = self._last_episode_update_time * 1000.0
        metrics["perf/batch_update_ms_last"] = self._last_batch_update_time * 1000.0
        metrics["perf/episode_update_count"] = float(self._episode_update_count)
        metrics["perf/batch_update_count"] = float(self._batch_update_count)
        if self._episode_update_count > 0:
            metrics["perf/episode_update_ms_avg"] = (
                (self._episode_update_time_accum / float(self._episode_update_count)) * 1000.0
            )
        if self._batch_update_count > 0:
            metrics["perf/batch_update_ms_avg"] = (
                (self._batch_update_time_accum / float(self._batch_update_count)) * 1000.0
            )
        metrics["rewards/nan_rewards_total"] = float(self._nan_rewards_seen)
        metrics["rewards/nan_reward_episodes"] = float(self._nan_reward_episodes)
        metrics["rewards/nan_reward_last_episode"] = (
            1.0 if self._last_episode_nan_repaired else 0.0
        )
        if self._total_reward_steps > 0:
            nan_rate = float(self._nan_rewards_seen) / float(self._total_reward_steps)
        else:
            nan_rate = 0.0
        metrics["rewards/nan_reward_rate"] = nan_rate
        metrics["alerts/nan_reward_rate_high"] = 1.0 if nan_rate > 0.001 else 0.0
        return metrics

    def _update_policy(self) -> None:
        self._apply_optimizer_step(adjust_for_partial=False)

    def finalize_updates(self) -> None:
        self._apply_optimizer_step(adjust_for_partial=True)

    def _apply_optimizer_step(self, adjust_for_partial: bool) -> None:
        if self._episodes_in_batch <= 0:
            return
        start_time = time.perf_counter()
        batch_count = max(1, self._episodes_in_batch)
        scaler = self._grad_scaler
        self._apply_lr_schedule()
        if scaler is not None:
            try:
                scaler.unscale_(self.optimizer)
            except Exception:
                pass
        avg_scale = 1.0 / float(max(1, batch_count))
        for param in self.model.parameters():
            grad = param.grad
            if grad is not None:
                grad.mul_(avg_scale)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._max_grad_norm)
        if scaler is not None:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()
        self._updates += 1

        denom = float(max(1, batch_count))
        self._last_metrics = {
            "learner/policy_loss": self._batch_accum_policy_loss / denom,
            "learner/value_loss": self._batch_accum_value_loss / denom,
            "learner/entropy": self._batch_accum_entropy / denom,
            "learner/updates": float(self._updates),
            "critic/explained_variance": self._batch_accum_critic_ev / denom,
        }
        self.optimizer.zero_grad(set_to_none=True)
        self._episodes_in_batch = 0
        self._steps_in_batch = 0
        self._batch_accum_policy_loss = 0.0
        self._batch_accum_value_loss = 0.0
        self._batch_accum_entropy = 0.0
        self._batch_accum_critic_ev = 0.0
        duration = time.perf_counter() - start_time
        self._record_batch_update_time(duration)

    def get_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "updates": int(self._updates),
            "episodes_in_batch": int(self._episodes_in_batch),
            "steps_in_batch": int(self._steps_in_batch),
            "batch_steps": int(self._batch_steps),
            "batch_accum_policy_loss": float(self._batch_accum_policy_loss),
            "batch_accum_value_loss": float(self._batch_accum_value_loss),
            "batch_accum_entropy": float(self._batch_accum_entropy),
            "batch_accum_critic_ev": float(self._batch_accum_critic_ev),
            "batch_runs": int(self._batch_runs),
            "policy_arch": self._policy_arch,
            "obs_mode": self._obs_mode,
            "color_repr": self._color_repr,
            "latest_critic_ev": float(self._latest_critic_ev),
            "nan_rewards_seen": int(self._nan_rewards_seen),
            "nan_reward_episodes": int(self._nan_reward_episodes),
            "reward_step_total": int(self._total_reward_steps),
            "total_env_steps": int(self._total_env_steps),
            "base_learning_rate": float(self._base_learning_rate),
            "lr_schedule": self._lr_schedule,
            "lr_warmup_steps": int(self._lr_warmup_steps),
            "lr_cosine_steps": int(self._lr_cosine_steps),
            "lr_min_scale": float(self._lr_min_scale),
            "current_learning_rate": float(self._current_learning_rate),
        }
        if torch is None:
            return state
        try:
            model_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            optim_state = self.optimizer.state_dict()
            for group in optim_state.get("state", {}).values():
                for key, value in list(group.items()):
                    if isinstance(value, torch.Tensor):
                        group[key] = value.detach().cpu()
            state["model_state"] = model_state
            state["optimizer_state"] = optim_state
            if self._grad_scaler is not None:
                try:
                    state["grad_scaler_state"] = self._grad_scaler.state_dict()
                except Exception:
                    pass
        except Exception:
            pass
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        if torch is None:
            return
        try:
            model_state = state.get("model_state")
            if isinstance(model_state, dict):
                mapped_state = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in model_state.items()}
                self.model.load_state_dict(mapped_state)
        except Exception:
            pass
        try:
            optim_state = state.get("optimizer_state")
            if isinstance(optim_state, dict):
                for group in optim_state.get("state", {}).values():
                    for key, value in list(group.items()):
                        if isinstance(value, torch.Tensor):
                            group[key] = value.to(self._device)
                self.optimizer.load_state_dict(optim_state)
        except Exception:
            pass
        scaler_state = state.get("grad_scaler_state")
        if self._grad_scaler is not None and scaler_state is not None:
            try:
                self._grad_scaler.load_state_dict(scaler_state)
            except Exception:
                pass
        try:
            self._updates = int(state.get("updates", self._updates))
            self._episodes_in_batch = int(state.get("episodes_in_batch", 0))
            self._steps_in_batch = int(state.get("steps_in_batch", 0))
            self._batch_accum_policy_loss = float(state.get("batch_accum_policy_loss", 0.0))
            self._batch_accum_value_loss = float(state.get("batch_accum_value_loss", 0.0))
            self._batch_accum_entropy = float(state.get("batch_accum_entropy", 0.0))
            self._batch_accum_critic_ev = float(state.get("batch_accum_critic_ev", 0.0))
            self._latest_critic_ev = float(state.get("latest_critic_ev", 0.0))
            self._batch_runs = max(0, int(state.get("batch_runs", self._batch_runs)))
            self._batch_steps = max(0, int(state.get("batch_steps", self._batch_steps)))
            self._nan_rewards_seen = int(state.get("nan_rewards_seen", self._nan_rewards_seen))
            self._nan_reward_episodes = int(
                state.get("nan_reward_episodes", self._nan_reward_episodes)
            )
            self._total_reward_steps = int(state.get("reward_step_total", self._total_reward_steps))
            self._total_env_steps = int(state.get("total_env_steps", self._total_env_steps))
            self._base_learning_rate = float(state.get("base_learning_rate", self._base_learning_rate))
            self._lr_schedule = str(state.get("lr_schedule", self._lr_schedule))
            self._lr_warmup_steps = max(0, int(state.get("lr_warmup_steps", self._lr_warmup_steps)))
            self._lr_cosine_steps = max(1, int(state.get("lr_cosine_steps", self._lr_cosine_steps)))
            self._lr_min_scale = max(0.0, float(state.get("lr_min_scale", self._lr_min_scale)))
            self._current_learning_rate = float(
                state.get("current_learning_rate", self._current_learning_rate)
            )
        except Exception:
            pass
        self.optimizer.zero_grad(set_to_none=True)
        self._apply_lr_schedule(force=True)


@dataclass
class _MLXEpisodeContext:
    observations: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    recurrent_state: Optional[Any] = None
    last_spawn_id: Optional[int] = None


class PolicyGradientAgentMLX:
    def __init__(
        self,
        action_space,
        prototype_obs: np.ndarray,
        gamma: float,
        learning_rate: float,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        policy_arch: str = "drmario_cnn",
        color_repr: str = "none",
        obs_mode: str = "state",
        batch_runs: int = 1,
        batch_steps: Optional[int] = None,
        lr_schedule: str = "constant",
        lr_warmup_steps: int = 0,
        lr_cosine_steps: int = 100000,
        lr_min_scale: float = 0.0,
        use_last_frame_inference: bool = True,
    ) -> None:
        if mx is None or nn_mlx is None or optim_mlx is None:
            raise RuntimeError("MLX backend is not available. Please install mlx to use --policy-backend mlx.")
        if DrMarioStatePolicyMLX is None:
            raise RuntimeError("DrMarioStatePolicyMLX is unavailable.")
        if obs_mode != "state":
            raise ValueError("MLX backend currently supports state observations only.")
        if policy_arch not in {"drmario_cnn", "drmario_color_cnn"}:
            raise ValueError("MLX backend requires --policy-arch drmario_cnn or drmario_color_cnn.")
        self._action_space = action_space
        self._action_dim = int(action_space.n)
        self._gamma = float(gamma)
        self._entropy_coef = float(entropy_coef)
        self._value_coef = float(value_coef)
        self._color_repr = color_repr
        self._obs_mode = obs_mode
        self._max_grad_norm = float(max_grad_norm)
        self._batch_runs = max(0, int(batch_runs))
        self._batch_steps = max(0, int(batch_steps)) if batch_steps is not None else 0
        self._base_learning_rate = float(learning_rate)
        self._lr_schedule = str(lr_schedule)
        self._lr_warmup_steps = max(0, int(lr_warmup_steps))
        self._lr_cosine_steps = max(1, int(lr_cosine_steps))
        self._lr_min_scale = max(0.0, float(lr_min_scale))
        self._current_learning_rate = self._base_learning_rate
        self._total_env_steps = 0
        self._use_last_frame_inference = bool(use_last_frame_inference)
        self._contexts: Dict[int, _MLXEpisodeContext] = {}
        augmented_stack = _apply_color_representation(np.asarray(prototype_obs, dtype=np.float32), color_repr)
        if augmented_stack.ndim < 3:
            raise ValueError("Prototype observation must have at least 3 dimensions.")
        self._stack_depth = augmented_stack.shape[0] if augmented_stack.ndim >= 3 else 1
        in_channels = augmented_stack.shape[1] if augmented_stack.ndim >= 3 else ram_specs.STATE_CHANNELS
        self.model = DrMarioStatePolicyMLX(
            action_dim=self._action_dim,
            in_channels=int(in_channels),
        )
        self.optimizer = optim_mlx.Adam(learning_rate=self._base_learning_rate)
        self._apply_lr_schedule(force=True)
        try:  # pragma: no cover - optional evaluation for lazy arrays
            mx.eval(self.model.parameters(), self.optimizer.state)
        except Exception:
            pass
        self._updates = 0
        self._last_metrics: Dict[str, float] = {}
        self._last_episode_update_time = 0.0
        self._episode_update_time_accum = 0.0
        self._episode_update_count = 0
        self._last_batch_update_time = 0.0
        self._batch_update_time_accum = 0.0
        self._batch_update_count = 0
        self._latest_critic_ev = 0.0
        self._episodes_in_batch = 0
        self._steps_in_batch = 0
        self._batch_accum_policy_loss = 0.0
        self._batch_accum_value_loss = 0.0
        self._batch_accum_entropy = 0.0
        self._batch_accum_critic_ev = 0.0
        self._grad_accum: Optional[Any] = None
        self._nan_rewards_seen = 0
        self._nan_reward_episodes = 0
        self._total_reward_steps = 0
        self._last_episode_nan_repaired = False

    def _set_optimizer_lr(self, value: float) -> None:
        try:
            if hasattr(self.optimizer, "learning_rate"):
                setattr(self.optimizer, "learning_rate", float(value))
            elif hasattr(self.optimizer, "lr"):
                setattr(self.optimizer, "lr", float(value))
        except Exception:
            pass
        self._current_learning_rate = float(value)

    def _compute_scheduled_lr(self) -> float:
        base_lr = self._base_learning_rate
        if self._lr_schedule != "cosine":
            return base_lr
        min_lr = base_lr * self._lr_min_scale
        steps = self._total_env_steps
        warmup = self._lr_warmup_steps
        if warmup > 0 and steps < warmup:
            progress = float(steps) / float(max(1, warmup))
            return min_lr + (base_lr - min_lr) * progress
        decay_steps = max(0, steps - warmup)
        cosine_progress = min(1.0, float(decay_steps) / float(self._lr_cosine_steps))
        cosine_value = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
        return min_lr + (base_lr - min_lr) * cosine_value

    def _apply_lr_schedule(self, force: bool = False) -> None:
        if self._lr_schedule == "constant" and not force:
            return
        new_lr = self._compute_scheduled_lr()
        if not force and abs(new_lr - self._current_learning_rate) < 1e-12:
            return
        self._set_optimizer_lr(new_lr)

    def _context_key(self, context_id: Optional[int]) -> int:
        if context_id is None:
            return 0
        return int(context_id)

    def _get_context(self, context_id: Optional[int]) -> _MLXEpisodeContext:
        key = self._context_key(context_id)
        ctx = self._contexts.get(key)
        if ctx is None:
            ctx = _MLXEpisodeContext()
            self._contexts[key] = ctx
        return ctx

    def _reset_context(self, context_id: Optional[int]) -> _MLXEpisodeContext:
        ctx = self._get_context(context_id)
        ctx.observations.clear()
        ctx.rewards.clear()
        ctx.actions.clear()
        ctx.dones.clear()
        ctx.recurrent_state = None
        ctx.last_spawn_id = None
        return ctx

    def _prepare_obs(self, obs: Any) -> Tuple[Any, np.ndarray]:
        stack = _extract_state_stack(obs)
        stack_np = np.asarray(stack, dtype=np.float32)
        if self._obs_mode == "state":
            stack_np = _apply_color_representation(stack_np, self._color_repr)
        stack_tensor = mx.array(stack_np, dtype=mx.float32)
        return stack_tensor, stack_np

    def _detach_recurrent_state(self, hx: Optional[Any]) -> Optional[Any]:
        if hx is None:
            return None
        if hasattr(mx, "stop_gradient"):
            stop_grad = mx.stop_gradient
        else:  # pragma: no cover - fallback for older MLX builds
            def stop_grad(value: Any) -> Any:
                array_value = np.array(value, copy=True)
                dtype = getattr(value, "dtype", None)
                if dtype is not None:
                    return mx.array(array_value, dtype=dtype)
                return mx.array(array_value)

        def _detach_leaf(value: Any) -> Any:
            try:
                detached_value = stop_grad(value)
            except Exception:
                return None
            if hasattr(mx, "eval"):
                try:  # pragma: no cover - optional evaluation for lazy arrays
                    mx.eval(detached_value)
                except Exception:
                    pass
            return detached_value

        return self._tree_map(hx, _detach_leaf)

    def _zero_recurrent_state(self, ctx: _MLXEpisodeContext) -> None:
        ctx.recurrent_state = None

    def begin_episode(self, context_id: Optional[int] = None) -> None:
        self._reset_context(context_id)

    def observe_step(
        self,
        reward: float,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
        context_id: Optional[int] = None,
    ) -> None:
        ctx = self._get_context(context_id)
        ctx.rewards.append(float(reward))
        ctx.dones.append(bool(done))
        spawn_id: Optional[int] = None
        if info is not None:
            raw_spawn = info.get("spawn_id")
            if isinstance(raw_spawn, numbers.Integral):
                spawn_id = int(raw_spawn)
            elif isinstance(raw_spawn, numbers.Number):
                spawn_id = int(raw_spawn)
        if spawn_id is not None:
            if ctx.last_spawn_id is None:
                ctx.last_spawn_id = spawn_id
            elif spawn_id != ctx.last_spawn_id:
                self._zero_recurrent_state(ctx)
                ctx.last_spawn_id = spawn_id
        if done:
            self._zero_recurrent_state(ctx)
            ctx.last_spawn_id = None

    def select_action(self, obs: Any, _info: Dict[str, Any], context_id: Optional[int] = None) -> int:
        ctx = self._get_context(context_id)
        stack_tensor, stack_np = self._prepare_obs(obs)
        if stack_tensor.ndim == 3:
            sequence_tensor = mx.expand_dims(stack_tensor, axis=0)
            ctx.observations.append(np.array(stack_np, dtype=np.float32, copy=True))
        elif stack_tensor.ndim == 4:
            latest_frame_tensor = stack_tensor[-1]
            latest_frame_np = np.array(stack_np[-1], dtype=np.float32, copy=True)
            ctx.observations.append(latest_frame_np)
            if self._use_last_frame_inference:
                sequence_tensor = mx.expand_dims(latest_frame_tensor, axis=0)
            else:
                sequence_tensor = stack_tensor
        else:
            raise RuntimeError("MLX backend expects observation shape (C, H, W) or (T, C, H, W).")
        obs_tensor = mx.expand_dims(sequence_tensor, axis=0)  # (1, T, C, H, W)
        logits, value, hx = self.model(obs_tensor, ctx.recurrent_state, last_only=True)
        logits_vec = logits[0]
        ctx.recurrent_state = self._detach_recurrent_state(hx)

        probs = mx.softmax(logits_vec, axis=-1)
        probs_np = np.asarray(probs, dtype=np.float64)
        if probs_np.ndim != 1:
            probs_np = probs_np.reshape(-1)
        probs_np = np.clip(probs_np, 1e-8, None)
        probs_np = probs_np / probs_np.sum()
        action = int(np.random.choice(self._action_dim, p=probs_np))
        ctx.actions.append(action)
        return action

    def _mx_to_float(self, value: Any) -> float:
        try:
            return float(value)
        except Exception:
            try:
                return float(np.asarray(value))
            except Exception:
                return float(np.array(value).reshape(()))

    def _record_episode_update_time(self, duration: float) -> None:
        self._last_episode_update_time = float(duration)
        self._episode_update_time_accum += float(duration)
        self._episode_update_count += 1

    def _record_batch_update_time(self, duration: float) -> None:
        self._last_batch_update_time = float(duration)
        self._batch_update_time_accum += float(duration)
        self._batch_update_count += 1

    def timing_snapshot(self) -> Dict[str, float]:
        snapshot: Dict[str, float] = {
            "episode_update_last_s": float(self._last_episode_update_time),
            "batch_update_last_s": float(self._last_batch_update_time),
            "episode_update_count": float(self._episode_update_count),
            "batch_update_count": float(self._batch_update_count),
        }
        if self._episode_update_count > 0:
            snapshot["episode_update_avg_s"] = (
                self._episode_update_time_accum / float(self._episode_update_count)
            )
        else:
            snapshot["episode_update_avg_s"] = 0.0
        if self._batch_update_count > 0:
            snapshot["batch_update_avg_s"] = (
                self._batch_update_time_accum / float(self._batch_update_count)
            )
        else:
            snapshot["batch_update_avg_s"] = 0.0
        return snapshot

    def end_episode(self, _score: float, context_id: Optional[int] = None) -> None:
        start_time = time.perf_counter()
        try:
            ctx = self._get_context(context_id)
            self._last_episode_nan_repaired = False
            if not ctx.actions or not ctx.observations or not ctx.rewards:
                self._reset_context(context_id)
                return

            episode_len = min(
                len(ctx.actions), len(ctx.observations), len(ctx.rewards), len(ctx.dones)
            )
            if episode_len <= 0:
                self._reset_context(context_id)
                return

            rewards_np = np.asarray(ctx.rewards[:episode_len], dtype=np.float32)
            nan_mask = np.isnan(rewards_np)
            nan_count = int(nan_mask.sum())
            self._total_reward_steps += episode_len
            if nan_count:
                self._nan_rewards_seen += nan_count
                self._nan_reward_episodes += 1
                self._last_episode_nan_repaired = True
                rewards_np = rewards_np.copy()
                rewards_np[nan_mask] = 0.0

            obs_items = ctx.observations[:episode_len]
            obs_np = np.stack([np.asarray(item, dtype=np.float32) for item in obs_items], axis=0)
            obs_batch = mx.array(obs_np, dtype=mx.float32)
            obs_batch = mx.expand_dims(obs_batch, axis=0)  # (1, T, C, H, W)

            action_array = ctx.actions[:episode_len]
            actions_batch = mx.expand_dims(
                mx.array(action_array, dtype=mx.int32), axis=-1
            )
            rewards_tensor = mx.array(rewards_np, dtype=mx.float32)
            dones_tensor = mx.array(ctx.dones[:episode_len], dtype=mx.bool_)
            target_shape = rewards_tensor.shape
            last_done_flag = bool(ctx.dones[episode_len - 1])

            metrics_cache: Dict[str, Any] = {}

            def loss_fn(obs_tensor, action_tensor):
                logits_seq, values_seq, _ = self.model(obs_tensor, None, last_only=False)
                logits_seq = logits_seq[0]
                values_seq = values_seq[0]
                log_probs = _mlx_log_softmax(logits_seq, axis=-1)
                values_seq_shaped = mx.reshape(values_seq, target_shape)
                if hasattr(mx, "stop_gradient"):
                    bootstrap_candidates = mx.stop_gradient(values_seq_shaped[-1])
                    values_detached = mx.stop_gradient(values_seq_shaped)
                else:  # pragma: no cover - fallback for older MLX builds
                    bootstrap_candidates = mx.array(
                        np.array(values_seq_shaped[-1], copy=True), dtype=values_seq_shaped.dtype
                    )
                    values_detached = mx.array(
                        np.array(values_seq_shaped, copy=True), dtype=values_seq_shaped.dtype
                    )

                bootstrap_arg = None if last_done_flag else bootstrap_candidates

                returns_tensor = discounted_returns_mlx(
                    rewards_tensor, self._gamma, dones=dones_tensor, bootstrap=bootstrap_arg
                )
                selected = mx.squeeze(mx.take_along_axis(log_probs, action_tensor, axis=-1), axis=-1)
                advantages_policy = returns_tensor - values_detached
                adv_mean = mx.mean(advantages_policy)
                advantages_policy = advantages_policy - adv_mean
                adv_sq = advantages_policy * advantages_policy
                adv_std = mx.sqrt(mx.mean(adv_sq))
                epsilon = mx.array(1e-6, dtype=advantages_policy.dtype)
                advantages_policy = advantages_policy / mx.maximum(adv_std, epsilon)
                policy_loss = -mx.mean(selected * advantages_policy)
                advantages_value = returns_tensor - values_seq_shaped
                value_sq = advantages_value * advantages_value
                value_loss = 0.5 * mx.mean(mx.clip(value_sq, a_min=None, a_max=10.0))
                probs = mx.softmax(logits_seq, axis=-1)
                entropy = -mx.mean(mx.sum(probs * log_probs, axis=-1))
                metrics_cache["policy_loss"] = self._mx_to_float(policy_loss)
                metrics_cache["value_loss"] = self._mx_to_float(value_loss)
                metrics_cache["entropy"] = self._mx_to_float(entropy)
                value_mse = self._mx_to_float(mx.mean(value_sq))
                if hasattr(mx, "var"):
                    returns_var = self._mx_to_float(mx.var(returns_tensor))
                else:  # pragma: no cover - fallback for older MLX versions
                    mean_returns = mx.mean(returns_tensor)
                    diff = returns_tensor - mean_returns
                    returns_var = self._mx_to_float(mx.mean(diff * diff))
                if returns_var <= 1e-8:
                    ev_value = 0.0
                else:
                    ev_value = 1.0 - (value_mse / (returns_var + 1e-8))
                metrics_cache["critic_ev"] = max(-1.0, min(1.0, float(ev_value)))
                return policy_loss + self._value_coef * value_loss - self._entropy_coef * entropy

            if self._episodes_in_batch == 0:
                self._grad_accum = None
                self._batch_accum_policy_loss = 0.0
                self._batch_accum_value_loss = 0.0
                self._batch_accum_entropy = 0.0
                self._batch_accum_critic_ev = 0.0

            loss_fn_grad = nn_mlx.value_and_grad(self.model, loss_fn)
            _loss_value, grads = loss_fn_grad(obs_batch, actions_batch)
            if self._grad_accum is None:
                self._grad_accum = grads
            else:
                self._grad_accum = self._tree_combine(self._grad_accum, grads, lambda a, b: a + b)

            policy_loss_val = float(metrics_cache.get("policy_loss", 0.0))
            value_loss_val = float(metrics_cache.get("value_loss", 0.0))
            entropy_val = float(metrics_cache.get("entropy", 0.0))
            ev = float(metrics_cache.get("critic_ev", 0.0))
            self._batch_accum_policy_loss += policy_loss_val
            self._batch_accum_value_loss += value_loss_val
            self._batch_accum_entropy += entropy_val
            self._batch_accum_critic_ev += ev
            self._latest_critic_ev = ev
            episode_steps = int(episode_len)
            self._episodes_in_batch += 1
            self._steps_in_batch += episode_steps
            self._total_env_steps += episode_steps

            self._reset_context(context_id)

            should_update = (
                (self._batch_runs > 0 and self._episodes_in_batch >= self._batch_runs)
                or (self._batch_steps > 0 and self._steps_in_batch >= self._batch_steps)
            )
            if should_update:
                self._apply_optimizer_step(adjust_for_partial=False)
        finally:
            duration = time.perf_counter() - start_time
            self._record_episode_update_time(duration)

    def latest_metrics(self) -> Dict[str, float]:
        metrics = dict(self._last_metrics)
        metrics["learner/pending_runs"] = float(self._episodes_in_batch)
        metrics["learner/pending_steps"] = float(self._steps_in_batch)
        metrics["learner/env_steps_total"] = float(self._total_env_steps)
        metrics["optimizer/lr"] = float(self._current_learning_rate)
        metrics["critic/latest_ev"] = float(self._latest_critic_ev)
        metrics["perf/episode_update_ms_last"] = self._last_episode_update_time * 1000.0
        metrics["perf/batch_update_ms_last"] = self._last_batch_update_time * 1000.0
        metrics["perf/episode_update_count"] = float(self._episode_update_count)
        metrics["perf/batch_update_count"] = float(self._batch_update_count)
        if self._episode_update_count > 0:
            metrics["perf/episode_update_ms_avg"] = (
                (self._episode_update_time_accum / float(self._episode_update_count)) * 1000.0
            )
        if self._batch_update_count > 0:
            metrics["perf/batch_update_ms_avg"] = (
                (self._batch_update_time_accum / float(self._batch_update_count)) * 1000.0
            )
        metrics["rewards/nan_rewards_total"] = float(self._nan_rewards_seen)
        metrics["rewards/nan_reward_episodes"] = float(self._nan_reward_episodes)
        metrics["rewards/nan_reward_last_episode"] = (
            1.0 if self._last_episode_nan_repaired else 0.0
        )
        if self._total_reward_steps > 0:
            nan_rate = float(self._nan_rewards_seen) / float(self._total_reward_steps)
        else:
            nan_rate = 0.0
        metrics["rewards/nan_reward_rate"] = nan_rate
        metrics["alerts/nan_reward_rate_high"] = 1.0 if nan_rate > 0.001 else 0.0
        return metrics

    def finalize_updates(self) -> None:
        self._apply_optimizer_step(adjust_for_partial=True)

    def _apply_optimizer_step(self, adjust_for_partial: bool) -> None:
        if self._episodes_in_batch <= 0 or self._grad_accum is None:
            return
        start_time = time.perf_counter()
        batch_count = max(1, self._episodes_in_batch)
        grads = self._grad_accum
        avg_scale = 1.0 / float(max(1, batch_count))
        grads = self._tree_map(grads, lambda g: g * avg_scale if g is not None else None)

        if self._max_grad_norm > 0.0:
            total = self._tree_sum_squares(grads)
            norm = self._mx_to_float(mx.sqrt(total))
            if norm > self._max_grad_norm and norm > 0.0:
                clip_scale = self._max_grad_norm / (norm + 1e-8)
                grads = self._tree_map(grads, lambda g: g * clip_scale if g is not None else None)

        self._apply_lr_schedule()
        self.optimizer.update(self.model, grads)
        try:  # pragma: no cover - optional evaluation for lazy arrays
            mx.eval(self.model.parameters(), self.optimizer.state)
        except Exception:
            pass
        self._updates += 1
        denom = float(max(1, batch_count))
        self._last_metrics = {
            "learner/policy_loss": self._batch_accum_policy_loss / denom,
            "learner/value_loss": self._batch_accum_value_loss / denom,
            "learner/entropy": self._batch_accum_entropy / denom,
            "learner/updates": float(self._updates),
            "critic/explained_variance": self._batch_accum_critic_ev / denom,
        }
        self._episodes_in_batch = 0
        self._steps_in_batch = 0
        self._batch_accum_policy_loss = 0.0
        self._batch_accum_value_loss = 0.0
        self._batch_accum_entropy = 0.0
        self._batch_accum_critic_ev = 0.0
        self._grad_accum = None
        duration = time.perf_counter() - start_time
        self._record_batch_update_time(duration)

    def _tree_map(self, tree: Any, fn) -> Any:
        if tree is None:
            return None
        if isinstance(tree, dict):
            return {key: self._tree_map(value, fn) for key, value in tree.items()}
        if isinstance(tree, list):
            return [self._tree_map(value, fn) for value in tree]
        if isinstance(tree, tuple):
            return tuple(self._tree_map(value, fn) for value in tree)
        return fn(tree)

    def _tree_combine(self, left: Any, right: Any, op) -> Any:
        if left is None:
            return right
        if right is None:
            return left
        if isinstance(left, dict) and isinstance(right, dict):
            return {key: self._tree_combine(left[key], right[key], op) for key in left.keys()}
        if isinstance(left, list) and isinstance(right, list):
            return [self._tree_combine(lv, rv, op) for lv, rv in zip(left, right)]
        if isinstance(left, tuple) and isinstance(right, tuple):
            return tuple(self._tree_combine(lv, rv, op) for lv, rv in zip(left, right))
        return op(left, right)

    def _tree_sum_squares(self, tree: Any) -> Any:
        if tree is None:
            return mx.array(0.0, dtype=mx.float32)
        if isinstance(tree, dict):
            total = None
            for value in tree.values():
                current = self._tree_sum_squares(value)
                total = current if total is None else total + current
            return total if total is not None else mx.array(0.0, dtype=mx.float32)
        if isinstance(tree, list):
            total = None
            for value in tree:
                current = self._tree_sum_squares(value)
                total = current if total is None else total + current
            return total if total is not None else mx.array(0.0, dtype=mx.float32)
        if isinstance(tree, tuple):
            total = None
            for value in tree:
                current = self._tree_sum_squares(value)
                total = current if total is None else total + current
            return total if total is not None else mx.array(0.0, dtype=mx.float32)
        return mx.sum(tree * tree)

    def get_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "updates": int(self._updates),
            "latest_critic_ev": float(self._latest_critic_ev),
            "episodes_in_batch": int(self._episodes_in_batch),
            "steps_in_batch": int(self._steps_in_batch),
            "batch_steps": int(self._batch_steps),
            "batch_accum_policy_loss": float(self._batch_accum_policy_loss),
            "batch_accum_value_loss": float(self._batch_accum_value_loss),
            "batch_accum_entropy": float(self._batch_accum_entropy),
            "batch_accum_critic_ev": float(self._batch_accum_critic_ev),
            "batch_runs": int(self._batch_runs),
            "supports_checkpoint": True,
            "nan_rewards_seen": int(self._nan_rewards_seen),
            "nan_reward_episodes": int(self._nan_reward_episodes),
            "reward_step_total": int(self._total_reward_steps),
            "total_env_steps": int(self._total_env_steps),
            "base_learning_rate": float(self._base_learning_rate),
            "lr_schedule": self._lr_schedule,
            "lr_warmup_steps": int(self._lr_warmup_steps),
            "lr_cosine_steps": int(self._lr_cosine_steps),
            "lr_min_scale": float(self._lr_min_scale),
            "current_learning_rate": float(self._current_learning_rate),
        }
        try:
            mx.eval(self.model.parameters(), self.optimizer.state)
            state["model_parameters"] = self._tree_map(self.model.parameters(), lambda arr: np.asarray(arr))
            state["optimizer_state"] = self._tree_map(self.optimizer.state, lambda arr: np.asarray(arr))
        except Exception:
            state["supports_checkpoint"] = False
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        try:
            self._updates = int(state.get("updates", self._updates))
            self._latest_critic_ev = float(state.get("latest_critic_ev", self._latest_critic_ev))
            self._episodes_in_batch = int(state.get("episodes_in_batch", 0))
            self._steps_in_batch = int(state.get("steps_in_batch", 0))
            self._batch_accum_policy_loss = float(state.get("batch_accum_policy_loss", 0.0))
            self._batch_accum_value_loss = float(state.get("batch_accum_value_loss", 0.0))
            self._batch_accum_entropy = float(state.get("batch_accum_entropy", 0.0))
            self._batch_accum_critic_ev = float(state.get("batch_accum_critic_ev", 0.0))
            self._nan_rewards_seen = int(state.get("nan_rewards_seen", self._nan_rewards_seen))
            self._nan_reward_episodes = int(
                state.get("nan_reward_episodes", self._nan_reward_episodes)
            )
            self._total_reward_steps = int(state.get("reward_step_total", self._total_reward_steps))
            self._batch_steps = max(0, int(state.get("batch_steps", self._batch_steps)))
            self._batch_runs = max(0, int(state.get("batch_runs", self._batch_runs)))
            self._total_env_steps = int(state.get("total_env_steps", self._total_env_steps))
            self._base_learning_rate = float(state.get("base_learning_rate", self._base_learning_rate))
            self._lr_schedule = str(state.get("lr_schedule", self._lr_schedule))
            self._lr_warmup_steps = max(0, int(state.get("lr_warmup_steps", self._lr_warmup_steps)))
            self._lr_cosine_steps = max(1, int(state.get("lr_cosine_steps", self._lr_cosine_steps)))
            self._lr_min_scale = max(0.0, float(state.get("lr_min_scale", self._lr_min_scale)))
            self._current_learning_rate = float(
                state.get("current_learning_rate", self._current_learning_rate)
            )
        except Exception:
            pass

        params = state.get("model_parameters")
        if params is not None:
            try:
                mx_params = self._tree_map(params, lambda arr: mx.array(arr))
                self.model.update(mx_params)  # type: ignore[attr-defined]
            except Exception:
                pass
        optimizer_state = state.get("optimizer_state")
        if optimizer_state is not None:
            try:
                mx_state = self._tree_map(optimizer_state, lambda arr: mx.array(arr))
                self.optimizer.state = mx_state  # type: ignore[attr-defined]
            except Exception:
                pass
        self._apply_lr_schedule(force=True)
        self._grad_accum = None

def _extract_state_stack(observation: Any) -> np.ndarray:
    core = observation["obs"] if isinstance(observation, dict) else observation
    return np.asarray(core)


def _apply_color_representation(stack: np.ndarray, mode: str) -> np.ndarray:
    if mode != "shared_color":
        return stack
    if stack.ndim < 3:
        return stack
    if ram_specs.STATE_USE_BITPLANES:
        return stack
    base_channels = ram_specs.STATE_CHANNELS
    augmented_channels = ram_specs.STATE_CHANNELS + 3
    if stack.shape[1] == augmented_channels:
        return stack
    if stack.shape[1] != base_channels:
        return stack
    viruses = stack[:, 0:3]
    fixed = stack[:, 3:6]
    falling = stack[:, 6:9]
    color_planes = np.minimum(1.0, viruses + fixed + falling)
    return np.concatenate([stack, color_planes], axis=1)


def _format_hyperparams_for_monitor(args: argparse.Namespace, agent: Any) -> Dict[str, Any]:
    payload = {
        "runs": args.runs,
        "max_steps": args.max_steps,
        "strategy": args.strategy,
        "learner": args.learner,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "policy_arch": args.policy_arch,
        "policy_backend": args.policy_backend,
        "color_repr": args.color_repr,
        "batch_runs": args.batch_runs,
        "batch_steps": args.batch_steps,
        "lr_schedule": args.lr_schedule,
        "lr_warmup_steps": args.lr_warmup_steps,
        "lr_cosine_steps": args.lr_cosine_steps,
        "lr_min_scale": args.lr_min_scale,
        "randomize_rng": bool(args.randomize_rng),
        "frame_offset": args.frame_offset,
        "parallel_envs": getattr(args, "parallel_envs", getattr(args, "num_envs", None)),
        "seed_index": getattr(args, "seed_index", None),
        "intent_mode": bool(getattr(args, "intent_action_space", False)),
    }
    checkpoint_attr = getattr(args, "checkpoint_path", None)
    if checkpoint_attr:
        try:
            payload["checkpoint_path"] = str(Path(checkpoint_attr).expanduser())
        except Exception:
            payload["checkpoint_path"] = str(checkpoint_attr)
    else:
        payload["checkpoint_path"] = ""
    if getattr(args, "reward_config", None):
        payload["reward_config_path"] = str(Path(args.reward_config).expanduser())
    if hasattr(agent, "epsilon"):
        payload["epsilon"] = round(float(getattr(agent, "epsilon")), 4)
        payload["epsilon_decay"] = args.epsilon_decay
        payload["epsilon_min"] = args.epsilon_min
    if args.learner != "none":
        payload["entropy_coef"] = args.entropy_coef
        payload["value_coef"] = args.value_coef
        payload["max_grad_norm"] = args.max_grad_norm
        if args.device:
            payload["device"] = args.device
    payload["perf_mode"] = bool(getattr(args, "perf_mode", False))
    if args.policy_backend == "mlx":
        payload["mlx_inference_window"] = getattr(args, "mlx_inference_window", "last")
        mlx_device_label = getattr(args, "mlx_device_label", None)
        if mlx_device_label:
            payload["mlx_device"] = mlx_device_label
    else:
        payload["torch_amp"] = bool(getattr(args, "torch_amp", False))
        payload["torch_compile"] = bool(getattr(args, "torch_compile", False))
        payload["torch_matmul_high"] = bool(getattr(args, "torch_matmul_high", False))
        payload["torch_seed"] = getattr(args, "torch_seed", None)
    return payload


def seed_policy_rng(seed_val: int) -> None:
    """Seed torch RNGs used by the policy for deterministic-but-unique sampling."""
    if torch is None:
        return
    try:
        torch.manual_seed(seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_val)
    except Exception:
        pass
    if hasattr(torch, "mps") and callable(getattr(torch.mps, "manual_seed", None)):
        try:
            torch.mps.manual_seed(seed_val)  # type: ignore[attr-defined]
        except Exception:
            pass


def _soft_reset_env(env: Any) -> bool:
    """Attempt an in-place backend reset and force the next reset to run the triple-start."""
    raw_env = getattr(env, "unwrapped", env)
    try:
        backend_reset = getattr(raw_env, "backend_reset", None)
        if callable(backend_reset):
            backend_reset()
    except Exception as exc:
        print(f"Backend reset failed, will recreate env: {exc}", file=sys.stderr)
        return False

    if hasattr(raw_env, "_first_boot"):
        try:
            raw_env._first_boot = True  # type: ignore[attr-defined]
        except Exception:
            pass

    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Level-0 speedrun experiment harness")
    ap.add_argument("--mode", choices=["pixel", "state"], default="state")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--runs", type=int, default=10, help="Number of training runs (episodes)")
    ap.add_argument("--max-steps", type=int, default=2000, help="Max environment steps per run")
    ap.add_argument("--risk-tau", type=float, default=0.5)
    ap.add_argument("--frame-offset", type=int, default=0)
    ap.add_argument("--randomize-rng", action="store_true")
    ap.add_argument("--seed-index", type=int, default=0, help="Seed index used when RNG randomization is disabled.")
    ap.add_argument("--backend", type=str, default=None)
    ap.add_argument("--core-path", type=str, default=None)
    ap.add_argument("--rom-path", type=str, default=None)
    ap.add_argument("--reward-config", type=str, default=None, help="Path to reward config override (JSON).")
    ap.add_argument("--checkpoint-path", type=str, default=None, help="File to save checkpoints between runs.")
    ap.add_argument("--checkpoint-interval", type=int, default=0, help="Save checkpoint every N completed runs (0 disables).")
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint if available.")
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
    ap.add_argument(
        "--perf-mode",
        action="store_true",
        help="Enable performance preset (no viewer/monitor, zero throttling, max parallel envs).",
    )
    ap.add_argument(
        "--strategy",
        choices=["random", "down", "down_hold", "noop", "policy"],
        default="random",
    )
    ap.add_argument("--learner", choices=["none", "reinforce"], default="none")
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--epsilon-decay", type=float, default=0.97)
    ap.add_argument("--epsilon-min", type=float, default=0.05)
    ap.add_argument(
        "--policy-arch",
        choices=["mlp", "drmario_cnn", "drmario_color_cnn", "drmario_unet_pixel"],
        default="mlp",
    )
    ap.add_argument(
        "--policy-backend",
        choices=["torch", "mlx"],
        default="torch",
        help="Select the deep learning backend for the policy network.",
    )
    ap.add_argument(
        "--mlx-inference-window",
        choices=["last", "stack"],
        default="last",
        help="MLX-only: use the latest frame ('last') or the full stack ('stack') when selecting actions.",
    )
    ap.add_argument(
        "--mlx-device",
        type=str,
        default=None,
        help="MLX-only: select the compute device (index or kind[:index]).",
    )
    ap.add_argument(
        "--mlx-list-devices",
        action="store_true",
        help="List all available MLX devices and exit.",
    )
    ap.add_argument("--color-repr", choices=["none", "shared_color"], default="none")
    ap.add_argument("--state-repr", choices=["extended", "bitplane"], default="extended")
    ap.add_argument(
        "--batch-runs",
        type=int,
        default=32,
        help="Episodes to accumulate before each policy update (0 disables).",
    )
    ap.add_argument(
        "--batch-steps",
        type=int,
        default=32768,
        help="Environment steps to accumulate before each policy update (0 disables).",
    )
    ap.add_argument(
        "--lr-schedule",
        choices=["constant", "cosine"],
        default="cosine",
        help="Learning rate schedule applied to policy gradient learners.",
    )
    ap.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=5000,
        help="Linear warmup steps before enabling the learning rate schedule.",
    )
    ap.add_argument(
        "--lr-cosine-steps",
        type=int,
        default=200000,
        help="Number of steps over which to anneal the cosine schedule after warmup.",
    )
    ap.add_argument(
        "--lr-min-scale",
        type=float,
        default=0.0,
        help="Minimum learning rate scale relative to the base rate for cosine scheduling.",
    )
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--torch-amp", action="store_true", help="Enable torch autocast for policy forward (CUDA/MPS only).")
    ap.add_argument("--torch-compile", action="store_true", help="Compile torch model where supported (not MPS).")
    ap.add_argument(
        "--torch-matmul-high",
        action="store_true",
        help="Request high precision float32 matmuls on CPU/CUDA backends.",
    )
    ap.add_argument(
        "--torch-seed",
        type=int,
        default=None,
        help="Seed torch (and numpy/random) RNGs for reproducible policy sampling.",
    )
    ap.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Parallel environments to run within each batch (default: cpu_count - 2).",
    )
    ap.add_argument(
        "--recreate-env",
        action="store_true",
        help="Close and recreate the environment between runs instead of using a soft reset.",
    )
    ap.add_argument(
        "--intent-action-space",
        action="store_true",
        help="Use the intent-based control wrapper instead of direct controller actions.",
    )
    ap.add_argument(
        "--intent-safe-writes",
        action="store_true",
        help="Allow the intent translator to perform safe backend writes (experimental).",
    )

    args = ap.parse_args()

    if args.mlx_list_devices:
        exit_code = _mlx_print_available_devices()
        sys.exit(exit_code)

    selected_mlx_device: Optional[_MLXDeviceInfo] = None
    if args.policy_backend == "mlx":
        if mx is None:
            print("MLX backend is unavailable; install mlx to use --policy-backend mlx.", file=sys.stderr)
        else:
            try:
                selected_mlx_device = _mlx_configure_device(args.mlx_device)
            except Exception as exc:
                raise RuntimeError(f"Failed to select MLX device: {exc}") from exc
            if selected_mlx_device is not None:
                args.mlx_device_label = selected_mlx_device.summary()
                args.mlx_device_identifier = selected_mlx_device.identifier
    elif args.mlx_device:
        print(
            "Warning: --mlx-device specified but --policy-backend is not 'mlx'; ignoring MLX device selection.",
            file=sys.stderr,
        )
        args.mlx_device = None

    if args.perf_mode:
        args.no_monitor = True
        args.no_show_window = True
        args.viz_sync = False
        args.emu_target_hz = 0.0
        if hasattr(args, "intent_action_space"):
            setattr(args, "intent_action_space", False)
        if args.num_envs is None:
            cpu_local = os.cpu_count() or 1
            args.num_envs = max(1, cpu_local - 1)

    if args.torch_seed is not None:
        seed_val = int(args.torch_seed)
        args.torch_seed = seed_val
        try:
            random.seed(seed_val)
        except Exception:
            pass
        try:
            np.random.seed(seed_val)
        except Exception:
            pass
        if torch is not None:
            try:
                torch.manual_seed(seed_val)
            except Exception:
                pass
            if hasattr(torch, "cuda") and callable(getattr(torch.cuda, "manual_seed_all", None)):
                try:
                    torch.cuda.manual_seed_all(seed_val)
                except Exception:
                    pass
            if hasattr(torch, "mps") and callable(getattr(torch.mps, "manual_seed", None)):
                try:
                    torch.mps.manual_seed(seed_val)  # type: ignore[attr-defined]
                except Exception:
                    pass

    ram_specs.set_state_representation(args.state_repr)

    register_env_id()
    if args.intent_action_space:
        register_intent_env_id()

    if args.mode != "state" and args.color_repr != "none":
        raise ValueError("--color-repr shared_color is only supported for state mode observations")

    env_kwargs: Dict[str, Any] = {
        "obs_mode": args.mode,
        "level": args.level,
        "risk_tau": args.risk_tau,
        "render_mode": "rgb_array",
        "learner_discount": args.gamma,
        "state_repr": args.state_repr,
    }
    if args.backend:
        env_kwargs["backend"] = args.backend
    if args.core_path:
        env_kwargs["core_path"] = Path(args.core_path).expanduser()
    if args.rom_path:
        env_kwargs["rom_path"] = Path(args.rom_path).expanduser()
    if args.reward_config:
        env_kwargs["reward_config_path"] = Path(args.reward_config).expanduser()
    if args.no_auto_start:
        env_kwargs["auto_start"] = False

    def build_env() -> Any:
        return make("DrMarioRetroEnv-v0", **env_kwargs)

    def build_training_env() -> Any:
        env_instance = build_env()
        if args.intent_action_space:
            env_instance = DrMarioIntentEnv(env_instance, safe_writes=args.intent_safe_writes)
        return env_instance

    randomize_rng_enabled = bool(args.randomize_rng)
    current_seed_index = int(getattr(args, "seed_index", 0))
    reset_options: Dict[str, Any] = {"frame_offset": args.frame_offset}
    if randomize_rng_enabled:
        reset_options["randomize_rng"] = True
    setattr(args, "seed_index", current_seed_index)
    checkpoint_path: Optional[Path]
    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path).expanduser()
    else:
        checkpoint_path = None
    checkpoint_interval = max(0, int(args.checkpoint_interval))
    if args.resume and checkpoint_path is None:
        raise ValueError("--resume requires --checkpoint-path")

    completed_rewards: list[float] = []
    recent_rewards: deque[float] = deque(maxlen=5)
    total_steps = 0
    completed_runs = 0
    next_run_idx = 0
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
        reset_options["start_wait_frames"] = int(args.start_wait_frames)

    def reset_environment(environment: Any) -> Tuple[Any, Dict[str, Any]]:
        opts = dict(reset_options)
        if randomize_rng_enabled:
            return environment.reset(options=opts)
        return environment.reset(seed=current_seed_index, options=opts)

    env = build_training_env()
    obs, info = reset_environment(env)

    use_learning_agent = args.learner == "reinforce" or args.strategy == "policy"
    if use_learning_agent and args.learner == "none":
        args.learner = "reinforce"  # align with requested strategy
    if use_learning_agent and args.strategy != "policy":
        args.strategy = "policy"

    if args.mode == "state":
        prototype_stack = _extract_state_stack(obs)
    else:
        core = obs["obs"] if isinstance(obs, dict) else obs
        prototype_stack = np.asarray(core)

    if use_learning_agent:
        if args.mode == "pixel" and args.policy_arch not in {"mlp", "drmario_unet_pixel"}:
            raise RuntimeError("Pixel observations require --policy-arch mlp or drmario_unet_pixel.")
        if prototype_stack is None:
            raise RuntimeError("Observation stack unavailable for learner initialisation.")
        if args.policy_backend == "mlx":
            agent = PolicyGradientAgentMLX(
                env.action_space,
                prototype_obs=prototype_stack,
                gamma=args.gamma,
                learning_rate=args.learning_rate,
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef,
                max_grad_norm=args.max_grad_norm,
                policy_arch=args.policy_arch,
                color_repr=args.color_repr,
                obs_mode=args.mode,
                batch_runs=args.batch_runs,
                batch_steps=args.batch_steps,
                lr_schedule=args.lr_schedule,
                lr_warmup_steps=args.lr_warmup_steps,
                lr_cosine_steps=args.lr_cosine_steps,
                lr_min_scale=args.lr_min_scale,
                use_last_frame_inference=(args.mlx_inference_window == "last"),
            )
            if selected_mlx_device is not None:
                setattr(agent, "mlx_device", selected_mlx_device.identifier)
                setattr(agent, "mlx_device_label", selected_mlx_device.summary())
        else:
            agent = PolicyGradientAgent(
                env.action_space,
                prototype_obs=prototype_stack,
                gamma=args.gamma,
                learning_rate=args.learning_rate,
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef,
                max_grad_norm=args.max_grad_norm,
                device=args.device,
                policy_arch=args.policy_arch,
                color_repr=args.color_repr,
                obs_mode=args.mode,
                batch_runs=args.batch_runs,
                batch_steps=args.batch_steps,
                lr_schedule=args.lr_schedule,
                lr_warmup_steps=args.lr_warmup_steps,
                lr_cosine_steps=args.lr_cosine_steps,
                lr_min_scale=args.lr_min_scale,
                torch_amp=bool(getattr(args, "torch_amp", False)),
                torch_compile=bool(getattr(args, "torch_compile", False)),
                torch_matmul_precision="high" if bool(getattr(args, "torch_matmul_high", False)) else None,
            )
    else:
        agent = SpeedrunAgent(
            env.action_space,
            strategy=args.strategy,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min,
        )

    network_tracker = NetworkDiagnosticsTracker(agent)

    monitor: Optional[ExperimentMonitor]
    if args.no_monitor:
        monitor = None
    else:
        monitor = ExperimentMonitor(_format_hyperparams_for_monitor(args, agent), args.strategy)
        initial_metrics = network_tracker.collect()
        if initial_metrics:
            monitor.publish_diagnostics(initial_metrics)

    def process_monitor_commands() -> None:
        nonlocal randomize_rng_enabled, reset_options, current_seed_index
        nonlocal checkpoint_path, next_run_idx, completed_runs, completed_rewards, recent_rewards
        nonlocal total_steps
        if monitor is None:
            return
        commands = monitor.poll_commands()
        if not commands:
            return
        changed_rng = False
        seed_changed = False
        for cmd in commands:
            cmd_type = cmd.get("type")
            if cmd_type == "rng_toggle":
                enabled = bool(cmd.get("enabled", randomize_rng_enabled))
                if enabled != randomize_rng_enabled:
                    randomize_rng_enabled = enabled
                    changed_rng = True
            elif cmd_type == "seed_adjust":
                seed_val = cmd.get("seed_index")
                try:
                    seed_val = int(seed_val)
                    if seed_val != current_seed_index:
                        current_seed_index = seed_val
                        seed_changed = True
                except Exception:
                    pass
            elif cmd_type == "checkpoint_save":
                path_val = cmd.get("path")
                target_path: Optional[Path]
                try:
                    target_path = Path(str(path_val)).expanduser() if path_val else None
                except Exception:
                    target_path = None
                if target_path is None:
                    target_path = checkpoint_path
                if target_path is None:
                    if monitor is not None:
                        try:
                            monitor.publish_status("Checkpoint save failed: no path selected")
                        except Exception:
                            pass
                    continue
                checkpoint_path = target_path
                setattr(args, "checkpoint_path", str(target_path))
                save_checkpoint(completed_runs, path=target_path)
                if monitor is not None:
                    monitor.update_hyperparams(_format_hyperparams_for_monitor(args, agent))
                    try:
                        monitor.publish_status(f"Checkpoint saved to {target_path}")
                    except Exception:
                        pass
            elif cmd_type == "checkpoint_load":
                path_val = cmd.get("path")
                target_path: Optional[Path]
                try:
                    target_path = Path(str(path_val)).expanduser() if path_val else None
                except Exception:
                    target_path = None
                if target_path is None:
                    target_path = checkpoint_path
                if target_path is None:
                    if monitor is not None:
                        try:
                            monitor.publish_status("Checkpoint load failed: no path selected")
                        except Exception:
                            pass
                    continue
                payload = load_checkpoint(path=target_path)
                if payload:
                    checkpoint_path = target_path
                    setattr(args, "checkpoint_path", str(target_path))
                    apply_checkpoint_payload(payload)
                    if monitor is not None:
                        monitor.reset_scores(list(enumerate(completed_rewards)))
                        monitor.update_hyperparams(_format_hyperparams_for_monitor(args, agent))
                        try:
                            monitor.publish_status(f"Checkpoint loaded from {target_path}")
                        except Exception:
                            pass
                    try:
                        print(f"Loaded checkpoint from '{target_path}'", flush=True)
                    except Exception:
                        pass
                else:
                    if monitor is not None:
                        try:
                            monitor.publish_status("Checkpoint load failed")
                        except Exception:
                            pass
        if changed_rng:
            if randomize_rng_enabled:
                reset_options["randomize_rng"] = True
            else:
                reset_options.pop("randomize_rng", None)
            setattr(args, "randomize_rng", randomize_rng_enabled)
            if monitor is not None:
                monitor.update_hyperparams(_format_hyperparams_for_monitor(args, agent))
        if seed_changed:
            setattr(args, "seed_index", current_seed_index)
            if monitor is not None:
                monitor.update_hyperparams(_format_hyperparams_for_monitor(args, agent))
                try:
                    monitor.publish_status(f"Seed index set to {current_seed_index}")
                except Exception:
                    pass

    def check_reward_config_updates() -> None:
        changed = False
        for slot in slots:
            base_env = getattr(slot.env, "unwrapped", slot.env)
            reload_fn = getattr(slot.env, "reload_reward_config", None)
            if reload_fn is None:
                reload_fn = getattr(base_env, "reload_reward_config", None)
            if callable(reload_fn):
                try:
                    if reload_fn():
                        changed = True
                except Exception as exc:
                    print(f"Warning: failed to reload reward config: {exc}", file=sys.stderr)
        if changed and monitor is not None:
            monitor.update_hyperparams(_format_hyperparams_for_monitor(args, agent))
            try:
                monitor.publish_status("Reloaded reward configuration")
            except Exception:
                pass

    def build_checkpoint_payload(next_run: int) -> Dict[str, Any]:
        agent_state: Optional[Dict[str, Any]] = None
        state_fn = getattr(agent, "get_state", None)
        if callable(state_fn):
            try:
                agent_state = state_fn()
            except Exception:
                agent_state = None
        payload: Dict[str, Any] = {
            "version": 1,
            "timestamp": time.time(),
            "args": vars(args),
            "runs_completed": next_run,
            "completed_rewards": list(completed_rewards),
            "recent_rewards": list(recent_rewards),
            "total_steps": total_steps,
            "randomize_rng_enabled": randomize_rng_enabled,
            "seed_index": current_seed_index,
            "agent_state": agent_state,
        }
        return payload

    def apply_checkpoint_payload(payload: Dict[str, Any]) -> None:
        nonlocal completed_rewards, recent_rewards, total_steps
        nonlocal randomize_rng_enabled, current_seed_index, completed_runs, next_run_idx, reset_options
        if not isinstance(payload, dict):
            return
        try:
            loaded_runs = int(payload.get("runs_completed", 0))
        except Exception:
            loaded_runs = 0
        rewards_source = payload.get("completed_rewards", [])
        new_rewards: list[float] = []
        if isinstance(rewards_source, (list, tuple)):
            for item in rewards_source:
                try:
                    new_rewards.append(float(item))
                except Exception:
                    continue
        if loaded_runs > 0 and loaded_runs < len(new_rewards):
            new_rewards = new_rewards[:loaded_runs]
        if loaded_runs == 0:
            completed_runs_local = len(new_rewards)
        else:
            completed_runs_local = min(loaded_runs, len(new_rewards))
        new_rewards = new_rewards[:completed_runs_local]
        completed_rewards = new_rewards
        recent_rewards = deque(completed_rewards[-5:], maxlen=5)
        completed_runs = completed_runs_local
        next_run_idx = completed_runs
        try:
            total_steps = int(payload.get("total_steps", total_steps))
        except Exception:
            pass
        randomize_rng_enabled = bool(payload.get("randomize_rng_enabled", randomize_rng_enabled))
        if randomize_rng_enabled:
            reset_options["randomize_rng"] = True
        else:
            reset_options.pop("randomize_rng", None)
        setattr(args, "randomize_rng", randomize_rng_enabled)
        try:
            current_seed_index = int(payload.get("seed_index", current_seed_index))
        except Exception:
            pass
        setattr(args, "seed_index", current_seed_index)
        agent_state = payload.get("agent_state")
        state_fn = getattr(agent, "set_state", None)
        if callable(state_fn) and agent_state is not None:
            try:
                state_fn(agent_state)
            except Exception as exc:
                print(f"Warning: failed to restore agent state: {exc}", file=sys.stderr)

    def save_checkpoint(next_run: int, *, path: Optional[Path] = None) -> None:
        target_path = path or checkpoint_path
        if target_path is None:
            return
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("wb") as f:
                pickle.dump(build_checkpoint_payload(next_run), f)
        except Exception as exc:
            print(f"Failed to save checkpoint: {exc}", file=sys.stderr)

    def load_checkpoint(*, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        target_path = path or checkpoint_path
        if target_path is None or not target_path.is_file():
            return None
        try:
            with target_path.open("rb") as f:
                return pickle.load(f)
        except Exception as exc:
            print(f"Failed to load checkpoint: {exc}", file=sys.stderr)
            return None

    if args.resume and checkpoint_path:
        checkpoint_data = load_checkpoint(path=checkpoint_path)
        if checkpoint_data:
            apply_checkpoint_payload(checkpoint_data)
            if monitor is not None:
                monitor.reset_scores(list(enumerate(completed_rewards)))
                monitor.update_hyperparams(_format_hyperparams_for_monitor(args, agent))
            print(f"Resumed from checkpoint '{checkpoint_path}' at run {completed_runs}")
        else:
            print("No checkpoint data found; starting fresh.")

    show_window = not args.no_show_window
    state_shape: Optional[Tuple[int, ...]] = None
    state_dtype: Optional[np.dtype] = None
    if args.mode == "state":
        state_shape = prototype_stack.shape
        state_dtype = prototype_stack.dtype

    cpu_count = os.cpu_count() or 1
    default_envs = max(1, cpu_count - 2)
    requested_envs = default_envs if args.num_envs is None else max(1, int(args.num_envs))
    num_envs = max(1, min(requested_envs, args.runs))
    args.parallel_envs = num_envs
    if monitor is not None:
        monitor.update_hyperparams(_format_hyperparams_for_monitor(args, agent))

    target_hz = max(0.0, float(args.emu_target_hz))
    step_period = 1.0 / target_hz if target_hz > 0 else 0.0
    experiment_start_time = time.perf_counter()

    def create_viewer(slot_index: int) -> Optional[_ProcessViewer]:
        nonlocal show_window
        if not show_window:
            return None
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
            title = "Dr. Mario Trainer" if num_envs == 1 else f"Dr. Mario Trainer #{slot_index + 1}"
            return _ProcessViewer(
                title,
                float(args.display_scale),
                with_stats=(args.mode == "state"),
                **viewer_kwargs,
            )
        except Exception as exc:
            print(f"Viewer unavailable: {exc}", file=sys.stderr)
            show_window = False
            return None

    def publish_frame(
        slot: _EnvSlot,
        observation: Any,
        info_payload: Dict[str, Any],
        action_val: Optional[int],
        reward_val: float,
        step_idx: int,
        episode_reward: float,
    ) -> None:
        viewer_local = slot.viewer
        if viewer_local is None:
            return
        stats = {
            "info": info_payload,
            "step": step_idx,
            "reward": reward_val,
            "cumulative": episode_reward,
            "action": None if action_val is None else int(action_val),
            "run": -1 if slot.run_idx is None else int(slot.run_idx),
            "emu_fps": slot.emu_fps,
            "env_slot": slot.index,
            "frame": slot.frame_index,
            "time": time.perf_counter(),
        }
        now_time = stats["time"]
        wall_elapsed = 0.0
        if slot.run_start_time > 0.0:
            wall_elapsed = max(0.0, now_time - slot.run_start_time)
        inference_time = float(slot.inference_time)
        compute_time = float(slot.step_compute_time)
        perf_stats: Dict[str, Any] = {
            "inference_s": inference_time,
            "compute_s": compute_time,
            "wall_s": wall_elapsed,
            "inference_pct_wall": (inference_time / wall_elapsed * 100.0)
            if wall_elapsed > 1e-9
            else 0.0,
            "inference_pct_compute": (inference_time / compute_time * 100.0)
            if compute_time > 1e-9
            else 0.0,
            "inference_calls": int(slot.inference_calls),
            "last_inference_ms": float(slot.last_inference_duration) * 1000.0,
            "last_step_ms": float(slot.last_step_duration) * 1000.0,
        }
        timing_snapshot_fn = getattr(agent, "timing_snapshot", None)
        timing_info: Dict[str, Any] = {}
        if callable(timing_snapshot_fn):
            try:
                timing_candidate = timing_snapshot_fn()
            except Exception:
                timing_candidate = {}
            if isinstance(timing_candidate, dict):
                timing_info = timing_candidate
        if timing_info:
            episode_last = float(timing_info.get("episode_update_last_s", 0.0))
            batch_last = float(timing_info.get("batch_update_last_s", 0.0))
            perf_stats["episode_update_last_ms"] = episode_last * 1000.0
            perf_stats["batch_update_last_ms"] = batch_last * 1000.0
            episode_avg = float(timing_info.get("episode_update_avg_s", 0.0))
            batch_avg = float(timing_info.get("batch_update_avg_s", 0.0))
            perf_stats["episode_update_avg_ms"] = episode_avg * 1000.0
            perf_stats["batch_update_avg_ms"] = batch_avg * 1000.0
            perf_stats["episode_update_count"] = float(
                timing_info.get("episode_update_count", 0.0)
            )
            perf_stats["batch_update_count"] = float(timing_info.get("batch_update_count", 0.0))
        stats["perf"] = perf_stats
        try:
            if args.mode == "state":
                stack = _extract_state_stack(observation)
                if stack is None:
                    return
                if not viewer_local.push(stack, stats):
                    slot.viewer = None
                    return
            else:
                try:
                    frame = slot.env.render()
                except Exception:
                    slot.viewer = None
                    return
                if frame is None:
                    slot.viewer = None
                    return
                frame_np = np.asarray(frame)
                if frame_np.ndim == 3 and frame_np.shape[-1] == 3:
                    if not viewer_local.push(frame_np, stats):
                        slot.viewer = None
                        return
                else:
                    slot.viewer = None
                    return
        except Exception:
            slot.viewer = None
            return
        slot.frame_index += 1

    def assign_run(slot: _EnvSlot, run_idx: int) -> None:
        slot.run_idx = run_idx
        slot.episode_reward = 0.0
        slot.episode_steps = 0
        slot.frame_index = 0
        slot.emu_fps = 0.0
        slot.step_times.clear()
        slot.step_times.append(time.perf_counter())
        slot.next_step_time = time.perf_counter()
        slot.active = True
        slot.info = dict(slot.info or {})
        slot.component_sums.clear()
        slot.inference_time = 0.0
        slot.step_compute_time = 0.0
        slot.inference_calls = 0
        slot.run_start_time = time.perf_counter()
        slot.last_inference_duration = 0.0
        slot.last_step_duration = 0.0
        if (
            use_learning_agent
            and not randomize_rng_enabled
            and args.policy_backend == "torch"
            and args.torch_seed is None
        ):
            policy_seed = current_seed_index + run_idx * max(1, num_envs) + slot.index
            seed_policy_rng(policy_seed)
        agent.begin_episode(context_id=slot.context_id)
        publish_frame(slot, slot.obs, slot.info, None, 0.0, total_steps, 0.0)
        if monitor is not None:
            try:
                monitor.publish_status(
                    f"Run {run_idx + 1}/{args.runs} assigned to env {slot.index + 1}/{num_envs}"
                )
            except Exception:
                pass

    slots: list[_EnvSlot] = []
    for slot_idx in range(num_envs):
        if slot_idx == 0:
            slot_env = env
            slot_obs, slot_info = obs, info
        else:
            slot_env = build_training_env()
            slot_obs, slot_info = reset_environment(slot_env)
        viewer_instance = create_viewer(slot_idx)
        slots.append(
            _EnvSlot(
                index=slot_idx,
                env=slot_env,
                viewer=viewer_instance,
                context_id=slot_idx,
                obs=slot_obs,
                info=dict(slot_info or {}),
            )
        )

    check_reward_config_updates()

    next_run_idx = len(completed_rewards)
    for slot in slots:
        if next_run_idx >= args.runs:
            break
        assign_run(slot, next_run_idx)
        next_run_idx += 1

    completed_runs = len(completed_rewards)

    try:
        while completed_runs < args.runs:
            process_monitor_commands()
            check_reward_config_updates()
            active_any = False
            for slot in slots:
                if not slot.active or slot.run_idx is None:
                    continue
                active_any = True
                if target_hz > 0:
                    now = time.perf_counter()
                    if now < slot.next_step_time:
                        time.sleep(max(0.0, slot.next_step_time - now))
                        now = time.perf_counter()
                    slot.next_step_time = now + step_period

                step_start = time.perf_counter()
                action = agent.select_action(slot.obs, slot.info, context_id=slot.context_id)
                inference_end = time.perf_counter()
                slot.inference_time += inference_end - step_start
                slot.inference_calls += 1
                slot.last_inference_duration = inference_end - step_start
                next_obs, reward, term, trunc, step_info = slot.env.step(action)
                step_info = dict(step_info or {})
                step_done = bool(term or trunc)

                slot.episode_reward += reward
                slot.episode_steps += 1
                total_steps += 1

                agent.observe_step(reward, step_done, step_info, context_id=slot.context_id)

                slot.step_times.append(time.perf_counter())
                if len(slot.step_times) >= 2:
                    span = slot.step_times[-1] - slot.step_times[0]
                    if span > 0:
                        slot.emu_fps = float(len(slot.step_times) - 1) / span

                publish_frame(slot, next_obs, step_info, int(action), reward, total_steps, slot.episode_reward)

                slot.obs = next_obs
                slot.info = step_info
                for comp_key, _ in COMPONENT_FIELDS:
                    if comp_key == "episode_reward":
                        continue
                    raw_val = step_info.get(comp_key)
                    if isinstance(raw_val, numbers.Number):
                        val = float(raw_val)
                        if comp_key == "action_penalty":
                            val = -val
                        slot.component_sums[comp_key] = slot.component_sums.get(comp_key, 0.0) + val

                step_end = time.perf_counter()
                slot.step_compute_time += step_end - step_start
                slot.last_step_duration = step_end - step_start

                done = step_done or slot.episode_steps >= args.max_steps
                if not done:
                    continue

                agent.end_episode(slot.episode_reward, context_id=slot.context_id)
                run_reward = slot.episode_reward
                completed_rewards.append(run_reward)
                recent_rewards.append(run_reward)

                if monitor is not None:
                    monitor.publish_score(slot.run_idx, run_reward)
                    monitor.update_hyperparams(_format_hyperparams_for_monitor(args, agent))
                    now_time = time.perf_counter()
                    elapsed = max(1e-6, now_time - experiment_start_time)
                    run_wall = max(1e-6, now_time - slot.run_start_time)
                    inference_s = float(slot.inference_time)
                    compute_s = float(slot.step_compute_time)
                    diagnostics_payload = {
                        "episodes_completed": len(completed_rewards),
                        "episode_reward": run_reward,
                        "reward_mean_5": float(np.mean(recent_rewards)) if recent_rewards else 0.0,
                        "reward_std_5": float(np.std(recent_rewards)) if len(recent_rewards) > 1 else 0.0,
                        "best_reward": float(np.max(completed_rewards)),
                        "episode_steps": slot.episode_steps,
                        "total_steps": total_steps,
                        "steps_per_second": total_steps / elapsed,
                        "perf/inference_time_s": inference_s,
                        "perf/run_wall_time_s": run_wall,
                        "perf/loop_compute_time_s": compute_s,
                        "perf/inference_wait_wall_pct": (
                            inference_s / run_wall * 100.0
                            if run_wall > 1e-9
                            else 0.0
                        ),
                        "perf/inference_wait_compute_pct": (
                            inference_s / compute_s * 100.0
                            if compute_s > 1e-9
                            else 0.0
                        ),
                        "perf/inference_calls": float(slot.inference_calls),
                    }
                    for comp_key, _ in COMPONENT_FIELDS:
                        if comp_key == "episode_reward":
                            continue
                        diagnostics_payload[comp_key] = slot.component_sums.get(comp_key, 0.0)
                    diagnostics_payload["parallel_envs"] = num_envs
                    diagnostics_payload["active_envs"] = sum(1 for s in slots if s.active)
                    if hasattr(agent, "epsilon"):
                        diagnostics_payload["epsilon"] = float(getattr(agent, "epsilon"))
                    network_metrics = network_tracker.collect()
                    if network_metrics:
                        diagnostics_payload.update(network_metrics)
                    agent_metrics_fn = getattr(agent, "latest_metrics", None)
                    if callable(agent_metrics_fn):
                        agent_metrics = agent_metrics_fn()
                        if agent_metrics:
                            diagnostics_payload.update(agent_metrics)
                    timing_snapshot_fn = getattr(agent, "timing_snapshot", None)
                    if callable(timing_snapshot_fn):
                        try:
                            timing_snapshot = timing_snapshot_fn()
                        except Exception:
                            timing_snapshot = {}
                        if isinstance(timing_snapshot, dict) and timing_snapshot:
                            episode_last = float(timing_snapshot.get("episode_update_last_s", 0.0))
                            batch_last = float(timing_snapshot.get("batch_update_last_s", 0.0))
                            diagnostics_payload["perf/update_episode_ms_last"] = (
                                episode_last * 1000.0
                            )
                            diagnostics_payload["perf/update_batch_ms_last"] = batch_last * 1000.0
                            episode_avg = float(
                                timing_snapshot.get("episode_update_avg_s", 0.0)
                            )
                            batch_avg = float(
                                timing_snapshot.get("batch_update_avg_s", 0.0)
                            )
                            diagnostics_payload["perf/update_episode_ms_avg"] = (
                                episode_avg * 1000.0
                            )
                            diagnostics_payload["perf/update_batch_ms_avg"] = (
                                batch_avg * 1000.0
                            )
                            diagnostics_payload["perf/update_episode_count"] = float(
                                timing_snapshot.get("episode_update_count", 0.0)
                            )
                            diagnostics_payload["perf/update_batch_count"] = float(
                                timing_snapshot.get("batch_update_count", 0.0)
                            )
                    try:
                        monitor.publish_diagnostics(diagnostics_payload)
                        active_envs = sum(1 for s in slots if s.active)
                        status_text = (
                            f"Completed {completed_runs + 1}/{args.runs} | "
                            f"run {slot.run_idx + 1} reward {run_reward:.2f} | "
                            f"env {slot.index + 1} fps {slot.emu_fps:.1f} | "
                            f"active {active_envs}/{num_envs}"
                        )
                        if hasattr(agent, "epsilon"):
                            try:
                                status_text += f" | eps {float(getattr(agent, 'epsilon')):.3f}"
                            except Exception:
                                pass
                        monitor.publish_status(status_text)
                    except Exception:
                        pass

                completed_runs += 1
                slot.active = False

                if checkpoint_path is not None and checkpoint_interval > 0:
                    if completed_runs % checkpoint_interval == 0:
                        save_checkpoint(completed_runs)

                if next_run_idx < args.runs:
                    if args.recreate_env:
                        try:
                            slot.env.close()
                        except Exception:
                            pass
                        slot.env = build_training_env()
                    else:
                        if not _soft_reset_env(slot.env):
                            try:
                                slot.env.close()
                            except Exception:
                                pass
                            slot.env = build_training_env()
                    slot.obs, slot.info = reset_environment(slot.env)
                    slot.info = dict(slot.info or {})
                    assign_run(slot, next_run_idx)
                    next_run_idx += 1
                else:
                    slot.run_idx = None
                    slot.step_times.clear()

            if not active_any:
                process_monitor_commands()
                check_reward_config_updates()
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("Interrupted by user; stopping experiment.", file=sys.stderr)
    finally:
        finalize_updates_fn = getattr(agent, "finalize_updates", None)
        if callable(finalize_updates_fn):
            try:
                finalize_updates_fn()
            except Exception as exc:
                print(f"Finalize updates failed: {exc}", file=sys.stderr)
        if monitor is not None:
            monitor.close()
        for slot in slots:
            if slot.viewer is not None:
                slot.viewer.close()
            try:
                slot.env.close()
            except Exception:
                pass
        if checkpoint_path is not None:
            save_checkpoint(completed_runs)
        if completed_rewards:
            mean_reward = float(np.mean(completed_rewards))
            best_reward = float(np.max(completed_rewards))
            print(
                f"Completed {len(completed_rewards)} runs. Mean reward={mean_reward:.2f}, "
                f"best={best_reward:.2f}"
            )



if __name__ == "__main__":
    main()
    set_seed_label()
    seed_label = tk.Label(root, textvariable=seed_state_var, anchor="w", justify="left")
    seed_label.pack(fill="x", padx=8, pady=2)
