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
import multiprocessing as mp
import numbers
import os
import queue
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from envs.retro.demo import _ProcessViewer
from envs.retro.register_env import register_env_id
from envs.retro.drmario_env import Action
from gymnasium import make
from models.policy.networks import DrMarioStatePolicyNet, DrMarioPixelUNetPolicyNet

try:
    import torch
    from torch import nn, optim
    from torch.distributions import Categorical
except Exception:  # pragma: no cover - torch is optional for this experiment
    torch = None
    nn = None
    optim = None
    Categorical = None


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

    hyper_label = tk.Label(
        root, text=format_hyperparams(hyperparams), anchor="w", justify="left"
    )
    hyper_label.pack(fill="x", padx=8, pady=4)

    status_var = tk.StringVar(value="Waiting for runs...")
    status_label = tk.Label(root, textvariable=status_var, anchor="w", justify="left")
    status_label.pack(fill="x", padx=8, pady=2)

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
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
    canvas.pack(fill="both", expand=True, padx=8, pady=8)

    scores: list[Tuple[int, float]] = []
    score_by_run: Dict[int, float] = {}
    batch_medians: Dict[int, float] = {}
    runs_total = int(hyperparams.get("runs", 0) or 0)
    parallel_envs_var = int(hyperparams.get("parallel_envs", 1) or 1)
    batch_runs_var = int(hyperparams.get("batch_runs", 1) or 1)
    group_size_var = max(1, parallel_envs_var, batch_runs_var)
    latest_diagnostics: Dict[str, Any] = {}
    diagnostic_history: Dict[str, deque[float]] = {}
    diagnostics_window = 20

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
        canvas.create_line(
            margin, margin + plot_h, margin + plot_w, margin + plot_h, tags="plot", fill="#444"
        )

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
            text=f"Batches {min_run}–{max_run} | Median {values[-1]:.1f}",
            fill="#222",
        )

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

    def pump_queue() -> None:
        nonlocal runs_total, parallel_envs_var, batch_runs_var, group_size_var
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
                        updated = True
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
                        try:
                            batch_runs_var = int(
                                payload.get("batch_runs", batch_runs_var) or batch_runs_var or 1
                            )
                        except Exception:
                            pass
                        group_size_new = max(1, parallel_envs_var, batch_runs_var)
                        if group_size_new != group_size_var:
                            group_size_var = group_size_new
                            recompute_series(group_size_var)
                            updated = True
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


class NetworkDiagnosticsTracker:
    """Collect lightweight diagnostics from torch networks and optimizers."""

    def __init__(self, agent: Any):
        self._torch = torch
        self._nn = nn
        self._optim = optim
        self._agent = agent
        self._modules: Dict[str, Any] = {}
        self._optimizers: Dict[str, Any] = {}
        self._prev_params: Dict[str, "torch.Tensor"] = {}
        if self._torch is None:
            return
        self._discover_modules(agent)
        self._snapshot_initial_params()

    def _discover_modules(self, agent: Any) -> None:
        if self._torch is None:
            return
        attr_names = dir(agent)
        for name in attr_names:
            # getattr may raise for properties; guard accordingly.
            try:
                value = getattr(agent, name)
            except Exception:
                continue
            if self._nn is not None and isinstance(value, self._nn.Module):
                existing = self._modules.get(name)
                if existing is not value:
                    self._modules[name] = value
                    flat = self._flatten_parameters(value)
                    if flat is not None:
                        self._prev_params[name] = flat
            if self._optim is not None and isinstance(value, self._optim.Optimizer):
                self._optimizers[name] = value

    def _snapshot_initial_params(self) -> None:
        if self._torch is None:
            return
        for name, module in self._modules.items():
            flat = self._flatten_parameters(module)
            if flat is not None:
                self._prev_params[name] = flat

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

    def collect(self) -> Dict[str, Any]:
        """Return a dictionary of diagnostic scalars (and status strings)."""
        if self._torch is None:
            return {"networks/status": "torch unavailable"}
        # Refresh in case new modules/optimizers were attached since the last call.
        self._discover_modules(self._agent)
        metrics: Dict[str, Any] = {}
        if not self._modules:
            metrics["networks/status"] = "no modules detected"
            return metrics

        metrics["networks/status"] = f"{len(self._modules)} module(s) tracked"

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

        return metrics


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

    def publish_diagnostics(self, metrics: Dict[str, Any]) -> None:
        if self._closed or not metrics:
            return
        try:
            self._queue.put_nowait({"type": "diagnostics", "metrics": metrics})
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

    def begin_episode(self, context_id: Optional[int] = None) -> None:
        pass

    def observe_step(self, _reward: float, _done: bool, context_id: Optional[int] = None) -> None:
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
    log_probs: List["torch.Tensor"] = field(default_factory=list)
    values: List["torch.Tensor"] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    entropies: List["torch.Tensor"] = field(default_factory=list)
    recurrent_state: Optional[Any] = None


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
        self._batch_runs = max(1, int(batch_runs))
        self._episodes_in_batch = 0
        self._steps_in_batch = 0
        self._batch_accum_policy_loss = 0.0
        self._batch_accum_value_loss = 0.0
        self._batch_accum_entropy = 0.0
        self._contexts: Dict[int, _EpisodeContext] = {}
        augmented_stack = _apply_color_representation(np.asarray(prototype_obs, dtype=np.float32), color_repr)
        self._stack_depth = augmented_stack.shape[0] if augmented_stack.ndim >= 3 else 1
        self._input_dim = int(np.prod(augmented_stack.shape))
        if policy_arch == "mlp":
            self.model = SimpleStateActorCritic(self._input_dim, int(action_space.n)).to(self._device)
        elif policy_arch == "drmario_cnn":
            in_channels = augmented_stack.shape[1] if augmented_stack.ndim >= 3 else 14
            self.model = DrMarioStatePolicyNet(int(action_space.n), in_channels=in_channels).to(self._device)
        elif policy_arch == "drmario_color_cnn":
            in_channels = augmented_stack.shape[1] if augmented_stack.ndim >= 3 else 17
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(learning_rate))

        self._updates = 0
        self._last_metrics: Dict[str, float] = {}

    def _prepare_obs(self, obs: Any) -> torch.Tensor:
        stack = _extract_state_stack(obs)
        stack_np = stack.astype(np.float32, copy=False)
        if self._obs_mode == "state":
            stack_np = _apply_color_representation(stack_np, self._color_repr)
        if self._policy_arch == "mlp":
            flat = torch.from_numpy(stack_np).view(-1)
            return flat.to(self._device)
        if self._obs_mode == "pixel":
            tensor = torch.from_numpy(stack_np).permute(0, 3, 1, 2).contiguous()
        else:
            tensor = torch.from_numpy(stack_np)
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
        ctx.log_probs.clear()
        ctx.values.clear()
        ctx.rewards.clear()
        ctx.entropies.clear()
        ctx.recurrent_state = None
        return ctx

    def begin_episode(self, context_id: Optional[int] = None) -> None:
        self.model.train(True)
        ctx = self._reset_context(context_id)
        if self._policy_arch == "mlp":
            ctx.recurrent_state = None

    def select_action(self, obs: Any, _info: Dict[str, Any], context_id: Optional[int] = None) -> int:
        ctx = self._get_context(context_id)
        state_tensor = self._prepare_obs(obs)
        if self._policy_arch == "mlp":
            logits, value = self.model(state_tensor.unsqueeze(0))
            dist = Categorical(logits=logits.squeeze(0))
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value_tensor = value.squeeze(0)
            entropy = dist.entropy()
        else:
            if state_tensor.dim() == 3:
                input_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
            elif state_tensor.dim() == 4:
                input_tensor = state_tensor.unsqueeze(0)
            else:
                raise RuntimeError("Unexpected observation tensor shape")
            logits, value, hx = self.model(input_tensor, ctx.recurrent_state)
            logits = logits[:, -1, :]
            value_tensor = value[:, -1]
            dist = Categorical(logits=logits.squeeze(0))
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            if isinstance(hx, torch.Tensor):
                ctx.recurrent_state = hx.detach()
            elif isinstance(hx, tuple):
                ctx.recurrent_state = tuple(t.detach() for t in hx)
            else:
                ctx.recurrent_state = None

        ctx.log_probs.append(log_prob)
        ctx.values.append(value_tensor.squeeze(0))
        ctx.entropies.append(entropy)

        return int(action.item())

    def observe_step(self, reward: float, _done: bool, context_id: Optional[int] = None) -> None:
        ctx = self._get_context(context_id)
        ctx.rewards.append(float(reward))

    def end_episode(self, _score: float, context_id: Optional[int] = None) -> None:
        ctx = self._get_context(context_id)
        if not ctx.log_probs:
            return

        returns: list[float] = []
        G = 0.0
        # Drop the dummy terminal marker if present
        rewards = [r for r in ctx.rewards if not np.isnan(r)]
        for reward in reversed(rewards):
            G = reward + self._gamma * G
            returns.append(G)
        returns.reverse()
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self._device)

        log_probs = torch.stack(ctx.log_probs)
        values = torch.stack(ctx.values)
        entropies = torch.stack(ctx.entropies)
        if returns_tensor.shape[0] != log_probs.shape[0]:
            min_len = min(returns_tensor.shape[0], log_probs.shape[0])
            returns_tensor = returns_tensor[:min_len]
            log_probs = log_probs[:min_len]
            values = values[:min_len]
            entropies = entropies[:min_len]

        if self._episodes_in_batch == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self._batch_accum_policy_loss = 0.0
            self._batch_accum_value_loss = 0.0
            self._batch_accum_entropy = 0.0

        advantages = returns_tensor - values.detach()
        policy_loss = -(log_probs * advantages).mean()
        value_loss = 0.5 * (returns_tensor - values).pow(2).mean()
        entropy_term = entropies.mean()
        loss = policy_loss + self._value_coef * value_loss - self._entropy_coef * entropy_term

        scale = 1.0 / float(max(1, self._batch_runs))
        (loss * scale).backward()

        self._batch_accum_policy_loss += float(policy_loss.item())
        self._batch_accum_value_loss += float(value_loss.item())
        self._batch_accum_entropy += float(entropy_term.item())

        self._episodes_in_batch += 1
        self._steps_in_batch += int(returns_tensor.shape[0])

        ctx.log_probs.clear()
        ctx.values.clear()
        ctx.rewards.clear()
        ctx.entropies.clear()
        ctx.recurrent_state = None

        if self._episodes_in_batch >= self._batch_runs:
            self._update_policy()

    def latest_metrics(self) -> Dict[str, float]:
        metrics = dict(self._last_metrics)
        metrics["learner/pending_runs"] = float(self._episodes_in_batch)
        metrics["learner/pending_steps"] = float(self._steps_in_batch)
        return metrics

    def _update_policy(self) -> None:
        self._apply_optimizer_step(adjust_for_partial=False)

    def finalize_updates(self) -> None:
        self._apply_optimizer_step(adjust_for_partial=True)

    def _apply_optimizer_step(self, adjust_for_partial: bool) -> None:
        if self._episodes_in_batch <= 0:
            return
        batch_count = max(1, self._episodes_in_batch)
        if adjust_for_partial and self._episodes_in_batch < self._batch_runs:
            scale = float(self._batch_runs) / float(batch_count)
            for param in self.model.parameters():
                grad = param.grad
                if grad is not None:
                    grad.mul_(scale)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._max_grad_norm)
        self.optimizer.step()
        self._updates += 1

        denom = float(batch_count)
        self._last_metrics = {
            "learner/policy_loss": self._batch_accum_policy_loss / denom,
            "learner/value_loss": self._batch_accum_value_loss / denom,
            "learner/entropy": self._batch_accum_entropy / denom,
            "learner/updates": float(self._updates),
        }
        self.optimizer.zero_grad(set_to_none=True)
        self._episodes_in_batch = 0
        self._steps_in_batch = 0
        self._batch_accum_policy_loss = 0.0
        self._batch_accum_value_loss = 0.0
        self._batch_accum_entropy = 0.0


def _extract_state_stack(observation: Any) -> np.ndarray:
    core = observation["obs"] if isinstance(observation, dict) else observation
    return np.asarray(core)


def _apply_color_representation(stack: np.ndarray, mode: str) -> np.ndarray:
    if mode != "shared_color":
        return stack
    if stack.ndim < 3:
        return stack
    if stack.shape[1] not in (14, 17):
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
        "color_repr": args.color_repr,
        "batch_runs": args.batch_runs,
        "randomize_rng": bool(args.randomize_rng),
        "frame_offset": args.frame_offset,
        "parallel_envs": getattr(args, "parallel_envs", getattr(args, "num_envs", None)),
    }
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
    return payload


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
    ap.add_argument(
        "--strategy",
        choices=["random", "down", "down_hold", "noop", "policy"],
        default="random",
    )
    ap.add_argument("--learner", choices=["none", "reinforce"], default="none")
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--epsilon-decay", type=float, default=0.97)
    ap.add_argument("--epsilon-min", type=float, default=0.05)
    ap.add_argument(
        "--policy-arch",
        choices=["mlp", "drmario_cnn", "drmario_color_cnn", "drmario_unet_pixel"],
        default="mlp",
    )
    ap.add_argument("--color-repr", choices=["none", "shared_color"], default="none")
    ap.add_argument("--batch-runs", type=int, default=1, help="Episodes to accumulate before each policy update.")
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--device", type=str, default=None)
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

    args = ap.parse_args()

    register_env_id()

    if args.mode != "state" and args.color_repr != "none":
        raise ValueError("--color-repr shared_color is only supported for state mode observations")

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

    def build_env() -> Any:
        return make("DrMarioRetroEnv-v0", **env_kwargs)

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
        reset_options["start_wait_frames"] = int(args.start_wait_frames)

    env = build_env()
    obs, info = env.reset(options=reset_options)

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
        agent: Any = PolicyGradientAgent(
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

    total_steps = 0
    target_hz = max(0.0, float(args.emu_target_hz))
    step_period = 1.0 / target_hz if target_hz > 0 else 0.0
    experiment_start_time = time.perf_counter()
    completed_rewards: list[float] = []
    recent_rewards = deque(maxlen=5)

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
            slot_env = build_env()
            slot_obs, slot_info = slot_env.reset(options=reset_options)
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

    next_run_idx = 0
    for slot in slots:
        if next_run_idx >= args.runs:
            break
        assign_run(slot, next_run_idx)
        next_run_idx += 1

    completed_runs = 0

    try:
        while completed_runs < args.runs:
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

                action = agent.select_action(slot.obs, slot.info, context_id=slot.context_id)
                next_obs, reward, term, trunc, step_info = slot.env.step(action)
                step_info = dict(step_info or {})
                step_done = bool(term or trunc)

                slot.episode_reward += reward
                slot.episode_steps += 1
                total_steps += 1

                agent.observe_step(reward, step_done, context_id=slot.context_id)

                slot.step_times.append(time.perf_counter())
                if len(slot.step_times) >= 2:
                    span = slot.step_times[-1] - slot.step_times[0]
                    if span > 0:
                        slot.emu_fps = float(len(slot.step_times) - 1) / span

                publish_frame(slot, next_obs, step_info, int(action), reward, total_steps, slot.episode_reward)

                slot.obs = next_obs
                slot.info = step_info

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
                    elapsed = max(1e-6, time.perf_counter() - experiment_start_time)
                    diagnostics_payload = {
                        "episodes_completed": len(completed_rewards),
                        "episode_reward": run_reward,
                        "reward_mean_5": float(np.mean(recent_rewards)) if recent_rewards else 0.0,
                        "reward_std_5": float(np.std(recent_rewards)) if len(recent_rewards) > 1 else 0.0,
                        "best_reward": float(np.max(completed_rewards)),
                        "episode_steps": slot.episode_steps,
                        "total_steps": total_steps,
                        "steps_per_second": total_steps / elapsed,
                    }
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

                if next_run_idx < args.runs:
                    if args.recreate_env:
                        try:
                            slot.env.close()
                        except Exception:
                            pass
                        slot.env = build_env()
                    else:
                        if not _soft_reset_env(slot.env):
                            try:
                                slot.env.close()
                            except Exception:
                                pass
                            slot.env = build_env()
                    slot.obs, slot.info = slot.env.reset(options=reset_options)
                    slot.info = dict(slot.info or {})
                    assign_run(slot, next_run_idx)
                    next_run_idx += 1
                else:
                    slot.run_idx = None
                    slot.step_times.clear()

            if not active_any:
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
        if completed_rewards:
            mean_reward = float(np.mean(completed_rewards))
            best_reward = float(np.max(completed_rewards))
            print(
                f"Completed {len(completed_rewards)} runs. Mean reward={mean_reward:.2f}, "
                f"best={best_reward:.2f}"
            )



if __name__ == "__main__":
    main()
