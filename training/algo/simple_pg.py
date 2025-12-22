from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency in tests
    import torch
    from torch import nn, optim
    from torch.distributions import Categorical
except Exception:  # pragma: no cover - optional dependency in tests
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore
    Categorical = None  # type: ignore

from .base import AlgoAdapter
from training.utils.checkpoint_io import checkpoint_path, save_checkpoint
from training.utils.reproducibility import git_commit
from training.utils.spec import RolloutBatch


@dataclass(slots=True)
class AdvantageConfig:
    type: str = "mc"
    gae_lambda: float = 0.95
    gamma: float = 0.997


@dataclass(slots=True)
class SimplePGConfig:
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_steps: int = 16384
    advantage: AdvantageConfig = field(default_factory=AdvantageConfig)


if torch is not None:

    class _PolicyValueNet(nn.Module):
        def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(obs_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.policy_head = nn.Linear(hidden_size, action_dim)
            self.value_head = nn.Linear(hidden_size, 1)

        def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            features = self.backbone(obs)
            logits = self.policy_head(features)
            value = self.value_head(features).squeeze(-1)
            return logits, value


else:  # pragma: no cover - executed only when torch is missing

    class _PolicyValueNet:  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "SimplePGAdapter requires PyTorch. Install extras with 'pip install .[rl]' before use."
            )


class SimplePGAdapter(AlgoAdapter):
    """Lightweight policy-gradient adapter that mirrors the original speedrun loop."""

    def __init__(self, cfg: Any, env: Any, logger: Any, event_bus: Any, device: str | None = None) -> None:
        super().__init__(cfg, env, logger, event_bus, device=device)
        if torch is None:  # pragma: no cover - handled by dependency installation
            raise RuntimeError("SimplePGAdapter requires PyTorch. Install extras with 'pip install .[rl]'.")

        sp_cfg = getattr(cfg, "simple_pg", None)
        if sp_cfg is None:
            sp_cfg = {}
        elif hasattr(sp_cfg, "to_dict"):
            sp_cfg = sp_cfg.to_dict()
        adv_cfg = sp_cfg.get("advantage", {})
        self.hparams = SimplePGConfig(
            lr=float(sp_cfg.get("lr", 3e-4)),
            entropy_coef=float(sp_cfg.get("entropy_coef", 0.01)),
            value_coef=float(sp_cfg.get("value_coef", 0.5)),
            max_grad_norm=float(sp_cfg.get("max_grad_norm", 0.5)),
            batch_steps=int(sp_cfg.get("batch_steps", 16384)),
            advantage=AdvantageConfig(
                type=str(adv_cfg.get("type", "mc")),
                gae_lambda=float(adv_cfg.get("gae_lambda", 0.95)),
                gamma=float(adv_cfg.get("gamma", 0.997)),
            ),
        )

        obs_space = getattr(env, "single_observation_space", env.observation_space)
        obs_shape = obs_space.shape
        self.obs_dim = int(np.prod(obs_shape))
        self.action_dim = int(env.single_action_space.n)
        hidden_size = int(getattr(cfg.model, "hidden_size", 128)) if hasattr(cfg, "model") else 128
        self.net = _PolicyValueNet(self.obs_dim, self.action_dim, hidden_size=hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        self.global_step = 0
        self.total_steps = int(getattr(cfg.train, "total_steps", 2000000))
        self.checkpoint_interval = int(getattr(cfg.train, "checkpoint_interval", 100000))
        self.video_interval = int(getattr(cfg, "video_interval", 5000))
        self.batch_returns: deque[float] = deque(maxlen=100)
        self.batch_lengths: deque[int] = deque(maxlen=100)
        self.batch_viruses: deque[float] = deque(maxlen=100)
        self.batch_lines: deque[float] = deque(maxlen=100)
        self.batch_top_out: deque[float] = deque(maxlen=100)
        self.batch_combo: deque[float] = deque(maxlen=100)
        self.checkpoint_dir = Path(getattr(cfg, "logdir", "runs/auto")) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._next_checkpoint = self.checkpoint_interval

    # ------------------------------------------------------------------ training
    def train_forever(self) -> None:
        obs, _ = self.env.reset(seed=getattr(self.cfg, "seed", None))
        obs = obs.astype(np.float32)
        start_time = time.time()
        while self.global_step < self.total_steps:
            rollout_start = time.time()
            batch, obs = self._collect_rollout(obs)
            rollout_time = time.time() - rollout_start
            self.event_bus.emit(
                "rollout_end",
                step=self.global_step,
                n_steps=int(batch.actions.size),
                n_envs=int(self.env.num_envs),
                queue_time=0.0,
                compute_time=rollout_time,
            )
            update_start = time.time()
            metrics = self._update(batch)
            update_time = time.time() - update_start
            metrics["perf/update_sec"] = update_time
            elapsed = time.time() - start_time
            metrics["perf/sps"] = float(self.global_step / max(elapsed, 1e-6))
            self._log_metrics(metrics)
            self.event_bus.emit("update_end", step=self.global_step, **metrics)
            self._maybe_checkpoint()
            self.logger.flush()

    # ------------------------------------------------------------------ rollout
    def _collect_rollout(self, obs: np.ndarray) -> Tuple[RolloutBatch, np.ndarray]:
        T = max(self.hparams.batch_steps // self.env.num_envs, 1)
        obs_buf: List[np.ndarray] = []
        act_buf: List[np.ndarray] = []
        rew_buf: List[np.ndarray] = []
        done_buf: List[np.ndarray] = []
        val_buf: List[np.ndarray] = []
        logp_buf: List[np.ndarray] = []
        obs_t = obs.reshape(self.env.num_envs, -1)
        next_obs = obs
        for t in range(T):
            obs_tensor = torch.as_tensor(obs_t, dtype=torch.float32, device=self.device)
            logits, values = self.net(obs_tensor)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean().item()

            next_obs, rewards, terminated, truncated, infos = self.env.step(actions.cpu().numpy())
            done = np.logical_or(terminated, truncated).astype(np.float32)

            obs_buf.append(obs_t.copy())
            act_buf.append(actions.cpu().numpy())
            rew_buf.append(rewards.copy())
            done_buf.append(done.copy())
            val_buf.append(values.detach().cpu().numpy())
            logp_buf.append(log_probs.detach().cpu().numpy())

            self.global_step += int(self.env.num_envs)
            if self.video_interval > 0 and self.global_step % self.video_interval == 0:
                frame = self.env.render()
                self.event_bus.emit("video_frame", step=self.global_step, tag="rollout", frame=frame)

            for info in infos:
                if "episode" in info:
                    ep = info["episode"]
                    drm = info.get("drm", {})
                    self.batch_returns.append(float(ep.get("r", 0.0)))
                    self.batch_lengths.append(int(ep.get("l", 0)))
                    self.batch_viruses.append(float(drm.get("viruses_cleared", 0.0)))
                    self.batch_lines.append(float(drm.get("lines_cleared", 0.0)))
                    self.batch_top_out.append(float(drm.get("top_out", False)))
                    self.batch_combo.append(float(drm.get("combo_max", 0.0)))
                    self.event_bus.emit(
                        "episode_end",
                        step=self.global_step,
                        ret=float(ep.get("r", 0.0)),
                        len=int(ep.get("l", 0)),
                        **{f"drm/{k}": v for k, v in drm.items()},
                    )

            obs_t = next_obs.reshape(self.env.num_envs, -1)

            if self.global_step >= self.total_steps:
                break

        # Bootstrap
        obs_tensor = torch.as_tensor(obs_t, dtype=torch.float32, device=self.device)
        _, bootstrap_values = self.net(obs_tensor)
        rewards_arr = np.stack(rew_buf)
        values_arr = np.stack(val_buf)
        dones_arr = np.stack(done_buf)
        bootstrap = bootstrap_values.detach().cpu().numpy()
        adv, ret = self._compute_advantages(rewards_arr, values_arr, dones_arr, bootstrap)
        rollout = RolloutBatch(
            observations=np.stack(obs_buf, axis=0),
            actions=np.stack(act_buf, axis=0),
            rewards=rewards_arr,
            dones=dones_arr,
            values=values_arr,
            log_probs=np.stack(logp_buf, axis=0),
            advantages=adv,
            returns=ret,
        )
        return rollout, next_obs.astype(np.float32)

    # ------------------------------------------------------------------ update
    def _update(self, batch: RolloutBatch) -> Dict[str, float]:
        obs = torch.as_tensor(batch.observations.reshape(-1, self.obs_dim), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions.reshape(-1), dtype=torch.long, device=self.device)
        returns = torch.as_tensor(batch.returns.reshape(-1), dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch.advantages.reshape(-1), dtype=torch.float32, device=self.device)
        log_probs_old = torch.as_tensor(batch.log_probs.reshape(-1), dtype=torch.float32, device=self.device)

        logits, values = self.net(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        policy_loss = -(advantages.detach() * log_probs).mean()
        value_loss = 0.5 * (returns - values).pow(2).mean()
        loss = policy_loss + self.hparams.value_coef * value_loss - self.hparams.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.hparams.max_grad_norm)
        self.optimizer.step()

        kld = torch.mean(log_probs_old.exp() * (log_probs_old - log_probs)).item()
        explained_var = self._explained_variance(returns, values)
        metrics = {
            "loss/policy": float(policy_loss.item()),
            "loss/value": float(value_loss.item()),
            "policy/entropy": float(entropy.item()),
            "loss/total": float(loss.item()),
            "loss/kl": float(kld),
            "optim/lr": float(self.optimizer.param_groups[0]["lr"]),
            "optim/grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
            "adv/mean": float(advantages.mean().item()),
            "adv/std": float(advantages.std(unbiased=False).item()),
            "vf/explained_var": float(explained_var),
        }
        return metrics

    # ---------------------------------------------------------------- utilities
    def _compute_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        bootstrap: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        T, N = rewards.shape
        adv = np.zeros_like(rewards)
        last_adv = np.zeros(N, dtype=np.float32)
        returns = np.zeros_like(rewards)
        gamma = self.hparams.advantage.gamma
        lam = self.hparams.advantage.gae_lambda
        use_gae = self.hparams.advantage.type.lower() == "gae"
        next_values = bootstrap
        next_return = bootstrap
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            if t < T - 1:
                next_values = values[t + 1]
                next_return = returns[t + 1]
            delta = rewards[t] + gamma * next_values * mask - values[t]
            if use_gae:
                last_adv = delta + gamma * lam * mask * last_adv
                adv[t] = last_adv
            else:
                adv[t] = delta
            returns[t] = rewards[t] + gamma * next_return * mask
            next_return = returns[t]
        if not use_gae:
            adv = returns - values
        return adv, returns

    @staticmethod
    def _explained_variance(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        y_true_detached = y_true.detach()
        residual = (y_true - y_pred).detach()
        var_y = torch.var(y_true_detached)
        if var_y < 1e-6:
            return 0.0
        return float(1.0 - torch.var(residual) / (var_y + 1e-6))

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        step = self.global_step
        if self.batch_returns:
            returns = np.array(self.batch_returns, dtype=np.float32)
            self.logger.log_scalar("train/return_mean", float(returns.mean()), step)
            self.logger.log_scalar("train/return_std", float(returns.std()), step)
        if self.batch_lengths:
            lengths = np.array(self.batch_lengths, dtype=np.float32)
            self.logger.log_scalar("train/ep_len_mean", float(lengths.mean()), step)
        if self.batch_viruses:
            viruses = np.array(self.batch_viruses, dtype=np.float32)
            self.logger.log_scalar("drm/viruses_per_ep", float(viruses.mean()), step)
        if self.batch_lines:
            lines = np.array(self.batch_lines, dtype=np.float32)
            self.logger.log_scalar("drm/lines_per_ep", float(lines.mean()), step)
        if self.batch_top_out:
            top_out = np.array(self.batch_top_out, dtype=np.float32)
            self.logger.log_scalar("drm/top_out_rate", float(top_out.mean()), step)
        for key, value in metrics.items():
            self.logger.log_scalar(key, value, step)

    def _maybe_checkpoint(self) -> None:
        if self.global_step < self._next_checkpoint:
            return
        payload = {
            "state_dict": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": getattr(self.cfg, "to_dict", lambda: {} )(),
            "step": self.global_step,
            "sha": git_commit(),
        }
        path = checkpoint_path(self.checkpoint_dir, "simple_pg", self.global_step, compress=True)
        save_checkpoint(payload, path)
        self.event_bus.emit("checkpoint", step=self.global_step, path=str(path), walltime=time.time())
        self._next_checkpoint += self.checkpoint_interval
