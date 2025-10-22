from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Tuple

import numpy as np

from .base import AlgoAdapter
from training.utils.reproducibility import git_commit
from training.utils.spec import RolloutBatch

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn, optim
    from torch.distributions import Categorical
    from torch.nn import functional as F
    from torch.nn.utils import clip_grad_norm_
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore
    Categorical = None  # type: ignore
    F = None  # type: ignore
    clip_grad_norm_ = None  # type: ignore


@dataclass(slots=True)
class PPOConfig:
    rollout: int = 256
    batch_size: int = 2048
    mini_epochs: int = 4
    minibatch_size: int = 256
    lr: float = 3e-4
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


if torch is not None:

    class _PPOPolicy(nn.Module):
        def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256) -> None:
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
            values = self.value_head(features).squeeze(-1)
            return logits, values


else:  # pragma: no cover - executed only when torch is missing

    class _PPOPolicy:  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("SampleFactoryAdapter requires PyTorch. Install extras with 'pip install .[rl]'.")


class SampleFactoryAdapter(AlgoAdapter):
    """In-repo PPO implementation that mirrors Sample Factory defaults."""

    def __init__(self, cfg: Any, env: Any, logger: Any, event_bus: Any, device: str | None = None) -> None:
        super().__init__(cfg, env, logger, event_bus, device=device)
        if torch is None:  # pragma: no cover - dependency guard
            raise RuntimeError("SampleFactoryAdapter requires PyTorch. Install extras with 'pip install .[rl]'.")

        ppo_cfg = getattr(cfg, "ppo", {})
        if hasattr(ppo_cfg, "to_dict"):
            ppo_cfg = ppo_cfg.to_dict()

        self.hparams = PPOConfig(
            rollout=int(ppo_cfg.get("rollout", 256)),
            batch_size=int(ppo_cfg.get("batch_size", 2048)),
            mini_epochs=int(ppo_cfg.get("mini_epochs", 4)),
            minibatch_size=int(ppo_cfg.get("minibatch_size", max(1, int(ppo_cfg.get("batch_size", 2048)) // 8))),
            lr=float(ppo_cfg.get("lr", 3e-4)),
            gamma=float(ppo_cfg.get("gamma", 0.997)),
            gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
            clip_param=float(ppo_cfg.get("clip_param", 0.2)),
            value_loss_coef=float(ppo_cfg.get("value_loss_coef", 0.5)),
            entropy_coef=float(ppo_cfg.get("entropy_coef", 0.01)),
            max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        )

        obs_shape = env.observation_space.shape
        self.obs_dim = int(np.prod(obs_shape))
        self.action_dim = int(env.single_action_space.n)
        hidden_size = int(getattr(cfg.model, "hidden_size", 256)) if hasattr(cfg, "model") else 256
        self.net = _PPOPolicy(self.obs_dim, self.action_dim, hidden_size=hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        self.global_step = 0
        self.total_steps = int(getattr(cfg.train, "total_steps", 2000000))
        self.checkpoint_interval = int(getattr(cfg.train, "checkpoint_interval", 100000))
        self.video_interval = int(getattr(cfg, "video_interval", 5000))
        self.checkpoint_dir = Path(getattr(cfg, "logdir", "runs/auto")) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._next_checkpoint = self.checkpoint_interval

        self.batch_returns: Deque[float] = deque(maxlen=100)
        self.batch_lengths: Deque[int] = deque(maxlen=100)
        self.batch_viruses: Deque[float] = deque(maxlen=100)
        self.batch_lines: Deque[float] = deque(maxlen=100)
        self.batch_top_out: Deque[float] = deque(maxlen=100)
        self.batch_combo: Deque[float] = deque(maxlen=100)

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
            elapsed = max(time.time() - start_time, 1e-6)
            metrics["perf/sps"] = float(self.global_step / elapsed)
            self._log_metrics(metrics)
            self.event_bus.emit("update_end", step=self.global_step, **metrics)
            self._maybe_checkpoint()
            self.logger.flush()

    # ------------------------------------------------------------------ rollout
    def _collect_rollout(self, obs: np.ndarray) -> Tuple[RolloutBatch, np.ndarray]:
        T = max(int(self.hparams.rollout), 1)
        num_envs = int(self.env.num_envs)
        obs_buf: List[np.ndarray] = []
        act_buf: List[np.ndarray] = []
        rew_buf: List[np.ndarray] = []
        done_buf: List[np.ndarray] = []
        val_buf: List[np.ndarray] = []
        logp_buf: List[np.ndarray] = []

        obs_t = obs.reshape(num_envs, -1)
        for _ in range(T):
            obs_tensor = torch.as_tensor(obs_t, dtype=torch.float32, device=self.device)
            logits, values = self.net(obs_tensor)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            next_obs, rewards, terminated, truncated, infos = self.env.step(actions.cpu().numpy())
            done = np.logical_or(terminated, truncated).astype(np.float32)

            obs_buf.append(obs_t.copy())
            act_buf.append(actions.cpu().numpy())
            rew_buf.append(rewards.astype(np.float32))
            done_buf.append(done)
            val_buf.append(values.detach().cpu().numpy().astype(np.float32))
            logp_buf.append(log_probs.detach().cpu().numpy().astype(np.float32))

            self.global_step += num_envs
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

            obs_t = next_obs.reshape(num_envs, -1)
            obs = next_obs.astype(np.float32)

        with torch.no_grad():
            last_obs_tensor = torch.as_tensor(obs_t, dtype=torch.float32, device=self.device)
            _, last_values = self.net(last_obs_tensor)

        rewards_arr = np.stack(rew_buf)
        values_arr = np.stack(val_buf)
        dones_arr = np.stack(done_buf)
        adv, ret = self._compute_gae(rewards_arr, values_arr, dones_arr, last_values.detach().cpu().numpy())

        obs_np = np.stack(obs_buf)
        actions_np = np.stack(act_buf)
        logp_np = np.stack(logp_buf)

        batch = RolloutBatch(
            observations=obs_np.reshape(T * num_envs, -1).astype(np.float32),
            actions=actions_np.reshape(T * num_envs).astype(np.int64),
            rewards=rewards_arr.reshape(T * num_envs).astype(np.float32),
            dones=dones_arr.reshape(T * num_envs).astype(np.float32),
            values=values_arr.reshape(T * num_envs).astype(np.float32),
            log_probs=logp_np.reshape(T * num_envs).astype(np.float32),
            advantages=adv.reshape(T * num_envs).astype(np.float32),
            returns=ret.reshape(T * num_envs).astype(np.float32),
        )
        return batch, obs

    # ------------------------------------------------------------------ update
    def _update(self, batch: RolloutBatch) -> Dict[str, float]:
        total_transitions = batch.actions.shape[0]
        obs = torch.as_tensor(batch.observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(batch.log_probs, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        metrics: Dict[str, float] = {
            "loss/policy": 0.0,
            "loss/value": 0.0,
            "loss/total": 0.0,
            "policy/entropy": 0.0,
            "policy/approx_kl": 0.0,
            "policy/clip_frac": 0.0,
            "adv/mean": float(batch.advantages.mean()),
            "adv/std": float(batch.advantages.std()),
        }

        updates = 0
        for _ in range(self.hparams.mini_epochs):
            indices = torch.randperm(total_transitions, device=self.device)
            minibatch = max(1, min(self.hparams.minibatch_size, total_transitions))
            for start in range(0, total_transitions, minibatch):
                end = min(start + minibatch, total_transitions)
                mb_idx = indices[start:end]

                logits, value_pred = self.net(obs[mb_idx])
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(actions[mb_idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - old_log_probs[mb_idx])
                adv_mb = advantages[mb_idx]
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.hparams.clip_param, 1.0 + self.hparams.clip_param) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value_pred, returns[mb_idx])
                loss = (
                    policy_loss
                    + self.hparams.value_loss_coef * value_loss
                    - self.hparams.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.net.parameters(), self.hparams.max_grad_norm)
                self.optimizer.step()

                approx_kl = torch.mean(old_log_probs[mb_idx] - log_probs).item()
                clip_frac = torch.mean((torch.abs(ratio - 1.0) > self.hparams.clip_param).float()).item()

                metrics["loss/policy"] += float(policy_loss.item())
                metrics["loss/value"] += float(value_loss.item())
                metrics["loss/total"] += float(loss.item())
                metrics["policy/entropy"] += float(entropy.item())
                metrics["policy/approx_kl"] += approx_kl
                metrics["policy/clip_frac"] += clip_frac
                updates += 1

        updates = max(updates, 1)
        for key in ["loss/policy", "loss/value", "loss/total", "policy/entropy", "policy/approx_kl", "policy/clip_frac"]:
            metrics[key] /= updates

        with torch.no_grad():
            _, values_eval = self.net(obs)
        metrics["value/explained_var"] = self._explained_variance(returns, values_eval)

        return metrics

    # ------------------------------------------------------------------ utils
    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        T, N = rewards.shape
        adv = np.zeros((T, N), dtype=np.float32)
        last_gae = np.zeros(N, dtype=np.float32)
        next_value = last_values.astype(np.float32)
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.hparams.gamma * next_value * mask - values[t]
            last_gae = delta + self.hparams.gamma * self.hparams.gae_lambda * mask * last_gae
            adv[t] = last_gae
            next_value = values[t]
        returns = adv + values
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
            "cfg": getattr(self.cfg, "to_dict", lambda: {})(),
            "step": self.global_step,
            "sha": git_commit(),
        }
        path = self.checkpoint_dir / f"ppo_step{self.global_step}.pt"
        torch.save(payload, path)
        self.event_bus.emit("checkpoint", step=self.global_step, path=str(path), walltime=time.time())
        self._next_checkpoint += self.checkpoint_interval
