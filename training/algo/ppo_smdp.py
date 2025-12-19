"""SMDP-PPO trainer for placement policies.

Implements PPO with SMDP (Semi-Markov Decision Process) discounting where
actions span variable durations τ and credit assignment uses γ^τ.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import optim
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    optim = None  # type: ignore

from training.algo.base import AlgoAdapter
from training.rollout.decision_buffer import DecisionBatch, DecisionRolloutBuffer, DecisionStep
from training.utils.reproducibility import git_commit
from models.policy.placement_heads import PlacementPolicyNet
from models.policy.placement_dist import MaskedPlacementDist, unflatten_placement

import envs.specs.ram_to_state as ram_specs

@dataclass(slots=True)
class SMDPPPOConfig:
    """Configuration for SMDP-PPO."""
    
    # Learning
    lr: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    
    # PPO
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Rollout
    decisions_per_update: int = 512
    num_epochs: int = 4
    minibatch_size: int = 128
    
    # Policy head
    head_type: str = "dense"  # dense, shift_score, or factorized
    pill_embed_dim: int = 32
    encoder_blocks: int = 0
    
    # Exploration
    entropy_schedule_end: float = 0.003
    entropy_schedule_steps: int = 1000000
    use_gumbel_topk: bool = False
    gumbel_k: int = 2
    
    # Misc
    value_loss_type: str = "mse"  # mse or huber


class SMDPPPOAdapter(AlgoAdapter):
    """PPO trainer for placement policies with SMDP discounting."""
    
    def __init__(
        self,
        cfg: Any,
        env: Any,
        logger: Any,
        event_bus: Any,
        device: Optional[str] = None,
    ):
        super().__init__(cfg, env, logger, event_bus, device=device)
        
        if torch is None:
            raise RuntimeError("PyTorch is required for SMDP-PPO")
            
        # Parse config
        ppo_cfg_dict = getattr(cfg, "smdp_ppo", {})
        if hasattr(ppo_cfg_dict, "to_dict"):
            ppo_cfg_dict = ppo_cfg_dict.to_dict()
            
        self.hparams = SMDPPPOConfig(
            lr=float(ppo_cfg_dict.get("lr", 3e-4)),
            gamma=float(ppo_cfg_dict.get("gamma", 0.995)),
            gae_lambda=float(ppo_cfg_dict.get("gae_lambda", 0.95)),
            clip_epsilon=float(ppo_cfg_dict.get("clip_epsilon", 0.2)),
            value_coef=float(ppo_cfg_dict.get("value_coef", 0.5)),
            entropy_coef=float(ppo_cfg_dict.get("entropy_coef", 0.01)),
            max_grad_norm=float(ppo_cfg_dict.get("max_grad_norm", 0.5)),
            decisions_per_update=int(ppo_cfg_dict.get("decisions_per_update", 512)),
            num_epochs=int(ppo_cfg_dict.get("num_epochs", 4)),
            minibatch_size=int(ppo_cfg_dict.get("minibatch_size", 128)),
            head_type=str(ppo_cfg_dict.get("head_type", "dense")),
            pill_embed_dim=int(ppo_cfg_dict.get("pill_embed_dim", 32)),
            encoder_blocks=int(ppo_cfg_dict.get("encoder_blocks", 0)),
            entropy_schedule_end=float(ppo_cfg_dict.get("entropy_schedule_end", 0.003)),
            entropy_schedule_steps=int(ppo_cfg_dict.get("entropy_schedule_steps", 1000000)),
            use_gumbel_topk=bool(ppo_cfg_dict.get("use_gumbel_topk", False)),
            gumbel_k=int(ppo_cfg_dict.get("gumbel_k", 2)),
            value_loss_type=str(ppo_cfg_dict.get("value_loss_type", "mse")),
        )
        
        # Environment info
        obs_space = getattr(env, "single_observation_space", env.observation_space)
        obs_shape = obs_space.shape  # expected [C, 16, 8] when frame_stack == 1
        in_channels = obs_shape[0] if len(obs_shape) == 3 else 12
        
        # Create policy network
        self.net = PlacementPolicyNet(
            in_channels=in_channels,
            head_type=self.hparams.head_type,
            pill_embed_dim=self.hparams.pill_embed_dim,
            encoder_blocks=self.hparams.encoder_blocks,
            num_colors=3,
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        
        # Rollout buffer (decision-wise)
        self.buffer = DecisionRolloutBuffer(
            capacity=self.hparams.decisions_per_update * 2,
            obs_shape=obs_shape,
            num_envs=env.num_envs,
            gamma=self.hparams.gamma,
            gae_lambda=self.hparams.gae_lambda,
        )
        
        # Tracking
        self.global_step = 0  # Total environment steps (frames)
        self.decision_step = 0  # Total decisions made
        self.total_steps = int(getattr(cfg.train, "total_steps", 5000000))
        self.checkpoint_interval = int(getattr(cfg.train, "checkpoint_interval", 100000))
        
        self.batch_returns: deque[float] = deque(maxlen=100)
        self.batch_lengths: deque[int] = deque(maxlen=100)
        self.batch_viruses: deque[float] = deque(maxlen=100)
        self.batch_decisions: deque[int] = deque(maxlen=100)

        # Lightweight perf counters (used by debug UI via RateLimitedVecEnv hooks).
        self._perf_inference_calls: int = 0
        self._perf_inference_sec_total: float = 0.0
        self._perf_last_inference_sec: float = 0.0
        self._last_update_step: int = 0
        
        self.checkpoint_dir = Path(getattr(cfg, "logdir", "runs/smdp_ppo")) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._next_checkpoint = self.checkpoint_interval
        
        # Entropy annealing
        self._entropy_coef_initial = self.hparams.entropy_coef
        
    # ---------------------------------------------------------------- training
    def train_forever(self) -> None:
        """Main training loop."""
        obs, info = self.env.reset(seed=getattr(self.cfg, "seed", None))
        obs = obs.astype(np.float32)
        
        # Decision-level tracking per environment
        decision_obs = obs.copy()
        decision_info = [dict(i) for i in info] if isinstance(info, (list, tuple)) else [info.copy()]
        decision_start_steps = np.zeros(self.env.num_envs, dtype=np.int32)
        decision_rewards = np.zeros(self.env.num_envs, dtype=np.float32)
        
        start_time = time.time()
        
        while self.global_step < self.total_steps:
            # Collect decisions until buffer is full
            decisions_collected = 0
            
            while decisions_collected < self.hparams.decisions_per_update:
                # Make decision for each environment
                for env_idx in range(self.env.num_envs):
                    if decisions_collected >= self.hparams.decisions_per_update:
                        break
                        
                    # Extract info for this env
                    env_obs = decision_obs[env_idx]
                    env_info = decision_info[env_idx] if env_idx < len(decision_info) else {}
                    
                    # Get action mask and pill colors
                    mask = self._extract_mask(env_info)
                    pill_colors = self._extract_pill_colors(env_info)
                    preview_pill_colors = self._extract_preview_pill_colors(env_info)
                    
                    # Policy forward
                    with torch.no_grad():
                        action_idx, log_prob, value = self._select_action(
                            env_obs, mask, pill_colors, preview_pill_colors, deterministic=False
                        )
                        
                    # Execute action (placement happens over τ frames)
                    start_step = self.global_step
                    obs_after, reward_accum, term, trunc, info_after = self._execute_placement(
                        env_idx, action_idx
                    )
                    tau = self.global_step - start_step
                    
                    # Store decision
                    step = DecisionStep(
                        obs=env_obs,
                        mask=mask,
                        pill_colors=pill_colors,
                        preview_pill_colors=preview_pill_colors,
                        action=action_idx,
                        log_prob=log_prob,
                        value=value,
                        tau=tau,
                        reward=reward_accum,
                        obs_next=obs_after[env_idx],
                        done=term[env_idx] or trunc[env_idx],
                        info=info_after[env_idx] if env_idx < len(info_after) else {},
                    )
                    self.buffer.add(step)
                    decisions_collected += 1
                    self.decision_step += 1
                    
                    # Update decision state
                    decision_obs[env_idx] = obs_after[env_idx]
                    if env_idx < len(info_after):
                        decision_info[env_idx] = info_after[env_idx]
                        
                    # Track episodes
                    if step.done:
                        ep_info = step.info.get("episode", {})
                        drm_info = step.info.get("drm", {})
                        
                        self.batch_returns.append(float(ep_info.get("r", 0.0)))
                        self.batch_lengths.append(int(ep_info.get("l", 0)))
                        self.batch_viruses.append(float(drm_info.get("viruses_cleared", 0.0)))
                        
                        self.event_bus.emit(
                            "episode_end",
                            step=self.global_step,
                            ret=float(ep_info.get("r", 0.0)),
                            len=int(ep_info.get("l", 0)),
                            **{f"drm/{k}": v for k, v in drm_info.items()},
                        )
                        
            # Update policy
            update_start = time.time()
            
            # Bootstrap value for last observation
            with torch.no_grad():
                bootstrap_values = []
                for env_idx in range(self.env.num_envs):
                    env_obs = decision_obs[env_idx]
                    env_info = decision_info[env_idx]
                    mask = self._extract_mask(env_info)
                    pill_colors = self._extract_pill_colors(env_info)
                    preview_pill_colors = self._extract_preview_pill_colors(env_info)
                    _, value_bootstrap = self._forward_policy(
                        env_obs, mask, pill_colors, preview_pill_colors
                    )
                    bootstrap_values.append(value_bootstrap)
                bootstrap = float(np.mean(bootstrap_values))
                
            batch = self.buffer.get_batch(bootstrap_value=bootstrap)
            metrics = self._update_policy(batch)
            
            self.buffer.clear()
            
            update_time = time.time() - update_start
            metrics["perf/update_sec"] = update_time

            frames_since_update = int(self.global_step - self._last_update_step)
            self._last_update_step = int(self.global_step)
            try:
                if hasattr(self.env, "record_update"):
                    self.env.record_update(float(update_time), frames=frames_since_update)
            except Exception:
                pass
            
            elapsed = time.time() - start_time
            metrics["perf/sps"] = float(self.global_step / max(elapsed, 1e-6))
            metrics["perf/dps"] = float(self.decision_step / max(elapsed, 1e-6))  # decisions/sec

            # Inference timing (policy forward passes outside the PPO update).
            metrics["perf/inference_calls"] = float(self._perf_inference_calls)
            metrics["perf/inference_sec_total"] = float(self._perf_inference_sec_total)
            metrics["perf/last_inference_ms"] = float(self._perf_last_inference_sec) * 1000.0
            if self._perf_inference_calls > 0:
                metrics["perf/inference_ms_avg"] = (
                    float(self._perf_inference_sec_total) * 1000.0 / float(self._perf_inference_calls)
                )
            if self.global_step > 0:
                metrics["perf/inference_ms_per_frame"] = (
                    float(self._perf_inference_sec_total) * 1000.0 / float(self.global_step)
                )
            
            self._log_metrics(metrics)
            self.event_bus.emit("update_end", step=self.global_step, **metrics)
            self._maybe_checkpoint()
            self.logger.flush()
            
    # ---------------------------------------------------------- policy methods
    def _forward_policy(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        pill_colors: np.ndarray,
        preview_pill_colors: np.ndarray,
    ) -> Tuple[torch.Tensor, float]:
        """Forward pass through policy network.
        
        Returns:
            Tuple of (logits_map [4, 16, 8], value scalar)
        """
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        colors_t = torch.from_numpy(pill_colors).unsqueeze(0).to(self.device)
        preview_t = torch.from_numpy(preview_pill_colors).unsqueeze(0).to(self.device)
        
        t0 = time.perf_counter()
        logits_map, value = self.net(obs_t, colors_t, preview_t, mask_t)
        dt = float(time.perf_counter() - t0)
        self._perf_inference_calls += 1
        self._perf_inference_sec_total += dt
        self._perf_last_inference_sec = dt
        try:
            if hasattr(self.env, "record_inference"):
                self.env.record_inference(dt)
        except Exception:
            pass
        
        return logits_map.squeeze(0), float(value.squeeze().item())
        
    def _select_action(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        pill_colors: np.ndarray,
        preview_pill_colors: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """Select action using masked policy.
        
        Returns:
            Tuple of (action_idx, log_prob, value)
        """
        logits_map, value = self._forward_policy(obs, mask, pill_colors, preview_pill_colors)
        
        # Create masked distribution
        dist = MaskedPlacementDist(logits_map, torch.from_numpy(mask).to(self.device))
        
        # Sample
        action_idx, log_prob = dist.sample(deterministic=deterministic)
        
        return int(action_idx.item()), float(log_prob.item()), value
        
    def _execute_placement(
        self,
        env_idx: int,
        action_idx: int,
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, List[Dict]]:
        """Execute placement action (may span multiple frames).
        
        The placement wrapper handles the multi-frame execution internally.
        We just step once and it returns the cumulative result.
        
        Returns:
            Tuple of (obs, total_reward, terminated, truncated, info)
        """
        # Create action array for vectorized env
        actions = np.full(self.env.num_envs, action_idx, dtype=np.int64)
        
        # Step environment (wrapper handles τ frames internally)
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        
        # Track frames consumed (get from info if available)
        tau = 1  # Default
        if env_idx < len(infos):
            # Try to extract actual tau from info
            tau = infos[env_idx].get("placements/tau", 1)
            if isinstance(tau, np.ndarray):
                tau = int(tau.item())
            else:
                tau = int(tau) if tau else 1
        
        self.global_step += tau  # Track actual frames consumed
        
        return obs, float(rewards[env_idx]), terminated, truncated, infos
        
    # ------------------------------------------------------------------ update
    def _update_policy(self, batch: DecisionBatch) -> Dict[str, float]:
        """Update policy using PPO on decision-level batch."""
        T = len(batch.actions)
        
        # Convert to tensors
        obs = torch.from_numpy(batch.observations).to(self.device)
        masks = torch.from_numpy(batch.masks).to(self.device)
        pill_colors = torch.from_numpy(batch.pill_colors).to(self.device)
        preview_pill_colors = torch.from_numpy(batch.preview_pill_colors).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.device)
        log_probs_old = torch.from_numpy(batch.log_probs).to(self.device)
        returns = torch.from_numpy(batch.returns).to(self.device)
        advantages = torch.from_numpy(batch.advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Current entropy coefficient (annealed)
        entropy_coef = self._get_entropy_coef()
        
        # Multiple epochs over the batch
        metrics_accum = {
            "loss/policy": 0.0,
            "loss/value": 0.0,
            "loss/total": 0.0,
            "policy/entropy": 0.0,
            "policy/kl": 0.0,
            "policy/clip_frac": 0.0,
        }
        
        for epoch in range(self.hparams.num_epochs):
            # Shuffle indices
            indices = torch.randperm(T, device=self.device)
            
            for start in range(0, T, self.hparams.minibatch_size):
                end = min(start + self.hparams.minibatch_size, T)
                mb_indices = indices[start:end]
                
                # Mini-batch
                mb_obs = obs[mb_indices]
                mb_masks = masks[mb_indices]
                mb_colors = pill_colors[mb_indices]
                mb_preview = preview_pill_colors[mb_indices]
                mb_actions = actions[mb_indices]
                mb_log_probs_old = log_probs_old[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                
                # Forward pass
                logits_map, values = self.net(mb_obs, mb_colors, mb_preview, mb_masks)
                
                # Compute log probs and entropy
                dist = MaskedPlacementDist(logits_map, mb_masks)
                log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                # PPO policy loss
                ratio = torch.exp(log_probs - mb_log_probs_old)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.hparams.clip_epsilon,
                    1.0 + self.hparams.clip_epsilon,
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                if self.hparams.value_loss_type == "huber":
                    value_loss = F.huber_loss(values.squeeze(-1), mb_returns)
                else:
                    value_loss = F.mse_loss(values.squeeze(-1), mb_returns)
                    
                # Total loss
                loss = (
                    policy_loss
                    + self.hparams.value_coef * value_loss
                    - entropy_coef * entropy
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.net.parameters(),
                    self.hparams.max_grad_norm,
                )
                self.optimizer.step()
                
                # Track metrics
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.hparams.clip_epsilon).float().mean()
                    # Approximate KL(old || new) under the sampled actions.
                    # Use the mini-batch's old log-probs to match `ratio` and avoid shape mismatch.
                    kl = (mb_log_probs_old - log_probs).mean()
                    
                metrics_accum["loss/policy"] += policy_loss.item()
                metrics_accum["loss/value"] += value_loss.item()
                metrics_accum["loss/total"] += loss.item()
                metrics_accum["policy/entropy"] += entropy.item()
                metrics_accum["policy/kl"] += kl.item()
                metrics_accum["policy/clip_frac"] += clip_frac.item()
                
        # Average metrics
        num_updates = self.hparams.num_epochs * max(1, T // self.hparams.minibatch_size)
        for key in metrics_accum:
            metrics_accum[key] /= num_updates
            
        metrics_accum["optim/lr"] = self.optimizer.param_groups[0]["lr"]
        metrics_accum["optim/entropy_coef"] = entropy_coef
        
        return metrics_accum
        
    # --------------------------------------------------------------- utilities
    def _extract_mask(self, info: Dict) -> np.ndarray:
        """Extract action mask from info dict."""
        for key in ("placements/feasible_mask", "placements/legal_mask", "mask"):
            mask = info.get(key)
            if mask is not None:
                if isinstance(mask, np.ndarray) and mask.shape == (4, 16, 8):
                    return mask.astype(bool)
        # Fallback: all valid
        return np.ones((4, 16, 8), dtype=bool)
        
    def _extract_pill_colors(self, info: Dict) -> np.ndarray:
        """Extract current pill colors (canonical indices 0=R,1=Y,2=B) from info dict."""
        colors = info.get("next_pill_colors")
        if colors is not None:
            arr = np.asarray(colors, dtype=np.int64)
            if arr.shape == (2,):
                return arr
        # Fallback: [0, 0]
        return np.array([0, 0], dtype=np.int64)

    def _extract_preview_pill_colors(self, info: Dict) -> np.ndarray:
        """Extract preview pill colors (canonical indices 0=R,1=Y,2=B) from info dict."""

        raw_left: Optional[int] = None
        raw_right: Optional[int] = None

        raw_ram = info.get("raw_ram")
        try:
            if isinstance(raw_ram, (bytes, bytearray, memoryview)) and len(raw_ram) > 0x031B:
                raw_left = int(raw_ram[0x031A]) & 0x03
                raw_right = int(raw_ram[0x031B]) & 0x03
        except Exception:
            raw_left = None
            raw_right = None

        if raw_left is None or raw_right is None:
            preview = info.get("preview_pill")
            if isinstance(preview, dict):
                try:
                    raw_left = int(preview.get("first_color", 0)) & 0x03
                    raw_right = int(preview.get("second_color", 0)) & 0x03
                except Exception:
                    raw_left = None
                    raw_right = None
            elif isinstance(preview, (list, tuple)) and len(preview) >= 2:
                try:
                    raw_left = int(preview[0]) & 0x03
                    raw_right = int(preview[1]) & 0x03
                except Exception:
                    raw_left = None
                    raw_right = None

        if raw_left is None or raw_right is None:
            return np.array([0, 0], dtype=np.int64)

        def _map_color(raw: int) -> int:
            return int(ram_specs.COLOR_VALUE_TO_INDEX.get(int(raw) & 0x03, 0))

        return np.array([_map_color(raw_left), _map_color(raw_right)], dtype=np.int64)
        
    def _get_entropy_coef(self) -> float:
        """Get current entropy coefficient (annealed over training)."""
        progress = min(1.0, self.global_step / self.hparams.entropy_schedule_steps)
        return (
            self._entropy_coef_initial
            + (self.hparams.entropy_schedule_end - self._entropy_coef_initial) * progress
        )
        
    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to logger."""
        step = self.global_step
        
        if self.batch_returns:
            returns = np.array(self.batch_returns, dtype=np.float32)
            self.logger.log_scalar("train/return_mean", float(returns.mean()), step)
            self.logger.log_scalar("train/return_std", float(returns.std()), step)
            
        if self.batch_viruses:
            viruses = np.array(self.batch_viruses, dtype=np.float32)
            self.logger.log_scalar("drm/viruses_per_ep", float(viruses.mean()), step)
            
        for key, value in metrics.items():
            self.logger.log_scalar(key, value, step)
            
    def _maybe_checkpoint(self) -> None:
        """Save checkpoint if interval reached."""
        if self.global_step < self._next_checkpoint:
            return
            
        payload = {
            "state_dict": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": getattr(self.cfg, "to_dict", lambda: {})(),
            "step": self.global_step,
            "decision_step": self.decision_step,
            "sha": git_commit(),
        }
        
        path = self.checkpoint_dir / f"smdp_ppo_step{self.global_step}.pt"
        torch.save(payload, path)
        
        self.event_bus.emit("checkpoint", step=self.global_step, path=str(path), walltime=time.time())
        self._next_checkpoint += self.checkpoint_interval


__all__ = ["SMDPPPOAdapter", "SMDPPPOConfig"]
