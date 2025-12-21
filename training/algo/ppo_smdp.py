"""SMDP-PPO trainer for placement policies.

Implements PPO with SMDP (Semi-Markov Decision Process) discounting where
actions span variable durations τ and credit assignment uses γ^τ.
"""
from __future__ import annotations

import time
from collections import Counter, deque
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
from models.policy.placement_dist import MaskedPlacementDist

import envs.specs.ram_to_state as ram_specs

_AUX_SPEC_NONE = "none"
_AUX_SPEC_V1 = "v1"

_AUX_V1_LEVEL_MIN = -15
_AUX_V1_LEVEL_MAX = 20
_AUX_V1_LEVEL_DIM = _AUX_V1_LEVEL_MAX - _AUX_V1_LEVEL_MIN + 1  # 36
_AUX_V1_VIRUS_NORM = 84.0  # Max viruses at level 20: (20+1)*4 = 84

# v1 feature layout (float32, [B, 57]):
#   speed_onehot[3]
#   virus_total/84
#   virus_by_color/84 [3] (R,Y,B)
#   level_onehot[36] for levels [-15..20] (out-of-range => all zeros)
#   frame_count_norm [1] (task/frames_used normalized)
#   max_height/16 [1]
#   col_heights/16 [8]
#   clearance_progress [1] (matches or viruses)
#   feasible_fraction [1] (placements/options / 512)
#   occupancy_fraction [1] (occupied tiles / 128)
#   virus_max_height/16 [1]
_AUX_V1_DIM = 3 + 1 + 3 + _AUX_V1_LEVEL_DIM + 1 + 1 + 8 + 1 + 1 + 1 + 1  # 57

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

    # Optional auxiliary vector inputs (derived from obs + info).
    aux_spec: str = "none"  # none|v1
    
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
            aux_spec=str(ppo_cfg_dict.get("aux_spec", "none")),
            entropy_schedule_end=float(ppo_cfg_dict.get("entropy_schedule_end", 0.003)),
            entropy_schedule_steps=int(ppo_cfg_dict.get("entropy_schedule_steps", 1000000)),
            use_gumbel_topk=bool(ppo_cfg_dict.get("use_gumbel_topk", False)),
            gumbel_k=int(ppo_cfg_dict.get("gumbel_k", 2)),
            value_loss_type=str(ppo_cfg_dict.get("value_loss_type", "mse")),
        )

        aux_spec_norm = str(self.hparams.aux_spec or "none").strip().lower()
        if aux_spec_norm not in {_AUX_SPEC_NONE, _AUX_SPEC_V1}:
            raise ValueError(f"Unknown smdp_ppo.aux_spec: {self.hparams.aux_spec!r}")
        self.aux_spec = aux_spec_norm
        self.aux_dim = int(_AUX_V1_DIM) if self.aux_spec == _AUX_SPEC_V1 else 0
        
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
            aux_dim=self.aux_dim,
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        
        # Rollout buffer (decision-wise)
        self.buffer = DecisionRolloutBuffer(
            capacity=self.hparams.decisions_per_update * 2,
            obs_shape=obs_shape,
            num_envs=env.num_envs,
            gamma=self.hparams.gamma,
            gae_lambda=self.hparams.gae_lambda,
            aux_dim=self.aux_dim,
        )
        
        # Tracking
        self.global_step = 0  # Total environment steps (frames)
        self.decision_step = 0  # Total decisions made
        self.total_steps = int(getattr(cfg.train, "total_steps", 5000000))
        self.checkpoint_interval = int(getattr(cfg.train, "checkpoint_interval", 100000))
        self._episodes_total = 0
        self._curriculum_last_level: Optional[int] = None
        self._curriculum_last_frames: int = 0
        self._curriculum_last_episodes: int = 0
        
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
        obs_arr = self._ensure_batched_obs(self._unwrap_obs(obs)).astype(np.float32)

        # Decision-level tracking per environment
        decision_obs = obs_arr.copy()
        decision_info = self._normalize_infos(info)

        start_time = time.time()
        
        while self.global_step < self.total_steps:
            # Collect decisions until buffer is full
            decisions_collected = 0
            
            while decisions_collected < self.hparams.decisions_per_update:
                # Make a vectorized decision for all environments
                (
                    actions,
                    log_probs,
                    values,
                    masks,
                    pill_colors,
                    preview_pill_colors,
                    aux_batch,
                ) = self._select_actions_batch(decision_obs, decision_info, deterministic=False)

                # Step environment once for the full batch.
                obs_after, rewards, terminated, truncated, info_after = self.env.step(actions)

                obs_after_arr = self._ensure_batched_obs(self._unwrap_obs(obs_after)).astype(
                    np.float32
                )
                info_after_list = self._normalize_infos(info_after)
                rewards_arr = np.asarray(rewards, dtype=np.float32).reshape(self.env.num_envs)
                terminated_arr = np.asarray(terminated, dtype=bool).reshape(self.env.num_envs)
                truncated_arr = np.asarray(truncated, dtype=bool).reshape(self.env.num_envs)

                tau_arr = np.array(
                    [self._extract_tau(info_after_list[i]) for i in range(self.env.num_envs)],
                    dtype=np.int32,
                )

                frames_total = int(np.sum(tau_arr))
                self.global_step += frames_total
                self.decision_step += int(self.env.num_envs)
                decisions_collected += int(self.env.num_envs)

                advance_from: Optional[int] = None
                advance_to: Optional[int] = None
                for env_idx in range(self.env.num_envs):
                    info_i = info_after_list[env_idx] if env_idx < len(info_after_list) else {}
                    step = DecisionStep(
                        obs=decision_obs[env_idx],
                        mask=masks[env_idx],
                        pill_colors=pill_colors[env_idx],
                        preview_pill_colors=preview_pill_colors[env_idx],
                        aux=None if aux_batch is None else aux_batch[env_idx],
                        action=int(actions[env_idx]),
                        log_prob=float(log_probs[env_idx]),
                        value=float(values[env_idx]),
                        tau=int(tau_arr[env_idx]),
                        reward=float(rewards_arr[env_idx]),
                        obs_next=obs_after_arr[env_idx],
                        done=bool(terminated_arr[env_idx] or truncated_arr[env_idx]),
                        env_id=int(env_idx),
                        info=dict(info_i),
                    )
                    self.buffer.add(step)

                    # Track episodes
                    if step.done:
                        self._episodes_total += 1
                        ep_info = step.info.get("episode", {})
                        drm_info = step.info.get("drm", {})

                        self.batch_returns.append(float(ep_info.get("r", 0.0)))
                        self.batch_lengths.append(int(ep_info.get("l", 0)))
                        self.batch_viruses.append(float(drm_info.get("viruses_cleared", 0.0)))
                        self.batch_decisions.append(int(ep_info.get("decisions", 0)))

                        payload = {
                            "step": self.global_step,
                            "ret": float(ep_info.get("r", 0.0)),
                            "len": int(ep_info.get("l", 0)),
                            "env_index": int(env_idx),
                        }
                        if "decisions" in ep_info:
                            payload["decisions"] = int(ep_info.get("decisions", 0))
                        payload.update({f"drm/{k}": v for k, v in drm_info.items()})
                        self.event_bus.emit("episode_end", **payload)

                    if advance_to is None:
                        adv_to = self._extract_int(info_i.get("curriculum/advanced_to"))
                        if adv_to is not None:
                            advance_to = int(adv_to)
                            advance_from = self._extract_int(
                                info_i.get("curriculum/advanced_from")
                            )

                if advance_to is not None:
                    self._log_curriculum_advance(advance_from, advance_to)

                # Update decision state for next batch.
                decision_obs = obs_after_arr.copy()
                decision_info = info_after_list

                # Keep PPO rollouts stage-pure: once the curriculum advances,
                # stop collecting and update immediately so we don't mix levels
                # within a single PPO update batch.
                if advance_to is not None:
                    break
                        
            # Update policy
            update_start = time.time()
            
            # Bootstrap values for the last observation per environment.
            with torch.no_grad():
                (
                    _actions,
                    _log_probs,
                    bootstrap_values,
                    _masks,
                    _pill_colors,
                    _preview_pill_colors,
                    _aux_batch,
                ) = self._select_actions_batch(decision_obs, decision_info, deterministic=True)

            batch = self.buffer.get_batch(
                bootstrap_value=np.asarray(bootstrap_values, dtype=np.float32)
            )
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
            metrics["perf/sps_frames_total"] = float(self.global_step / max(elapsed, 1e-6))
            metrics["perf/dps_decisions_total"] = float(self.decision_step / max(elapsed, 1e-6))
            # Backwards-compatible aliases (used by existing TUI/event handlers).
            metrics["perf/sps"] = float(metrics["perf/sps_frames_total"])
            metrics["perf/dps"] = float(metrics["perf/dps_decisions_total"])

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

            curriculum_snapshot = self._extract_curriculum_snapshot(decision_info)
            if curriculum_snapshot is not None:
                metrics.update(self._curriculum_scalar_metrics(curriculum_snapshot))
            
            self._log_metrics(metrics)
            if curriculum_snapshot is not None:
                self.event_bus.emit(
                    "update_end", step=self.global_step, curriculum=curriculum_snapshot, **metrics
                )
            else:
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
        aux: Optional[np.ndarray],
    ) -> Tuple[torch.Tensor, float]:
        """Forward pass through policy network.
        
        Returns:
            Tuple of (logits_map [4, 16, 8], value scalar)
        """
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        colors_t = torch.from_numpy(pill_colors).unsqueeze(0).to(self.device)
        preview_t = torch.from_numpy(preview_pill_colors).unsqueeze(0).to(self.device)
        aux_t = None if aux is None else torch.from_numpy(aux).unsqueeze(0).to(self.device)
        
        t0 = time.perf_counter()
        logits_map, value = self.net(obs_t, colors_t, preview_t, mask_t, aux=aux_t)
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

    def _forward_policy_batch(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        pill_colors: np.ndarray,
        preview_pill_colors: np.ndarray,
        aux: Optional[np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through policy network for a batch.

        Returns:
            Tuple of (logits_map [B, 4, 16, 8], values [B])
        """
        obs_t = torch.from_numpy(obs).to(self.device)
        mask_t = torch.from_numpy(mask).to(self.device)
        colors_t = torch.from_numpy(pill_colors).to(self.device)
        preview_t = torch.from_numpy(preview_pill_colors).to(self.device)
        aux_t = None if aux is None else torch.from_numpy(aux).to(self.device)

        t0 = time.perf_counter()
        logits_map, value = self.net(obs_t, colors_t, preview_t, mask_t, aux=aux_t)
        dt = float(time.perf_counter() - t0)
        batch_size = int(obs_t.shape[0])
        self._perf_inference_calls += batch_size
        self._perf_inference_sec_total += dt
        self._perf_last_inference_sec = dt
        try:
            if hasattr(self.env, "record_inference"):
                self.env.record_inference(dt, calls=batch_size)
        except Exception:
            pass

        return logits_map, value.squeeze(-1)
        
    def _select_action(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        pill_colors: np.ndarray,
        preview_pill_colors: np.ndarray,
        aux: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """Select action using masked policy.
        
        Returns:
            Tuple of (action_idx, log_prob, value)
        """
        logits_map, value = self._forward_policy(obs, mask, pill_colors, preview_pill_colors, aux)
        
        # Create masked distribution
        dist = MaskedPlacementDist(logits_map, torch.from_numpy(mask).to(self.device))
        
        # Sample
        action_idx, log_prob = dist.sample(deterministic=deterministic)
        
        return int(action_idx.item()), float(log_prob.item()), value

    def _select_actions_batch(
        self,
        obs_batch: np.ndarray,
        infos: List[Dict[str, Any]],
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Select actions for a batch of environments."""
        num_envs = int(self.env.num_envs)
        obs_arr = self._ensure_batched_obs(obs_batch).astype(np.float32, copy=False)

        masks = np.zeros((num_envs, 4, 16, 8), dtype=bool)
        pill_colors = np.zeros((num_envs, 2), dtype=np.int64)
        preview_pill_colors = np.zeros((num_envs, 2), dtype=np.int64)
        aux_batch: Optional[np.ndarray] = None
        if self.aux_dim > 0:
            aux_batch = np.zeros((num_envs, self.aux_dim), dtype=np.float32)
        for i in range(num_envs):
            info_i = infos[i] if i < len(infos) else {}
            masks[i] = self._extract_mask(info_i)
            pill_colors[i] = self._extract_pill_colors(info_i)
            preview_pill_colors[i] = self._extract_preview_pill_colors(info_i)
            if aux_batch is not None:
                aux_batch[i] = self._build_aux(obs_arr[i], info_i)

        logits_map, values = self._forward_policy_batch(
            obs_arr, masks, pill_colors, preview_pill_colors, aux_batch
        )
        dist = MaskedPlacementDist(logits_map, torch.from_numpy(masks).to(self.device))
        if deterministic:
            action_idx = dist.mode()
            log_probs = dist.log_prob(action_idx)
        else:
            action_idx, log_probs = dist.sample(deterministic=False)

        actions_np = action_idx.detach().cpu().numpy().astype(np.int64)
        log_probs_np = log_probs.detach().cpu().numpy().astype(np.float32)
        values_np = values.detach().cpu().numpy().astype(np.float32)

        return actions_np, log_probs_np, values_np, masks, pill_colors, preview_pill_colors, aux_batch
        
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
        aux = None if batch.aux is None else torch.from_numpy(batch.aux).to(self.device)
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
                mb_aux = None if aux is None else aux[mb_indices]
                mb_actions = actions[mb_indices]
                mb_log_probs_old = log_probs_old[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                
                # Forward pass
                logits_map, values = self.net(mb_obs, mb_colors, mb_preview, mb_masks, aux=mb_aux)
                
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

    def _unwrap_obs(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict) and "obs" in obs:
            obs = obs.get("obs")
        return np.asarray(obs)

    def _ensure_batched_obs(self, obs: np.ndarray) -> np.ndarray:
        obs_arr = np.asarray(obs)
        if obs_arr.ndim == len(self.buffer.obs_shape):
            return obs_arr[None, ...]
        return obs_arr

    def _normalize_infos(self, infos: Any) -> List[Dict[str, Any]]:
        num_envs = int(self.env.num_envs)
        if infos is None:
            return [{} for _ in range(num_envs)]
        if isinstance(infos, (list, tuple)):
            out = [dict(i) if isinstance(i, dict) else {} for i in infos]
        elif isinstance(infos, dict):
            out = [dict(infos) for _ in range(num_envs)]
        else:
            out = [{} for _ in range(num_envs)]
        if len(out) < num_envs:
            out.extend({} for _ in range(num_envs - len(out)))
        return out[:num_envs]

    @staticmethod
    def _extract_tau(info: Dict[str, Any]) -> int:
        tau = info.get("placements/tau", 1)
        if isinstance(tau, np.ndarray):
            try:
                tau = tau.item()
            except Exception:
                return 1
        try:
            return max(1, int(tau))
        except Exception:
            return 1
        
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

    @staticmethod
    def _extract_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            try:
                value = value.item()
            except Exception:
                return None
        try:
            return int(value)
        except Exception:
            return None

    def _build_aux(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        if self.aux_spec == _AUX_SPEC_V1:
            return self._build_aux_v1(obs, info)
        raise ValueError(f"aux_spec={self.aux_spec!r} does not define auxiliary inputs")

    @staticmethod
    def _column_heights(mask: np.ndarray) -> np.ndarray:
        occ = np.asarray(mask, dtype=bool)
        if occ.shape != (16, 8):
            raise ValueError(f"Expected mask shape (16,8), got {occ.shape!r}")
        heights = np.zeros((8,), dtype=np.int32)
        for c in range(8):
            rows = np.nonzero(occ[:, c])[0]
            if rows.size == 0:
                heights[c] = 0
            else:
                heights[c] = int(16 - int(rows.min()))
        return heights

    def _build_aux_v1(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        if self.aux_dim != _AUX_V1_DIM:
            raise ValueError(f"aux_dim mismatch: expected {_AUX_V1_DIM}, got {self.aux_dim}")
        frame = np.asarray(obs, dtype=np.float32)
        if frame.ndim == 4 and frame.shape[-2:] == (16, 8):
            # Allow passing a fixed frame stack (T,C,16,8); use the latest frame.
            frame = frame[-1]
        if frame.ndim != 3 or frame.shape[1:] != (16, 8):
            raise ValueError(f"Expected obs shape (C,16,8), got {frame.shape!r}")

        out = np.zeros((_AUX_V1_DIM,), dtype=np.float32)
        k = 0

        # speed_onehot[3]
        speed = self._extract_int(info.get("pill/speed_setting"))
        if speed is None:
            speed = self._extract_int(info.get("speed_setting"))
        if speed is None:
            speed = 2
        speed = int(max(0, min(int(speed), 2)))
        out[k + speed] = 1.0
        k += 3

        # Virus counts from the bottle planes.
        virus_mask = ram_specs.get_virus_mask(frame)
        virus_total = int(virus_mask.sum())
        out[k] = float(np.clip(float(virus_total) / float(_AUX_V1_VIRUS_NORM), 0.0, 1.0))
        k += 1

        virus_planes = ram_specs.get_virus_color_planes(frame)
        if virus_planes.shape[0] != 3:
            raise ValueError(f"Expected 3 virus color planes, got {virus_planes.shape!r}")
        for c in range(3):
            out[k + c] = float(
                np.clip(float((virus_planes[c] > 0.5).sum()) / float(_AUX_V1_VIRUS_NORM), 0.0, 1.0)
            )
        k += 3

        # level_onehot[36] for [-15..20]
        lvl = self._extract_int(info.get("curriculum/env_level"))
        if lvl is None:
            lvl = self._extract_int(info.get("curriculum_level"))
        if lvl is None:
            lvl = self._extract_int(info.get("level"))
        if lvl is None:
            lvl = 0
        lvl_i = int(lvl)
        if _AUX_V1_LEVEL_MIN <= lvl_i <= _AUX_V1_LEVEL_MAX:
            out[k + (lvl_i - _AUX_V1_LEVEL_MIN)] = 1.0
        k += _AUX_V1_LEVEL_DIM

        # frame_count_norm (task timer; normalized to [0,1])
        frames_used = self._extract_int(info.get("task/frames_used"))
        if frames_used is None:
            frames_used = 0
        max_frames = self._extract_int(info.get("task/max_frames"))
        if max_frames is not None and int(max_frames) > 0:
            out[k] = float(np.clip(float(frames_used) / float(max_frames), 0.0, 1.0))
        else:
            out[k] = float(np.tanh(float(frames_used) / 8000.0))
        k += 1

        # heights from occupancy mask (bottle-only for bitplane_bottle*).
        occ = ram_specs.get_occupancy_mask(frame)
        heights = self._column_heights(occ)
        max_h = int(heights.max())
        out[k] = float(np.clip(float(max_h) / 16.0, 0.0, 1.0))
        k += 1

        out[k : k + 8] = np.clip(heights.astype(np.float32) / 16.0, 0.0, 1.0)
        k += 8

        # clearance_progress (matches or viruses)
        task_mode = str(info.get("task_mode") or "viruses").strip().lower()
        progress = 0.0
        if task_mode in {"matches", "any_clear"}:
            mc = self._extract_int(info.get("matches_completed")) or 0
            target = self._extract_int(info.get("match_target")) or 0
            if target > 0:
                progress = float(mc) / float(max(1, target))
        else:
            v0 = self._extract_int(info.get("drm/viruses_initial"))
            if v0 is None:
                v0 = self._extract_int(info.get("viruses_initial"))
            v_now = self._extract_int(info.get("viruses_remaining"))
            if v_now is None:
                v_now = virus_total
            if v0 is not None and int(v0) > 0:
                progress = float(int(v0) - int(v_now)) / float(int(v0))
        out[k] = float(np.clip(progress, 0.0, 1.0))
        k += 1

        # feasible_fraction (placements/options / 512)
        options = self._extract_int(info.get("placements/options"))
        if options is None:
            options = 0
        out[k] = float(np.clip(float(options) / 512.0, 0.0, 1.0))
        k += 1

        # occupancy_fraction (occupied / 128)
        out[k] = float(np.clip(float(occ.sum()) / 128.0, 0.0, 1.0))
        k += 1

        # virus_max_height/16
        virus_heights = self._column_heights(virus_mask)
        virus_max_h = int(virus_heights.max())
        out[k] = float(np.clip(float(virus_max_h) / 16.0, 0.0, 1.0))
        k += 1

        if k != _AUX_V1_DIM:
            raise RuntimeError(f"aux_v1 packing mismatch: k={k} dim={_AUX_V1_DIM}")
        return out

    def _extract_curriculum_snapshot(self, infos: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract a compact curriculum snapshot from per-env info dicts."""

        if not infos:
            return None

        source: Optional[Dict[str, Any]] = None
        for info in infos:
            if isinstance(info, dict) and "curriculum/current_level" in info:
                source = info
                break
        if source is None:
            return None

        current_level = self._extract_int(source.get("curriculum/current_level"))
        if current_level is None:
            return None

        def _float(key: str, default: float = 0.0) -> float:
            val = source.get(key, default)
            if isinstance(val, np.ndarray):
                try:
                    val = val.item()
                except Exception:
                    return float(default)
            try:
                return float(val)
            except Exception:
                return float(default)

        def _int(key: str, default: int = 0) -> int:
            val = self._extract_int(source.get(key))
            return int(default) if val is None else int(val)

        snapshot: Dict[str, Any] = {
            "current_level": int(current_level),
            "rate_current": _float("curriculum/rate_current", 0.0),
            "window_n": _int("curriculum/window_n", 0),
            "window_size": _int("curriculum/window_size", 0),
            "episodes_current_total": _int("curriculum/episodes_current_total", 0),
            "start_level": _int("curriculum/start_level", 0),
            "max_level": _int("curriculum/max_level", 0),
            "success_threshold": _float("curriculum/success_threshold", 0.0),
            "min_episodes": _int("curriculum/min_episodes", 0),
            "rehearsal_prob": _float("curriculum/rehearsal_prob", 0.0),
        }

        confidence_sigmas = source.get("curriculum/confidence_sigmas")
        if confidence_sigmas is not None:
            try:
                snapshot["confidence_sigmas"] = float(confidence_sigmas)
            except Exception:
                pass
        confidence_lb = source.get("curriculum/confidence_lower_bound")
        if confidence_lb is not None:
            try:
                snapshot["confidence_lower_bound"] = float(confidence_lb)
            except Exception:
                pass
        window_successes = self._extract_int(source.get("curriculum/window_successes"))
        if window_successes is not None:
            snapshot["window_successes"] = int(window_successes)

        time_budget_frames = self._extract_int(source.get("curriculum/time_budget_frames"))
        if time_budget_frames is not None:
            snapshot["time_budget_frames"] = int(time_budget_frames)
        time_budget_spawns = self._extract_int(source.get("curriculum/time_budget_spawns"))
        if time_budget_spawns is not None:
            snapshot["time_budget_spawns"] = int(time_budget_spawns)
        time_mean_frames = source.get("curriculum/time_mean_frames")
        if time_mean_frames is not None:
            try:
                snapshot["time_mean_frames"] = float(time_mean_frames)
            except Exception:
                pass
        time_mad_frames = source.get("curriculum/time_mad_frames")
        if time_mad_frames is not None:
            try:
                snapshot["time_mad_frames"] = float(time_mad_frames)
            except Exception:
                pass
        time_mean_spawns = source.get("curriculum/time_mean_spawns")
        if time_mean_spawns is not None:
            try:
                snapshot["time_mean_spawns"] = float(time_mean_spawns)
            except Exception:
                pass
        time_mad_spawns = source.get("curriculum/time_mad_spawns")
        if time_mad_spawns is not None:
            try:
                snapshot["time_mad_spawns"] = float(time_mad_spawns)
            except Exception:
                pass
        time_k = self._extract_int(source.get("curriculum/time_k"))
        if time_k is not None:
            snapshot["time_k"] = int(time_k)
        time_target = source.get("curriculum/time_target")
        if time_target is not None:
            try:
                snapshot["time_target"] = float(time_target)
            except Exception:
                pass

        mode = source.get("curriculum/mode")
        if isinstance(mode, str) and mode:
            snapshot["mode"] = str(mode)

        stage_index = self._extract_int(source.get("curriculum/stage_index"))
        if stage_index is not None:
            snapshot["stage_index"] = int(stage_index)
        stage_count = self._extract_int(source.get("curriculum/stage_count"))
        if stage_count is not None:
            snapshot["stage_count"] = int(stage_count)

        probe_threshold = _float("curriculum/probe_threshold", 0.0)
        if probe_threshold > 0.0:
            snapshot["probe_threshold"] = float(probe_threshold)

        decisions_current_total = self._extract_int(source.get("curriculum/decisions_current_total"))
        if decisions_current_total is not None:
            snapshot["decisions_current_total"] = int(decisions_current_total)
        min_stage_decisions = self._extract_int(source.get("curriculum/min_stage_decisions"))
        if min_stage_decisions is not None:
            snapshot["min_stage_decisions"] = int(min_stage_decisions)

        # Distribution of active env levels.
        env_levels: List[int] = []
        for info in infos:
            if not isinstance(info, dict):
                continue
            lvl = self._extract_int(info.get("curriculum/env_level"))
            if lvl is not None:
                env_levels.append(int(lvl))
        if env_levels:
            counts = Counter(env_levels)
            snapshot["env_level_counts"] = {str(k): int(v) for k, v in sorted(counts.items())}

        # Advancement (present only on steps that trigger it).
        adv_from = None
        adv_to = None
        for info in infos:
            if not isinstance(info, dict):
                continue
            adv_to = self._extract_int(info.get("curriculum/advanced_to"))
            if adv_to is None:
                continue
            adv_from = self._extract_int(info.get("curriculum/advanced_from"))
            break
        if adv_to is not None:
            snapshot["advanced_to"] = int(adv_to)
            if adv_from is not None:
                snapshot["advanced_from"] = int(adv_from)

        return snapshot

    @staticmethod
    def _curriculum_scalar_metrics(snapshot: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            out["curriculum/current_level"] = float(int(snapshot.get("current_level", 0)))
            out["curriculum/rate_current"] = float(snapshot.get("rate_current", 0.0) or 0.0)
            out["curriculum/window_n"] = float(int(snapshot.get("window_n", 0)))
            out["curriculum/window_size"] = float(int(snapshot.get("window_size", 0)))
            out["curriculum/episodes_current_total"] = float(int(snapshot.get("episodes_current_total", 0)))
            decisions_total = snapshot.get("decisions_current_total")
            if decisions_total is not None:
                out["curriculum/decisions_current_total"] = float(int(decisions_total))
            min_stage_decisions = snapshot.get("min_stage_decisions")
            if min_stage_decisions is not None:
                out["curriculum/min_stage_decisions"] = float(int(min_stage_decisions))
            out["curriculum/start_level"] = float(int(snapshot.get("start_level", 0)))
            out["curriculum/max_level"] = float(int(snapshot.get("max_level", 0)))
            out["curriculum/success_threshold"] = float(snapshot.get("success_threshold", 0.0) or 0.0)
            out["curriculum/min_episodes"] = float(int(snapshot.get("min_episodes", 0)))
            out["curriculum/rehearsal_prob"] = float(snapshot.get("rehearsal_prob", 0.0) or 0.0)
            stage_index = snapshot.get("stage_index")
            if stage_index is not None:
                out["curriculum/stage_index"] = float(int(stage_index))
            conf_sigmas = snapshot.get("confidence_sigmas")
            if conf_sigmas is not None:
                out["curriculum/confidence_sigmas"] = float(conf_sigmas)
            conf_lb = snapshot.get("confidence_lower_bound")
            if conf_lb is not None:
                out["curriculum/confidence_lower_bound"] = float(conf_lb)
            window_successes = snapshot.get("window_successes")
            if window_successes is not None:
                out["curriculum/window_successes"] = float(int(window_successes))
            time_budget_frames = snapshot.get("time_budget_frames")
            if time_budget_frames is not None:
                out["curriculum/time_budget_frames"] = float(int(time_budget_frames))
            time_budget_spawns = snapshot.get("time_budget_spawns")
            if time_budget_spawns is not None:
                out["curriculum/time_budget_spawns"] = float(int(time_budget_spawns))
            time_mean = snapshot.get("time_mean_frames")
            if time_mean is not None:
                out["curriculum/time_mean_frames"] = float(time_mean)
            time_mad = snapshot.get("time_mad_frames")
            if time_mad is not None:
                out["curriculum/time_mad_frames"] = float(time_mad)
            time_mean_spawns = snapshot.get("time_mean_spawns")
            if time_mean_spawns is not None:
                out["curriculum/time_mean_spawns"] = float(time_mean_spawns)
            time_mad_spawns = snapshot.get("time_mad_spawns")
            if time_mad_spawns is not None:
                out["curriculum/time_mad_spawns"] = float(time_mad_spawns)
            time_k = snapshot.get("time_k")
            if time_k is not None:
                out["curriculum/time_k"] = float(int(time_k))
            time_target = snapshot.get("time_target")
            if time_target is not None:
                out["curriculum/time_target"] = float(time_target)
        except Exception:
            return out

        counts = snapshot.get("env_level_counts")
        if isinstance(counts, dict) and counts:
            try:
                levels = [int(k) for k in counts.keys()]
                out["curriculum/env_level_min"] = float(min(levels))
                out["curriculum/env_level_max"] = float(max(levels))
                out["curriculum/env_levels_unique"] = float(len(levels))
            except Exception:
                pass
        return out

    def _log_curriculum_advance(self, level_from: Optional[int], level_to: int) -> None:
        if (
            self._curriculum_last_level is not None
            and int(level_to) <= int(self._curriculum_last_level)
        ):
            return
        frames_total = int(self.global_step)
        episodes_total = int(self._episodes_total)
        frames_delta = frames_total - int(self._curriculum_last_frames)
        episodes_delta = episodes_total - int(self._curriculum_last_episodes)

        step = int(self.global_step)
        if level_from is not None:
            self.logger.log_scalar("curriculum/advanced_from", float(level_from), step)
        self.logger.log_scalar("curriculum/advanced_to", float(level_to), step)
        self.logger.log_scalar(
            "curriculum/advanced_frames_total", float(frames_total), step
        )
        self.logger.log_scalar(
            "curriculum/advanced_episodes_total", float(episodes_total), step
        )
        self.logger.log_scalar(
            "curriculum/advanced_frames_delta", float(frames_delta), step
        )
        self.logger.log_scalar(
            "curriculum/advanced_episodes_delta", float(episodes_delta), step
        )

        self._curriculum_last_level = int(level_to)
        self._curriculum_last_frames = frames_total
        self._curriculum_last_episodes = episodes_total
        
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
