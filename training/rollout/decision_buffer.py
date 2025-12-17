"""Decision-wise rollout buffer for SMDP-PPO.

Stores transitions at decision granularity (per spawn) rather than per frame.
Each decision spans τ frames and accumulates reward R until the next decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(slots=True)
class DecisionStep:
    """Single decision-level transition."""
    
    obs: np.ndarray  # Observation at decision time
    mask: np.ndarray  # Action mask [4, 16, 8]
    pill_colors: np.ndarray  # Next pill colors [2]
    action: int  # Selected action index
    log_prob: float  # Log probability of action
    value: float  # Value estimate V(s)
    tau: int  # Number of frames consumed
    reward: float  # Cumulative reward over τ frames
    obs_next: np.ndarray  # Observation after placement
    done: bool  # Episode terminated
    info: Dict = field(default_factory=dict)  # Additional metadata


@dataclass(slots=True)
class DecisionBatch:
    """Batch of decision-level transitions."""
    
    observations: np.ndarray  # [T, ...]
    masks: np.ndarray  # [T, 4, 16, 8]
    pill_colors: np.ndarray  # [T, 2]
    actions: np.ndarray  # [T]
    log_probs: np.ndarray  # [T]
    values: np.ndarray  # [T]
    taus: np.ndarray  # [T] - frame durations
    rewards: np.ndarray  # [T] - cumulative rewards
    observations_next: np.ndarray  # [T, ...]
    dones: np.ndarray  # [T]
    advantages: Optional[np.ndarray] = None  # [T] - computed later
    returns: Optional[np.ndarray] = None  # [T] - computed later
    gammas: Optional[np.ndarray] = None  # [T] - γ^τ for each step


class DecisionRolloutBuffer:
    """Ring buffer for decision-level rollouts.
    
    Stores decisions across multiple parallel environments and supports
    efficient batching for SMDP-PPO updates.
    """
    
    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        num_envs: int = 1,
        gamma: float = 0.997,
        gae_lambda: float = 0.95,
    ):
        """Initialize decision rollout buffer.
        
        Args:
            capacity: Maximum number of decisions to store
            obs_shape: Shape of observations (e.g., [C, 16, 8])
            num_envs: Number of parallel environments
            gamma: Discount factor for returns
            gae_lambda: GAE lambda parameter
        """
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.num_envs = num_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Storage
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.masks = np.zeros((capacity, 4, 16, 8), dtype=np.bool_)
        self.pill_colors = np.zeros((capacity, 2), dtype=np.int64)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.taus = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.observations_next = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        self.ptr = 0
        self.size = 0
        
    def add(self, step: DecisionStep) -> None:
        """Add a decision step to the buffer."""
        idx = self.ptr
        
        self.observations[idx] = step.obs
        self.masks[idx] = step.mask
        self.pill_colors[idx] = step.pill_colors
        self.actions[idx] = step.action
        self.log_probs[idx] = step.log_prob
        self.values[idx] = step.value
        self.taus[idx] = step.tau
        self.rewards[idx] = step.reward
        self.observations_next[idx] = step.obs_next
        self.dones[idx] = step.done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def add_batch(self, steps: List[DecisionStep]) -> None:
        """Add multiple decision steps."""
        for step in steps:
            self.add(step)
            
    def compute_advantages(
        self,
        bootstrap_value: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute advantages, returns, and per-step gammas using GAE-SMDP.
        
        Args:
            bootstrap_value: Value estimate for s_{T+1}, shape [1] or None
            
        Returns:
            Tuple of (advantages, returns, gammas) each with shape [T]
        """
        T = self.size
        
        # Extract arrays
        rewards = self.rewards[:T]
        values = self.values[:T]
        dones = self.dones[:T]
        taus = self.taus[:T]
        
        # Compute Gamma_t = gamma^tau_t
        gammas = self.gamma ** taus.astype(np.float32)
        
        # Initialize
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        
        # Bootstrap
        if bootstrap_value is None:
            next_value = 0.0
            next_return = 0.0
        else:
            next_value = float(bootstrap_value)
            next_return = float(bootstrap_value)
            
        last_gae = 0.0
        
        # Backward pass for GAE
        for t in reversed(range(T)):
            if t < T - 1:
                next_value = values[t + 1]
                next_return = returns[t + 1]
            elif bootstrap_value is not None:
                next_value = float(bootstrap_value)
                next_return = float(bootstrap_value)
            else:
                next_value = 0.0
                next_return = 0.0
                
            # Mask for terminal states
            mask = 1.0 - float(dones[t])
            
            # TD error with SMDP discount
            delta = rewards[t] + gammas[t] * next_value * mask - values[t]
            
            # GAE accumulation
            last_gae = delta + gammas[t] * self.gae_lambda * mask * last_gae
            advantages[t] = last_gae
            
            # Return
            returns[t] = rewards[t] + gammas[t] * next_return * mask
            next_return = returns[t]
            
        return advantages, returns, gammas
        
    def get_batch(self, bootstrap_value: Optional[float] = None) -> DecisionBatch:
        """Get all stored decisions as a batch with computed advantages.
        
        Args:
            bootstrap_value: Value for bootstrapping (e.g., V(s_T))
            
        Returns:
            DecisionBatch with advantages and returns computed
        """
        T = self.size
        
        # Compute advantages
        bootstrap = np.array([bootstrap_value]) if bootstrap_value is not None else None
        advantages, returns, gammas = self.compute_advantages(bootstrap)
        
        return DecisionBatch(
            observations=self.observations[:T].copy(),
            masks=self.masks[:T].copy(),
            pill_colors=self.pill_colors[:T].copy(),
            actions=self.actions[:T].copy(),
            log_probs=self.log_probs[:T].copy(),
            values=self.values[:T].copy(),
            taus=self.taus[:T].copy(),
            rewards=self.rewards[:T].copy(),
            observations_next=self.observations_next[:T].copy(),
            dones=self.dones[:T].copy(),
            advantages=advantages,
            returns=returns,
            gammas=gammas,
        )
        
    def clear(self) -> None:
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0
        
    def __len__(self) -> int:
        return self.size


def compute_gae_smdp(
    values: np.ndarray,
    rewards: np.ndarray,
    gammas: np.ndarray,
    dones: Optional[np.ndarray] = None,
    lam: float = 0.95,
    bootstrap: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Standalone GAE computation with SMDP discounting.
    
    Args:
        values: Value estimates [T]
        rewards: Rewards [T]
        gammas: Per-step discount factors gamma^tau [T]
        dones: Terminal flags [T], optional
        lam: GAE lambda
        bootstrap: Bootstrap value for V(s_T)
        
    Returns:
        Tuple of (advantages [T], returns [T])
    """
    T = len(values)
    
    if dones is None:
        dones = np.zeros(T, dtype=np.float32)
    else:
        dones = dones.astype(np.float32)
        
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)
    
    next_value = bootstrap
    next_return = bootstrap
    last_gae = 0.0
    
    for t in reversed(range(T)):
        if t < T - 1:
            next_value = values[t + 1]
            next_return = returns[t + 1]
        else:
            next_value = bootstrap
            next_return = bootstrap
            
        mask = 1.0 - dones[t]
        
        # SMDP TD error
        delta = rewards[t] + gammas[t] * next_value * mask - values[t]
        
        # GAE
        last_gae = delta + gammas[t] * lam * mask * last_gae
        advantages[t] = last_gae
        
        # Return
        returns[t] = rewards[t] + gammas[t] * next_return * mask
        next_return = returns[t]
        
    return advantages, returns


__all__ = [
    "DecisionStep",
    "DecisionBatch",
    "DecisionRolloutBuffer",
    "compute_gae_smdp",
]
