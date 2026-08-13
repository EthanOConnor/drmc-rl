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
    pill_colors: np.ndarray  # Current pill colors [2]
    preview_pill_colors: np.ndarray  # Next (preview) pill colors [2]
    action: int  # Selected action index
    log_prob: float  # Log probability of action
    value: float  # Value estimate V(s)
    tau: int  # Number of frames consumed
    reward: float  # Cumulative reward over τ frames
    obs_next: np.ndarray  # Observation after placement
    done: bool  # Episode terminated
    cost_to_lock: Optional[np.ndarray] = None  # [4, 16, 8] float32; frames-to-lock (planner output)
    aux: Optional[np.ndarray] = None  # Optional auxiliary vector [aux_dim]
    env_id: int = 0  # Environment index for multi-env rollouts
    info: Dict = field(default_factory=dict)  # Additional metadata
    # Search-distillation targets (drmc_rl/training/algo/search_distill.py); only
    # stored when the buffer is built with search_target_dim > 0.
    search_target: Optional[np.ndarray] = None  # [Kmax] float32 improved policy
    search_value: float = 0.0  # search-consistent value estimate
    searched: bool = False  # True when this decision was searched


@dataclass(slots=True)
class DecisionBatch:
    """Batch of decision-level transitions."""
    
    observations: np.ndarray  # [T, ...]
    masks: np.ndarray  # [T, 4, 16, 8]
    pill_colors: np.ndarray  # [T, 2]
    preview_pill_colors: np.ndarray  # [T, 2]
    actions: np.ndarray  # [T]
    log_probs: np.ndarray  # [T]
    values: np.ndarray  # [T]
    taus: np.ndarray  # [T] - frame durations
    rewards: np.ndarray  # [T] - cumulative rewards
    observations_next: np.ndarray  # [T, ...]
    dones: np.ndarray  # [T]
    costs_to_lock: Optional[np.ndarray] = None  # [T, 4, 16, 8] float32
    aux: Optional[np.ndarray] = None  # [T, aux_dim]
    env_ids: Optional[np.ndarray] = None  # [T] - environment indices
    advantages: Optional[np.ndarray] = None  # [T] - computed later
    returns: Optional[np.ndarray] = None  # [T] - computed later
    gammas: Optional[np.ndarray] = None  # [T] - γ^τ for each step
    # Search-distillation targets (None when disabled).
    search_targets: Optional[np.ndarray] = None  # [T, Kmax] float32
    search_values: Optional[np.ndarray] = None  # [T] float32
    search_mask: Optional[np.ndarray] = None  # [T] bool


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
        aux_dim: int = 0,
        store_costs_to_lock: bool = False,
        search_target_dim: int = 0,
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
        self.aux_dim = int(max(0, int(aux_dim)))
        self._store_costs_to_lock = bool(store_costs_to_lock)
        self.search_target_dim = int(max(0, int(search_target_dim)))
        
        # Storage
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.masks = np.zeros((capacity, 4, 16, 8), dtype=np.bool_)
        self.costs_to_lock = (
            np.zeros((capacity, 4, 16, 8), dtype=np.float32) if self._store_costs_to_lock else None
        )
        self.pill_colors = np.zeros((capacity, 2), dtype=np.int64)
        self.preview_pill_colors = np.zeros((capacity, 2), dtype=np.int64)
        self.aux = (
            np.zeros((capacity, self.aux_dim), dtype=np.float32) if self.aux_dim > 0 else None
        )
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.taus = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.observations_next = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.env_ids = np.zeros(capacity, dtype=np.int32)
        if self.search_target_dim > 0:
            self.search_targets: Optional[np.ndarray] = np.zeros(
                (capacity, self.search_target_dim), dtype=np.float32
            )
            self.search_values: Optional[np.ndarray] = np.zeros(capacity, dtype=np.float32)
            self.search_mask: Optional[np.ndarray] = np.zeros(capacity, dtype=np.bool_)
        else:
            self.search_targets = None
            self.search_values = None
            self.search_mask = None

        self.ptr = 0
        self.size = 0
        
    def add(self, step: DecisionStep) -> None:
        """Add a decision step to the buffer."""
        idx = self.ptr
        
        self.observations[idx] = step.obs
        self.masks[idx] = step.mask
        if self.costs_to_lock is not None:
            if step.cost_to_lock is None:
                raise ValueError("DecisionStep.cost_to_lock is required when store_costs_to_lock=True")
            cost_arr = np.asarray(step.cost_to_lock, dtype=np.float32)
            if cost_arr.shape != (4, 16, 8):
                raise ValueError(f"Expected cost_to_lock shape (4,16,8), got {cost_arr.shape!r}")
            self.costs_to_lock[idx] = cost_arr
        self.pill_colors[idx] = step.pill_colors
        self.preview_pill_colors[idx] = step.preview_pill_colors
        if self.aux is not None:
            if step.aux is None:
                raise ValueError("DecisionStep.aux is required when aux_dim > 0")
            aux_arr = np.asarray(step.aux, dtype=np.float32).reshape(-1)
            if aux_arr.shape != (self.aux_dim,):
                raise ValueError(f"Expected aux shape ({self.aux_dim},), got {aux_arr.shape!r}")
            self.aux[idx] = aux_arr
        self.actions[idx] = step.action
        self.log_probs[idx] = step.log_prob
        self.values[idx] = step.value
        self.taus[idx] = step.tau
        self.rewards[idx] = step.reward
        self.observations_next[idx] = step.obs_next
        self.dones[idx] = step.done
        self.env_ids[idx] = int(step.env_id)
        if self.search_targets is not None:
            if step.search_target is None:
                self.search_targets[idx] = 0.0
            else:
                tgt = np.asarray(step.search_target, dtype=np.float32).reshape(-1)
                if tgt.shape != (self.search_target_dim,):
                    raise ValueError(
                        f"Expected search_target shape ({self.search_target_dim},), "
                        f"got {tgt.shape!r}"
                    )
                self.search_targets[idx] = tgt
            self.search_values[idx] = float(step.search_value)
            self.search_mask[idx] = bool(step.searched)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def add_batch(self, steps: List[DecisionStep]) -> None:
        """Add multiple decision steps."""
        for step in steps:
            self.add(step)

    def add_arrays(
        self,
        *,
        observations: np.ndarray,
        masks: np.ndarray,
        pill_colors: np.ndarray,
        preview_pill_colors: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        taus: np.ndarray,
        rewards: np.ndarray,
        observations_next: np.ndarray,
        dones: np.ndarray,
        env_ids: np.ndarray,
        costs_to_lock: Optional[np.ndarray] = None,
        aux: Optional[np.ndarray] = None,
        search_targets: Optional[np.ndarray] = None,
        search_values: Optional[np.ndarray] = None,
        search_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Append one vector-environment wave with contiguous array writes."""

        actions_arr = np.asarray(actions).reshape(-1)
        count = int(actions_arr.shape[0])
        end = self.ptr + count
        if end > self.capacity:
            raise BufferError(
                f"Decision rollout batch exceeds capacity: {self.ptr}+{count}>{self.capacity}"
            )
        dst = slice(self.ptr, end)

        self.observations[dst] = observations
        self.masks[dst] = masks
        if self.costs_to_lock is not None:
            if costs_to_lock is None:
                raise ValueError("costs_to_lock is required when store_costs_to_lock=True")
            self.costs_to_lock[dst] = costs_to_lock
        self.pill_colors[dst] = pill_colors
        self.preview_pill_colors[dst] = preview_pill_colors
        if self.aux is not None:
            if aux is None:
                raise ValueError("aux is required when aux_dim > 0")
            self.aux[dst] = aux
        self.actions[dst] = actions_arr
        self.log_probs[dst] = log_probs
        self.values[dst] = values
        self.taus[dst] = taus
        self.rewards[dst] = rewards
        self.observations_next[dst] = observations_next
        self.dones[dst] = dones
        self.env_ids[dst] = env_ids

        if self.search_targets is not None:
            if search_targets is None:
                self.search_targets[dst] = 0.0
            else:
                self.search_targets[dst] = search_targets
            self.search_values[dst] = 0.0 if search_values is None else search_values
            self.search_mask[dst] = False if search_mask is None else search_mask

        self.ptr = end
        self.size = max(self.size, end)
            
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
        env_ids = self.env_ids[:T]
        
        # Compute Gamma_t = gamma^tau_t
        gammas = self.gamma ** taus.astype(np.float32)
        
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)

        # Normalize bootstrap values to per-env array.
        if bootstrap_value is None:
            bootstrap_by_env = np.zeros(self.num_envs, dtype=np.float32)
        else:
            arr = np.asarray(bootstrap_value, dtype=np.float32).reshape(-1)
            if arr.size == 1:
                bootstrap_by_env = np.full(self.num_envs, float(arr.item()), dtype=np.float32)
            else:
                if arr.size != self.num_envs:
                    raise ValueError(
                        f"bootstrap_value size {arr.size} does not match num_envs={self.num_envs}"
                    )
                bootstrap_by_env = arr.astype(np.float32)

        # Compute GAE separately for each environment sequence.
        for env_idx in range(self.num_envs):
            idxs = np.nonzero(env_ids == env_idx)[0]
            if idxs.size == 0:
                continue

            last_gae = 0.0
            for pos in range(idxs.size - 1, -1, -1):
                t = int(idxs[pos])
                if pos < idxs.size - 1:
                    t_next = int(idxs[pos + 1])
                    next_value = float(values[t_next])
                    next_return = float(returns[t_next])
                else:
                    next_value = float(bootstrap_by_env[env_idx])
                    next_return = float(bootstrap_by_env[env_idx])

                mask = 1.0 - float(dones[t])
                delta = rewards[t] + gammas[t] * next_value * mask - values[t]
                last_gae = delta + gammas[t] * self.gae_lambda * mask * last_gae
                advantages[t] = last_gae
                returns[t] = rewards[t] + gammas[t] * next_return * mask
            
        return advantages, returns, gammas
        
    def get_batch(
        self,
        bootstrap_value: Optional[float] = None,
        *,
        copy_storage: bool = True,
    ) -> DecisionBatch:
        """Get all stored decisions as a batch with computed advantages.
        
        Args:
            bootstrap_value: Value for bootstrapping (e.g., V(s_T))
            
        Returns:
            DecisionBatch with advantages and returns computed
        """
        T = self.size
        
        # Compute advantages
        bootstrap = None if bootstrap_value is None else np.asarray(bootstrap_value)
        advantages, returns, gammas = self.compute_advantages(bootstrap)
        
        def stored(array: np.ndarray) -> np.ndarray:
            view = array[:T]
            return view.copy() if copy_storage else view

        return DecisionBatch(
            observations=stored(self.observations),
            masks=stored(self.masks),
            costs_to_lock=stored(self.costs_to_lock) if self.costs_to_lock is not None else None,
            pill_colors=stored(self.pill_colors),
            preview_pill_colors=stored(self.preview_pill_colors),
            aux=stored(self.aux) if self.aux is not None else None,
            actions=stored(self.actions),
            log_probs=stored(self.log_probs),
            values=stored(self.values),
            taus=stored(self.taus),
            rewards=stored(self.rewards),
            observations_next=stored(self.observations_next),
            dones=stored(self.dones),
            env_ids=stored(self.env_ids),
            advantages=advantages,
            returns=returns,
            gammas=gammas,
            search_targets=stored(self.search_targets) if self.search_targets is not None else None,
            search_values=stored(self.search_values) if self.search_values is not None else None,
            search_mask=stored(self.search_mask) if self.search_mask is not None else None,
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
