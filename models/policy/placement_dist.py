"""Masked categorical distribution over placement actions.

Provides utilities for sampling, log-prob computation, and entropy calculation
over the 464-way placement action space (4 orientations × 16 rows × 8 cols)
with invalid action masking.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.distributions import Categorical
except ImportError:
    torch = None  # type: ignore
    F = None  # type: ignore
    Categorical = None  # type: ignore

# Placement grid dimensions
ORIENTATIONS = 4
GRID_HEIGHT = 16
GRID_WIDTH = 8
TOTAL_ACTIONS = ORIENTATIONS * GRID_HEIGHT * GRID_WIDTH  # 512


def flatten_placement(o: int, i: int, j: int) -> int:
    """Convert (orientation, row, col) to flat action index.
    
    Args:
        o: Orientation in [0, 3] (H+, H−, V+, V−)
        i: Row in [0, 15]
        j: Col in [0, 7]
        
    Returns:
        Flat index in [0, 511]
    """
    return o * GRID_HEIGHT * GRID_WIDTH + i * GRID_WIDTH + j


def unflatten_placement(idx: int) -> Tuple[int, int, int]:
    """Convert flat action index to (orientation, row, col).
    
    Args:
        idx: Flat index in [0, 511]
        
    Returns:
        Tuple (o, i, j) with o in [0,3], i in [0,15], j in [0,7]
    """
    o = idx // (GRID_HEIGHT * GRID_WIDTH)
    remainder = idx % (GRID_HEIGHT * GRID_WIDTH)
    i = remainder // GRID_WIDTH
    j = remainder % GRID_WIDTH
    return int(o), int(i), int(j)


class MaskedPlacementDist:
    """Masked categorical distribution over placement actions.
    
    Handles invalid action masking and provides sampling, log-prob, and entropy
    computation with proper normalization over feasible actions only.
    """
    
    def __init__(
        self,
        logits_map: Union[torch.Tensor, np.ndarray],
        mask: Union[torch.Tensor, np.ndarray],
        eps: float = 1e-9,
    ):
        """Initialize masked placement distribution.
        
        Args:
            logits_map: Raw logits with shape [B, 4, 16, 8] or [4, 16, 8]
            mask: Boolean mask with shape [B, 4, 16, 8] or [4, 16, 8]
                  True = valid action, False = invalid
            eps: Small constant for numerical stability
        """
        if torch is None:
            raise RuntimeError("PyTorch is required for MaskedPlacementDist")
            
        # Convert to torch tensors if needed
        if isinstance(logits_map, np.ndarray):
            logits_map = torch.from_numpy(logits_map)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).bool()
            
        self.logits_map = logits_map
        self.mask = mask.bool()
        self.eps = eps
        
        # Add batch dimension if needed
        if self.logits_map.ndim == 3:
            self.logits_map = self.logits_map.unsqueeze(0)
            self.mask = self.mask.unsqueeze(0)
            self._batch_added = True
        else:
            self._batch_added = False
            
        # Flatten to [B, 512]
        B = self.logits_map.shape[0]
        self.logits_flat = self.logits_map.reshape(B, -1)
        self.mask_flat = self.mask.reshape(B, -1)
        
        # Apply mask: set invalid logits to large negative value
        self.masked_logits = self.logits_flat.clone()
        self.masked_logits[~self.mask_flat] = -1e9
        
        # Handle edge case: no valid actions (shouldn't happen in practice)
        # Fall back to first valid cell or uniform if truly none
        for b in range(B):
            if not self.mask_flat[b].any():
                # Find first True in original mask or use action 0
                first_valid = 0
                if mask.any():
                    first_valid = int(mask.reshape(-1).nonzero()[0].item())
                self.mask_flat[b, first_valid] = True
                self.masked_logits[b, first_valid] = 0.0
                
        # Compute probabilities
        self.probs = F.softmax(self.masked_logits, dim=-1)
        
    def sample(self, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from the masked distribution.
        
        Args:
            deterministic: If True, return argmax; otherwise sample
            
        Returns:
            Tuple of (flat_indices, log_probs) with shape [B] or scalar if batch added
        """
        if deterministic:
            indices = self.masked_logits.argmax(dim=-1)
        else:
            indices = Categorical(probs=self.probs).sample()
            
        log_probs = self.log_prob(indices)
        
        if self._batch_added:
            return indices.squeeze(0), log_probs.squeeze(0)
        return indices, log_probs
        
    def sample_placement(self, deterministic: bool = False) -> Tuple[Tuple[int, int, int], float]:
        """Sample and return as (o, i, j) tuple with log-prob.
        
        Args:
            deterministic: If True, return argmax; otherwise sample
            
        Returns:
            Tuple of ((o, i, j), log_prob) for single batch
        """
        indices, log_probs = self.sample(deterministic=deterministic)
        idx = int(indices.item())
        logp = float(log_probs.item())
        return unflatten_placement(idx), logp
        
    def log_prob(self, actions: Union[torch.Tensor, int]) -> torch.Tensor:
        """Compute log probabilities for given actions.
        
        Args:
            actions: Action indices with shape [B] or scalar
            
        Returns:
            Log probabilities with shape [B] or scalar
        """
        if isinstance(actions, int):
            actions = torch.tensor([actions], dtype=torch.long, device=self.logits_flat.device)
        elif actions.ndim == 0:
            actions = actions.unsqueeze(0)
            
        # Ensure batch dimension
        if actions.shape[0] != self.probs.shape[0]:
            actions = actions.expand(self.probs.shape[0])
            
        log_probs_all = torch.log(self.probs + self.eps)
        log_probs = log_probs_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        if self._batch_added:
            return log_probs.squeeze(0)
        return log_probs
        
    def entropy(self) -> torch.Tensor:
        """Compute entropy of the masked distribution.
        
        Returns:
            Entropy with shape [B] or scalar if batch added
        """
        # Only compute over valid actions
        log_probs = torch.log(self.probs + self.eps)
        entropy = -(self.probs * log_probs).sum(dim=-1)
        
        if self._batch_added:
            return entropy.squeeze(0)
        return entropy
        
    def mode(self) -> torch.Tensor:
        """Return the most probable action (argmax).
        
        Returns:
            Action indices with shape [B] or scalar
        """
        indices = self.masked_logits.argmax(dim=-1)
        if self._batch_added:
            return indices.squeeze(0)
        return indices


def gumbel_top_k_sample(
    logits_map: torch.Tensor,
    mask: torch.Tensor,
    k: int = 2,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample top-k candidates using Gumbel-Top-k trick.
    
    Useful for exploration: sample k diverse high-probability actions
    without replacement.
    
    Args:
        logits_map: Logits with shape [B, 4, 16, 8]
        mask: Boolean mask with shape [B, 4, 16, 8]
        k: Number of candidates to sample
        temperature: Gumbel temperature (lower = more deterministic)
        
    Returns:
        Tuple of (indices [B, k], gumbel_values [B, k])
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for gumbel_top_k_sample")
        
    B = logits_map.shape[0]
    logits_flat = logits_map.reshape(B, -1)
    mask_flat = mask.reshape(B, -1).bool()
    
    # Apply mask
    masked_logits = logits_flat.clone()
    masked_logits[~mask_flat] = -1e9
    
    # Add Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(masked_logits) + 1e-9) + 1e-9)
    perturbed = (masked_logits / temperature) + gumbel_noise
    
    # Top-k
    values, indices = torch.topk(perturbed, k=min(k, mask_flat.sum(dim=-1).min().item()), dim=-1)
    
    return indices, values


__all__ = [
    "MaskedPlacementDist",
    "flatten_placement",
    "unflatten_placement",
    "gumbel_top_k_sample",
    "ORIENTATIONS",
    "GRID_HEIGHT",
    "GRID_WIDTH",
    "TOTAL_ACTIONS",
]
