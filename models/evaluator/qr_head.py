from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QREvaluator(nn.Module):
    """Quantile Regression head for time-to-clear distribution.

    Expects an upstream encoder; this module maps features -> K quantiles.
    """

    def __init__(self, in_dim: int = 256, num_quantiles: int = 101):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.fc = nn.Linear(in_dim, num_quantiles)
        # Fixed quantile fractions [0.01 .. 0.99]
        taus = torch.linspace(0.01, 0.99, steps=num_quantiles)
        self.register_buffer("taus", taus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns (B, K)
        return self.fc(x)

    @staticmethod
    def quantile_huber_loss(
        preds: torch.Tensor,  # (B, K)
        targets: torch.Tensor,  # (B,)
        taus: torch.Tensor,  # (K,)
        kappa: float = 1.0,
    ) -> torch.Tensor:
        # Broadcast targets to (B, K)
        diff = targets.unsqueeze(1) - preds
        abs_diff = diff.abs()
        huber = torch.where(abs_diff <= kappa, 0.5 * diff.pow(2), kappa * (abs_diff - 0.5 * kappa))
        # Indicator for diff < 0
        indicator = (diff.detach() < 0).float()
        loss = (torch.abs(taus - indicator) * huber).mean()
        return loss

