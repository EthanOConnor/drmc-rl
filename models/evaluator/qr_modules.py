from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import envs.specs.ram_to_state as ram_specs

class QREvaluator(nn.Module):
    """Quantile Regression evaluator for T_clear distribution.

    Inputs:
        x: (B, C, H, W) state tensor (e.g., STATE_CHANNELS×16×8 upsampled to 128×128).
    Outputs:
        quantiles: (B, K) predicted quantiles of T_clear (in frames).
    """

    def __init__(self, in_channels: Optional[int] = None, k_quantiles: int = 101):
        super().__init__()
        if in_channels is None:
            in_channels = ram_specs.STATE_CHANNELS
        self.k = k_quantiles
        # Simple IMPALA-like encoder for demonstration; match your policy encoder shape.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),  # assuming input upsampled to 128×128
            nn.ReLU(inplace=True),
            nn.Linear(256, self.k),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        return self.head(z)


def quantile_huber_loss(
    pred_q: torch.Tensor, target_samples: torch.Tensor, taus: torch.Tensor, kappa: float = 1.0
) -> torch.Tensor:
    """Compute Quantile Huber loss.

    pred_q: (B, K)
    target_samples: (B, S) Monte-Carlo samples of T_clear
    taus: (K,) quantile fractions in (0,1)
    """
    B, K = pred_q.shape
    S = target_samples.shape[1]
    # Expand for pairwise differences: (B, K, S)
    pred = pred_q.unsqueeze(-1)
    target = target_samples.unsqueeze(1)
    diff = target - pred
    abs_diff = diff.abs()
    huber = torch.where(abs_diff <= kappa, 0.5 * diff.pow(2), kappa * (abs_diff - 0.5 * kappa))
    # Indicator: diff < 0
    indicator = (diff.detach() < 0).float()
    taus = taus.view(1, K, 1)
    loss = (torch.abs(taus - indicator) * huber).mean()
    return loss
