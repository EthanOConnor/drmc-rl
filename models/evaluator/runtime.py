"""Runtime helper to load a TorchScript evaluator and expose mean/quantile/CVaR.

This makes it easy to plug into reward shaping and planning without coupling to training code.
"""
from __future__ import annotations
import torch
import numpy as np
from typing import Any


class ScriptEvaluator:
    def __init__(self, path: str, device: str = 'cpu', in_channels: int = 14, input_hw=(128, 128)):
        self.model = torch.jit.load(path, map_location=device)
        self.model.eval()
        self.device = device
        self.in_channels = in_channels
        self.input_hw = input_hw

    def _to_tensor(self, state) -> torch.Tensor:
        # Expect state as (C,H,W) float32 in [0,1] or similar.
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return x

    @torch.no_grad()
    def quantiles(self, state, taus=None):
        x = self._to_tensor(state)
        q = self.model(x)  # (B,K)
        return q.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def mean_time(self, state) -> float:
        q = self.quantiles(state)
        # approximate mean via uniform average of quantiles
        return float(np.mean(q))

    @torch.no_grad()
    def quantile_time(self, state, tau: float) -> float:
        q = self.quantiles(state)
        idx = int(np.clip(round(tau * (len(q) - 1)), 0, len(q) - 1))
        return float(q[idx])

    @torch.no_grad()
    def cvar_time(self, state, alpha: float = 0.25) -> float:
        q = self.quantiles(state)
        k = max(1, int(alpha * len(q)))
        return float(np.mean(q[:k]))

