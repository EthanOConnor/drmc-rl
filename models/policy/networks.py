from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ImpalaBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class ImpalaEncoder(nn.Module):
    def __init__(self, in_ch: int, out_dim: int = 256):
        super().__init__()
        self.blocks = nn.Sequential(
            ImpalaBlock(in_ch, 32),
            ImpalaBlock(32, 64),
            ImpalaBlock(64, 64),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, out_dim),  # assuming 128x128 input -> /2/2/2 = 16
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(x))


class StateProjector(nn.Module):
    """Project 16x8 state planes to 128x128 image-like tensors.

    - Input: (B, C, 16, 8)
    - Output: (B, C, 128, 128)
    """

    def __init__(self, mode: str = "nearest"):
        super().__init__()
        self.upsample = nn.Upsample(size=(128, 128), mode=mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class PolicyNet(nn.Module):
    """PPO policy with risk conditioning and optional LSTM.

    Input shape assumptions:
      - pixel mode: (B, 4, 3, 128, 128) -> stack along channel dim -> (B, 12, 128, 128)
      - state mode: (B, 4, C, 16, 8) -> use StateProjector to 128x128, then stack frames
    """

    def __init__(self, action_dim: int, in_ch: int = 12, lstm_hidden: int = 128):
        super().__init__()
        self.encoder = ImpalaEncoder(in_ch=in_ch, out_dim=256)
        self.risk_embed = nn.Sequential(nn.Linear(1, 16), nn.ReLU(inplace=True))
        self.core = nn.LSTM(input_size=256 + 16, hidden_size=lstm_hidden, batch_first=True)
        self.policy = nn.Linear(lstm_hidden, action_dim)
        self.value = nn.Linear(lstm_hidden, 1)

    def forward(
        self,
        x: torch.Tensor,
        risk_tau: Optional[torch.Tensor] = None,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        # x: (B, T, C, H, W) -> flatten T into batch for encoding
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        z = self.encoder(x)
        z = z.view(B, T, -1)
        if risk_tau is None:
            risk_tau = torch.ones(B, T, 1, device=z.device)
        e = self.risk_embed(risk_tau)
        h_in = torch.cat([z, e], dim=-1)
        y, hx = self.core(h_in, hx)
        logits = self.policy(y)
        value = self.value(y)
        return logits, value, hx
