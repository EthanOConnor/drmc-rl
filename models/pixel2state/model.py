from __future__ import annotations

import torch
import torch.nn as nn

import envs.specs.ram_to_state as ram_specs


class SimpleUNet(nn.Module):
    """Lightweight UNet-like model for pixel->state segmentation (skeleton)."""

    def __init__(self, in_ch: int = 3, out_ch: int | None = None):
        super().__init__()
        if out_ch is None:
            out_ch = ram_specs.STATE_CHANNELS
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.enc3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(64, 64, 2, stride=2), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU())
        self.head = nn.ConvTranspose2d(32, out_ch, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        y = self.dec2(x3)
        y = self.dec1(y)
        y = self.head(y)
        return y  # logits per state channel
