from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

import envs.specs.ram_to_state as ram_specs

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


class DrMarioFrameEncoder(nn.Module):
    """Encode a 16x8 RAM-derived state frame into a compact feature vector."""

    def __init__(
        self,
        in_channels: Optional[int] = None,
        proj_dim: int = 192,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        if in_channels is None:
            in_channels = ram_specs.STATE_CHANNELS
        self._use_bitplanes = ram_specs.STATE_USE_BITPLANES
        idx = ram_specs.STATE_IDX
        if self._use_bitplanes:
            self._color_idx = [int(i) for i in idx.color_channels]
            self._virus_mask_idx = int(idx.virus_mask)
            self._locked_mask_idx = int(idx.locked_mask)
            self._static_idx = None
            self._virus_color_idx = None
        else:
            self._color_idx = None
            self._virus_mask_idx = None
            self._locked_mask_idx = None
            self._static_idx = [int(i) for i in idx.static_color_channels]
            self._virus_color_idx = [int(i) for i in idx.virus_color_channels]
        mid = hidden_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)  # (16x8) -> (8x4)
        self.conv3 = nn.Conv2d(mid, hidden_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)  # (8x4) -> (4x2)
        self.conv5 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

        conv_feat_dim = hidden_channels * 4 * 2
        feature_dim = conv_feat_dim + 3 + 8
        self.proj = nn.Sequential(nn.Linear(feature_dim, proj_dim), nn.ReLU(inplace=True))
        self.output_dim = proj_dim
        self.register_buffer(
            "_row_from_bottom",
            torch.arange(1, 17, dtype=torch.float32).flip(0).view(1, 16, 1),
            persistent=False,
        )
        self.register_buffer(
            "_row_scale",
            torch.tensor(16.0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_virus_normalizer",
            torch.tensor(1.0 / (16.0 * 8.0), dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = x
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.pool1(x)
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.pool2(x)
        x = self.act(self.conv5(x))
        conv_feat = x.flatten(1)

        if self._use_bitplanes:
            color_planes = original[:, self._color_idx]
            locked_mask = original[:, self._locked_mask_idx].unsqueeze(1)
            static_planes = color_planes * locked_mask
            static_mask = (static_planes > 0.1).any(dim=1).float()
            virus_planes = color_planes * original[:, self._virus_mask_idx].unsqueeze(1)
        else:
            static_planes = original[:, self._static_idx]
            static_mask = (static_planes > 0.1).any(dim=1).float()
            virus_planes = original[:, self._virus_color_idx]
        row_from_bottom = self._row_from_bottom.to(device=original.device, dtype=original.dtype)
        row_scale = self._row_scale.to(device=original.device, dtype=original.dtype)
        column_heights = (static_mask * row_from_bottom).amax(dim=1) / row_scale
        virus_norm = self._virus_normalizer.to(device=original.device, dtype=original.dtype)
        virus_counts = virus_planes.reshape(original.size(0), 3, -1).sum(dim=-1) * virus_norm

        features = torch.cat([conv_feat, virus_counts, column_heights], dim=1)
        return self.proj(features)


class DrMarioStatePolicyNet(nn.Module):
    """Policy/value head tailored for Dr. Mario 16x8 state inputs.

    The encoder reasons over local tile structure, column/row statistics, and virus
    counts. A GRU core maintains temporal context across steps (spawn cadence, lock
    timers), and linear heads map to action logits and value estimates. The recurrent
    core can be configured as a GRU or LSTM and defaults to a lightweight 128-unit
    hidden size suited for low-latency MPS execution.
    """

    def __init__(
        self,
        action_dim: int,
        in_channels: Optional[int] = None,
        frame_embed_dim: int = 192,
        core_hidden: int = 128,
        core_type: str = "gru",
    ) -> None:
        super().__init__()
        if in_channels is None:
            in_channels = ram_specs.STATE_CHANNELS
        self.encoder = DrMarioFrameEncoder(in_channels=in_channels, proj_dim=frame_embed_dim)
        self.pre_core_ln = nn.LayerNorm(frame_embed_dim)
        core_type = core_type.lower()
        if core_type not in {"gru", "lstm"}:
            raise ValueError("core_type must be either 'gru' or 'lstm'")
        self._core_type = core_type
        if self._core_type == "gru":
            self.core: nn.Module = nn.GRU(
                input_size=frame_embed_dim,
                hidden_size=core_hidden,
                batch_first=True,
            )
        else:
            self.core = nn.LSTM(
                input_size=frame_embed_dim,
                hidden_size=core_hidden,
                batch_first=True,
            )
        self._core_hidden = core_hidden
        self.post_core_ln = nn.LayerNorm(core_hidden)
        self.policy = nn.Linear(core_hidden, action_dim)
        self.value = nn.Linear(core_hidden, 1)
        self._flatten_called = False

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ]:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        frames = x.view(B * T, C, H, W)
        encoded = self.encoder(frames).view(B, T, -1)
        encoded = self.pre_core_ln(encoded)
        if not self._flatten_called and hasattr(self.core, "flatten_parameters"):
            self.core.flatten_parameters()
            self._flatten_called = True
        outputs, hx = self.core(encoded, hx)
        outputs = self.post_core_ln(outputs)
        logits = self.policy(outputs)
        values = self.value(outputs).squeeze(-1)
        return logits, values, hx


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, out_ch // 16), num_channels=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, out_ch // 16), num_channels=out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetFeatureExtractor(nn.Module):
    def __init__(self, in_ch: int = 12, base: int = 64, proj_dim: int = 256) -> None:
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 4
        self.enc1 = ConvBlock(in_ch, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.bottleneck = ConvBlock(c3, c4)
        self.down = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.up0 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c3 + c3, c3)
        self.dec1 = ConvBlock(c2 + c2, c2)
        self.dec0 = ConvBlock(c1 + c1, c1)
        self.proj = nn.Sequential(
            nn.Linear(c1 + c2 + c3 + c4, proj_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 128, 128)
        s1 = self.enc1(x)
        d1 = self.down(s1)
        s2 = self.enc2(d1)
        d2 = self.down(s2)
        s3 = self.enc3(d2)
        d3 = self.down(s3)
        bottleneck = self.bottleneck(d3)

        u2 = self.up2(bottleneck)
        u2 = torch.cat([u2, s3], dim=1)
        u2 = self.dec2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, s2], dim=1)
        u1 = self.dec1(u1)

        u0 = self.up0(u1)
        u0 = torch.cat([u0, s1], dim=1)
        u0 = self.dec0(u0)

        gap_s1 = s1.mean(dim=(2, 3))
        gap_s2 = s2.mean(dim=(2, 3))
        gap_s3 = s3.mean(dim=(2, 3))
        gap_bot = bottleneck.mean(dim=(2, 3))

        features = torch.cat([gap_s1, gap_s2, gap_s3, gap_bot], dim=1)
        return self.proj(features)


class DrMarioPixelUNetPolicyNet(nn.Module):
    """Policy/value network operating on 4×128×128 pixel stacks."""

    def __init__(
        self,
        action_dim: int,
        in_channels: int = 12,
        frame_embed_dim: int = 256,
        core_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = UNetFeatureExtractor(in_ch=in_channels, proj_dim=frame_embed_dim)
        self.core = nn.GRU(input_size=frame_embed_dim, hidden_size=core_hidden, batch_first=True)
        self.policy = nn.Linear(core_hidden, action_dim)
        self.value = nn.Linear(core_hidden, 1)

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, T, C, H, W) with H=W=128
        B, T, C, H, W = x.shape
        frames = x.view(B * T, C, H, W)
        encoded = self.encoder(frames).view(B, T, -1)
        outputs, hx = self.core(encoded, hx)
        logits = self.policy(outputs)
        values = self.value(outputs).squeeze(-1)
        return logits, values, hx
