"""MLX-native policy/value networks tuned for Dr. Mario state inputs.

The module implements a lightweight convolutional encoder paired with a
custom GRU core written directly with MLX primitives. The design keeps the
temporal core in-register on Apple Silicon GPUs, maximising performance when
driving training or evaluation loops with ``mlx``.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import envs.specs.ram_to_state as ram_specs

try:  # pragma: no cover - optional dependency
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:  # pragma: no cover - optional dependency
    mx = None
    nn = None


if mx is not None and nn is not None:

    class _ResidualUnit(nn.Module):
        """Depthwise-enhanced residual bottleneck for the state encoder."""

        def __init__(self, channels: int, expansion: int = 2) -> None:
            super().__init__()
            hidden = channels * expansion
            self.expand = nn.Conv2d(channels, hidden, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv = nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, bias=False)
            self.project = nn.Conv2d(hidden, channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.act = nn.SiLU()

        def __call__(self, x: mx.array) -> mx.array:
            shortcut = x
            x = self.expand(x)
            x = self.act(x)
            x = self.conv(x)
            x = self.act(x)
            x = self.project(x)
            return self.act(x + shortcut)


    class _DownsampleBlock(nn.Module):
        """Strided convolution followed by a residual refinement block."""

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
            self.act = nn.SiLU()
            self.refine = _ResidualUnit(out_channels)

        def __call__(self, x: mx.array) -> mx.array:
            x = self.conv(x)
            x = self.act(x)
            return self.refine(x)


    class _StateEncoder(nn.Module):
        """Compact convolutional encoder for 16x8 RAM-derived state planes."""

        def __init__(
            self,
            in_channels: int,
            width: int = 48,
            proj_dim: int = 192,
        ) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, width, kernel_size=5, stride=1, padding=2, bias=False),
                nn.SiLU(),
                _ResidualUnit(width, expansion=2),
            )
            self.down1 = _DownsampleBlock(width, width * 2)
            self.down2 = _DownsampleBlock(width * 2, width * 4)
            self.norm = nn.LayerNorm(width * 4)
            self.project = nn.Sequential(
                nn.Linear(width * 4, proj_dim, bias=False),
                nn.SiLU(),
            )

        def __call__(self, x: mx.array) -> mx.array:
            # Input: (B, C, 16, 8)
            x = self.stem(x)
            x = self.down1(x)
            x = self.down2(x)
            x = mx.mean(x, axis=(2, 3))
            x = self.norm(x)
            return self.project(x)


    class _TemporalGRUCell(nn.Module):
        """Hand-optimised GRU cell using MLX primitives."""

        def __init__(self, input_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim * 3, bias=False)
            self.hidden_proj = nn.Linear(hidden_dim, hidden_dim * 3, bias=True)
            self.hidden_dim = hidden_dim

        def __call__(self, x: mx.array, h: mx.array) -> mx.array:
            projected_x = self.input_proj(x)
            projected_h = self.hidden_proj(h)
            hidden = self.hidden_dim

            z = mx.sigmoid(projected_x[:, :hidden] + projected_h[:, :hidden])
            r = mx.sigmoid(projected_x[:, hidden : 2 * hidden] + projected_h[:, hidden : 2 * hidden])
            n = mx.tanh(projected_x[:, 2 * hidden :] + r * projected_h[:, 2 * hidden :])
            return (1.0 - z) * n + z * h


    class _TemporalCore(nn.Module):
        """Unrolled GRU core that maintains per-batch temporal context."""

        def __init__(self, input_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.cell = _TemporalGRUCell(input_dim, hidden_dim)
            self.hidden_dim = hidden_dim

        def __call__(self, x: mx.array, h: mx.array) -> Tuple[mx.array, mx.array]:
            steps = int(x.shape[1])
            outputs = []
            state = h
            for t in range(steps):
                state = self.cell(x[:, t, :], state)
                outputs.append(state)
            return mx.stack(outputs, axis=1), state


    class DrMarioStatePolicyMLX(nn.Module):
        """End-to-end MLX actor-critic tailored for Dr. Mario state tensors."""

        def __init__(
            self,
            action_dim: int,
            in_channels: Optional[int] = None,
            embed_dim: int = 192,
            core_hidden: int = 128,
        ) -> None:
            super().__init__()
            if in_channels is None:
                in_channels = ram_specs.STATE_CHANNELS
            self.encoder = _StateEncoder(in_channels=in_channels, proj_dim=embed_dim)
            self.core = _TemporalCore(embed_dim, core_hidden)
            self.norm = nn.LayerNorm(core_hidden)
            self.policy = nn.Linear(core_hidden, action_dim, bias=False)
            self.value = nn.Linear(core_hidden, 1, bias=False)
            self._hidden_dim = core_hidden

        def __call__(
            self,
            x: mx.array,
            hx: Optional[mx.array] = None,
        ) -> Tuple[mx.array, mx.array, mx.array]:
            if x.ndim != 5:
                raise ValueError("Expected input of shape (B, T, C, H, W)")
            batch, time, channels, height, width = x.shape
            frames = mx.reshape(x, (batch * time, channels, height, width))
            encoded = self.encoder(frames)
            encoded = mx.reshape(encoded, (batch, time, -1))

            if hx is None:
                hx = mx.zeros((batch, self._hidden_dim), dtype=encoded.dtype)

            outputs, hx = self.core(encoded, hx)
            outputs = self.norm(outputs)

            flat = mx.reshape(outputs, (batch * time, self._hidden_dim))
            logits = self.policy(flat)
            values = self.value(flat)
            logits = mx.reshape(logits, (batch, time, -1))
            values = mx.reshape(values, (batch, time))
            return logits, values, hx

        def initial_state(self, batch_size: int, dtype: Optional[mx.Dtype] = None) -> mx.array:
            """Return a zero-initialised recurrent state for the given batch size."""

            dtype = dtype or mx.float32
            return mx.zeros((batch_size, self._hidden_dim), dtype=dtype)


else:  # pragma: no cover - fall-back when MLX is unavailable

    class DrMarioStatePolicyMLX:  # type: ignore[misc]
        """Placeholder that surfaces a helpful error when MLX is missing."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("MLX is required to instantiate DrMarioStatePolicyMLX.")


__all__ = ["DrMarioStatePolicyMLX"]

