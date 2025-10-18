"""Utilities for converting Dr. Mario state tensors to RGB debug frames."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def state_to_rgb(
    state_stack: np.ndarray,
    info: Optional[Dict[str, Any]] = None,  # info currently unused but kept for parity with demo hook
    *,
    tile_scale: int = 6,
) -> np.ndarray:
    """Convert a stacked state tensor to an RGB visualization suitable for logging."""
    latest = np.asarray(state_stack[-1])
    h, w = latest.shape[1], latest.shape[2]
    img = np.zeros((h, w, 3), dtype=np.uint8)

    def paint(channel: int, color: tuple[int, int, int], thresh: float = 0.1) -> None:
        mask = latest[channel] > thresh
        img[mask] = color

    # Viruses (channels 0-2)
    paint(0, (220, 40, 40))
    paint(1, (240, 220, 40))
    paint(2, (40, 120, 240))
    # Fixed pill halves (3-5)
    paint(3, (180, 0, 0))
    paint(4, (200, 180, 0))
    paint(5, (0, 80, 200))
    # Falling pill halves (6-8)
    paint(6, (255, 128, 128))
    paint(7, (255, 255, 120))
    paint(8, (120, 120, 255))
    # Background gradient for contrast
    gravity = np.clip(latest[10], 0.0, 1.0)
    base = (gravity * 40).astype(np.uint8)
    img[img.sum(axis=-1) == 0] = base[img.sum(axis=-1) == 0][:, None]

    scale = max(1, int(tile_scale))
    if scale > 1:
        img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

    virus_mask = (latest[0:3] > 0.1).any(axis=0)
    if virus_mask.any():
        centers = np.argwhere(virus_mask)
        for r, c in centers:
            rr = r * scale + scale // 2
            cc = c * scale + scale // 2
            if 0 <= rr < img.shape[0] and 0 <= cc < img.shape[1]:
                img[rr, cc] = (0, 0, 0)

    return img


__all__ = ["state_to_rgb"]
