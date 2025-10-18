"""Utilities for converting Dr. Mario state tensors to RGB debug frames."""
from __future__ import annotations

from typing import Any, Dict, Optional

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
    board = np.zeros((h, w, 3), dtype=np.uint8)

    def apply_mask(channel: int, color: tuple[int, int, int], thresh: float = 0.1) -> None:
        mask = latest[channel] > thresh
        board[mask] = color

    # Viruses (channels 0-2)
    apply_mask(0, (220, 40, 40))
    apply_mask(1, (240, 220, 40))
    apply_mask(2, (40, 120, 240))

    # Fixed pill halves (channels 3-5)
    static_palette = {
        3: (180, 0, 0),
        4: (200, 180, 0),
        5: (0, 80, 200),
    }
    for channel, color in static_palette.items():
        apply_mask(channel, color)

    # Falling pill halves (channels 6-8) — recolor as static when locked
    lock_norm = float(latest[11, 0, 0]) if latest.shape[0] > 11 else 0.0
    locked = lock_norm > 1e-3
    if isinstance(info, dict) and "is_locked" in info:
        locked = bool(info["is_locked"])
    falling_channels = {
        6: ((255, 128, 128), static_palette[3]),
        7: ((255, 255, 120), static_palette[4]),
        8: ((120, 120, 255), static_palette[5]),
    }
    for channel, (falling_color, locked_color) in falling_channels.items():
        mask = latest[channel] > 0.1
        if not mask.any():
            continue
        board[mask] = locked_color if locked else falling_color

    # Background gradient for contrast
    gravity = np.clip(latest[10], 0.0, 1.0)
    base = (gravity * 40).astype(np.uint8)
    empty_mask = board.sum(axis=-1) == 0
    if empty_mask.any():
        board[empty_mask] = base[empty_mask][:, None]

    # Compose preview area above the board
    preview_rows = 4
    canvas = np.zeros((h + preview_rows, w, 3), dtype=np.uint8)
    preview_bg = np.full((preview_rows, w, 3), 15, dtype=np.uint8)
    canvas[:preview_rows] = preview_bg
    canvas[preview_rows:] = board

    if isinstance(info, dict):
        preview = info.get("preview_pill")
    else:
        preview = None
    if isinstance(preview, dict):
        center_col = w // 2
        center_row = preview_rows // 2
        rotation = int(preview.get("rotation", 0)) & 0x03
        color_values = (
            preview.get("first_color"),
            preview.get("second_color"),
        )
        placements = {
            0: ((center_row, center_col - 1), (center_row, center_col)),
            1: ((center_row, center_col), (center_row - 1, center_col)),
            2: ((center_row, center_col), (center_row, center_col - 1)),
            3: ((center_row - 1, center_col), (center_row, center_col)),
        }
        coords = placements.get(rotation, placements[0])
        color_lookup = {
            0: static_palette[4],  # yellow
            1: static_palette[3],  # red
            2: static_palette[5],  # blue
        }
        for color_value, (rr, cc) in zip(color_values, coords):
            if color_value is None:
                continue
            color = color_lookup.get(int(color_value))
            if color is None:
                continue
            if 0 <= rr < preview_rows and 0 <= cc < w:
                canvas[rr, cc] = color

    scale = max(1, int(tile_scale))
    if scale > 1:
        canvas = np.repeat(np.repeat(canvas, scale, axis=0), scale, axis=1)

    virus_mask = (latest[0:3] > 0.1).any(axis=0)
    if virus_mask.any():
        centers = np.argwhere(virus_mask)
        for r, c in centers:
            rr = (r + preview_rows) * scale + scale // 2
            cc = c * scale + scale // 2
            if 0 <= rr < canvas.shape[0] and 0 <= cc < canvas.shape[1]:
                canvas[rr, cc] = (0, 0, 0)

    return canvas


__all__ = ["state_to_rgb"]
