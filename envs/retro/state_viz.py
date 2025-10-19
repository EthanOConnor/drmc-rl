"""Utilities for converting Dr. Mario state tensors to RGB debug frames."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

import envs.specs.ram_to_state as ram_specs


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
    color_planes = ram_specs.get_color_planes(latest)
    static_mask = ram_specs.get_static_mask(latest)
    falling_mask = ram_specs.get_falling_mask(latest)
    virus_mask = ram_specs.get_virus_mask(latest)
    preview_mask = ram_specs.get_preview_mask(latest)
    clearing_mask = (
        (latest[ram_specs.STATE_IDX.clearing_mask] > 0.5)
        if ram_specs.STATE_USE_BITPLANES and ram_specs.STATE_IDX.clearing_mask is not None
        else np.zeros((h, w), dtype=bool)
    )

    virus_palette = (
        (220, 40, 40),
        (240, 220, 40),
        (40, 120, 240),
    )
    static_palette = (
        (180, 0, 0),
        (200, 180, 0),
        (0, 80, 200),
    )
    falling_palette = (
        (255, 128, 128),
        (255, 255, 120),
        (120, 120, 255),
    )
    preview_palette = static_palette

    lock_norm = float(np.clip(ram_specs.get_lock_value(latest), 0.0, 1.0))
    locked = lock_norm > 1e-3
    if isinstance(info, dict) and "is_locked" in info:
        locked = bool(info["is_locked"])

    # Background gradient for contrast
    gravity = float(np.clip(ram_specs.get_gravity_value(latest), 0.0, 1.0))
    base = np.full((h, w), gravity * 40.0, dtype=np.float32)

    for idx in range(min(3, color_planes.shape[0])):
        color_mask = color_planes[idx] > 0.1
        if not color_mask.any():
            continue
        virus_cells = color_mask & virus_mask
        static_cells = color_mask & static_mask
        falling_cells = color_mask & falling_mask
        preview_cells = color_mask & preview_mask
        if virus_cells.any():
            board[virus_cells] = virus_palette[idx]
        if static_cells.any():
            board[static_cells] = static_palette[idx]
        if falling_cells.any():
            board[falling_cells] = static_palette[idx] if locked else falling_palette[idx]
        if preview_cells.any():
            board[preview_cells] = preview_palette[idx]

    if clearing_mask.any():
        # Do not obscure falling pills with the clearing overlay.
        safe_overlay = np.logical_and(clearing_mask, ~falling_mask)
        if safe_overlay.any():
            board[safe_overlay] = (250, 200, 240)

    empty_mask = board.sum(axis=-1) == 0
    if empty_mask.any():
        board[empty_mask] = base[empty_mask][:, None].astype(np.uint8)

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
            0: preview_palette[1],  # yellow
            1: preview_palette[0],  # red
            2: preview_palette[2],  # blue
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

    if virus_mask.any():
        centers = np.argwhere(virus_mask)
        for r, c in centers:
            rr = (r + preview_rows) * scale + scale // 2
            cc = c * scale + scale // 2
            if 0 <= rr < canvas.shape[0] and 0 <= cc < canvas.shape[1]:
                canvas[rr, cc] = (0, 0, 0)

    return canvas


__all__ = ["state_to_rgb"]
