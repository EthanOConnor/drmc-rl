"""Utilities to visualize the 14-channel state tensor and (optionally) overlay on pixels."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def grid_show(state: np.ndarray, title: str = "state", savepath: str | None = None):
    """Show a tiled grid of the 14 channels (C,H,W) as 2D images."""
    assert state.ndim == 3 and state.shape[0] == 14, f"expected (14,H,W), got {state.shape}"
    C, H, W = state.shape
    cols = 7
    rows = (C + cols - 1) // cols
    fig_w = 2.5 * cols
    fig_h = 2.5 * rows
    plt.figure(figsize=(fig_w, fig_h))
    for i in range(C):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(state[i], interpolation='nearest')
        ax.set_title(f"ch {i}")
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=140)
        print(f"saved {savepath}")
    else:
        plt.show()


def overlay_on_rgb(rgb: np.ndarray, state: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Overlay key planes on an RGB image for eyeballing correctness.
    - viruses (0..2) tint red, fixed (3..5) green, falling (6..8) blue.
    Returns an RGB array the same size as input.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    H, W, _ = rgb.shape
    C, h, w = state.shape
    if (h, w) != (H, W):
        # require cv2 to resize if available
        try:
            import cv2  # type: ignore
            state_resized = np.zeros((C, H, W), dtype=state.dtype)
            for c in range(C):
                state_resized[c] = cv2.resize(state[c], (W, H), interpolation=cv2.INTER_NEAREST)
            state = state_resized
        except Exception:
            pass
    out = rgb.astype(np.float32).copy()
    virus = np.clip(state[0] + state[1] + state[2], 0, 1)[..., None]
    fixed = np.clip(state[3] + state[4] + state[5], 0, 1)[..., None]
    fall = np.clip(state[6] + state[7] + state[8], 0, 1)[..., None]
    tint = np.concatenate([virus, fixed, fall], axis=-1)
    out = (1 - alpha) * out + alpha * 255.0 * tint
    return np.clip(out, 0, 255).astype(np.uint8)

