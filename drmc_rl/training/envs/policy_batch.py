from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PlacementPolicyBatch:
    """Typed, allocation-light policy inputs exposed by native vector envs."""

    feasible_mask: np.ndarray
    cost_to_lock: np.ndarray
    pill_colors: np.ndarray
    preview_pill_colors: np.ndarray
    aux: np.ndarray | None
