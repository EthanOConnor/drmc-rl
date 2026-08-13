"""Stable conditioning shared by human-policy training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class HumanSkillCondition:
    """Map Fightcade WHR-C onto a compact, bounded continuous feature vector.

    The quadratic term lets a small policy express different behavior at the
    tails without discretizing players into rating buckets. Values outside the
    observed range are clamped: the model is not licensed to extrapolate.
    """

    mean: float
    scale: float
    minimum: float
    maximum: float

    @classmethod
    def fit(cls, ratings: np.ndarray) -> "HumanSkillCondition":
        values = np.asarray(ratings, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("cannot fit skill conditioning without ratings")
        scale = float(values.std())
        return cls(
            mean=float(values.mean()),
            scale=max(scale, 1.0),
            minimum=float(values.min()),
            maximum=float(values.max()),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HumanSkillCondition":
        return cls(**{key: float(value[key]) for key in ("mean", "scale", "minimum", "maximum")})

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def encode(self, rating: float | np.ndarray) -> np.ndarray:
        values = np.asarray(rating, dtype=np.float32)
        clipped = np.clip(values, self.minimum, self.maximum)
        z = (clipped - self.mean) / self.scale
        return np.stack((z, z * z), axis=-1).astype(np.float32, copy=False)

    def resolve(self, rating: float) -> tuple[float, bool]:
        requested = float(rating)
        resolved = float(np.clip(requested, self.minimum, self.maximum))
        return resolved, resolved != requested
