"""Rating-residualized player/style latent space for independently tunable style."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class StyleSpace:
    schema: str
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    rating_mean: float
    rating_scale: float
    rating_coefficients: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    player_ids: np.ndarray
    player_embeddings: np.ndarray

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        ratings: np.ndarray,
        player_ids: np.ndarray,
        *,
        feature_names: Sequence[str] | None = None,
        dimensions: int = 6,
        min_decisions_per_player: int = 50,
    ) -> "StyleSpace":
        x = np.asarray(features, dtype=np.float64)
        rating = np.asarray(ratings, dtype=np.float64).reshape(-1)
        players = np.asarray(player_ids).reshape(-1)
        if x.ndim != 2 or len(x) != len(rating) or len(x) != len(players):
            raise ValueError("features, ratings, and player_ids must share the row dimension")
        valid = np.isfinite(x).all(axis=1) & np.isfinite(rating)
        x, rating, players = x[valid], rating[valid], players[valid]
        if len(x) < 2:
            raise ValueError("not enough valid style rows")
        names = tuple(feature_names or (f"feature_{index}" for index in range(x.shape[1])))
        if len(names) != x.shape[1]:
            raise ValueError("feature_names must match feature width")
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        standardized = (x - mean) / scale
        rating_mean = float(rating.mean())
        rating_scale = float(max(rating.std(), 1.0))
        z = (rating - rating_mean) / rating_scale
        design = np.column_stack((np.ones(len(z)), z, z * z))
        coefficients, *_ = np.linalg.lstsq(design, standardized, rcond=None)
        residual = standardized - design @ coefficients

        unique, inverse, counts = np.unique(players, return_inverse=True, return_counts=True)
        keep = counts >= int(min_decisions_per_player)
        kept_ids = unique[keep]
        if len(kept_ids) < 2:
            raise ValueError("not enough players meet min_decisions_per_player")
        matrix = np.vstack([residual[players == player].mean(axis=0) for player in kept_ids])
        matrix -= matrix.mean(axis=0, keepdims=True)
        _u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
        dims = min(int(dimensions), len(kept_ids) - 1, x.shape[1])
        components = vt[:dims]
        embeddings = matrix @ components.T
        variance = singular[:dims] ** 2
        explained = variance / max(float((singular**2).sum()), 1e-12)
        return cls(
            schema="drmc-style-space-v1",
            feature_names=names,
            feature_mean=mean.astype(np.float32),
            feature_scale=scale.astype(np.float32),
            rating_mean=rating_mean,
            rating_scale=rating_scale,
            rating_coefficients=coefficients.astype(np.float32),
            components=components.astype(np.float32),
            explained_variance=explained.astype(np.float32),
            player_ids=kept_ids,
            player_embeddings=embeddings.astype(np.float32),
        )

    @property
    def dimensions(self) -> int:
        return int(self.components.shape[0])

    def residualize(self, features: np.ndarray, ratings: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        rating = np.asarray(ratings, dtype=np.float64).reshape(-1)
        if x.ndim != 2 or len(x) != len(rating) or x.shape[1] != len(self.feature_names):
            raise ValueError("style residualization shape mismatch")
        standardized = (x - self.feature_mean) / self.feature_scale
        z = (rating - self.rating_mean) / self.rating_scale
        design = np.column_stack((np.ones(len(z)), z, z * z))
        return standardized - design @ self.rating_coefficients

    def encode(self, features: np.ndarray, ratings: np.ndarray) -> np.ndarray:
        return (self.residualize(features, ratings) @ self.components.T).astype(np.float32)

    def candidate_style_features(self, features: np.ndarray, *, rating: float) -> np.ndarray:
        rows = np.asarray(features, dtype=np.float64)
        ratings = np.full(len(rows), float(rating), dtype=np.float64)
        return self.encode(rows, ratings)

    def nearest_players(self, style: Sequence[float], *, limit: int = 10) -> list[tuple[object, float]]:
        vector = np.asarray(style, dtype=np.float64).reshape(self.dimensions)
        distance = np.linalg.norm(self.player_embeddings - vector, axis=1)
        order = np.argsort(distance, kind="stable")[: max(1, int(limit))]
        return [(self.player_ids[index].item(), float(distance[index])) for index in order]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "feature_names": list(self.feature_names),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "rating_mean": self.rating_mean,
            "rating_scale": self.rating_scale,
            "rating_coefficients": self.rating_coefficients.tolist(),
            "components": self.components.tolist(),
            "explained_variance": self.explained_variance.tolist(),
            "player_ids": self.player_ids.tolist(),
            "player_embeddings": self.player_embeddings.tolist(),
        }

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "StyleSpace":
        if value.get("schema") != "drmc-style-space-v1":
            raise ValueError("unsupported style-space schema")
        return cls(
            schema="drmc-style-space-v1",
            feature_names=tuple(str(item) for item in value["feature_names"]),  # type: ignore[arg-type]
            feature_mean=np.asarray(value["feature_mean"], dtype=np.float32),
            feature_scale=np.asarray(value["feature_scale"], dtype=np.float32),
            rating_mean=float(value["rating_mean"]),
            rating_scale=float(value["rating_scale"]),
            rating_coefficients=np.asarray(value["rating_coefficients"], dtype=np.float32),
            components=np.asarray(value["components"], dtype=np.float32),
            explained_variance=np.asarray(value["explained_variance"], dtype=np.float32),
            player_ids=np.asarray(value["player_ids"]),
            player_embeddings=np.asarray(value["player_embeddings"], dtype=np.float32),
        )


__all__ = ["StyleSpace"]
