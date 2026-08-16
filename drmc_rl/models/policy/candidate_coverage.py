"""Candidate-width evidence and hard no-truncation guards."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class CandidateCoverage:
    rows: int
    maximum_legal: int
    configured_width: int
    dropped_rows: int
    dropped_candidates: int

    @property
    def complete(self) -> bool:
        return self.dropped_candidates == 0

    def to_dict(self) -> dict[str, int | bool]:
        return {**asdict(self), "complete": self.complete}


def candidate_coverage(
    feasible_mask: np.ndarray | torch.Tensor | Sequence[object],
    *,
    max_candidates: int,
) -> CandidateCoverage:
    """Measure exact feasible counts before any candidate packing or sorting."""

    if isinstance(feasible_mask, torch.Tensor):
        mask = feasible_mask.detach().bool().reshape(feasible_mask.shape[0], -1)
        counts = mask.sum(dim=1).cpu().numpy().astype(np.int64)
    else:
        array = np.asarray(feasible_mask, dtype=np.bool_)
        if array.ndim < 2:
            raise ValueError("feasible mask must have a batch dimension")
        counts = array.reshape(array.shape[0], -1).sum(axis=1, dtype=np.int64)
    width = max(1, int(max_candidates))
    excess = np.maximum(counts - width, 0)
    return CandidateCoverage(
        rows=int(len(counts)),
        maximum_legal=int(counts.max(initial=0)),
        configured_width=width,
        dropped_rows=int((excess > 0).sum()),
        dropped_candidates=int(excess.sum()),
    )


def require_complete_candidate_coverage(
    feasible_mask: np.ndarray | torch.Tensor | Sequence[object],
    *,
    max_candidates: int,
    context: str = "candidate packing",
) -> CandidateCoverage:
    evidence = candidate_coverage(feasible_mask, max_candidates=max_candidates)
    if not evidence.complete:
        raise RuntimeError(
            f"{context} would drop {evidence.dropped_candidates} legal candidates "
            f"across {evidence.dropped_rows} rows; maximum legal count "
            f"{evidence.maximum_legal} exceeds width {evidence.configured_width}"
        )
    return evidence


__all__ = [
    "CandidateCoverage",
    "candidate_coverage",
    "require_complete_candidate_coverage",
]
