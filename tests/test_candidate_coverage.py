import numpy as np
import pytest

from drmc_rl.models.policy.candidate_coverage import (
    candidate_coverage,
    require_complete_candidate_coverage,
)


def test_candidate_coverage_reports_and_rejects_drops() -> None:
    mask = np.zeros((2, 4, 16, 8), dtype=bool)
    mask[0].reshape(-1)[:5] = True
    mask[1].reshape(-1)[:9] = True
    evidence = candidate_coverage(mask, max_candidates=8)
    assert evidence.maximum_legal == 9
    assert evidence.dropped_candidates == 1
    with pytest.raises(RuntimeError, match="would drop 1"):
        require_complete_candidate_coverage(mask, max_candidates=8)
