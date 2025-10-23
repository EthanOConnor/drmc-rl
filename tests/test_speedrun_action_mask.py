import numpy as np
import pytest

pytest.importorskip("torch")

from training.speedrun_experiment import (
    _extract_action_mask,
    _placement_action_reachable,
)


def test_extract_action_mask_flattens_nested_sequences():
    info = {"placements/feasible_mask": [[True, False, True, False]]}
    mask = _extract_action_mask(info)
    assert mask is not None
    assert mask.shape == (4,)
    assert mask.dtype == np.bool_
    assert mask.tolist() == [True, False, True, False]


def test_cached_action_remains_reachable_with_flattened_mask():
    mask = np.array([[True, False, True, True]])
    info = {"placements/feasible_mask": mask}

    assert _placement_action_reachable(info, 0) is True
    assert _placement_action_reachable(info, 1) is False
    assert _placement_action_reachable(info, 2) is True

